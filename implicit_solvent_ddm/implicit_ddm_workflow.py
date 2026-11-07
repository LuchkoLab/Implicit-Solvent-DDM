"""
Double Decoupling Method (DDM) workflow for free energy calculations.
"""

# from implicit_solvent_ddm.remd import run_remd
import logging
import os
import os.path
import re
import time

from pathlib import Path

import numpy as np
import yaml
from toil.common import Toil
from toil.job import Job, JobFunctionWrappingJob

from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.workflow_phases import (
    setup_workflow_components,
    run_endstate_simulations,
    decompose_system_and_generate_restraints,
    setup_intermediate_simulations,
    compute_free_energy_and_consolidate,
    run_intermediate_and_post,
    _aggregate_post_output,
    adaptive_restraint_pilot,
    merge_pilot_windows,
    initilized_jobs,
)

logger = logging.getLogger(__name__)
working_directory = os.getcwd()


def ddm_workflow(
    job: JobFunctionWrappingJob, config: Config
) -> tuple[Config, Config, Config]:
    """
    Double Decoupling Method (DDM) workflow for free energy calculations.

    This workflow performs a complete DDM calculation including:
    1. Setup and preparation of simulation components
    2. Endstate simulations (complex, receptor, ligand)
    3. System decomposition and restraint generation
    4. Intermediate state simulations with alchemical transformations
    5. Post-processing and analysis

    Parameters
    ----------
    job : JobFunctionWrappingJob
        Toil job wrapper for workflow execution
    config : Config
        Configuration object containing all simulation parameters

    Returns
    -------
    tuple[Config, Config, Config]
        Updated configuration objects for complex, ligand, and receptor systems
    """

    # Phase 1: Setup and Preparation
    setup_jobs = job.addChildJobFn(setup_workflow_components, config)
    updated_config = setup_jobs.rv()

    setup_jobs.addFollowOnJobFn(
        initilized_jobs, 
        message="✓ Phase 1 Complete: Workflow components setup finished"
    )
    setup_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="--> Moving to phase 2: Endstate Simulations"
    )
    
    # Phase 2: Endstate Simulations (depends on setup)
    endstate_jobs = setup_jobs.addFollowOnJobFn(
        run_endstate_simulations, 
        updated_config
    )
    endstate_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="✓ Phase 2 Complete: Endstate simulations finished"
    )
    endstate_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="--> Moving to phase 3: System Decomposition and Restraint Generation"
    )
    # Phase 3: System Decomposition and Restraint Generation (depends on endstate)
    decomposition_jobs = endstate_jobs.addFollowOnJobFn(
        decompose_system_and_generate_restraints,
        endstate_jobs.rv(),
        updated_config
    )
    decomposition_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="✓ Phase 3 Complete: System decomposition and restraint generation finished"
    )
    decomposition_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="--> Moving to phase 4: Intermediate State Simulations"
    )
    # Phase 4: Intermediate State Simulations (depends on decomposition)
    setup_intermediate_jobs = decomposition_jobs.addFollowOnJobFn(
        setup_intermediate_simulations,
        decomposition_jobs.rv(), 
        endstate_jobs.rv(),
        updated_config
    )

    setup_intermediate_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="✓ Phase 4 Complete: Intermidate simulations setup finished"
    )
    setup_intermediate_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="--> Moving to phase 5: Intermediate State Simulations"
    )

    # Phase 4.5: Adaptive Lambda Scheduler pilot (flag-gated, CLOSED LOOP). Runs a short-MD pilot of the
    # complex + receptor legs, drives the per-band R-ADD scheduler (dielectric -> charge -> restraints) to
    # convergence, then REBUILDS the production setups from the converged dielectric + charge schedule so
    # Phases 5/6/7 run full-length MD on the pilot-determined window count. Flag off -> no pilot job, Phase
    # 5 reads the Phase-4 seed setups exactly as before (byte-identical DAG).
    phase4_tail = setup_intermediate_jobs
    setup_source = setup_intermediate_jobs  # which setup job feeds Phases 5/6/7
    if config.workflow.adaptive_lambda:
        pilot = setup_intermediate_jobs.addFollowOnJobFn(
            adaptive_restraint_pilot,
            decomposition_jobs.rv(),
            endstate_jobs.rv(),
            updated_config,
        )
        # Apply the converged dielectric + charge + restraint windows onto a fresh production config
        # (full mdin, production dir).
        merged = pilot.addFollowOnJobFn(merge_pilot_windows, updated_config, pilot.rv())
        # Re-run Phase 3 on the merged schedule so RestraintMaker materializes a restraint file for every
        # pilot-inserted exponent (the original seed RestraintMaker only carries seed-window files, so an
        # inserted con/orient window's restraint_key would not resolve). Anchor protection keeps inserts
        # inside the seed (min, max), so binding modes, max_*_restraint, and the Boresch ΔG are unchanged
        # — this only adds the inserted interior restraint files. Then rebuild the production setups from
        # the merged config + re-generated restraints.
        redecomposition_jobs = merged.addFollowOnJobFn(
            decompose_system_and_generate_restraints,
            endstate_jobs.rv(),
            merged.rv(),
        )
        setup_source = redecomposition_jobs.addFollowOnJobFn(
            setup_intermediate_simulations,
            redecomposition_jobs.rv(),
            endstate_jobs.rv(),
            merged.rv(),
        )
        phase4_tail = setup_source

    # Phases 5 + 6 (merged): submit all intermediate MD windows first and couple each window's
    # post-analysis row to its OWN MD job -- dissolving the global Phase5->Phase6 barrier so the
    # dominant N^2 CPU post-analysis backfills cores as trajectories land. Returns
    # (complex, receptor, ligand, flat_bottom) post-output bundles.
    merged_jobs = phase4_tail.addFollowOnJobFn(
        run_intermediate_and_post,
        setup_source.rv(0), # config
        setup_source.rv(1), # complex simulations
        setup_source.rv(2), # receptor simulations
        setup_source.rv(3), # ligand simulations
        setup_source.rv(4), # flat bottom simulations
    )
    merged_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="✓ Phases 5-6 Complete: Intermediate MD + energy post-processing finished"
    )

    if config.workflow.md_only:
        # No post rows were created, so there is nothing to aggregate or solve.
        merged_jobs.addFollowOnJobFn(
            initilized_jobs,
            message="✓ md_only: MD complete. Re-run with md_only off and the same "
                    "output_directory_name to score the trajectories."
        )
        return merged_jobs

    # Aggregate: flatten each system's per-window post rows into one .post_output bundle. Wired as a
    # FOLLOW-ON of the merged dispatcher, so it waits for the dispatcher's entire MD+post subtree
    # (every post_runner.rv() in merged_jobs.rv(0..3) is resolved before it runs).
    aggregate_jobs = merged_jobs.addFollowOnJobFn(
        _aggregate_post_output,
        merged_jobs.rv(0), # complex per-window post rows
        merged_jobs.rv(1), # receptor per-window post rows
        merged_jobs.rv(2), # ligand per-window post rows
        merged_jobs.rv(3), # flat bottom per-window post rows
    )
    aggregate_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="--> Moving to phase 7: Free energy computation and consolidation"
    )

    # Phase 7: Compute Free Energy and Consolidate Results. Follow-on of the AGGREGATOR (consumes its
    # rv), so Phase 7 waits for aggregation -> the .post_output bundles are resolved.
    free_energy_difference_jobs = aggregate_jobs.addFollowOnJobFn(
        compute_free_energy_and_consolidate,
        aggregate_jobs.rv(0), # complex post-output bundle
        aggregate_jobs.rv(1), # receptor post-output bundle
        aggregate_jobs.rv(2), # ligand post-output bundle
        aggregate_jobs.rv(3), # flat bottom post-output bundle
        setup_source.rv(0), # config
    )

    free_energy_difference_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="✓ Phase 7 Complete: Free energy computation and consolidation finished"
    )

    # Final workflow completion message
    
    free_energy_difference_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="🎉 DDM Workflow Complete: All phases finished successfully!"
    )
    
    return free_energy_difference_jobs


def _allocated_gpu_count() -> int:
    """Number of GPUs allocated to this job, from the most reliable signal available
    in the *batch-script* environment.

    On this cluster SLURM_STEP_GPUS / CUDA_VISIBLE_DEVICES are only set inside srun
    steps, not in the batch script, but SLURM_GPUS_ON_NODE (a count) is -- and the
    cgroup still exposes the allocated GPUs to the process. Fall back to nvidia-smi,
    which inside the cgroup lists exactly the allocated devices.
    """
    on_node = os.environ.get("SLURM_GPUS_ON_NODE", "")
    if on_node.strip().isdigit():
        return int(on_node)
    for var in ("CUDA_VISIBLE_DEVICES", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        ids = [p for p in os.environ.get(var, "").split(",") if p.strip() != ""]
        if ids:
            return len(ids)
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        return sum(1 for ln in out.stdout.splitlines() if ln.strip().startswith("GPU "))
    except Exception:
        return 0


def _confine_single_machine_to_allocated_gpus(options):
    """Place one GPU MD window per Slurm-allocated GPU under Toil's single_machine batch system.

    When this workflow runs ``single_machine`` inside one Slurm allocation
    (``--gres=gpu:N``), three behaviours in Toil 8.2.0 (still present in 9.5.0,
    verified against the installed source) break GPU isolation, so every
    ``pmemd.cuda`` window lands on the first GPU while the others idle:

      1. ``toil/worker.py`` restores the leader's pickled ``os.environ`` onto each
         worker and ``CUDA_VISIBLE_DEVICES`` is not in its ``env_reject`` set, so
         the allocation-wide value the leader inherited from Slurm (e.g. ``"0,1"``)
         overwrites the per-job pin the batch system set on the worker -- Amber
         then defaults to device 0 for every window.
      2. ``toil.lib.accelerators.get_individual_local_accelerators`` counts *all*
         physical GPUs via ``nvidia-smi`` (ignores the allocation), so Toil would
         place windows on GPUs we were never granted.
      3. ``get_restrictive_environment_for_local_accelerators`` writes the bare
         acquired slot index rather than the real allocated GPU id.

    We correct all three at the leader, before the batch system is built, so Toil
    keeps global ownership of GPU assignment.  (A per-runner round-robin would
    double-book GPU 0 across concurrently-running runners -- see runner.py.)
    """
    if getattr(options, "batchSystem", None) not in (None, "single_machine"):
        # Other batch systems (e.g. slurm) submit each job separately and let the
        # scheduler pin GPUs per sub-job; this single-allocation fix-up is moot.
        return

    # GPUs this process may use, in ITS OWN namespace. Under cgroup-constrained
    # Slurm (this cluster) the allocated GPUs are exposed to the process as logical
    # ids 0..N-1. If CUDA_VISIBLE_DEVICES is present (batch scripts that get it, or
    # srun steps) use it directly; otherwise fall back to the GPU *count* and use
    # logical ids 0..N-1 -- this is what makes it work when the batch script has no
    # SLURM_*_GPUS/CVD set (those are only set per srun step on this cluster).
    cvd_ids = [p for p in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
               if p.strip().isdigit()]
    if cvd_ids:
        allocated_gpus = cvd_ids
        _src = "CUDA_VISIBLE_DEVICES"
    else:
        allocated_gpus = [str(i) for i in range(_allocated_gpu_count())]
        _src = "count->logical"
    # print() (not logger): this runs before Toil configures logging, so INFO would
    # be swallowed. This always lands in the job's stdout (.out).
    print(
        "[GPU] allocation detect: "
        f"SLURM_STEP_GPUS={os.environ.get('SLURM_STEP_GPUS')!r} "
        f"SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS')!r} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
        f"SLURM_GPUS_ON_NODE={os.environ.get('SLURM_GPUS_ON_NODE')!r} "
        f"-> allocated_gpus={allocated_gpus!r} (via {_src})",
        flush=True,
    )
    if not allocated_gpus:
        print(
            "[GPU] WARNING: no GPUs detected -> single_machine GPU pinning DISABLED; "
            "pmemd.cuda will fail with 'no CUDA-capable device'. Check the job's "
            "'#SBATCH --gres=gpu:N'.",
            flush=True,
        )
        return

    import toil.batchSystems.singleMachine as single_machine

    # (1) Keep the allocation-wide pin out of environment.pickle so it cannot
    #     clobber each worker's per-job CUDA_VISIBLE_DEVICES.
    for _var in (
        "CUDA_VISIBLE_DEVICES",
        "SINGULARITYENV_CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        os.environ.pop(_var, None)

    # (2) Advertise only the allocated GPUs so Toil runs one GPU window per GPU.
    def _allocated_accelerators():
        return [
            {"kind": "gpu", "brand": "nvidia", "api": "cuda", "count": 1}
            for _ in allocated_gpus
        ]

    # (3) Translate Toil's acquired slot back to the real allocated GPU id.
    #     Safe because Toil's accelerator slot space is range(len(allocated_gpus))
    #     -- it acquires from the same list advertised in _allocated_accelerators.
    def _restrictive_environment(acquired):
        try:
            gpu_list = ",".join(allocated_gpus[i] for i in sorted(acquired))
        except (IndexError, TypeError):
            # Unreachable under the invariant above; warn loudly rather than
            # silently emitting raw indices (which would re-introduce the GPU-0
            # double-booking this patch exists to fix).
            logger.warning(
                "[GPU] unexpected accelerator slots %r for allocation %r; "
                "using raw indices",
                acquired,
                allocated_gpus,
            )
            gpu_list = ",".join(str(i) for i in acquired)
        return {
            "CUDA_VISIBLE_DEVICES": gpu_list,
            "SINGULARITYENV_CUDA_VISIBLE_DEVICES": gpu_list,
        }

    single_machine.get_individual_local_accelerators = _allocated_accelerators
    single_machine.get_restrictive_environment_for_local_accelerators = (
        _restrictive_environment
    )

    logger.info(
        "[GPU] Toil single_machine confined to %d allocated GPU(s): %s",
        len(allocated_gpus),
        ",".join(allocated_gpus),
    )


def main():
    parser = Job.Runner.getDefaultArgumentParser()
    parser.add_argument(
        "--config_file",
        nargs="*",
        type=str,
        required=True,
        help="configuartion file with input parameters",
    )
    parser.add_argument(
        "--ignore_receptor",
        action="store_true",
        help=" Receptor MD caluculations with not be performed.",
    )
    options = parser.parse_args()
    options.clean = "onSuccess"
    # INFO is what carries the per-job [TIMING] records into the leader log that
    # timing_report.py parses; without it the timing breakdown is empty.
    options.logLevel = "INFO"
    config_file = options.config_file[0]
    ignore_receptor = options.ignore_receptor

    start = time.perf_counter()
    try:
        with open(config_file) as f:
            config_file = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)

    # setup configuration dataclass
    logger.info(f'[CONFIG] Loading in config')
    config = Config.from_config(config_file)
    logger.info(f'[CONFIG] Finished loading in config')

    # create top level directory to write output files
    if not os.path.exists(config.system_settings.top_directory_path):
        os.makedirs(config.system_settings.top_directory_path)

    complex_name = re.sub(
        r"\..*",
        "",
        os.path.basename(
            config_file["endstate_parameter_files"]["complex_parameter_filename"]
        ),
    )
    # create unique workflow log file
    job_number = 1
    while os.path.exists(
        f"{config.system_settings.top_directory_path}/{complex_name}_job_{job_number:03}.txt"
    ):
        job_number += 1
    Path(
        f"{config.system_settings.top_directory_path}/{complex_name}_job_{job_number:03}.txt"
    ).touch()

    options.logFile = f"{config.system_settings.top_directory_path}/{complex_name}_job_{job_number:03}.txt"
    # Pin one GPU window per Slurm-allocated GPU and stop the leader environment
    # from clobbering each worker's per-job GPU assignment (Toil single_machine).
    _confine_single_machine_to_allocated_gpus(options)
    # setup toil workflow
    with Toil(options) as toil:
        config.workflow.ignore_receptor_endstate = ignore_receptor

        # log the performance time
        file_handler = logging.FileHandler(
            os.path.join(
                config.system_settings.top_directory_path,
                f"{complex_name}_{job_number}_workflow_performance.log",
            ),
            mode="w",
        )
        formatter = logging.Formatter(
            "%(asctime)s~%(levelname)s~%(message)s~module:%(module)s"
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

        if not toil.options.restart:
            config.endstate_files.toil_import_parameters(toil=toil)
            config.intermediate_args.toil_import_user_mdin(toil=toil)
            # if the user doesn't provide there own endstate simulation
            if config.endstate_method.endstate_method_type != 0:
                # import files for remd
                if config.endstate_method.endstate_method_type == "remd":
                    config.endstate_method.remd_args.toil_import_replica_mdin(toil=toil)
                # import files for basic MD
                else:
                    config.endstate_method.basic_md_args.toil_import_basic_mdin(
                        toil=toil
                    )

            if config.intermediate_args.guest_restraint_files is not None:
                config.intermediate_args.toil_import_user_restraints(toil=toil)

            config.inputs["min_mdin"] = str(
                toil.import_file(
                    "file://"
                    + os.path.abspath(
                        os.path.dirname(os.path.realpath(__file__))
                        + "/templates/min.mdin"
                    )
                )
            )
            logger.info(f"config.endstate_files.complex_parameter_filename: {config.endstate_files.complex_parameter_filename}")
            update_config = toil.start(Job.wrapJobFn(ddm_workflow, config))
            logger.info(
                f" Total workflow time: {time.perf_counter() - start} seconds\n"
            )

        else:
            toil.restart()


if __name__ == "__main__":
    main()
