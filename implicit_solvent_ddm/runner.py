import os
import re
import time
from email import message
from importlib.metadata import files
from pathlib import Path
from sre_constants import ANY
from typing import Optional, Type, TypedDict, Union

import pandas as pd
from matplotlib.backend_bases import key_press_handler
from toil.batchSystems import abstractBatchSystem
from toil.job import FileID, Job, JobFunctionWrappingJob, PromisedRequirement

from implicit_solvent_ddm import block_mbar
from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.matrix_order import CycleSteps
from implicit_solvent_ddm.postTreatment import create_mdout_dataframe
from implicit_solvent_ddm.restraints import RestraintMaker
from implicit_solvent_ddm.simulations import Simulation
from itertools import islice


def chunked(iterable, size):
    """Yield successive chunks from iterable of given size."""
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])


class IntermidateRunner(Job):
    """
    Manages and executes MD simulations and post-analysis steps for a given system phase.

    This runner class handles the full lifecycle for a batch of simulation jobs 
    (e.g., for ligand, receptor, or complex), including:
    - Launching intermediate MD simulations (if not already run),
    - Collecting and caching output data (parsed from mdout),
    - Running post-analysis jobs (e.g., energy decomposition),
    - Supporting adaptive workflows and optional "post-only" modes.

    Parameters
    ----------
    simulations : list of Simulation
        List of simulation objects to run. Each one encapsulates the inputs for a ligand, receptor, or complex phase.
    restraints : RestraintMaker
        Object that manages creation of restraint files or logic for the simulations.
    post_process_no_solv_mdin : FileID
        Input file for post-processing jobs that should exclude implicit solvent (e.g., for igb=6).
    post_process_mdin : FileID
        Standard input file for post-analysis energy evaluation (e.g., sander).
    post_process_saltfree_mdin : FileID, optional
        Scoring input file for gb_dielectric columns (saltcon=0). Falls back to
        ``post_process_mdin`` when None, so resumed jobstores keep working.
    post_process_distruct : str
        Directory structure key or identifier used to organize post-processing job outputs.
    post_only : bool
        If True, skips MD simulations and runs post-analysis only.
    config : Config
        Global configuration object for the workflow.
    adaptive : bool, optional
        Enables adaptive lambda window or restraint scheduling if True.
    loaded_dataframe : list, optional
        Tracks previously parsed output directories to avoid reprocessing.
    post_output : list or list of pd.DataFrame, optional
        Stores collected energy analysis output dataframes from post-processing.
    memory : int or str, optional
        Memory allocation for the job (used by the underlying workflow engine).
    cores : int or float or str, optional
        Number of CPU cores to request for each job.
    disk : int or str, optional
        Disk allocation for the job (used by the underlying workflow engine).
    preemptable : bool or int or str, optional
        Flag to mark the job as preemptable (depending on scheduler).
    unitName : str, optional
        Name for job unit (optional; used in workflow diagnostics).
    checkpoint : bool, optional
        If True, enables checkpointing of the job state.
    displayName : str, optional
        Custom name for the job (for logs/monitoring).
    descriptionClass : str, optional
        Optional tag or label for job type.

    Attributes
    ----------
    post_output : list
        Contains parsed pandas DataFrames of energy terms from post-analysis jobs.
    ligand_output : list
        Placeholder list for ligand-specific outputs.
    receptor_output : list
        Placeholder list for receptor-specific outputs.
    complex_output : list
        Placeholder list for complex-specific outputs.
    _loaded_dataframe : list
        Tracks directories already parsed to prevent duplicate processing.

    Notes
    -----
    - Each Simulation object is checked for output; if missing, MD is run first.
    - If `post_only` is True, only analysis is performed using available trajectories.
    - Uses `sander.MPI` for post-processing and `create_mdout_dataframe` for parsing outputs.
    - Designed for use in generalized workflows involving restraint-free energies or DDM.

    Returns
    -------
    self : IntermidateRunner
        Returns itself to support chaining or retrieval in a workflow graph.
    """

    # simulations: dict[Simulation, int]
    def __init__(
        self,
        simulations: list[Simulation],
        restraints: RestraintMaker,
        post_process_no_solv_mdin: FileID,
        post_process_mdin: FileID,
        post_process_distruct: str,
        post_only: bool,
        config: Config,
        post_process_saltfree_mdin: Optional[FileID] = None,
        adaptive: bool = False,
        loaded_dataframe: Optional[list] = None,
        post_output: Optional[Union[list, list[pd.DataFrame]]] = None,
        traj_map: Optional[dict] = None,
        restrict_completed: Optional[set] = None,
        memory: Optional[Union[int, str]] = None,
        cores: Optional[Union[int, float, str]] = None,
        disk: Optional[Union[int, str]] = None,
        preemptable: Optional[Union[bool, int, str]] = None,
        unitName: Optional[str] = "",
        checkpoint: Optional[bool] = False,
        displayName: Optional[str] = "",
        descriptionClass: Optional[str] = None,
    ) -> None:
        super().__init__(
            memory,
            cores,
            disk,
            accelerators=None,
            preemptible="false",
            unitName=unitName,
            checkpoint=checkpoint,
            displayName=displayName,
        )

        self.simulations = simulations
        self.restraints = restraints
        self.no_solvent_mdin = post_process_no_solv_mdin
        self.mdin = post_process_mdin
        self.saltfree_mdin = post_process_saltfree_mdin
        self.post_only = post_only
        self.config = config
        self.adaptive = adaptive
        # Mutable-default fix: fresh per-instance lists, so production runners don't alias one
        # shared module-level list. Callers that WANT to share by reference (e.g. new_runner
        # threading post_output across pilot iterations) pass explicit non-None lists.
        self.post_output = post_output if post_output is not None else []
        self.ligand_output = []
        self.receptor_output = []
        self.complex_output = []
        self._loaded_dataframe = loaded_dataframe if loaded_dataframe is not None else []
        # {output_dir -> trajectory FileID promise}. Shared by reference across pilot
        # passes (threaded via new_runner) exactly like post_output / _loaded_dataframe, so
        # each post pass can set inptraj straight from the jobStore and never read the
        # network output_dir. Accumulates every window's trajectory across pilot iterations.
        self.traj_map = traj_map if traj_map is not None else {}
        # Per-window post-analysis scoping (merged MD->post pipeline). When set, the post_only pass
        # scores ONLY the completed window(s) whose output_dir is in this set -- the trajectory-row
        # for that window -- leaving the inner Hamiltonian loop (every state's re-score of that
        # trajectory) intact. None (default) preserves the legacy whole-list behaviour, so the ALS
        # pilot / new_runner paths are byte-identical. Robust to on-disk state: unlike a lone
        # traj_map entry, this filter holds even on warm/export-on runs where other windows' mdouts
        # exist on disk.
        self.restrict_completed = restrict_completed
        self.post_process_distruct = post_process_distruct
        # {system_type -> set of (traj_state, hamiltonian_state)} for banded post-analysis.
        # Built lazily per system_type in _post_analysis_pairs, None when banding is off.
        self._block_pairs_cache: dict = {}

    def _post_analysis_pairs(self, system_type: str, fileStore) -> Optional[set]:
        """Return the (trajectory, Hamiltonian) pairs post-analysis should evaluate.

        Parameters
        ----------
        system_type : str
            ``"complex"``, ``"ligand"`` or ``"receptor"``.
        fileStore : FileStore-like
            Used for logging.

        Returns
        -------
        tuple of (set, set) or None
            ``(required_pairs, known_states)`` for the configured block chain, or None when
            banding is off and the full N^2 should be evaluated.
        """
        block_size = getattr(
            self.config.intermediate_args, "post_analysis_block_size", None
        )
        if not block_size:
            return None
        if system_type in self._block_pairs_cache:
            return self._block_pairs_cache[system_type]

        cycle_steps = CycleSteps(
            conformation_forces=self.config.intermediate_args.exponent_conformational_forces_list,
            orientational_forces=self.config.intermediate_args.exponent_orientational_forces_list,
            charges_windows=self.config.intermediate_args.charges_lambda_window,
            external_dielectic=self.config.intermediate_args.gb_extdiel_windows,
        )
        cycle_steps.round(3)
        order = {
            "complex": cycle_steps.complex_order,
            "ligand": cycle_steps.ligand_order,
            "receptor": cycle_steps.receptor_order,
        }[system_type]

        bounds = block_mbar.band_boundaries(order)
        pairs = block_mbar.required_pairs(order, block_size, bounds)
        fileStore.logToMaster(
            f"[block_mbar] {system_type}: block_size={block_size}, {len(order)} states -> "
            f"{len(pairs)} evaluations instead of {len(order) ** 2} "
            f"({len(order) ** 2 / len(pairs):.1f}x fewer sander jobs)"
        )
        self._block_pairs_cache[system_type] = (pairs, set(order))
        return self._block_pairs_cache[system_type]

    @staticmethod
    def _column_is_required(source_key, target_key, pairs) -> bool:
        """Should this (trajectory, Hamiltonian) cell be evaluated under a block chain?

        Parameters
        ----------
        source_key : tuple
            Canonical state tuple of the completed trajectory.
        target_key : tuple or None
            Canonical state tuple of the Hamiltonian, or None when it resolves to no state
            in the cycle order.
        pairs : set of tuple
            Pairs the block chain needs, from ``_post_analysis_pairs``.

        Returns
        -------
        bool
            True to evaluate. An unresolved ``target_key`` returns True: skipping it would
            punch a hole in a block the chained solve later requires, surfacing as a
            ValueError in ``chained_mbar_result`` long after the cheap sander job could
            have filled it. One extra evaluation is the safe direction to err.
        """
        if target_key is None:
            return True
        return (source_key, target_key) in pairs

    def _select_post_mdin(self, post_simulation):
        """Pick the scoring mdin for one post-analysis column.

        Parameters
        ----------
        post_simulation : Simulation
            The window supplying the Hamiltonian this cell is scored under.

        Returns
        -------
        FileID
            ``no_solvent_mdin`` for gas (igb=6) columns; ``saltfree_mdin`` for
            ``gb_dielectric`` columns, whose MD is written salt-free by
            ``generate_extdiel_mdin`` so scoring must match it; ``mdin`` otherwise.

        Notes
        -----
        The gb_dielectric branch dispatches on ``state_label``, not ``igb_value``:
        ``setup_gb_external_dielectric`` sets the string ``"igb_2"`` while the ALS insertion
        path sets the int. ``saltfree_mdin`` is None on jobstores predating that fix, in
        which case this falls back to ``mdin`` (the original, salted behaviour).
        """
        if post_simulation.directory_args["igb_value"] == 6:
            return self.no_solvent_mdin
        if (
            post_simulation.directory_args.get("state_label") == "gb_dielectric"
            and self.saltfree_mdin is not None
        ):
            return self.saltfree_mdin
        return self.mdin

    def run(self, fileStore):
        """
        Submits molecular dynamics (MD) or post-processing jobs based on configuration.

        This method checks whether to run MD simulations (`post_only=False`) or to perform
        post-analysis on previously completed MD runs (`post_only=True`). In MD mode, all 
        simulations in `self.simulations` are submitted as Toil child jobs. In post-only mode,
        it checks if the corresponding MD outputs exist, then launches energy analysis jobs
        using `sander.MPI` and appends the results.

        Parameters
        ----------
        fileStore : toil.job.FileStore
            A Toil file store object for handling file access, logging, and output exporting 
            within the job store environment.

        Returns
        -------
        self : IntermidateRunner
            The current job instance, after scheduling MD or post-analysis jobs.

        Notes
        -----
        - When `post_only` is True, this function will skip simulations for which MD output 
        is missing or incomplete.
        - Trajectories from MD runs are imported into the job store before being passed to
        post-processing steps.
        - Simulations with `state_label == "no_flat_bottom"` are ignored entirely.
        - This function should be run twice in a complete workflow: first to schedule MD, then
        again with `post_only=True` to schedule analysis jobs after MD has completed.
        """
        fileStore.logToMaster(f"IntermidateRunner: Running a total of {len(self.simulations)} simulations")
        fileStore.logToMaster(f"post only is {self.post_only}")

        md_jobs = []

        # Collect simulations that still need to run
        for simulation in self.simulations:
            if simulation.directory_args.get("state_label") == "no_flat_bottom":
                continue

            if self.post_only:
                # Per-window scoping (merged MD->post pipeline): this runner is the trajectory-row
                # for exactly one completed window, so skip every OTHER window as a completed_sim.
                # The inner Hamiltonian loop in only_post_analysis is unaffected -- the surviving
                # window is still re-scored under every state. None -> legacy whole-list behaviour.
                if (
                    self.restrict_completed is not None
                    and simulation.output_dir not in self.restrict_completed
                ):
                    continue
                # No-network hand-off: if this window's trajectory was produced with export
                # off, its FileID is in the shared traj_map -> set inptraj straight from the
                # jobStore so the guard below is satisfied and _get_md_traj/os.listdir (the
                # network read) is never reached. Covers production Phase 6 and every pilot pass.
                if simulation.inptraj is None and simulation.output_dir in self.traj_map:
                    simulation.inptraj = self.traj_map[simulation.output_dir]
                # Post-analysis logic
                if self._check_mdout(simulation) or simulation.inptraj is not None:
                    if simulation.inptraj is None:
                        fileStore.logToMaster(f"Importing MD traj from: {simulation.output_dir}")
                        simulation.inptraj = [
                            fileStore.import_file(
                                "file://" + self._get_md_traj(simulation, fileStore)
                            )
                        ]
                    self.only_post_analysis(
                        completed_sim=simulation,
                        md_traj=simulation.inptraj,
                        fileStore=fileStore
                    )
                else:
                    fileStore.logToMaster(
                        f"[WARNING] Expected MD output missing for {simulation.output_dir}, skipping post-analysis."
                    )
            else:
                # MD execution logic
                if self._check_mdout(simulation) or simulation.inptraj is not None:
                    fileStore.logToMaster(f"[SKIP] MD already complete for: {simulation.output_dir}")
                    continue
                fileStore.logToMaster(f"Running MD for: {simulation.output_dir}")
                md_jobs.append(simulation)

        # Submit all MD jobs as children and let Toil schedule them.
        #
        # GPU simulations already declare ``accelerators=1`` (set in
        # setup_simulations.py for complex/receptor systems), so Toil's
        # accelerator-aware scheduler runs at most one GPU job per available GPU
        # and pins each job to a distinct device through CUDA_VISIBLE_DEVICES.
        #
        # We deliberately do NOT assign CUDA_VISIBLE_DEVICES or chain jobs per-GPU
        # by hand: run() executes once per system (complex, receptor, ligand,
        # flat-bottom) and those runners execute CONCURRENTLY, so a per-runner
        # round-robin (`gpu_id = i % num_gpus`) restarts at GPU 0 every time and
        # double-books device 0 — the bottleneck this replaces. Delegating to
        # Toil coordinates GPU usage globally across every runner.
        if md_jobs:
            fileStore.logToMaster(f"Submitting {len(md_jobs)} MD job(s) to the Toil scheduler")
        _export = self.config.system_settings.export_intermediate_files
        # Map output_dir -> trajectory FileID promise. sim.run() returns
        # (restart_ID, trajectory_ID); rv(1) is the trajectory FileID list. The
        # post-analysis pass (Phase 6) reads this map and sets each window's inptraj
        # so it re-scores straight from the jobStore -- the Toil-promise hand-off used
        # when export is off (no network trajectory copy). Carried on `self` and read
        # off the resolved runner exactly like self.post_output reaches MBAR. MERGE (do not
        # reset) so it accumulates every window's trajectory across pilot passes.
        for sim in md_jobs:
            sim.export_network = _export
            self.addChild(sim)
            self.traj_map[sim.output_dir] = sim.rv(1)

        return self
    
    def only_post_analysis(self, completed_sim: Simulation, md_traj, fileStore):
        """
        Run post-analysis calculations on a completed MD simulation.

        This function schedules post-processing jobs that analyze energy terms
        using existing trajectory and restart files.

        Parameters
        ----------
        completed_sim : Simulation
            A `Simulation` object representing the completed MD run to be analyzed.
            This provides context such as working directories and system parameters.
        md_traj : str
            Path to the input trajectory file (`.nc`, `.dcd`, etc.) generated by the completed MD simulation.
        fileStore : FileStore-like
            An object used for logging and managing output and job submission context (e.g., within a workflow engine).

        Returns
        -------
        None
            All results are appended to `self.post_output` and tracked via `self._loaded_dataframe`.

        Notes
        -----
        - If analysis output (`simulation_mdout.parquet.gzip`) already exists and is cached, the job is skipped.
        - Post-analysis jobs are scheduled using `sander.MPI`, and energy data is parsed into pandas DataFrames.
        - This method does not perform MD; it assumes all dynamics are already complete.
        """
        fileStore.logToMaster("RUNNING POST only\n")
        fileStore.logToMaster(f"loaded dataframe: {self._loaded_dataframe}")

        # Banded post-analysis: evaluate only the cells a K-state block chain needs.
        block_pairs = self._post_analysis_pairs(completed_sim.system_type, fileStore)
        source_key = None
        if block_pairs is not None:
            pairs, known_states = block_pairs
            source_key = block_mbar.canonical_state_key(
                completed_sim.directory_args, known_states
            )
            if source_key is None:
                # The completed window is not in the cycle order -- schedule/data drift, or a
                # state this runner does not own. Fall back to the full row rather than
                # silently dropping every evaluation for it.
                fileStore.logToMaster(
                    "[block_mbar] WARNING trajectory state "
                    f"{block_mbar.state_key_from_dirargs(completed_sim.directory_args)} "
                    f"is absent from the {completed_sim.system_type} cycle order; "
                    "scoring it against ALL states"
                )
                block_pairs = None

        for post_simulation in self.simulations:
            if block_pairs is not None:
                target_key = block_mbar.canonical_state_key(
                    post_simulation.directory_args, known_states
                )
                if target_key is None:
                    # Unrecognised Hamiltonian: evaluate it rather than skip. Skipping would
                    # punch a hole in a block the chained solve later requires, and that
                    # surfaces as a ValueError in chained_mbar_result long after the cheap
                    # sander job could have filled it. One extra evaluation is the safe error.
                    fileStore.logToMaster(
                        "[block_mbar] WARNING Hamiltonian state "
                        f"{block_mbar.state_key_from_dirargs(post_simulation.directory_args)} "
                        f"is absent from the {completed_sim.system_type} cycle order; "
                        "evaluating it rather than dropping the cell"
                    )
                if not self._column_is_required(source_key, target_key, pairs):
                    continue

            directory_args = post_simulation.directory_args.copy()
            #fileStore.logToMaster(f"directory args before update: {directory_args}\n")
            # fileStore.logToMaster(f"args {completed_sim.directory_args} & {md_traj}")
            directory_args.update(self.update_postprocess_dirstruct(completed_sim.directory_args))  # type: ignore
            #fileStore.logToMaster(f"directory args after update: {directory_args}\n")
            mdin = self._select_post_mdin(post_simulation)

            # run simulation if its not endstate with endstate
            post_dirstruct = self.get_system_dirs(post_simulation.system_type)
            fileStore.logToMaster(f"post dirstruct {post_dirstruct}\n")

            post_process_job = Simulation(
                executable="sander.MPI",
                mpi_command=post_simulation.mpi_command, #mpi_command=post_simulation.mpi_command,
                num_cores=1,
                CUDA=False,
                prmtop=post_simulation.prmtop,
                incrd=post_simulation.incrd,
                input_file=mdin,
                restraint_file=post_simulation.restraint_file,
                working_directory=post_simulation.working_directory,
                directory_args=directory_args,
                dirstruct=post_dirstruct,
                inptraj=md_traj,
                post_analysis=True,
                restraint_key=post_simulation.restraint_key,
                sim_debug=True,
            )

            if completed_sim.directory_args["runtype"] == "lambda_window":
                fileStore.logToMaster(f"COMPLETED MD simulation of lambda window")
                fileStore.logToMaster(
                    f"Using trajectory from {completed_sim.output_dir}\n"
                )

            # In-memory dedup (authoritative): skip any cell whose output_dir was already
            # scored/scheduled this run. Dedup previously hung off has_post_analysis_data (the
            # NETWORK parquet check); with export_intermediate_files=False that parquet never
            # exists, so without this guard the same cell (repeated labels, or the same cell
            # re-encountered across pilot passes that share _loaded_dataframe by reference) would
            # be recomputed and re-appended -> duplicate MBAR rows ("Index contains duplicates").
            if post_process_job.output_dir in self._loaded_dataframe:
                continue

            if not self.has_post_analysis_data(post_process_job.output_dir):
                fileStore.logToMaster(
                    f"simulations_mdout.parquet is not found in  {post_process_job.output_dir} or is empty\n"
                )

                fileStore.logToMaster(
                    f"RUNNING post analysis with inptraj trajecory: {md_traj}"
                )

                fileStore.logToMaster(
                    f"State potential energy {post_simulation.directory_args['state_label']}"
                )
                # fileStore.logToMaster(f"state args: {post_simulation.directory_args}")

                _export = self.config.system_settings.export_intermediate_files
                post_process_job.export_network = _export
                self.addChild(post_process_job)

                data_frame = post_process_job.addFollowOnJobFn(
                    create_mdout_dataframe,
                    post_process_job.directory_args,
                    post_process_job.dirstruct,
                    post_process_job.output_dir,
                    # When export is off the mdout is never written to the network;
                    # read it from the jobStore via the post job's returned FileID
                    # promise, and skip the parquet cache write (compress=False) so
                    # nothing touches output_dir. The dataframe still reaches MBAR via rv().
                    compress=_export,
                    mdout_id=(None if _export else post_process_job.rv()),
                )

                self.post_output.append(data_frame.rv())

            else:
                # Warm/resume: the parquet exists on disk (export was on in a prior run); load
                # it instead of re-scoring. The top-of-loop guard already handled the
                # already-loaded-this-run case, so this only runs once per cell.
                if post_simulation.directory_args["state_label"] == "lambda_window":
                    fileStore.logToMaster(f"Energy post-analysis already completed and loading the results") 
                    fileStore.logToMaster(
                        f"Loading the Energy post-analysis results in the directory {post_process_job.output_dir}\n"
                    )
                _pa_t0 = time.perf_counter()
                self.post_output.append(
                    pd.read_parquet(
                        os.path.join(
                            post_process_job.output_dir, "simulation_mdout.parquet.gzip"
                        ),
                    )
                )
                # Cache-load timing record (parsed by timing_report.py). The imin=5 re-scoring for this
                # window was computed in a PRIOR run; here we only re-read its cached parquet, so this
                # counts as ~0 compute for the post_analysis phase (vs. the full sander re-score that
                # simulations.Calculation.run times on a COLD run). Emitting it keeps the phase VISIBLE
                # on warm/resumed runs (jobs=N, cpu_h~=load cost) instead of vanishing. Routed through
                # logToMaster so it lands in the same leader log as every other phase.
                fileStore.logToMaster(
                    f"[TIMING] phase=post_analysis wall_s={time.perf_counter() - _pa_t0:.2f} "
                    f"cores={post_process_job.num_cores} gpu=0 end={time.time():.0f} "
                    f"| post_analysis cache-load {post_process_job.directory_args.get('state_label', '')}"
                )
            self._loaded_dataframe.append(post_process_job.output_dir)
    
    def has_post_analysis_data(self, output_dir):
        """Return True if the parquet file exists and is not empty; otherwise False."""
        path = os.path.join(output_dir, "simulation_mdout.parquet.gzip")

        if not os.path.exists(path):
            return False

        try:
            df = pd.read_parquet(path)
            return not df.empty
        except (ValueError, OSError):
            return False
    
    def _add_complex_simulation(
        self,
        conformational,
        orientational,
        mdin,
        restraint_file,
        charge=1.0,
        charge_parm=None,
        gb_extdiel=None,
    ):
        con_force = float(round(conformational, 3))
        orient_force = float(round(orientational, 3))

        dirs_args = (
            {
                "topology": self.config.endstate_files.complex_parameter_filename,
                "state_label": "lambda_window",
                "extdiel": 78.5,
                "charge": charge,
                "igb": f"igb_{self.config.intermediate_args.igb_solvent}",
                "igb_value": self.config.intermediate_args.igb_solvent,
                "conformational_restraint": con_force,
                "orientational_restraints": orient_force,
                "filename": f"state_8_{con_force}_{orient_force}_prod",
                "runtype": f"Running restraint window. Conformational restraint: {con_force} and orientational restraint: {orient_force}",
                "topdir": self.config.system_settings.top_directory_path,
            },
        )

        parm_file = self.config.endstate_files.complex_parameter_filename

        # scaling GB external dielectric
        if gb_extdiel is not None:
            # prmtop was create with charge = 0
            parm_file = charge_parm.rv()  # type: ignore
            dirs_args[0].update({"state_label": "gb_dielectric"})
            dirs_args[0].update({"extdiel": gb_extdiel})
            dirs_args[0].update({"charge": charge})
        # scaling ligand charge windows
        elif charge_parm is not None:
            parm_file = charge_parm.rv()
            dirs_args[0].update({"state_label": "electrostatics"})  # type: ignore

        # scaling restraint windows
        else:
            restraint_file = restraint_file.rv()

        new_job = Simulation(
            executable=self.config.system_settings.executable,
            mpi_command=self.config.system_settings.mpi_command,
            num_cores=self.config.num_cores_per_system.complex_ncores,
            # ALS pilot (and the revived adaptive path) must honour the CUDA flag exactly like the
            # static path (setup_simulations.setup_apply_restraint_windows): complex runs on GPU for
            # protein-ligand. The previous hardcoded CUDA=False only happened to match host-guest
            # (cb7), where config.system_settings.CUDA is already falsy.
            CUDA=self.config.system_settings.CUDA,
            prmtop=parm_file,
            incrd=self.config.inputs["endstate_complex_lastframe"],
            input_file=mdin,
            restraint_file=restraint_file,
            working_directory=self.config.system_settings.working_directory,
            system_type="complex",
            directory_args=dirs_args[0],
            dirstruct="dirstruct_halo",
            accelerators=(
                self.config.system_settings.num_accelerators
                if self.config.system_settings.CUDA
                else None
            ),
        )

        self.simulations.append(new_job)

    def _add_ligand_simulation(
        self,
        conformational,
        mdin,
        restraint_file,
        charge=1.0,
        charge_parm=None,
    ):
        con_force = float(round(conformational, 3))

        dirs_args = (
            {
                "topology": self.config.endstate_files.ligand_parameter_filename,
                "state_label": "lambda_window",
                "conformational_restraint": con_force,
                "igb": f"igb_{self.config.intermediate_args.igb_solvent}",
                "extdiel": 78.5,
                "charge": charge,
                "igb_value": self.config.intermediate_args.igb_solvent,
                "filename": f"state_2_{con_force}_prod",
                "runtype": f"Running restraint window, Conformational restraint: {con_force}",
                "topdir": self.config.system_settings.top_directory_path,
            },
        )

        parm_file = self.config.endstate_files.ligand_parameter_filename
        if charge_parm is not None:
            parm_file = charge_parm.rv()
            dirs_args[0].update({"state_label": "electrostatics"})  # type: ignore
            dirs_args[0].update({"igb": "igb_6"})
            dirs_args[0].update({"filename": "state_4_prod"})
            dirs_args[0].update({"extdiel": 0.0})
            dirs_args[0].update(
                {
                    "runtype": f"Scailing ligand charges: {charge}",
                }
            )
        else:
            restraint_file = restraint_file.rv()
        new_job = Simulation(
            executable=self.config.system_settings.executable,
            mpi_command=self.config.system_settings.mpi_command,
            num_cores=self.config.num_cores_per_system.ligand_ncores,
            CUDA=self.config.system_settings.CUDA,
            prmtop=parm_file,
            incrd=self.config.inputs["ligand_endstate_frame"],
            input_file=mdin,
            restraint_file=restraint_file,
            working_directory=self.config.system_settings.working_directory,
            system_type="ligand",
            directory_args=dirs_args[0],
            dirstruct="dirstruct_apo",
        )

        self.simulations.append(new_job)

    def _add_receptor_simulation(
        self,
        conformational,
        mdin,
        restraint_file,
        gb_extdiel=None,
    ):
        con_force = float(round(conformational, 3))

        dirs_args = {
            "topology": self.config.endstate_files.receptor_parameter_filename,
            "state_label": "lambda_window",
            "extdiel": 78.5,
            "charge": 1.0,
            "igb": f"igb_{self.config.intermediate_args.igb_solvent}",
            "igb_value": self.config.intermediate_args.igb_solvent,
            "conformational_restraint": con_force,
            "filename": f"state_2_{con_force}_prod",
            "runtype": f"Running restraint window, Conformational restraint: {con_force}",
            "topdir": self.config.system_settings.top_directory_path,
        }

        # GB-dielectric window: the apo host carries its FULL charge (q=1; no ligand to decharge), so the
        # default receptor topology is reused — only the external dielectric (and state label) change. The
        # restraint_file is passed as a RESOLVED max-restraint file (not a promise), mirroring the complex
        # gb_extdiel branch in _add_complex_simulation.
        if gb_extdiel is not None:
            dirs_args.update(
                {
                    "state_label": "gb_dielectric",
                    "extdiel": gb_extdiel,
                    "charge": 1.0,
                    "filename": f"state_8_{gb_extdiel}_prod",
                    "runtype": f"Running receptor GB window. extdiel: {gb_extdiel}",
                }
            )
        else:
            restraint_file = restraint_file.rv()

        new_job = Simulation(
            executable=self.config.system_settings.executable,
            mpi_command=self.config.system_settings.mpi_command,
            num_cores=self.config.num_cores_per_system.receptor_ncores,
            CUDA=self.config.system_settings.CUDA,
            prmtop=self.config.endstate_files.receptor_parameter_filename,
            incrd=self.config.inputs["receptor_endstate_frame"],
            input_file=mdin,
            restraint_file=restraint_file,
            working_directory=self.config.system_settings.working_directory,
            system_type="receptor",
            directory_args=dirs_args,
            dirstruct="dirstruct_apo",
            # Receptor runs on GPU when CUDA is set (mirrors setup_apply_restraint_windows: a
            # non-ligand system requests an accelerator so Toil pins it to a distinct device).
            accelerators=(
                self.config.system_settings.num_accelerators
                if self.config.system_settings.CUDA
                else None
            ),
        )

        self.simulations.append(new_job)

    @classmethod
    def new_runner(
        cls: Type["IntermidateRunner"],
        config: Config,
        obj: dict,
        post_only: bool = True,
        simulations: Optional[list] = None,
    ):
        """Build a runner that shares ``post_output``/``_loaded_dataframe`` by reference with ``obj``.

        ``post_only`` selects the MD pass (``False``) or the post-analysis pass (``True``) of the ALS
        pilot's two-phase sub-runner; ``simulations`` overrides the simulation list (e.g. to run ONLY
        the newly-inserted pilot windows) while still accumulating into the shared ``post_output``.
        Defaults preserve the original behaviour (post-only over ``obj['simulations']``).
        """
        return cls(
            simulations=simulations if simulations is not None else obj["simulations"],
            restraints=obj["restraints"],
            config=config,
            post_process_distruct=obj["post_process_distruct"],
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_saltfree_mdin=config.inputs.get("post_saltfree_mdin"),
            adaptive=True,
            post_only=post_only,
            post_output=obj["post_output"],
            loaded_dataframe=obj["_loaded_dataframe"],
            traj_map=obj["traj_map"],
        )

    @staticmethod
    def _get_md_traj(simulation: Simulation, fileStore):
        """Return an absolute path to completed AMBER (.nc) trajectory filename.

        Parameters
        ----------
        simulation: Simulation
            Simulation class object which contains all required MD input arguments.
        fileStore: job.fileStore
            Toil interface to read and write files.
        Returns
        -------
        Filepath to AMBER trajectory (.nc) file.
        """
        return os.path.join(
            simulation.output_dir,
            list(
                filter(
                    lambda file: re.match(r"^.*\.nc$", file),
                    os.listdir(simulation.output_dir),
                )
            )[0],
        )

    @staticmethod
    def _check_mdout(simulation: Simulation) -> bool:
        if "mdout" in os.listdir(simulation.output_dir):
            for line in reversed(
                open(os.path.join(simulation.output_dir, "mdout")).readlines()
            ):
                if "Final Performance Info" in line:
                    return True
        return False

    @staticmethod
    def update_postprocess_dirstruct(
        run_time_args: dict,
    ) -> dict[str, Union[str, object]]:
        if "orientational_restraints" in run_time_args.keys():
            return {
                "traj_state_label": run_time_args["state_label"],
                "trajectory_restraint_conrest": run_time_args[
                    "conformational_restraint"
                ],
                "trajectory_restraint_orenrest": run_time_args[
                    "orientational_restraints"
                ],
                "traj_extdiel": run_time_args["extdiel"],
                "traj_igb": run_time_args["igb"],
                "traj_charge": run_time_args["charge"],
                "filename": f"{run_time_args['filename']}_postprocess",
            }

        return {
            "traj_state_label": run_time_args["state_label"],
            "trajectory_restraint_conrest": run_time_args["conformational_restraint"],
            "traj_igb": run_time_args["igb"],
            "traj_extdiel": run_time_args["extdiel"],
            "traj_charge": run_time_args["charge"],
            "filename": f"{run_time_args['filename']}_postprocess",
        }

    @staticmethod
    def get_system_dirs(system_type):
        if system_type == "ligand" or system_type == "receptor":
            return "post_process_apo"

        return "post_process_halo"


def run_post_pass_with_traj(job, runner, config, md_runner, simulations=None):
    """Follow-on of an ALS-pilot MD pass: run its post-analysis pass with NO network read.

    The MD pass (``md_runner``) and the post pass are different Toil jobs, so the freshly-run
    window's trajectory FileID(s) live in ``md_runner.traj_map`` and are NOT in the pre-MD
    ``runner``'s dict. Fold them into ``runner``'s shared, accumulated ``traj_map`` (which
    already carries every EARLIER pass's windows), then build the ``post_only=True`` runner
    from ``runner.__dict__`` so it shares that merged map by reference. ``run(post_only=True)``
    sets ``inptraj`` from ``traj_map`` (jobStore) instead of listing the network ``output_dir``.
    Returns the post runner (rv) carrying ``post_output`` + the accumulated ``traj_map`` for the
    next pass. Mirrors the production Phase 5->6 seam (workflow_phases.run_post_analysis_...).
    """
    runner.traj_map.update(md_runner.traj_map)
    post_runner = runner.new_runner(
        config, runner.__dict__, post_only=True, simulations=simulations
    )
    job.addChild(post_runner)
    return post_runner.rv()

    # /nas0/ayoub/sampl9_runs/sampl9_extend_windows_diel/WP6_G2_Hmass/lambda_window/1.0/78.5/-0.2857142857142864/3.7142857142857135/WP6_G2_Hmass_state_8_0.8203353560076375_13.1253656961222_prod_traj.nc
