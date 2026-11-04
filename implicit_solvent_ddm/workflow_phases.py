"""
Workflow phase functions for the Double Decoupling Method (DDM) workflow.

This module contains the individual phase functions that make up the DDM workflow,
providing a clean separation of concerns and improved maintainability.
"""

import copy
import glob
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from toil.job import JobFunctionWrappingJob

from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.mdin import (
    get_mdins,
    generate_extdiel_mdin,
    get_long_window_mdin,
)
from implicit_solvent_ddm.restraints import (
    BoreschRestraints,
    FlatBottom,
    RestraintMaker,
    write_empty_restraint,
)
from implicit_solvent_ddm.alchemical import alter_topology, split_complex_system
from implicit_solvent_ddm.setup_simulations import SimulationSetup
from implicit_solvent_ddm.runner import IntermidateRunner
from implicit_solvent_ddm import als_refine
from implicit_solvent_ddm.adaptive_restraints import (
    run_exponential_averaging,
    run_compute_mbar,
)
from implicit_solvent_ddm.run_endstate import (
    run_remd,
    run_basic_md,
    user_defined_endstate,
)
from implicit_solvent_ddm.postTreatment import ConsolidateData, cache_output_path

logger = logging.getLogger(__name__)


def setup_workflow_components(job: JobFunctionWrappingJob, config: Config):
    """
    Phase 1: Setup workflow components and MD input files.
    
    This phase:
    1. Generates MD input files for intermediate states
    2. Creates empty restraint templates
    3. Sets up flat bottom restraint potentials
    4. Prepares restraint generation components
    
    Parameters
    ----------
    job : JobFunctionWrappingJob
        Parent Toil job
    config : Config
        Configuration object
        
    Returns
    -------
    JobFunctionWrappingJob
        Job containing all setup components
    """
    # Generate MD input files for intermediate states
    # RMSD mode: the two MD mdins get infe=1 + a &pmd block pointing at the per-window cv.in. The
    # imin=5 scoring mdins deliberately do NOT -- the restraint term is added analytically.
    nfe_cv_file = "cv.in" if config.intermediate_args.rmsd_mode else None
    mdins = job.addChildJobFn(
        get_mdins,
        config.intermediate_args.mdin_intermediate_file,
        nfe_cv_file=nfe_cv_file,
        nfe_output_freq=config.intermediate_args.rmsd_output_freq,
    )
    
    # Store MDIN file references in config
    MDIN_TYPES = {
        'default': 0,
        'no_solvent': 1,
        'post': 2,
        'post_nosolv': 3,
        'post_saltfree': 4  # scoring mdin for the gb_dielectric band (saltcon=0, matching its MD)
    }

    config.inputs["default_mdin"] = mdins.rv(MDIN_TYPES['default'])
    config.inputs["no_solvent_mdin"] = mdins.rv(MDIN_TYPES['no_solvent'])
    config.inputs["post_mdin"] = mdins.rv(MDIN_TYPES['post'])
    config.inputs["post_nosolv_mdin"] = mdins.rv(MDIN_TYPES['post_nosolv'])
    config.inputs["post_saltfree_mdin"] = mdins.rv(MDIN_TYPES['post_saltfree'])

    # Stretched mdin for the lowest restraint window (setup_simulations._restraint_window_mdin).
    # Gated so a config with the knob off adds no job and no inputs key.
    if config.intermediate_args.long_restraint_window_exponent is not None:
        long_mdin = mdins.addChildJobFn(
            get_long_window_mdin,
            config.intermediate_args.mdin_intermediate_file,
            config.intermediate_args.long_restraint_window_ns,
            nfe_cv_file=nfe_cv_file,
            nfe_output_freq=config.intermediate_args.rmsd_output_freq,
        )
        config.inputs["long_window_mdin"] = long_mdin.rv(0)
        config.inputs["long_window_nstlim"] = long_mdin.rv(1)
        config.inputs["long_window_ntwx"] = long_mdin.rv(2)

    # Create empty restraint file
    empty_restraint = mdins.addChildJobFn(write_empty_restraint)
    config.inputs["empty_restraint"] = empty_restraint.rv()

    # Setup flat bottom restraint potentials
    flat_bottom_template = mdins.addChild(FlatBottom(config=config))
    config.inputs["flat_bottom_restraint"] = flat_bottom_template.rv(0)

    return config

def run_endstate_simulations(job, config: Config):
    """
    Phase 2: Run endstate simulations for complex, receptor, and ligand systems.
    
    This phase executes long MD simulations at the end states to generate
    representative conformations for the thermodynamic cycle.
    
    Parameters
    ----------
    job : JobFunctionWrappingJob
        Current Toil job
    config : Config
        Configuration object
        
    Returns
    -------
    endstate_job: JobFunctionWrappingJob
        Returns complex, receptor, and ligand endstate simulation results
    """
    job.fileStore.logToMaster(f"type config: {type(config)}")
    # Determine endstate simulation method
    if config.workflow.run_endstate_method:
        if config.endstate_method.endstate_method_type == "remd":
            endstate_job = job.addChildJobFn(run_remd, config)
        elif config.endstate_method.endstate_method_type == "basic_md":
            endstate_job = job.addChildJobFn(run_basic_md, config)
        else:
            endstate_job = job.addChildJobFn(user_defined_endstate, config)
    else:
        endstate_job = job.addChildJobFn(user_defined_endstate, config)

    reference = config.endstate_files.restraint_reference_coordinate
    if reference is not None:
        return endstate_job.addFollowOnJobFn(use_reference_frame, endstate_job.rv(), reference).rv()
    return endstate_job.rv()


def use_reference_frame(job, endstate, reference):
    """
    Replace the endstate's binding mode with the fixed restraint reference.

    Element 0 is the one frame Phase 3 splits and builds every restraint from, and every
    window's starting coordinate, so swapping it here gives all three legs the same reference
    in every replicate. The trajectories are untouched.

    Parameters
    ----------
    job : JobFunctionWrappingJob
        Current Toil job.
    endstate : tuple
        ``(complex binding mode, complex traj, receptor traj, ligand traj)``.
    reference : FileID
        Imported single-frame complex restart.

    Returns
    -------
    tuple
        ``endstate`` with element 0 replaced.
    """
    job.fileStore.logToMaster(f"restraint reference: fixed file {reference} (replaces {endstate[0]})")
    return (reference, *endstate[1:])


def decompose_system_and_generate_restraints(job, endstate_jobs, config: Config):
    """
    Phase 3: Decompose system and generate restraints.
    
    This phase:
    1. Splits the complex into receptor and ligand components
    2. Generates Boresch orientational restraints
    3. Creates restraint files for intermediate simulations
    4. Sets up flat bottom contribution calculations
    
    Parameters
    ----------
    job : JobFunctionWrappingJob
        Current Toil job
    endstate_jobs : JobFunctionWrappingJob
        Job containing endstate simulation results
    config : Config
        Configuration object
        
    Returns
    -------
    JobFunctionWrappingJob
        Job containing decomposed system and restraints
    """
    job.fileStore.logToMaster(f"type for endstate_jobs: {type(endstate_jobs)}")
    job.fileStore.logToMaster(f"endstate_jobs: {endstate_jobs}")

    # Split complex into receptor and ligand using endstate trajectory
    split_job = job.addChildJobFn(
        split_complex_system,
        config.endstate_files.complex_parameter_filename,
        endstate_jobs[0],  # complex binding mode
        config.amber_masks.ligand_mask,
        config.amber_masks.receptor_mask,
    )
    

    # Generate Boresch orientational restraints
    boresch_restraints = split_job.addChild(
        BoreschRestraints(
            complex_prmtop=config.endstate_files.complex_parameter_filename,
            complex_coordinate=endstate_jobs[0],
            restraint_type=config.intermediate_args.restraint_type,
            ligand_mask=config.amber_masks.ligand_mask,
            receptor_mask=config.amber_masks.receptor_mask,
            K_r=config.intermediate_args.boresch_Kr,
            K_thetaA=config.intermediate_args.max_orientational_restraint,
            K_thetaB=config.intermediate_args.max_orientational_restraint,
            K_phiA=config.intermediate_args.max_orientational_restraint,
            K_phiB=config.intermediate_args.max_orientational_restraint,
            K_phiC=config.intermediate_args.max_orientational_restraint,
            anchor_pool=config.intermediate_args.boresch_anchors,
            pocket_radius=config.intermediate_args.boresch_pocket_radius,
            excluded_receptor_mask=config.intermediate_args.freed_receptor_mask,
        )
    )
    
    # Create restraint files for intermediate simulations
    restraints = boresch_restraints.addChild(
        RestraintMaker(
            config=config,
            complex_binding_mode=endstate_jobs[0],
            boresch_restraints=boresch_restraints.rv(),
            flat_bottom=config.inputs["flat_bottom_restraint"],
            # Holo trajectory, for rmsd_reference_source='ensemble_average'. A single snapshot sits
            # about one thermal RMSF off the apo-to-holo segment, which both legs then pay for.
            endstate_trajectory=endstate_jobs[1],
        )
    ).rv()

    boresch_restraints_completed = boresch_restraints.addFollowOnJobFn(
        initilized_jobs,
        message="✓ Phase 3 Complete: Boresch restraints generated"
    )

    return split_job.rv(0), split_job.rv(1), restraints


def setup_intermediate_simulations(job, decomposition_jobs, endstate_jobs, config: Config):
    """
    Phase 4: Setups the intermediate state simulations with alchemical transformations.

    This phase:
    1. Sets up simulation systems for complex, receptor, and ligand
    2. Performs alchemical transformations (charge scaling, GB scaling, etc.)

    Parameters
    ----------
    job : JobFunctionWrappingJob
        Current Toil job
    decomposition_jobs : JobFunctionWrappingJob
        Job containing split receptor and ligand systems with generated restraints
    endstate_jobs : JobFunctionWrappingJob
        Job containing endstate simulation results
    config : Config
        Configuration object
        
    Returns
    -------
    JobFunctionWrappingJob
        Job containing intermediate simulation results
    """
    job.fileStore.logToMaster(f"Calling setup_intermediate_simulations")

    # Update config with binding modes
    updated_config = job.addChildJobFn(
        update_config, 
        config, 
        endstate_jobs[0],  # complex binding mode
        decomposition_jobs[0],    # receptor binding mode  
        decomposition_jobs[1],    # ligand binding mode
        decomposition_jobs[2],   # restraints
    ).rv()

    complex_simulations = SimulationSetup(
        config=config,
        system_type="complex",
        endstate_traj=endstate_jobs[1],  # complex trajectory
        binding_mode=endstate_jobs[0],   # complex binding mode
        restraints=decomposition_jobs[2],     # restraints
    )

    receptor_simulations = SimulationSetup(
        config=config,
        system_type="receptor",
        restraints=decomposition_jobs[2],     # restraints
        binding_mode=decomposition_jobs[0],   # receptor binding mode
        endstate_traj=endstate_jobs[2],  # receptor trajectory (index 2; [3] is the ligand traj)
    )
    
    ligand_simulations = SimulationSetup(
        config=config,
        system_type="ligand",
        restraints=decomposition_jobs[2],     # restraints
        binding_mode=decomposition_jobs[1],   # ligand binding mode
        endstate_traj=endstate_jobs[3],  # ligand trajectory
    )
    
    # Setup flat bottom contribution calculations
    flat_bottom_setup = SimulationSetup(
        config=config,
        system_type="complex",
        endstate_traj=endstate_jobs[1],   # complex trajectory
        binding_mode=endstate_jobs[0],     # complex binding mode
        restraints=decomposition_jobs[2],       # restraints
    )
    
    # Setup post-endstate analysis if enabled
    if config.workflow.end_state_postprocess:
        # Setup flat bottom contribution
        flat_bottom_setup.setup_post_endstate_simulation(flat_bottom=True)
        flat_bottom_setup.setup_post_endstate_simulation()

        # Setup endstate post-process analysis
        complex_simulations.setup_post_endstate_simulation(flat_bottom=True)
        receptor_simulations.setup_post_endstate_simulation()
        ligand_simulations.setup_post_endstate_simulation()

    # Calculate the pinned restraint anchor (mode-aware: max spring, or min RMSD tolerance)
    max_conformational_force = config.intermediate_args.strongest_conformational_restraint
    max_orientational_force = max(config.intermediate_args.orientational_restraint_forces)
    in_solvent = config.intermediate_args.decouple_in_solvent
    # in solvent the decoupled states also drop the ligand GB screening
    decoupled_screen = 0.0 if in_solvent else None
    complex_anchor_key = f"complex_{max_conformational_force}_{max_orientational_force}_rst"

    
    # Setup runner jobs for MD simulations
    runner_jobs = job.addChildJobFn(
        initilized_jobs, 
        message="Setting up MD simulations"
    )
    
    # Setup ligand charge scaling simulations
    for charge in config.intermediate_args.charges_lambda_window:
        # Scale ligand charges in isolated ligand system; in solvent q=1 is the restraint anchor
        if not (in_solvent and charge == 1.0):
            ligand_simulations.setup_ligand_charge_simulation(
                prmtop=job.addChildJobFn(
                    alter_topology,
                    solute_amber_parm=config.endstate_files.ligand_parameter_filename,
                    solute_amber_coordinate=config.endstate_files.ligand_coordinate_filename,
                    ligand_mask=config.amber_masks.ligand_mask,
                    receptor_mask=config.amber_masks.receptor_mask,
                    set_charge=charge,
                ).rv(),
                charge=charge,
                restraint_key=f"ligand_{max_conformational_force}_rst",
            )
        
        # Scale ligand charges within the complex
        complex_simulations.setup_ligand_charge_simulation(
            prmtop=job.addChildJobFn(
                alter_topology,
                solute_amber_parm=config.endstate_files.complex_parameter_filename,
                solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                ligand_mask=config.amber_masks.ligand_mask,
                receptor_mask=config.amber_masks.receptor_mask,
                set_charge=charge,
            ).rv(),
            charge=charge,
            restraint_key=f"complex_{max_conformational_force}_{max_orientational_force}_rst",
        )

    # Setup receptor desolvation simulations
    if config.workflow.remove_GB_solvent_receptor:
        receptor_simulations.setup_remove_gb_solvent_simulation(
            restraint_key=f"receptor_{max_conformational_force}_rst",
            prmtop=config.endstate_files.receptor_parameter_filename,
        )

    # Setup complex ligand exclusion simulations (gas phase; LJ 0 + SCREEN 0 in solvent)
    if config.workflow.complex_ligand_exclusions:
        complex_simulations.setup_remove_gb_solvent_simulation(
            restraint_key=complex_anchor_key,
            prmtop=job.addChildJobFn(
                alter_topology,
                solute_amber_parm=config.endstate_files.complex_parameter_filename,
                solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                ligand_mask=config.amber_masks.ligand_mask,
                receptor_mask=config.amber_masks.receptor_mask,
                set_charge=0.0,
                exculsions=not in_solvent,
                lj_scale=0.0 if in_solvent else None,
                screen_scale=decoupled_screen,
            ).rv(),
        )

    # Intermediate ligand-receptor LJ windows between no_interactions and interactions
    if config.workflow.complex_ligand_exclusions and config.workflow.complex_turn_off_exclusions:
        for lj_scale in config.intermediate_args.lj_scale_windows:
            complex_simulations.setup_lj_scaled_simulation(
                restraint_key=f"complex_{max_conformational_force}_{max_orientational_force}_rst",
                prmtop=job.addChildJobFn(
                    alter_topology,
                    solute_amber_parm=config.endstate_files.complex_parameter_filename,
                    solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                    ligand_mask=config.amber_masks.ligand_mask,
                    receptor_mask=config.amber_masks.receptor_mask,
                    set_charge=0.0,
                    lj_scale=lj_scale,
                    screen_scale=decoupled_screen,
                ).rv(),
                lj_scale=lj_scale,
            )

    # Setup LJ interaction simulations
    if config.workflow.complex_turn_off_exclusions:
        complex_simulations.setup_lj_interations_simulation(
            restraint_key=f"complex_{max_conformational_force}_{max_orientational_force}_rst",
            prmtop=job.addChildJobFn(
                alter_topology,
                solute_amber_parm=config.endstate_files.complex_parameter_filename,
                solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                ligand_mask=config.amber_masks.ligand_mask,
                receptor_mask=config.amber_masks.receptor_mask,
                set_charge=0.0,
                screen_scale=decoupled_screen,
            ).rv(),
        )

    # Ligand GB screening windows between interactions and uncharged electrostatics
    for screen_scale in config.intermediate_args.screening_windows:
        complex_simulations.setup_screening_simulation(
            restraint_key=complex_anchor_key,
            prmtop=job.addChildJobFn(
                alter_topology,
                solute_amber_parm=config.endstate_files.complex_parameter_filename,
                solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                ligand_mask=config.amber_masks.ligand_mask,
                receptor_mask=config.amber_masks.receptor_mask,
                set_charge=0.0,
                screen_scale=screen_scale,
            ).rv(),
            screen_scale=screen_scale,
        )
    
    # Setup GB external dielectric scaling simulations
    if config.workflow.gb_extdiel_windows:
        # Create complex with ligand electrostatics = 0
        complex_ligand_no_charge = job.addChildJobFn(
            alter_topology,
            solute_amber_parm=config.endstate_files.complex_parameter_filename,
            solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
            ligand_mask=config.amber_masks.ligand_mask,
            receptor_mask=config.amber_masks.receptor_mask,
            set_charge=0.0,
        ).rv()
        
        # Interpolate GB external dielectric constant
        for dielectric in config.intermediate_args.gb_extdiel_windows:
            complex_simulations.setup_gb_external_dielectric(
                restraint_key=f"complex_{max_conformational_force}_{max_orientational_force}_rst",
                prmtop=complex_ligand_no_charge,
                extdiel=dielectric,
                mdin=job.addChildJobFn(
                    generate_extdiel_mdin,
                    user_mdin_ID=config.intermediate_args.mdin_intermediate_file,
                    gb_extdiel=dielectric,
                    nfe_cv_file="cv.in" if config.intermediate_args.rmsd_mode else None,
                    nfe_output_freq=config.intermediate_args.rmsd_output_freq,
                ).rv(),
            )

            # Matched receptor (apo-host) GB band: same epsilon schedule at FULL host charge so the host
            # desolvation cancels between the complex and receptor legs. Uses the default (full-charge)
            # receptor topology — no alter_topology needed (the apo system has no ligand to decharge).
            # Mandatory whenever the band is enabled: receptor_GB_exl_windows is now in receptor_order, so
            # compute_mbar's _ordered() invariant requires these columns to exist.
            receptor_simulations.setup_gb_external_dielectric(
                restraint_key=f"receptor_{max_conformational_force}_rst",
                prmtop=config.endstate_files.receptor_parameter_filename,
                extdiel=dielectric,
                mdin=job.addChildJobFn(
                    generate_extdiel_mdin,
                    user_mdin_ID=config.intermediate_args.mdin_intermediate_file,
                    gb_extdiel=dielectric,
                    nfe_cv_file="cv.in" if config.intermediate_args.rmsd_mode else None,
                    nfe_output_freq=config.intermediate_args.rmsd_output_freq,
                ).rv(),
            )

    # Setup restraint force windows
    config.intermediate_args.exponent_conformational_forces_list = []
    config.intermediate_args.exponent_orientational_forces_list = []
    
    for con_force, orien_force in zip(
        config.intermediate_args.conformational_restraints_forces,
        config.intermediate_args.orientational_restraint_forces,
    ):
        # In rmsd mode the conformational slot already HOLDS the state label (an Angstrom tolerance),
        # so there is no 2^p to undo -- log2 would turn the ladder into exponents and send the
        # r3 = 0 anchor to -inf.
        exponent_conformational = (
            round(float(con_force), 3)
            if config.intermediate_args.rmsd_mode
            else round(np.log2(con_force), 3)
        )
        exponent_orientational = round(np.log2(orien_force), 3)

        config.intermediate_args.exponent_conformational_forces_list.append(exponent_conformational)
        config.intermediate_args.exponent_orientational_forces_list.append(exponent_orientational)

        # Add conformational restraints to ligand
        if config.workflow.add_ligand_conformational_restraints:
            ligand_simulations.setup_apply_restraint_windows(
                restraint_key=f"ligand_{con_force}_rst",
                exponent_conformational=exponent_conformational,
            )

        # Add conformational restraints to receptor
        if config.workflow.add_receptor_conformational_restraints:
            receptor_simulations.setup_apply_restraint_windows(
                restraint_key=f"receptor_{con_force}_rst",
                exponent_conformational=exponent_conformational,
            )

        # Remove conformational and orientational restraints from complex
        if config.workflow.complex_remove_restraint and max_conformational_force != con_force:
            complex_simulations.setup_apply_restraint_windows(
                restraint_key=f"complex_{con_force}_{orien_force}_rst",
                exponent_conformational=exponent_conformational,
                exponent_orientational=exponent_orientational,
            )


    return updated_config, complex_simulations, receptor_simulations, ligand_simulations, flat_bottom_setup



@dataclass
class HamiltonianSpec:
    """Lightweight, picklable stand-in for a ``Simulation`` carrying ONLY the read-only fields that
    ``IntermidateRunner.run(post_only=True)`` / ``only_post_analysis`` read off each
    ``self.simulations`` element. The merged MD->post dispatcher gives every per-window post-runner
    this spec list instead of the live ``Simulation`` Job objects, which otherwise drag Toil
    framework state plus a ``sim -> _followOns -> post_runner`` pickle cycle and blow up to N^2
    tangled deep-pickles at the leader. ``only_post_analysis`` builds a FRESH post ``Simulation``
    from these fields and never re-schedules the element, so behaviour is byte-identical; the spec
    just duck-types ``Simulation`` for the handful of attributes touched (see the field list).
    """

    directory_args: dict
    output_dir: str
    prmtop: object
    incrd: object
    restraint_file: object
    restraint_key: object
    working_directory: object
    system_type: object
    mpi_command: object
    inptraj: object = None
    cv_file: object = None
    cv_key: object = None
    cv_spec: object = None

    @classmethod
    def from_simulation(cls, sim):
        return cls(
            directory_args=dict(sim.directory_args),
            output_dir=sim.output_dir,
            prmtop=sim.prmtop,
            incrd=sim.incrd,
            restraint_file=sim.restraint_file,
            restraint_key=sim.restraint_key,
            cv_file=getattr(sim, "cv_file", None),
            cv_key=getattr(sim, "cv_key", None),
            cv_spec=getattr(getattr(sim, "restraint_file", None), "cv_spec", None),
            working_directory=sim.working_directory,
            system_type=sim.system_type,
            mpi_command=sim.mpi_command,
            inptraj=sim.inptraj,
        )


class PostOutputBundle:
    """Minimal picklable holder exposing ``.post_output`` -- the only attribute
    ``run_compute_mbar`` / ``run_exponential_averaging`` read off the per-system runner object. The
    merged dispatcher's per-system aggregator hands Phase 7 one of these (a flat list of the window
    rows) instead of re-pickling a whole ``IntermidateRunner``.
    """

    def __init__(self, post_output):
        self.post_output = post_output


def _needs_md(sim, is_flat_bottom, config: Config) -> bool:
    """Per-window replacement for the legacy Phase-5 MD skip logic: True -> run MD then score its
    trajectory; False -> score an EXISTING trajectory directly (no MD parent).

    * ``inptraj`` preset (endstate post-process windows) -> score only, NEVER MD. These are
      ``post_analysis=True`` jobs whose ``run()`` returns a scalar mdout FileID, so ``md.rv(1)``
      would be garbage -- and MBAR's ``_ordered()`` requires their ``endstate`` column.
    * MD already on disk (warm resume) -> score only.
    * flat-bottom leg under ``post_analysis_only`` -> score only (mirrors the legacy ``flat_bottom``
      ``post_only=post_analysis_only`` gate; complex/receptor/ligand always attempt MD, skipping only
      on-disk-complete windows).
    """
    if sim.inptraj is not None:
        return False
    if IntermidateRunner._check_mdout(sim):
        return False
    if is_flat_bottom and config.workflow.post_analysis_only:
        return False
    return True


def _post_row_runner(config: Config, spec_list, distruct, restrict, traj_map):
    """One per-window post-analysis runner: a ``post_only`` pass over the full Hamiltonian
    ``spec_list``, restricted (``restrict_completed``) to a single completed window and fed that
    window's trajectory via ``traj_map`` (or the window's own preset ``inptraj`` / on-disk file when
    ``traj_map`` is empty). Its ``.post_output`` is that window's ROW of the MBAR matrix.
    """
    return IntermidateRunner(
        spec_list,
        config.inputs["restraints"],
        post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
        post_process_mdin=config.inputs["post_mdin"],
        post_process_saltfree_mdin=config.inputs.get("post_saltfree_mdin"),
        post_process_distruct=distruct,
        post_only=True,
        config=config,
        traj_map=traj_map,
        restrict_completed=restrict,
    )


def run_intermediate_and_post(
    job, config: Config, complex_simulations, receptor_simulations, ligand_simulations, flat_bottom_setup
):
    """Phases 5 + 6 merged (dissolves the global Phase5->Phase6 barrier).

    All intermediate MD windows are submitted first, in ``complex -> receptor -> ligand ->
    flat_bottom`` order (front of Toil single_machine's FIFO queue). Each window's post-analysis ROW
    -- its trajectory re-scored under every Hamiltonian in the leg -- is attached as a **follow-on of
    that window's OWN MD job**, so the dominant N^2 CPU post-analysis backfills free cores as each
    trajectory lands (chiefly the cores idle while complex/receptor MD runs on the GPU) instead of
    waiting for every MD window to finish. Endstate / warm-resume windows (no MD parent) are scored
    directly. Per-system aggregators flatten the rows into the ``.post_output`` list Phase 7 already
    consumes. Same N^2 post ``Simulation`` jobs as the legacy path -> byte-identical energies; only
    the scheduling edges change. Replaces ``run_intermediate_simulations`` +
    ``run_post_analysis_intermediate_simulations``.

    Returns ``(complex, receptor, ligand, flat_bottom)`` post-output bundles for
    ``compute_free_energy_and_consolidate``.
    """
    job.fileStore.logToMaster("Calling run_intermediate_and_post (merged MD->post pipeline)")

    export = config.system_settings.export_intermediate_files
    # Expensive-first system order so MD enters the FIFO queue ahead of any post job.
    systems = [
        (complex_simulations, "post_process_halo", False),
        (receptor_simulations, "post_process_apo", False),
        (ligand_simulations, "post_process_apo", False),
        (flat_bottom_setup, "post_process_halo", True),
    ]

    rows_by_system = []
    for setup, distruct, is_flat_bottom in systems:
        # Hamiltonian COLUMN list built ONCE per system (ALL windows, incl. no_flat_bottom), as
        # lightweight specs -- shared read-only by every per-window row runner in this leg.
        spec_list = [HamiltonianSpec.from_simulation(s) for s in setup.simulations]
        # C2: per-system scope. A global set would false-positive because flat_bottom_setup is a
        # second system_type="complex" setup that shares complex's endstate output_dir.
        seen = set()
        system_rows = []
        for sim in setup.simulations:
            # no_flat_bottom stays a Hamiltonian COLUMN (kept in spec_list) but is never a completed
            # trajectory-ROW -- matches the legacy run() outer-loop skip.
            if sim.directory_args.get("state_label") == "no_flat_bottom":
                continue
            assert (
                sim.output_dir not in seen
            ), f"duplicate MD-window output_dir within system: {sim.output_dir}"
            seen.add(sim.output_dir)
            sim.export_network = export

            if config.workflow.md_only:
                # Phases 5+6 are fused, so not creating the post row is the only way to run MD alone.
                if _needs_md(sim, is_flat_bottom, config):
                    job.addChild(sim)
                continue

            if _needs_md(sim, is_flat_bottom, config):
                md = job.addChild(sim)  # MD window enters the FIFO queue in system order
                post_runner = _post_row_runner(
                    config,
                    spec_list,
                    distruct,
                    restrict={sim.output_dir},
                    traj_map={sim.output_dir: md.rv(1)},  # trajectory FileID promise
                )
                md.addFollowOn(post_runner)  # post ROW couples to THIS window's MD (back of queue)
            else:
                # endstate (preset inptraj) / warm-resume: no MD parent. Trajectory flows from the
                # window's own inptraj or from disk via _get_md_traj (export-on).
                post_runner = _post_row_runner(
                    config, spec_list, distruct, restrict={sim.output_dir}, traj_map={}
                )
                job.addChild(post_runner)
            system_rows.append(post_runner.rv())
        rows_by_system.append(system_rows)

    # Return the 4 per-system lists of per-window post-row runner promises. The aggregator is wired
    # as a FOLLOW-ON of this dispatcher (in ddm_workflow), not inside it: a follow-on waits for this
    # dispatcher's ENTIRE children-subtree (all MD windows + their coupled post rows + the score-only
    # rows + the create_mdout follow-ons), so every post_runner.rv() below is resolved by aggregation
    # time. This mirrors the proven Phase6->Phase7 pattern (consumer follow-on waits for the
    # rv-producing subtree). Order: complex, receptor, ligand, flat_bottom.
    return (
        rows_by_system[0],
        rows_by_system[1],
        rows_by_system[2],
        rows_by_system[3],
    )


def _aggregate_post_output(job, complex_rows, receptor_rows, ligand_rows, flat_rows):
    """Flatten each system's per-window post rows (each a resolved post-runner carrying
    ``.post_output``) into one flat list, wrapped in a ``PostOutputBundle`` for Phase 7. Toil
    resolves the nested ``create_mdout_dataframe`` dataframe promises on unpickle into this job
    (same as today's single-runner -> MBAR path). Order is irrelevant: ``compute_mbar`` concatenates
    and reindexes by state label.
    """

    def _flatten(rows):
        flat = []
        for runner in rows:
            flat.extend(runner.post_output)
        return PostOutputBundle(flat)

    return (
        _flatten(complex_rows),
        _flatten(receptor_rows),
        _flatten(ligand_rows),
        _flatten(flat_rows),
    )



def compute_free_energy_and_consolidate(job, post_complex_analysis, post_receptor_analysis, post_ligand_analysis, flat_bottom_analysis, config: Config):
    """
    Phase 7: Compute free energy and consolidate results.
    
    This phase:
    1. Runs MBAR analysis for complex, ligand, and receptor systems
    2. Runs MBAR analysis for flat bottom contribution
    3. Consolidates output data if enabled
    4. Generates final results and plots
    
    Parameters
    ----------
    job : JobFunctionWrappingJob
        Current Toil job
    post_complex_analysis : JobFunctionWrappingJob
        Job containing complex energy post-analysis results
    post_receptor_analysis : JobFunctionWrappingJob
        Job containing receptor energy post-analysis results
    post_ligand_analysis : JobFunctionWrappingJob
        Job containing ligand energy post-analysis results
    flat_bottom_analysis : JobFunctionWrappingJob
        Job containing flat bottom energy post-analysis results
    config : Config
        Configuration object
        
    Returns
    -------
    dict or Config
        ``{"consolidate": <one-row deltaG DataFrame or None>, "pairs": {leg: DataFrame} or None}``,
        as promises: the binding free energy ConsolidateData wrote, and (with ``adaptive_lambda``) each
        leg's adjacent two-state overlaps for post-production ALS. The config when post-analysis is off.
    """
    # Run MBAR analysis for each system
    # intermediate_jobs returns: [complex_post_analysis, receptor_post_analysis, ligand_post_analysis, flat_bottom_exp, restraints]
    
    
    if config.workflow.run_post_analysis:
        # perform exponntial averaging -> flat bottom restraint contribution
        flat_bottom_exp = job.addChildJobFn(
            run_exponential_averaging,
            flat_bottom_analysis,  # flat bottom post-analysis results
            config.intermediate_args.temperature,
            accelerators=config.system_settings.mbar_accelerators,  # 0 = CPU (pymbar); keeps the analysis tail GPU-free so GPUs release during post-processing. Set system_settings.mbar_accelerators=1 when MBAR is JAX/GPU-accelerated.
        )
        complex_mbar_job = job.addChildJobFn(
            run_compute_mbar,
            post_complex_analysis,  # complex post-analysis results
            config,
            "complex",
            accelerators=config.system_settings.mbar_accelerators,  # 0 = CPU (pymbar); keeps the analysis tail GPU-free so GPUs release during post-processing. Set system_settings.mbar_accelerators=1 when MBAR is JAX/GPU-accelerated.
        )
        ligand_mbar_job = job.addChildJobFn(
            run_compute_mbar,
            post_ligand_analysis,  # ligand post-analysis results
            config,
            "ligand",
            accelerators=config.system_settings.mbar_accelerators,  # 0 = CPU (pymbar); keeps the analysis tail GPU-free so GPUs release during post-processing. Set system_settings.mbar_accelerators=1 when MBAR is JAX/GPU-accelerated.
        )
        receptor_mbar_job = job.addChildJobFn(
            run_compute_mbar,
            post_receptor_analysis,  # receptor post-analysis results
            config,
            "receptor",
            accelerators=config.system_settings.mbar_accelerators,  # 0 = CPU (pymbar); keeps the analysis tail GPU-free so GPUs release during post-processing. Set system_settings.mbar_accelerators=1 when MBAR is JAX/GPU-accelerated.
        )
        
        # Post-production ALS reads each leg's adjacent two-state overlaps; one job per leg, after
        # its MBAR, re-subsamples the leg's energy matrix and solves the pairs.
        pairs = None
        if config.workflow.adaptive_lambda:
            pairs = {
                leg: mbar_job.addFollowOnJobFn(leg_pair_overlaps, mbar_job.rv(), leg).rv()
                for leg, mbar_job in (
                    ("complex", complex_mbar_job),
                    ("ligand", ligand_mbar_job),
                    ("receptor", receptor_mbar_job),
                )
            }

        # Consolidate output data if enabled
        consolidation_job = None
        if config.workflow.consolidate_output:
            consolidation_job = job.addFollowOn(
                ConsolidateData(
                    complex_adative_run=complex_mbar_job.rv(),
                    ligand_adaptive_run=ligand_mbar_job.rv(),
                    receptor_adaptive_run=receptor_mbar_job.rv(),
                    flat_botton_run=flat_bottom_exp.rv(),  # flat bottom results
                    temperature=config.intermediate_args.temperature,
                    max_conformation_force=config.intermediate_args.strongest_conformational_exponent,
                    max_orientational_force=max(config.intermediate_args.exponent_orientational_forces),
                    boresch_df=config.inputs["restraints"],  # restraints
                    complex_filename=config.endstate_files.complex_parameter_filename,
                    ligand_filename=config.endstate_files.ligand_parameter_filename,
                    receptor_filename=config.endstate_files.receptor_parameter_filename,
                    working_path=config.system_settings.cache_directory_output,
                    endstate_label=config.endstate_method.endstate_state_label,
                    plot_overlap_matrix=config.workflow.plot_overlap_matrix,
                    decouple_in_solvent=config.intermediate_args.decouple_in_solvent,
                )
            )
            
            # Final completion message
            analysis_complete = consolidation_job.addFollowOnJobFn(
                initilized_jobs,
                message="✓ Phase 7 Complete: Free energy computation and consolidation completed"
            )
        
        return {
            "consolidate": consolidation_job.rv() if consolidation_job is not None else None,
            "pairs": pairs,
        }

    return config


# --------------------------------------------------------------------------------------------------
# Phases 4 -> 7, and post-production adaptive lambda scheduling (ALS)
# --------------------------------------------------------------------------------------------------
def chain_phases_4_to_7(decomposition, endstate, config, flags: Config, audit: bool = False):
    """Wire Phase 4 -> Phases 5/6 -> aggregation -> Phase 7 behind a Phase-3 job.

    Parameters
    ----------
    decomposition : Job
        Phase-3 job (``decompose_system_and_generate_restraints``).
    endstate : Promise or tuple
        Phase-2 endstate results.
    config : Config or Promise
        Config handed to Phase 4.
    flags : Config
        A resolved config whose workflow switches are read while wiring.
    audit : bool, optional
        Put ``audit_existing_windows`` between Phase 4 and the MD, so windows already on disk are
        checked against the restraints and topologies this pass regenerated.

    Returns
    -------
    tuple
        ``(phase4, phases56, phase7)``; ``phase7`` is None under ``md_only``, which stops after MD.
    """
    decomposition.addFollowOnJobFn(
        initilized_jobs,
        message="✓ Phase 3 Complete: System decomposition and restraint generation finished",
    )
    phase4 = decomposition.addFollowOnJobFn(
        setup_intermediate_simulations, decomposition.rv(), endstate, config
    )
    phase4.addFollowOnJobFn(
        initilized_jobs, message="✓ Phase 4 Complete: Intermidate simulations setup finished"
    )
    gate = phase4
    if audit:
        gate = phase4.addFollowOnJobFn(
            audit_existing_windows, phase4.rv(1), phase4.rv(2), phase4.rv(3)
        )

    # Phases 5 + 6 (merged): every MD window is submitted first and each window's post-analysis row
    # is coupled to its own MD job, so the N^2 CPU post-analysis backfills cores as trajectories land.
    phases56 = gate.addFollowOnJobFn(
        run_intermediate_and_post,
        phase4.rv(0),  # config
        phase4.rv(1),  # complex simulations
        phase4.rv(2),  # receptor simulations
        phase4.rv(3),  # ligand simulations
        phase4.rv(4),  # flat bottom simulations
    )
    phases56.addFollowOnJobFn(
        initilized_jobs,
        message="✓ Phases 5-6 Complete: Intermediate MD + energy post-processing finished",
    )
    if flags.workflow.md_only:
        # No post rows were created, so there is nothing to aggregate or solve.
        phases56.addFollowOnJobFn(
            initilized_jobs,
            message="✓ md_only: MD complete. Re-run with md_only off and the same "
                    "output_directory_name to score the trajectories.",
        )
        return phase4, phases56, None

    # A follow-on of the dispatcher waits for its whole MD + post subtree, so every row is resolved.
    aggregate = phases56.addFollowOnJobFn(
        _aggregate_post_output, phases56.rv(0), phases56.rv(1), phases56.rv(2), phases56.rv(3)
    )
    aggregate.addFollowOnJobFn(
        initilized_jobs, message="--> Moving to phase 7: Free energy computation and consolidation"
    )
    phase7 = aggregate.addFollowOnJobFn(
        compute_free_energy_and_consolidate,
        aggregate.rv(0),  # complex post-output bundle
        aggregate.rv(1),  # receptor post-output bundle
        aggregate.rv(2),  # ligand post-output bundle
        aggregate.rv(3),  # flat bottom post-output bundle
        phase4.rv(0),  # config
    )
    phase7.addFollowOnJobFn(
        initilized_jobs, message="✓ Phase 7 Complete: Free energy computation and consolidation finished"
    )
    return phase4, phases56, phase7


def _restraint_blocks(text: str) -> list:
    """The ``&rst ... /`` blocks of a restraint file, whitespace-normalised and sorted.

    The conformational network comes from a set of atom pairs, so two generations of the same
    network can list identical blocks in a different order.
    """
    blocks = [" ".join(block.split()) for block in text.split("&rst")[1:]]
    return sorted(blocks)


def _topology_body(path: str) -> list:
    """Topology lines without the ``%VERSION`` stamp parmed writes with the current time."""
    with open(path) as fh:
        return [line for line in fh if not line.startswith("%VERSION")]


def audit_existing_windows(job, *setups):
    """Check every window already on disk against the inputs this pass regenerated.

    A warm resume reuses a window's trajectory and its scored cells, and scores new cells for it
    under the regenerated Hamiltonians. That is only consistent if the restraint file and topology
    the window originally ran with are the ones this pass rebuilt. Each exported window directory
    keeps a copy of both, so they are compared before any new MD starts.

    Parameters
    ----------
    job : JobFunctionWrappingJob
    *setups : SimulationSetup
        Phase-4 setups (complex, receptor, ligand).

    Raises
    ------
    RuntimeError
        If any existing window's restraint file or topology differs from the regenerated one.
    """
    checked, differ = 0, []
    for setup in setups:
        for sim in setup.simulations:
            if sim.inptraj is not None or sim.directory_args.get("state_label") == "no_flat_bottom":
                continue
            if not IntermidateRunner._check_mdout(sim):
                continue
            checked += 1
            problems = []
            on_disk = os.path.join(sim.output_dir, "restraint.RST")
            restraint = sim.restraint_file
            if isinstance(restraint, RestraintMaker):
                restraint = restraint.restraints.get(sim.restraint_key)
            if restraint is not None and os.path.exists(on_disk):
                with open(job.fileStore.readGlobalFile(restraint)) as fh:
                    new = fh.read()
                with open(on_disk) as fh:
                    old = fh.read()
                if new != old and _restraint_blocks(new) != _restraint_blocks(old):
                    problems.append("restraint file")
            parms = glob.glob(os.path.join(sim.output_dir, "*.parm7"))
            if sim.prmtop is not None and len(parms) == 1:
                if _topology_body(job.fileStore.readGlobalFile(sim.prmtop)) != _topology_body(parms[0]):
                    problems.append("topology")
            if problems:
                differ.append(f"{sim.output_dir}: {', '.join(problems)}")
    job.fileStore.logToMaster(
        f"[ALS] audit: {checked} existing windows checked against the regenerated inputs, "
        f"{len(differ)} differ"
    )
    if differ:
        shown = "\n  ".join(differ[:20])
        raise RuntimeError(
            f"{len(differ)} existing window(s) ran with inputs that differ from the ones this pass "
            f"regenerated; resuming would score their trajectories under a different Hamiltonian:\n  {shown}"
        )


def leg_pair_overlaps(job, mbar_run, leg: str) -> pd.DataFrame:
    """Two-state overlap of every adjacent pair in one leg, from its Phase-7 energy matrix.

    Parameters
    ----------
    mbar_run : tuple
        ``compute_mbar`` output; ``[1]`` is the cycle-ordered reduced-energy matrix.
    leg : str
        ``complex``, ``ligand`` or ``receptor`` (for the log).

    Returns
    -------
    pandas.DataFrame
        ``als_refine.adjacent_pair_overlaps`` table.
    """
    pairs = als_refine.adjacent_pair_overlaps(als_refine.subsample_like_production(mbar_run[1]))
    job.fileStore.logToMaster(
        f"[ALS] {leg}: {len(pairs)} adjacent pairs, smallest two-state overlap "
        f"{pairs.o_pair.min():.4f}"
    )
    return pairs


def als_round(job, round_index: int, config: Config, endstate):
    """One ALS round: Phase 3 -> window audit -> Phase 4 -> Phases 5/6 -> Phase 7.

    Round 0 is ordinary production. Later rounds run on a config whose window lists were extended;
    windows and energy cells already on disk are reused, so only the new ones are computed. The
    endstate (Phase 2) is the same in every round.

    Returns
    -------
    dict
        ``{"config": <Phase-4 config>, "phase7": <Phase-7 dict>}`` as promises.
    """
    job.fileStore.logToMaster(
        f"[ALS] round {round_index}: "
        f"{len(config.intermediate_args.exponent_conformational_forces)} restraint windows, "
        f"charges {sorted(config.intermediate_args.charges_lambda_window)}, "
        f"lj {config.intermediate_args.lj_scale_windows}, "
        f"screening {config.intermediate_args.screening_windows}"
    )
    decomposition = job.addChildJobFn(decompose_system_and_generate_restraints, endstate, config)
    phase4, _, phase7 = chain_phases_4_to_7(decomposition, endstate, config, config, audit=True)
    return {"config": phase4.rv(0), "phase7": phase7.rv()}


_ROUND_FILES = ("deltaG_*.h5", "*_fe.h5", "*_error.h5", "*_O_MBAR.h5", "boresch_*.h5")


def _write_round_snapshot(out_dir: str, round_index: int, plan, schedule_lists: dict):
    """Copy this round's results and decisions to ``<out_dir>/als/round_XX/``."""
    round_dir = os.path.join(out_dir, "als", f"round_{round_index:02d}")
    os.makedirs(round_dir, exist_ok=True)
    for pattern in _ROUND_FILES:
        for path in glob.glob(os.path.join(out_dir, pattern)):
            shutil.copy2(path, round_dir)
    plan.records.to_csv(os.path.join(round_dir, "pairs.csv"), index=False)
    with open(os.path.join(round_dir, "plan.json"), "w") as fh:
        json.dump({**plan.summary(), "schedule": schedule_lists}, fh, indent=2)


def _unresolved_gaps(records: pd.DataFrame, stop: str) -> list:
    """Gap rows still below the threshold when ALS stops.

    On ``max-rounds`` the gaps the plan would have filled next are unresolved too; they are reported
    as ``weak_rounds_exhausted`` so they cannot be read as filled.
    """
    statuses = ["weak_pool_exhausted", "weak_budget_deferred", "weak_band_disabled"]
    if stop == "max-rounds":
        statuses.append("weak_fill")
    out = records[records.status.isin(statuses)].copy()
    out["status"] = out.status.replace("weak_fill", "weak_rounds_exhausted")
    return out.to_dict("records")


def _schedule_lists(config: Config) -> dict:
    args = config.intermediate_args
    return dict(
        restraint=[list(pair) for pair in zip(args.exponent_conformational_forces_list,
                                               args.exponent_orientational_forces_list)],
        charges=sorted(args.charges_lambda_window),
        lj=list(args.lj_scale_windows),
        screening=list(args.screening_windows),
        dielectric=list(args.gb_extdiel_windows),
    )


def refine_schedule(job, round_index: int, round_result: dict, endstate, history: list):
    """Plan the next ALS round from this round's pairwise overlaps, then run it or stop.

    Parameters
    ----------
    round_index : int
        The round whose results are in ``round_result``.
    round_result : dict
        ``als_round`` output: the round's Phase-4 config and Phase-7 dict.
    endstate : Promise or tuple
        Phase-2 endstate results, reused unchanged.
    history : list of dict
        One summary row per earlier round.

    Returns
    -------
    dict
        Final stop reason, number of rounds, and the final window lists.
    """
    config = round_result["config"]
    phase7 = round_result["phase7"]
    args = config.intermediate_args
    out_dir = cache_output_path(
        config.system_settings.cache_directory_output,
        config.endstate_files.complex_parameter_filename,
    )
    deltaG = phase7["consolidate"]
    dG = float(deltaG["deltaG"].iloc[0]) if deltaG is not None else float("nan")

    if round_index == 0 and args.refine_expected_deltaG is not None:
        if not abs(dG - float(args.refine_expected_deltaG)) <= 1e-6:
            raise RuntimeError(
                f"[ALS] round 0 reproduced dG = {dG!r}, expected {args.refine_expected_deltaG!r}; the "
                "resumed run is not scoring the same data, so nothing will be inserted"
            )

    plan = als_refine.plan_refinement(
        phase7["pairs"],
        als_refine.Schedule.from_config(config),
        args.min_degree_overlap,
        bands=args.als_bands,
        pools={
            "restraint": args.candidate_conformational_pool,
            "charge": args.candidate_charge_pool,
            "lj": args.candidate_lj_pool,
            "screening": args.candidate_screening_pool,
            "dielectric": args.candidate_dielectric_pool,
        },
        min_steps=args.als_min_step,
        max_new_windows=args.als_max_new_windows_per_round,
    )
    schedule = _schedule_lists(config)
    _write_round_snapshot(out_dir, round_index, plan, schedule)
    for gap in plan.endstate_gaps.itertuples(index=False):
        job.fileStore.logToMaster(
            f"[ALS] round {round_index}: weak endstate pair ({gap.leg}) {gap.a} -> {gap.b}: "
            f"{gap.o_pair:.4f} (reported only)"
        )
    row = dict(round=round_index, deltaG=dG, threshold=plan.threshold, reason=plan.reason or "inserted",
               n_insertions=plan.n_insertions, n_md_windows=plan.n_md_windows,
               weak_endstate_gaps=len(plan.endstate_gaps),
               insertions=json.dumps(plan.summary()["insertions"]),
               **{f"n_{key}": len(value) for key, value in schedule.items()})
    history = list(history) + [row]
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "als", "history.csv"), index=False)
    job.fileStore.logToMaster(f"[ALS] round {round_index}: dG {dG:.3f}, plan {plan.summary()}")

    stop = None
    if plan.n_insertions == 0:
        stop = plan.reason
    elif round_index >= int(args.als_max_rounds):
        stop = "max-rounds"
    if stop is not None:
        final = dict(reason=stop, rounds=round_index, deltaG=dG, schedule=schedule,
                     unresolved=_unresolved_gaps(plan.records, stop),
                     endstate_gaps=plan.endstate_gaps.to_dict("records"))
        with open(os.path.join(out_dir, "als", "final.json"), "w") as fh:
            json.dump(final, fh, indent=2, default=str)
        job.fileStore.logToMaster(f"[ALS] stopped after round {round_index}: {stop}")
        return {"reason": stop, "rounds": round_index, "deltaG": dG, "schedule": schedule}

    extended = als_refine.extend_schedule(config, plan.insertions)
    next_round = job.addChildJobFn(als_round, round_index + 1, extended, endstate)
    return next_round.addFollowOnJobFn(
        refine_schedule, round_index + 1, next_round.rv(), endstate, history
    ).rv()


def update_config(job, config: Config, complex_binding_mode, receptor_binding_mode, ligand_binding_mode, restraints):
    """
    Update configuration with binding modes from endstate simulations.
    
    Parameters
    ----------
    job : JobFunctionWrappingJob
        Current Toil job
    config : Config
        Configuration object to update
    complex_binding_mode : Any
        Complex binding mode from endstate simulation
    receptor_binding_mode : Any
        Receptor binding mode from endstate simulation
    ligand_binding_mode : Any
        Ligand binding mode from endstate simulation
    restraints : Any
        Restraints from system decomposition
    Returns
    -------
    Config
        Updated configuration object
    """
    config.inputs["endstate_complex_lastframe"] = complex_binding_mode
    config.inputs["receptor_endstate_frame"] = receptor_binding_mode
    config.inputs["ligand_endstate_frame"] = ligand_binding_mode
    config.inputs["restraints"] = restraints
    job.fileStore.logToMaster(f"returning updated config: {config}")
    return config


def update_config_endstate(job, config: Config, message: str):
    """
    Update configuration with binding modes from endstate simulations.
    """
    job.fileStore.logToMaster(message)
    return config

def initilized_jobs(job, message: str):
    """
    Placeholder job for synchronization and logging.
    
    This function serves as a synchronization point in the workflow,
    ensuring that dependent jobs only start after previous jobs complete.
    It also provides logging for workflow progress tracking.
    
    Parameters
    ----------
    job : JobFunctionWrappingJob
        Current Toil job
    message : str
        Log message to output
        
    Returns
    -------
    None
    """
    job.fileStore.logToMaster(message)
    return
