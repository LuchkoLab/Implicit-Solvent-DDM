"""
Workflow phase functions for the Double Decoupling Method (DDM) workflow.

This module contains the individual phase functions that make up the DDM workflow,
providing a clean separation of concerns and improved maintainability.
"""

import copy
import logging
import numpy as np
from toil.job import JobFunctionWrappingJob

from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.mdin import get_mdins, generate_extdiel_mdin, get_pilot_mdin
from implicit_solvent_ddm.restraints import (
    BoreschRestraints,
    FlatBottom,
    RestraintMaker,
    write_empty_restraint,
)
from implicit_solvent_ddm.alchemical import alter_topology, split_complex_system
from implicit_solvent_ddm.setup_simulations import SimulationSetup
from implicit_solvent_ddm.runner import IntermidateRunner
from implicit_solvent_ddm.adaptive_restraints import (
    run_exponential_averaging,
    run_compute_mbar,
    adaptive_lambda_windows,
    pilot_dielectric_scheduler,
)
from implicit_solvent_ddm.run_endstate import (
    run_remd,
    run_basic_md,
    user_defined_endstate,
)
from implicit_solvent_ddm.postTreatment import ConsolidateData

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
    mdins = job.addChildJobFn(
        get_mdins, 
        config.intermediate_args.mdin_intermediate_file
    )
    
    # Store MDIN file references in config
    MDIN_TYPES = {
        'default': 0,
        'no_solvent': 1, 
        'post': 2,
        'post_nosolv': 3
    }
    
    config.inputs["default_mdin"] = mdins.rv(MDIN_TYPES['default'])
    config.inputs["no_solvent_mdin"] = mdins.rv(MDIN_TYPES['no_solvent'])
    config.inputs["post_mdin"] = mdins.rv(MDIN_TYPES['post'])
    config.inputs["post_nosolv_mdin"] = mdins.rv(MDIN_TYPES['post_nosolv'])
    
    # Create empty restraint file
    empty_restraint = mdins.addChildJobFn(write_empty_restraint)
    config.inputs["empty_restraint"] = empty_restraint.rv()

    # Setup flat bottom restraint potentials
    flat_bottom_template = mdins.addChild(FlatBottom(config=config))
    config.inputs["flat_bottom_restraint"] = flat_bottom_template.rv(0)

    # ALS pilot: emit a short-MD intermediate mdin used ONLY by the adaptive pilot (Phase 4.5). Behind
    # the adaptive_lambda flag so the static path is byte-identical (no extra job, no inputs key).
    # Always 50 ps by default (pilot_ps), with the step count derived from the user mdin's timestep and
    # ntwx set for ~pilot_frames frames (get_pilot_mdin). pilot_nstlim is an explicit step override for
    # tiny test systems where 50 ps is absurd.
    if config.workflow.adaptive_lambda:
        pilot_mdin = mdins.addChildJobFn(
            get_pilot_mdin,
            config.intermediate_args.mdin_intermediate_file,
            config.intermediate_args.pilot_ps,
            config.intermediate_args.pilot_frames,
            config.intermediate_args.pilot_nstlim,
        )
        # rv(0) = default (solvated) pilot mdin; rv(1) = no-solvent (igb=6 gas-phase) pilot mdin. Both
        # are swapped into the pilot config so EVERY pilot MD state runs at 50 ps (not just the
        # default-mdin states).
        config.inputs["pilot_mdin"] = pilot_mdin.rv(0)
        config.inputs["pilot_no_solvent_mdin"] = pilot_mdin.rv(1)
        # Pilot MD length (steps) + trajectory interval, reused by the GB-dielectric band to shorten the
        # per-epsilon extdiel mdins it generates for ALS-inserted windows.
        config.inputs["pilot_nstlim_steps"] = pilot_mdin.rv(2)
        config.inputs["pilot_ntwx"] = pilot_mdin.rv(3)

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
    
    return endstate_job.rv()


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
            K_r=config.intermediate_args.max_conformational_restraint,
            K_thetaA=config.intermediate_args.max_orientational_restraint,
            K_thetaB=config.intermediate_args.max_orientational_restraint,
            K_phiA=config.intermediate_args.max_orientational_restraint,
            K_phiB=config.intermediate_args.max_orientational_restraint,
            K_phiC=config.intermediate_args.max_orientational_restraint,
        )
    )
    
    # Create restraint files for intermediate simulations
    restraints = boresch_restraints.addChild(
        RestraintMaker(
            config=config,
            complex_binding_mode=endstate_jobs[0],
            boresch_restraints=boresch_restraints.rv(),
            flat_bottom=config.inputs["flat_bottom_restraint"],
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
        endstate_traj=endstate_jobs[3],  # receptor trajectory
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

    # Calculate maximum restraint forces
    max_conformational_force = max(config.intermediate_args.conformational_restraints_forces)
    max_orientational_force = max(config.intermediate_args.orientational_restraint_forces)

    
    # Setup runner jobs for MD simulations
    runner_jobs = job.addChildJobFn(
        initilized_jobs, 
        message="Setting up MD simulations"
    )
    
    # Setup ligand charge scaling simulations
    for charge in config.intermediate_args.charges_lambda_window:
        # Scale ligand charges in isolated ligand system
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

    # Setup complex ligand exclusion simulations (gas phase)
    if config.workflow.complex_ligand_exclusions:
        complex_simulations.setup_remove_gb_solvent_simulation(
            restraint_key=f"complex_{max_conformational_force}_{max_orientational_force}_rst",
            prmtop=job.addChildJobFn(
                alter_topology,
                solute_amber_parm=config.endstate_files.complex_parameter_filename,
                solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                ligand_mask=config.amber_masks.ligand_mask,
                receptor_mask=config.amber_masks.receptor_mask,
                set_charge=0.0,
                exculsions=True,
            ).rv(),
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
            ).rv(),
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
        
        # Pilot setups shorten the per-epsilon extdiel mdins to the pilot MD length; production keeps full
        # length (nstlim/ntwx = None -> no-op). The als_pilot marker is set only on the pilot deepcopy in
        # adaptive_restraint_pilot, so the static/production path is byte-identical.
        gb_nstlim = config.inputs.get("pilot_nstlim_steps") if config.inputs.get("als_pilot") else None
        gb_ntwx = config.inputs.get("pilot_ntwx") if config.inputs.get("als_pilot") else None

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
                    nstlim=gb_nstlim,
                    ntwx=gb_ntwx,
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
                    nstlim=gb_nstlim,
                    ntwx=gb_ntwx,
                ).rv(),
            )

    # Setup restraint force windows
    config.intermediate_args.exponent_conformational_forces_list = []
    config.intermediate_args.exponent_orientational_forces_list = []
    
    for con_force, orien_force in zip(
        config.intermediate_args.conformational_restraints_forces,
        config.intermediate_args.orientational_restraint_forces,
    ):
        exponent_conformational = round(np.log2(con_force), 3)
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



def run_intermediate_simulations(job, config: Config, complex_simulations, receptor_simulations, ligand_simulations, flat_bottom_setup):

    """
    Phase 5: Run intermediate state simulations.
    
    This phase:
    1. Runs intermediate MD simulations with restraints
    2. Runs flat bottom simulation
    Parameters
    ----------
    job : JobFunctionWrappingJob
        Current Toil job
    config : Config
        Configuration object
    complex_simulations : SimulationSetup
        Complex simulation setup
    receptor_simulations : SimulationSetup
        Receptor simulation setup
    ligand_simulations : SimulationSetup
        Ligand simulation setup
    flat_bottom_simulations : SimulationSetup
        Flat bottom simulation setup
    """
    job.fileStore.logToMaster(f"Calling run_intermediate_simulations")


    # Setup flat bottom contribution analysis
    flat_bottom_analysis = job.addChild(
        IntermidateRunner(
            flat_bottom_setup.simulations,
            config.inputs["restraints"],  # restraints
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_halo",
            post_only=config.workflow.post_analysis_only,
            config=config,
        )
    )

    
    # Run complex intermediate simulations
    intermediate_complex = job.addChild(
        IntermidateRunner(
            complex_simulations.simulations,
            config.inputs["restraints"],  # restraints
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_halo",
            post_only=False,
            config=config,
        )
    )
    
    # Run receptor intermediate simulations
    intermediate_receptor = job.addChild(
        IntermidateRunner(
            receptor_simulations.simulations,
            config.inputs["restraints"],  # restraints
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_apo",
            post_only=False,
            config=config,
        )
    )

    # Run ligand intermediate simulations
    intermediate_ligand = job.addChild(
        IntermidateRunner(
            ligand_simulations.simulations,
            config.inputs["restraints"],  # restraints
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_apo",
            post_only=False,
            config=config,
        )
    )
    

def run_post_analysis_intermediate_simulations(job, config: Config, complex_simulations, receptor_simulations, ligand_simulations, flat_bottom_simulations):
    """
    Phase 6: Run Energy post-processing with sander imin=5
    This phase:
    1. Runs post-analysis for complex, ligand, and receptor systems
    2. Runs post-analysis for flat bottom simulation

    Parameters
    ----------
    job : JobFunctionWrappingJob
        Current Toil job
    config : Config
        Configuration object
    complex_simulations : SimulationSetup
        Complex simulation setup
    receptor_simulations : SimulationSetup
        Receptor simulation setup
    ligand_simulations : SimulationSetup
        Ligand simulation setup
    flat_bottom_simulations : SimulationSetup
        Flat bottom simulation setup
    
    Returns
    -------
    post_analyses_intermediate_complex : JobFunctionWrappingJob
        Complex energy post-analysis results
    post_analyses_intermediate_receptor : JobFunctionWrappingJob
        Receptor energy post-analysis results
    post_analyses_intermediate_ligand : JobFunctionWrappingJob
        Ligand energy post-analysis results
    flat_bottom_analysis : JobFunctionWrappingJob
        Flat bottom energy post-analysis results
    """
    # Mark MD jobs as completed - this ensures all MD simulations finish before post-analysis

    flat_bottom_analysis = job.addChild(
        IntermidateRunner(
            flat_bottom_simulations.simulations,
            config.inputs["restraints"],  # restraints
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_halo",
            post_only=True,
            config=config,
        )
    ).rv()

    # Perform post-analysis on intermediate simulations (only after all MD jobs complete)
    post_analyses_intermediate_complex = job.addChild(
        IntermidateRunner(
            complex_simulations.simulations,
            config.inputs["restraints"],  # restraints
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_halo",
            post_only=True,
            config=config,
        )
    ).rv()
    
    post_analyses_intermediate_receptor = job.addChild(
        IntermidateRunner(
            receptor_simulations.simulations,
            config.inputs["restraints"],  # restraints
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_apo",
            post_only=True,
            config=config,
        )
    ).rv()
    
    post_analyses_intermediate_ligand = job.addChild(
        IntermidateRunner(
            ligand_simulations.simulations,
            config.inputs["restraints"],  # restraints
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_apo",
            post_only=True,
            config=config,
        )
    ).rv()
    
    return post_analyses_intermediate_complex, post_analyses_intermediate_receptor, post_analyses_intermediate_ligand, flat_bottom_analysis



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
    tuple[JobFunctionWrappingJob, JobFunctionWrappingJob, JobFunctionWrappingJob]
        MBAR analysis jobs for complex, ligand, and receptor systems and flat bottom energy post-analysis results
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
        
        # Consolidate output data if enabled
        if config.workflow.consolidate_output:
            consolidation_job = job.addFollowOn(
                ConsolidateData(
                    complex_adative_run=complex_mbar_job.rv(),
                    ligand_adaptive_run=ligand_mbar_job.rv(),
                    receptor_adaptive_run=receptor_mbar_job.rv(),
                    flat_botton_run=flat_bottom_exp.rv(),  # flat bottom results
                    temperature=config.intermediate_args.temperature,
                    max_conformation_force=max(config.intermediate_args.exponent_conformational_forces),
                    max_orientational_force=max(config.intermediate_args.exponent_orientational_forces),
                    boresch_df=config.inputs["restraints"],  # restraints
                    complex_filename=config.endstate_files.complex_parameter_filename,
                    ligand_filename=config.endstate_files.ligand_parameter_filename,
                    receptor_filename=config.endstate_files.receptor_parameter_filename,
                    working_path=config.system_settings.cache_directory_output,
                    plot_overlap_matrix=config.workflow.plot_overlap_matrix,
                )
            )
            
            # Final completion message
            analysis_complete = consolidation_job.addFollowOnJobFn(
                initilized_jobs,
                message="✓ Phase 7 Complete: Free energy computation and consolidation completed"
            )
        
        return consolidation_job.rv()

    return config


def adaptive_restraint_pilot(job, decomposition_jobs, endstate_jobs, config: Config):
    """Phase 4.5: Adaptive Lambda Scheduler pilot (OBSERVATIONAL, flag-gated).

    Runs a SHORT-MD copy of the intermediate cycle and drives the R-ADD restraint scheduler
    (``adaptive_lambda_windows``) to convergence on the COMPLEX leg, then LOGS the converged restraint
    schedule. **It does not feed production**: the production Phases 5/6/7 still read the Phase-4 seed
    setups, so the validated free-energy path is byte-for-byte unchanged. The converged config is
    returned for a future Step 5b (rebuild + re-thread) but is ignored by the DAG in 5a.

    The pilot is isolated from production two ways, both set on a deepcopy ``pilot_config`` (the
    production ``config`` is never mutated):
      * **short MD** — ``inputs["default_mdin"]`` is swapped to the short ``inputs["pilot_mdin"]`` so
        every pilot window (including ALS-inserted ones) runs at ``pilot_nstlim``;
      * **separate output tree** — ``system_settings.output_directory_name`` gets a ``_pilot`` suffix so
        the pilot's short trajectories never land in the production dirs (else production
        ``_check_mdout`` would skip MD and compute ΔG on 50-ps data).

    Only the complex leg is run: ``adaptive_lambda_windows("complex", ...)`` needs only the complex
    ``post_output``, and the complex ``SimulationSetup`` already carries every ``complex_order`` state
    (anchors, charges, restraint windows, endstate) so the pilot MBAR has all columns.

    Parameters
    ----------
    decomposition_jobs, endstate_jobs : resolved Phase-3 / Phase-2 outputs (restraint templates and
        endstate trajectories) — reused read-only, exactly as Phase 4 consumes them.
    config : Config
        The Phase-1 config (carries ``inputs["pilot_mdin"]`` when the flag is on).
    """
    if config.inputs.get("pilot_mdin") is None:
        # Misconfigured (flag on but no pilot_nstlim): fail loud rather than silently piloting at
        # production length, which would defeat the whole point of a cheap pilot.
        raise ValueError(
            "adaptive_restraint_pilot: config.inputs['pilot_mdin'] is unset. Set "
            "intermediate_args.pilot_nstlim with workflow.adaptive_lambda=True."
        )

    pilot_config = copy.deepcopy(config)
    # Short MD for the whole pilot cycle. The cycle uses TWO MD mdins: default (solvated) for restraint
    # windows + solvated charge states, and no_solvent (igb=6) for the gas-phase no_interactions /
    # interactions / igb=6 charge states. Swap BOTH to their 50 ps pilot versions — swapping only
    # default_mdin leaves the gas-phase states running at full production length.
    pilot_config.inputs["default_mdin"] = pilot_config.inputs["pilot_mdin"]
    pilot_config.inputs["no_solvent_mdin"] = pilot_config.inputs["pilot_no_solvent_mdin"]
    # The ALS restraint overlap is a banded MBAR over the restraint windows + their max-restraint
    # anchor only (adaptive_lambda_windows -> compute_mbar(restraint_band=True)); the endstate is not in
    # that band. So skip re-scoring it entirely — it removes the expensive full-length endstate
    # re-scoring from the pilot and guarantees no endstate column/rows leak into the pilot data.
    pilot_config.workflow.end_state_postprocess = False
    # Mark this as the pilot setup so the GB-dielectric setup loop shortens its per-epsilon extdiel mdins
    # to the pilot MD length (setup_intermediate_simulations checks inputs["als_pilot"]); production is
    # unaffected since this marker lives only on the pilot deepcopy.
    pilot_config.inputs["als_pilot"] = True
    # Isolate the pilot output tree (top_directory_path = working_directory/output_directory_name).
    pilot_config.system_settings.output_directory_name = (
        pilot_config.system_settings.output_directory_name + "_pilot"
    )

    job.fileStore.logToMaster(
        f"[ALS][pilot] Phase 4.5 starting. pilot output dir: "
        f"{pilot_config.system_settings.top_directory_path}"
    )

    # Reuse Phase 4 verbatim on the pilot config -> pilot SimulationSetup objects (short mdin, isolated
    # dir). rv(0) carries the binding modes + pilot mdin + seed _list; rv(1) is the complex setup.
    setup_pilot = job.addChildJobFn(
        setup_intermediate_simulations,
        decomposition_jobs,
        endstate_jobs,
        pilot_config,
    )

    return setup_pilot.addFollowOnJobFn(
        _pilot_md_post_drive,
        setup_pilot.rv(0),  # pilot config (binding modes + pilot mdin + seed _list)
        setup_pilot.rv(1),  # complex SimulationSetup
        setup_pilot.rv(2),  # receptor SimulationSetup (for the matched dielectric band)
    ).rv()


def _pilot_runner(simulations, pilot_config, distruct, post_only):
    """Build one pilot IntermidateRunner (short MD or post-analysis pass)."""
    return IntermidateRunner(
        simulations,
        pilot_config.inputs["restraints"],
        post_process_no_solv_mdin=pilot_config.inputs["post_nosolv_mdin"],
        post_process_mdin=pilot_config.inputs["post_mdin"],
        post_process_distruct=distruct,
        post_only=post_only,
        config=pilot_config,
    )


def _pilot_md_post_drive(job, pilot_config: Config, complex_setup, receptor_setup):
    """Run the complex AND receptor pilot legs (short MD -> post-analysis), then drive the band passes.

    Both legs are needed because the GB-dielectric band is scheduled on a MATCHED complex+receptor
    schedule (so the host desolvation cancels). Each leg runs MD (``post_only=False``) then post-analysis
    (``post_only=True``); a follow-on join then drives the per-band R-ADD passes in
    ``_pilot_band_passes``. ``run()`` returns the runner with a populated ``post_output``.
    """
    c_md = job.addChild(_pilot_runner(complex_setup.simulations, pilot_config, "post_process_halo", False))
    c_post = c_md.addFollowOn(
        _pilot_runner(complex_setup.simulations, pilot_config, "post_process_halo", True)
    )
    r_md = job.addChild(_pilot_runner(receptor_setup.simulations, pilot_config, "post_process_apo", False))
    r_post = r_md.addFollowOn(
        _pilot_runner(receptor_setup.simulations, pilot_config, "post_process_apo", True)
    )

    # The band passes run after BOTH legs' post-analysis (a follow-on of `job` waits on all its children).
    drive = job.addFollowOnJobFn(
        _pilot_band_passes, c_post.rv(), r_post.rv(), pilot_config
    )
    return drive.rv()


def _pilot_band_passes(job, complex_runner, receptor_runner, pilot_config: Config):
    """Drive the per-band R-ADD passes sequentially, threading the converged config forward.

    Order: matched dielectric (complex + receptor) -> charge (complex) -> restraints (complex). Each pass
    reads its own banded overlap and inserts in its own coordinate; the converged config (expanded window
    lists) is threaded into the next pass and ultimately returned for the production rebuild (Stage D).
    """
    # Pass 1 — matched GB-dielectric on both legs. Returns (complex_runner, receptor_runner, config).
    diel = job.addChildJobFn(
        pilot_dielectric_scheduler, complex_runner, receptor_runner, pilot_config
    )
    # Pass 2 — charge band on the complex leg. adaptive_lambda_windows -> (results, config, runner).
    charge = diel.addFollowOnJobFn(
        adaptive_lambda_windows,
        diel.rv(0),  # complex runner (post_output now carries the inserted dielectric windows)
        diel.rv(2),  # dielectric-converged config
        "complex",
        charge_scaling=True,
    )
    # Pass 3 — restraint band on the complex leg.
    restr = charge.addFollowOnJobFn(
        adaptive_lambda_windows,
        charge.rv(2),  # complex runner
        charge.rv(1),  # charge-converged config
        "complex",
        restraints_scaling=True,
    )
    restr.addFollowOnJobFn(_log_pilot_schedule, restr.rv(1))
    return restr.rv(1)


def apply_converged_schedule(production_config: Config, converged_config: Config) -> Config:
    """Pure: return a deep copy of ``production_config`` with the pilot's converged dielectric, charge,
    AND restraint schedules applied (full-length mdin / production dir preserved).

    * Dielectric + charge bands: the production setup loops iterate ``gb_extdiel_windows`` /
      ``charges_lambda_window`` directly, so replacing those window-value lists is sufficient.
    * Restraint band: the production setup (and ``RestraintMaker``) iterate the FORCE lists
      ``conformational_restraints_forces`` / ``orientational_restraint_forces`` (= ``2**exponent``). The
      pilot records its converged schedule in the PAIRED ``exponent_*_forces_list`` (seed + R-ADD
      insertions, ``con``/``orient`` aligned by index), so we carry those into the production seed
      exponents and recompute the forces. The caller must re-run
      ``decompose_system_and_generate_restraints`` on the result so ``RestraintMaker`` materializes a
      restraint file for every window — including the inserted ones — keyed by force.

    R-ADD anchor protection keeps every inserted exponent strictly inside ``(min, max)``, so
    ``max_*_restraint`` and the analytical Boresch ΔG are unchanged; re-decomposition only adds the
    inserted *interior* restraint files.
    """
    merged = copy.deepcopy(production_config)
    conv = converged_config.intermediate_args
    m = merged.intermediate_args

    m.gb_extdiel_windows = list(conv.gb_extdiel_windows)
    m.charges_lambda_window = list(conv.charges_lambda_window)

    con_exps = list(conv.exponent_conformational_forces_list)
    orient_exps = list(conv.exponent_orientational_forces_list)
    if con_exps and len(con_exps) == len(orient_exps):
        m.exponent_conformational_forces = con_exps
        m.exponent_orientational_forces = orient_exps
        m.conformational_restraints_forces = np.exp2(con_exps)
        m.orientational_restraint_forces = np.exp2(orient_exps)
    return merged


def merge_pilot_windows(job, production_config: Config, converged_config: Config):
    """Close the loop (Toil wrapper): apply the pilot's converged dielectric + charge + RESTRAINT
    schedules to a PRODUCTION config (full-length mdin, production output dir — NOT the pilot's short
    mdin / ``_pilot`` dir) and return it.

    The caller re-runs ``decompose_system_and_generate_restraints`` on the returned config so
    ``RestraintMaker`` materializes restraint files for the pilot-inserted exponents (the seed
    RestraintMaker only carries seed-window files), then rebuilds the production ``SimulationSetup``s —
    so Phases 5/6/7 run full-length MD on the ALL pilot-determined windows (dielectric, charge, AND
    restraint). The pure schedule merge lives in :func:`apply_converged_schedule` (unit-tested).
    """
    merged = apply_converged_schedule(production_config, converged_config)
    m = merged.intermediate_args
    job.fileStore.logToMaster(
        f"[ALS][pilot] rebuilding production from converged schedule: "
        f"{len(m.gb_extdiel_windows)} GB dielectric windows, "
        f"{len(m.charges_lambda_window)} charge windows, "
        f"{len(m.conformational_restraints_forces)} restraint windows"
    )
    return merged


def _log_pilot_schedule(job, converged_config: Config):
    """Terminal pilot follow-on: log the converged schedules across all bands (loop-close deliverable)."""
    con = sorted(converged_config.intermediate_args.exponent_conformational_forces_list)
    orient = sorted(converged_config.intermediate_args.exponent_orientational_forces_list)
    charges = sorted(converged_config.intermediate_args.charges_lambda_window)
    eps = sorted(converged_config.intermediate_args.gb_extdiel_windows)
    job.fileStore.logToMaster(
        f"[ALS][pilot] CONVERGED schedules feeding production:\n"
        f"[ALS][pilot]   restraint conformational exponents ({len(con)}): {con}\n"
        f"[ALS][pilot]   restraint orientational  exponents ({len(orient)}): {orient}\n"
        f"[ALS][pilot]   charge windows ({len(charges)}): {charges}\n"
        f"[ALS][pilot]   GB dielectric windows ({len(eps)}): {eps}"
    )
    return converged_config


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
