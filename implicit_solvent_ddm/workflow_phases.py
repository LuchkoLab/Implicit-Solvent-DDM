"""
Workflow phase functions for the Double Decoupling Method (DDM) workflow.

This module contains the individual phase functions that make up the DDM workflow,
providing a clean separation of concerns and improved maintainability.
"""

import logging
import numpy as np
from toil.job import JobFunctionWrappingJob

from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.mdin import get_mdins, generate_extdiel_mdin
from implicit_solvent_ddm.restraints import (
    BoreschRestraints,
    FlatBottom,
    RestraintMaker,
    write_empty_restraint,
)
from implicit_solvent_ddm.alchemical import alter_topology, split_complex_system
from implicit_solvent_ddm.setup_simulations import SimulationSetup
from implicit_solvent_ddm.runner import IntermidateRunner
from implicit_solvent_ddm.adaptive_restraints import run_exponential_averaging, run_compute_mbar
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
            accelerators=config.system_settings.num_accelerators,  # GPU slot for MBAR (JAX target; pymbar still runs CPU until jax is enabled)
        )
        complex_mbar_job = job.addChildJobFn(
            run_compute_mbar,
            post_complex_analysis,  # complex post-analysis results
            config,
            "complex",
            accelerators=config.system_settings.num_accelerators,  # GPU slot for MBAR (JAX target; pymbar still runs CPU until jax is enabled)
        )
        ligand_mbar_job = job.addChildJobFn(
            run_compute_mbar,
            post_ligand_analysis,  # ligand post-analysis results
            config,
            "ligand",
            accelerators=config.system_settings.num_accelerators,  # GPU slot for MBAR (JAX target; pymbar still runs CPU until jax is enabled)
        )
        receptor_mbar_job = job.addChildJobFn(
            run_compute_mbar,
            post_receptor_analysis,  # receptor post-analysis results
            config,
            "receptor",
            accelerators=config.system_settings.num_accelerators,  # GPU slot for MBAR (JAX target; pymbar still runs CPU until jax is enabled)
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
