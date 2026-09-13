"""
Functions that setup REMD, Basic MD or user defind endstate simulations. 
"""

import copy

# from implicit_solvent_ddm.remd import run_remd
import os
import os.path

from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.mdin import generate_replica_mdin
from implicit_solvent_ddm.simulations import (
    ExtractTrajectories,
    REMDSimulation,
    Simulation,
)

working_directory = os.getcwd()


def run_remd(job, user_config: Config):
    """Setup and run REMD.

    Args:
        job (_type_): _description_
        user_config (Config): _description_
    """
    equil_mdins = job.addChildJobFn(
        generate_replica_mdin,
        user_config.endstate_method.remd_args.equil_template_mdin,
        user_config.endstate_method.remd_args.temperatures,
        runtype="relax",
    )
    remd_mdins = equil_mdins.addChildJobFn(
        generate_replica_mdin,
        user_config.endstate_method.remd_args.remd_template_mdin,
        user_config.endstate_method.remd_args.temperatures,
        runtype="remd",
    )

    minimization_complex = remd_mdins.addChild(
        Simulation(
            executable=user_config.system_settings.executable,
            mpi_command=user_config.system_settings.mpi_command,
            num_cores=(0.1 if user_config.system_settings.CUDA else user_config.num_cores_per_system.complex_ncores),
            CUDA=user_config.system_settings.CUDA,
            prmtop=user_config.endstate_files.complex_parameter_filename,
            incrd=user_config.endstate_files.complex_coordinate_filename,
            input_file=user_config.inputs["min_mdin"],
            restraint_file=user_config.inputs["flat_bottom_restraint"],
            system_type="complex",
            directory_args={
                "runtype": "minimization",
                "filename": "min",
                "topology": user_config.endstate_files.complex_parameter_filename,
                "topdir": user_config.system_settings.top_directory_path,
            },
            working_directory=user_config.system_settings.working_directory,
        )
    )
    # config.endstate_method.remd_args.nthreads
    equilibrate_complex = minimization_complex.addFollowOn(
        REMDSimulation(
            executable=user_config.system_settings.executable,
            mpi_command=user_config.system_settings.mpi_command,
            num_cores=user_config.endstate_method.remd_args.nthreads_complex,
            CUDA=user_config.system_settings.CUDA,
            prmtop=user_config.endstate_files.complex_parameter_filename,
            incrd=minimization_complex.rv(0),
            input_file=equil_mdins.rv(),
            restraint_file=user_config.inputs["flat_bottom_restraint"],
            system_type="complex",
            runtype="equil",
            remd_debug=user_config.workflow.debug,
            ngroups=user_config.endstate_method.remd_args.ngroups,
            directory_args={
                "runtype": "equilibration",
                "topology": user_config.endstate_files.complex_parameter_filename,
                "topdir": user_config.system_settings.top_directory_path,
            },
            working_directory=user_config.system_settings.working_directory,
            memory=user_config.system_settings.memory,
            disk=user_config.system_settings.disk,
        )
    )

    remd_complex = equilibrate_complex.addFollowOn(
        REMDSimulation(
            executable=user_config.system_settings.executable,
            mpi_command=user_config.system_settings.mpi_command,
            num_cores=user_config.endstate_method.remd_args.nthreads_complex,
            CUDA=user_config.system_settings.CUDA,
            ngroups=user_config.endstate_method.remd_args.ngroups,
            prmtop=user_config.endstate_files.complex_parameter_filename,
            incrd=equilibrate_complex.rv(0),
            input_file=remd_mdins.rv(),
            system_type="complex",
            working_directory=user_config.system_settings.working_directory,
            restraint_file=user_config.inputs["flat_bottom_restraint"],
            runtype="remd",
            directory_args={
                "runtype": "remd",
                "topology": user_config.endstate_files.complex_parameter_filename,
                "topdir": user_config.system_settings.top_directory_path,
            },
            memory=user_config.system_settings.memory,
            disk=user_config.system_settings.disk,
            remd_debug=user_config.workflow.debug,
        )
    )
    # extact target temparture trajetory and last frame
    extract_complex = remd_complex.addFollowOn(
        ExtractTrajectories(
            user_config.endstate_files.complex_parameter_filename,
            remd_complex.rv(1),
            user_config.intermediate_args.temperature,
        )
    )

    # user_config.inputs["endstate_complex_traj"] = extract_complex.rv(0)

    user_config.inputs["endstate_complex_lastframe"] = extract_complex.rv(1)

    # run minimization at the end states for ligand system only
    minimization_ligand = minimization_complex.addFollowOn(
        Simulation(
            executable=user_config.system_settings.executable,
            mpi_command=user_config.system_settings.mpi_command,
            num_cores=user_config.num_cores_per_system.ligand_ncores,
            CUDA=user_config.system_settings.CUDA,
            prmtop=user_config.endstate_files.ligand_parameter_filename,
            incrd=user_config.endstate_files.ligand_coordinate_filename,
            input_file=user_config.inputs["min_mdin"],
            restraint_file=user_config.inputs["empty_restraint"],
            system_type="ligand",
            directory_args={
                "runtype": "minimization",
                "filename": "min",
                "topology": user_config.endstate_files.ligand_parameter_filename,
                "topdir": user_config.system_settings.top_directory_path,
            },
            working_directory=user_config.system_settings.working_directory,
            memory=user_config.system_settings.memory,
            disk=user_config.system_settings.disk,
        )
    )
    # check to see if PMEMD is specified
    num_ligand_cores = int(user_config.endstate_method.remd_args.nthreads_ligand)
    ligand_endstate_exe = user_config.system_settings.executable
    if "pmemd.MPI" in user_config.system_settings.executable:
        num_ligand_cores = int(
            user_config.endstate_method.remd_args.nthreads_ligand / 2
        )
        ligand_endstate_exe = "sander.MPI"

    equilibrate_ligand = minimization_ligand.addFollowOn(
        REMDSimulation(
            executable=ligand_endstate_exe,
            mpi_command=user_config.system_settings.mpi_command,
            num_cores=num_ligand_cores,
            CUDA=user_config.system_settings.CUDA,
            ngroups=user_config.endstate_method.remd_args.ngroups,
            prmtop=user_config.endstate_files.ligand_parameter_filename,
            incrd=minimization_ligand.rv(0),
            input_file=equil_mdins.rv(),
            restraint_file=user_config.inputs["empty_restraint"],
            runtype="equil",
            system_type="ligand",
            directory_args={
                "runtype": "equilibration",
                "topology": user_config.endstate_files.ligand_parameter_filename,
                "topdir": user_config.system_settings.top_directory_path,
            },
            working_directory=user_config.system_settings.working_directory,
            memory=user_config.system_settings.memory,
            disk=user_config.system_settings.disk,
            remd_debug=user_config.workflow.debug,
        )
    )

    remd_ligand = equilibrate_ligand.addFollowOn(
        REMDSimulation(
            executable=ligand_endstate_exe,
            mpi_command=user_config.system_settings.mpi_command,
            num_cores=num_ligand_cores,
            CUDA=user_config.system_settings.CUDA,
            prmtop=user_config.endstate_files.ligand_parameter_filename,
            incrd=equilibrate_ligand.rv(0),
            input_file=remd_mdins.rv(),
            restraint_file=user_config.inputs["empty_restraint"],
            runtype="remd",
            ngroups=user_config.endstate_method.remd_args.ngroups,
            system_type="ligand",
            directory_args={
                "runtype": "remd",
                "topology": user_config.endstate_files.ligand_parameter_filename,
                "topdir": user_config.system_settings.top_directory_path,
            },
            working_directory=user_config.system_settings.working_directory,
            memory=user_config.system_settings.memory,
            disk=user_config.system_settings.disk,
            remd_debug=user_config.workflow.debug,
        )
    )
    # extact target temparture trajetory and last frame
    extract_ligand_traj = remd_ligand.addFollowOn(
        ExtractTrajectories(
            user_config.endstate_files.ligand_parameter_filename,
            remd_ligand.rv(1),
            user_config.intermediate_args.temperature,
        )
    )
    # user_config.inputs["endstate_ligand_traj"] = extract_ligand_traj.rv(0)
    user_config.inputs["endstate_ligand_lastframe"] = extract_ligand_traj.rv(1)

    if not user_config.workflow.ignore_receptor_endstate:
        minimization_receptor = minimization_complex.addFollowOn(
            Simulation(
                executable=user_config.system_settings.executable,
                mpi_command=user_config.system_settings.mpi_command,
                num_cores=(0.1 if user_config.system_settings.CUDA else user_config.num_cores_per_system.receptor_ncores),
                CUDA=user_config.system_settings.CUDA,
                prmtop=user_config.endstate_files.receptor_parameter_filename,
                incrd=user_config.endstate_files.receptor_coordinate_filename,
                input_file=user_config.inputs["min_mdin"],
                restraint_file=user_config.inputs["empty_restraint"],
                system_type="receptor",
                directory_args={
                    "runtype": "minimization",
                    "filename": "min",
                    "topology": user_config.endstate_files.receptor_parameter_filename,
                    "topdir": user_config.system_settings.top_directory_path,
                },
                working_directory=user_config.system_settings.working_directory,
                memory=user_config.system_settings.memory,
                disk=user_config.system_settings.disk,
            )
        )

        equilibrate_receptor = minimization_receptor.addFollowOn(
            REMDSimulation(
                executable=user_config.system_settings.executable,
                mpi_command=user_config.system_settings.mpi_command,
                num_cores=user_config.endstate_method.remd_args.nthreads_receptor,
                CUDA=user_config.system_settings.CUDA,
                prmtop=user_config.endstate_files.receptor_parameter_filename,
                incrd=minimization_receptor.rv(0),
                input_file=equil_mdins.rv(),
                restraint_file=user_config.inputs["empty_restraint"],
                runtype="equil",
                remd_debug=user_config.workflow.debug,
                ngroups=user_config.endstate_method.remd_args.ngroups,
                system_type="receptor",
                directory_args={
                    "runtype": "equilibration",
                    "topology": user_config.endstate_files.receptor_parameter_filename,
                    "topdir": user_config.system_settings.top_directory_path,
                },
                working_directory=user_config.system_settings.working_directory,
                memory=user_config.system_settings.memory,
                disk=user_config.system_settings.disk,
            )
        )

        remd_receptor = equilibrate_receptor.addFollowOn(
            REMDSimulation(
                executable=user_config.system_settings.executable,
                mpi_command=user_config.system_settings.mpi_command,
                num_cores=user_config.endstate_method.remd_args.nthreads_receptor,
                CUDA=user_config.system_settings.CUDA,
                prmtop=user_config.endstate_files.receptor_parameter_filename,
                incrd=equilibrate_receptor.rv(0),
                input_file=remd_mdins.rv(),
                restraint_file=user_config.inputs["empty_restraint"],
                runtype="remd",
                ngroups=user_config.endstate_method.remd_args.ngroups,
                system_type="receptor",
                directory_args={
                    "runtype": "remd",
                    "topology": user_config.endstate_files.receptor_parameter_filename,
                    "topdir": user_config.system_settings.output_directory_name,
                },
                working_directory=user_config.system_settings.working_directory,
                memory=user_config.system_settings.memory,
                disk=user_config.system_settings.disk,
                remd_debug=user_config.workflow.debug,
            )
        )
        # extact target temparture trajetory and last frame
        extract_receptor = remd_receptor.addFollowOn(
            ExtractTrajectories(
                user_config.endstate_files.receptor_parameter_filename,
                remd_receptor.rv(1),
                user_config.intermediate_args.temperature,
            )
        )
        # user_config.inputs["endstate_receptor_traj"] = extract_receptor.rv(0)
        user_config.inputs["endstate_receptor_lastframe"] = extract_receptor.rv(1)
    # use loaded receptor completed trajectory
    else:
        extract_receptor = remd_complex.addChild(
            ExtractTrajectories(
                user_config.endstate_files.receptor_parameter_filename,
                user_config.endstate_files.receptor_coordinate_filename,
            )
        )
        # user_config.inputs["endstate_receptor_traj"] = extract_receptor.rv(0)
        user_config.inputs["endstate_receptor_lastframe"] = extract_receptor.rv(1)
        user_config.endstate_files.receptor_coordinate_filename = extract_receptor.rv(1)
    job.log(
        f"user_config['endstate_complex_lastframe']: {user_config.inputs['endstate_complex_lastframe']}"
    )
    return (
        extract_complex.rv(1),
        extract_complex.rv(0),
        extract_receptor.rv(0),
        extract_ligand_traj.rv(0),
    )


def run_basic_md(job, user_config: Config):
    """Setup and run basic MD.

    Args:
        job (_type_): _description_
        user_config (Config): _description_
    """

    minimization_complex = job.addChild(
        Simulation(
            executable=user_config.system_settings.executable,
            mpi_command=user_config.system_settings.mpi_command,
            num_cores=(0.1 if user_config.system_settings.CUDA else user_config.num_cores_per_system.complex_ncores),
            CUDA=user_config.system_settings.CUDA,
            accelerators=user_config.system_settings.num_accelerators,
            prmtop=user_config.endstate_files.complex_parameter_filename,
            incrd=user_config.endstate_files.complex_coordinate_filename,
            input_file=user_config.inputs["min_mdin"],
            restraint_file=user_config.inputs["flat_bottom_restraint"],
            system_type="complex",
            directory_args={
                "runtype": "minimization",
                "filename": "min",
                "topology": user_config.endstate_files.complex_parameter_filename,
                "topdir": user_config.system_settings.top_directory_path,
            },
            working_directory=user_config.system_settings.working_directory,
            sim_debug=user_config.workflow.debug,
        )
    )

    endstate_complex = minimization_complex.addFollowOn(
        Simulation(
            executable=user_config.system_settings.executable,
            mpi_command=user_config.system_settings.mpi_command,
            num_cores=(0.1 if user_config.system_settings.CUDA else user_config.num_cores_per_system.complex_ncores),
            CUDA=user_config.system_settings.CUDA,
            accelerators=user_config.system_settings.num_accelerators,
            prmtop=user_config.endstate_files.complex_parameter_filename,
            incrd=minimization_complex.rv(0),
            input_file=user_config.endstate_method.basic_md_args.md_template_mdin,
            working_directory=user_config.system_settings.working_directory,
            restraint_file=user_config.inputs["flat_bottom_restraint"],
            system_type="complex",
            directory_args={
                "runtype": "basicMD_endstate",
                "filename": "basicMD",
                "topology": user_config.endstate_files.complex_parameter_filename,
                "topdir": user_config.system_settings.top_directory_path,
            },
            memory=user_config.system_settings.memory,
            disk=user_config.system_settings.disk,
            sim_debug=user_config.workflow.debug,
        )
    )
    # extact target temparture trajetory and last frame
    extract_complex = endstate_complex.addFollowOn(
        ExtractTrajectories(
            user_config.endstate_files.complex_parameter_filename,
            endstate_complex.rv(1),
        )
    )

    user_config.inputs["endstate_complex_lastframe"] = extract_complex.rv(1)

    # run minimization at the end states for ligand system only
    minimization_ligand = minimization_complex.addFollowOn(
        Simulation(
            executable=user_config.system_settings.executable,
            mpi_command=user_config.system_settings.mpi_command,
            num_cores=user_config.num_cores_per_system.ligand_ncores,
            CUDA=user_config.system_settings.CUDA,
            prmtop=user_config.endstate_files.ligand_parameter_filename,
            incrd=user_config.endstate_files.ligand_coordinate_filename,
            input_file=user_config.inputs["min_mdin"],
            restraint_file=user_config.inputs["empty_restraint"],
            system_type="ligand",
            directory_args={
                "runtype": "minimization",
                "filename": "min",
                "topology": user_config.endstate_files.ligand_parameter_filename,
                "topdir": user_config.system_settings.top_directory_path,
            },
            working_directory=user_config.system_settings.working_directory,
            memory=user_config.system_settings.memory,
            disk=user_config.system_settings.disk,
            sim_debug=user_config.workflow.debug,
        )
    )

    endstate_ligand = minimization_ligand.addFollowOn(
        Simulation(
            executable=user_config.system_settings.executable,
            mpi_command=user_config.system_settings.mpi_command,
            num_cores=user_config.num_cores_per_system.ligand_ncores,
            CUDA=user_config.system_settings.CUDA,
            prmtop=user_config.endstate_files.ligand_parameter_filename,
            incrd=minimization_ligand.rv(0),
            input_file=user_config.endstate_method.basic_md_args.md_template_mdin,
            restraint_file=user_config.inputs["empty_restraint"],
            system_type="ligand",
            directory_args={
                "runtype": "basicMD_endstate",
                "filename": "basicMD",
                "topology": user_config.endstate_files.ligand_parameter_filename,
                "topdir": user_config.system_settings.top_directory_path,
            },
            working_directory=user_config.system_settings.working_directory,
            memory=user_config.system_settings.memory,
            disk=user_config.system_settings.disk,
            sim_debug=user_config.workflow.debug,
        )
    )
    # extact target temparture trajetory and last frame
    extract_ligand_traj = endstate_ligand.addFollowOn(
        ExtractTrajectories(
            user_config.endstate_files.ligand_parameter_filename,
            endstate_ligand.rv(1),
        )
    )
    # user_config.inputs["endstate_ligand_traj"] = extract_ligand_traj.rv(0)
    user_config.inputs["endstate_ligand_lastframe"] = extract_ligand_traj.rv(1)

    if not user_config.workflow.ignore_receptor_endstate:
        minimization_receptor = minimization_complex.addFollowOn(
            Simulation(
                executable=user_config.system_settings.executable,
                mpi_command=user_config.system_settings.mpi_command,
                num_cores=(0.1 if user_config.system_settings.CUDA else user_config.num_cores_per_system.receptor_ncores),
                CUDA=user_config.system_settings.CUDA,
                prmtop=user_config.endstate_files.receptor_parameter_filename,
                incrd=user_config.endstate_files.receptor_coordinate_filename,
                input_file=user_config.inputs["min_mdin"],
                restraint_file=user_config.inputs["empty_restraint"],
                system_type="receptor",
                directory_args={
                    "runtype": "minimization",
                    "filename": "min",
                    "topology": user_config.endstate_files.receptor_parameter_filename,
                    "topdir": user_config.system_settings.top_directory_path,
                },
                working_directory=user_config.system_settings.working_directory,
                memory=user_config.system_settings.memory,
                disk=user_config.system_settings.disk,
                sim_debug=user_config.workflow.debug,
                accelerators=user_config.system_settings.num_accelerators,
            )
        )

        endstate_receptor = minimization_receptor.addFollowOn(
            Simulation(
                executable=user_config.system_settings.executable,
                mpi_command=user_config.system_settings.mpi_command,
                num_cores=(0.1 if user_config.system_settings.CUDA else user_config.num_cores_per_system.receptor_ncores),
                CUDA=user_config.system_settings.CUDA,
                accelerators=user_config.system_settings.num_accelerators,
                prmtop=user_config.endstate_files.receptor_parameter_filename,
                incrd=minimization_receptor.rv(0),
                input_file=user_config.endstate_method.basic_md_args.md_template_mdin,
                restraint_file=user_config.inputs["empty_restraint"],
                system_type="receptor",
                directory_args={
                    "runtype": "basicMD_endstate",
                    "filename": "basicMD",
                    "topology": user_config.endstate_files.receptor_parameter_filename,
                    "topdir": user_config.system_settings.output_directory_name,
                },
                working_directory=user_config.system_settings.working_directory,
                memory=user_config.system_settings.memory,
                disk=user_config.system_settings.disk,
                sim_debug=user_config.workflow.debug,
            )
        )
        # extact target temparture trajetory and last frame
        extract_receptor = endstate_receptor.addFollowOn(
            ExtractTrajectories(
                user_config.endstate_files.receptor_parameter_filename,
                endstate_receptor.rv(1),
            )
        )
        # user_config.inputs["endstate_receptor_traj"] = extract_receptor.rv(0)
        user_config.inputs["endstate_receptor_lastframe"] = extract_receptor.rv(1)
    # use loaded receptor completed trajectory
    else:
        extract_receptor = endstate_complex.addChild(
            ExtractTrajectories(
                user_config.endstate_files.receptor_parameter_filename,
                user_config.endstate_files.receptor_coordinate_filename,
            )
        )
        # user_config.inputs["endstate_receptor_traj"] = extract_receptor.rv(0)
        user_config.inputs["endstate_receptor_lastframe"] = extract_receptor.rv(1)
        user_config.endstate_files.receptor_coordinate_filename = extract_receptor.rv(1)
    job.log(
        f"user_config['endstate_complex_lastframe']: {user_config.inputs['endstate_complex_lastframe']}"
    )
    return (
        extract_complex.rv(1),
        extract_complex.rv(0),
        extract_receptor.rv(0),
        extract_ligand_traj.rv(0),
    )


def user_defined_endstate(job, user_config: Config):
    """Extract target temperature from user provided endstate simulation.

    Args:
        job (_type_): _description_
        user_config (Config): _description_
    """
    job.fileStore.logToMaster("Extracting Complex Trajectory")
    extract_complex = job.addChild(
        ExtractTrajectories(
            user_config.endstate_files.complex_parameter_filename,
            user_config.endstate_files.complex_coordinate_filename,
        )
    )
    # user_config.inputs["endstate_complex_traj"] = extract_complex.rv(0)
    user_config.inputs["endstate_complex_lastframe"] = extract_complex.rv(1)

    extract_ligand_traj = extract_complex.addChild(
        ExtractTrajectories(
            user_config.endstate_files.ligand_parameter_filename,
            user_config.endstate_files.ligand_coordinate_filename,
        )
    )
    # user_config.inputs["endstate_ligand_traj"] = extract_ligand_traj.rv(0)
    user_config.inputs["endstate_ligand_lastframe"] = extract_ligand_traj.rv(1)

    extract_receptor_traj = extract_complex.addChild(
        ExtractTrajectories(
            user_config.endstate_files.receptor_parameter_filename,
            user_config.endstate_files.receptor_coordinate_filename,
        )
    )
    user_config.inputs["endstate_receptor_traj"] = (
        user_config.endstate_files.receptor_coordinate_filename
    )
    user_config.inputs["endstate_receptor_lastframe"] = extract_receptor_traj.rv(1)

    if user_config.workflow.vina_dock:
        user_config.inputs["endstate_complex_lastframe"] = (
            user_config.endstate_files.complex_coordinate_filename
        )

    return (
        extract_complex.rv(1),
        extract_complex.rv(0),
        extract_receptor_traj.rv(0),
        extract_ligand_traj.rv(0),
    )
