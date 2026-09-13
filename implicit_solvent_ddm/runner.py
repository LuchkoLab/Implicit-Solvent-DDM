import os
import re
from email import message
from importlib.metadata import files
from pathlib import Path
from sre_constants import ANY
from typing import Optional, Type, TypedDict, Union

import pandas as pd
from matplotlib.backend_bases import key_press_handler
from toil.batchSystems import abstractBatchSystem
from toil.job import FileID, Job, JobFunctionWrappingJob, PromisedRequirement

from implicit_solvent_ddm.config import Config
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
        adaptive: bool = False,
        loaded_dataframe: list = [],
        post_output: Union[list, list[pd.DataFrame]] = [],
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
        self.post_only = post_only
        self.config = config
        self.adaptive = adaptive
        self.post_output = post_output
        self.ligand_output = []
        self.receptor_output = []
        self.complex_output = []
        self._loaded_dataframe = loaded_dataframe
        self.post_process_distruct = post_process_distruct

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
        for sim in md_jobs:
            self.addChild(sim)

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

        for post_simulation in self.simulations:
            directory_args = post_simulation.directory_args.copy()
            #fileStore.logToMaster(f"directory args before update: {directory_args}\n")
            # fileStore.logToMaster(f"args {completed_sim.directory_args} & {md_traj}")
            directory_args.update(self.update_postprocess_dirstruct(completed_sim.directory_args))  # type: ignore
            #fileStore.logToMaster(f"directory args after update: {directory_args}\n")
            mdin = self.mdin
            if post_simulation.directory_args["igb_value"] == 6:
                mdin = self.no_solvent_mdin

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

                self.addChild(post_process_job)

                data_frame = post_process_job.addFollowOnJobFn(
                    create_mdout_dataframe,
                    post_process_job.directory_args,
                    post_process_job.dirstruct,
                    post_process_job.output_dir,
                )

                self.post_output.append(data_frame.rv())

            elif post_process_job.output_dir in self._loaded_dataframe:
                fileStore.logToMaster(f"Energy post-analysis already completed and already loaded the results")
                fileStore.logToMaster(
                    f"Already loaded the Energy post-analysis results in the directory {post_process_job.output_dir}\n"
                )
                continue

            else:
                if post_simulation.directory_args["state_label"] == "lambda_window":
                    fileStore.logToMaster(f"Energy post-analysis already completed and loading the results") 
                    fileStore.logToMaster(
                        f"Loading the Energy post-analysis results in the directory {post_process_job.output_dir}\n"
                    )
                self.post_output.append(
                    pd.read_parquet(
                        os.path.join(
                            post_process_job.output_dir, "simulation_mdout.parquet.gzip"
                        ),
                    )
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
            CUDA=False,
            prmtop=parm_file,
            incrd=self.config.inputs["endstate_complex_lastframe"],
            input_file=mdin,
            restraint_file=restraint_file,
            working_directory=self.config.system_settings.working_directory,
            system_type="complex",
            directory_args=dirs_args[0],
            dirstruct="dirstruct_halo",
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

    def _add_receptor_simulation(self, conformational, mdin, restraint_file):
        con_force = float(round(conformational, 3))
        new_job = Simulation(
            executable=self.config.system_settings.executable,
            mpi_command=self.config.system_settings.mpi_command,
            num_cores=self.config.num_cores_per_system.receptor_ncores,
            CUDA=self.config.system_settings.CUDA,
            prmtop=self.config.endstate_files.receptor_parameter_filename,
            incrd=self.config.inputs["receptor_endstate_frame"],
            input_file=mdin,
            restraint_file=restraint_file.rv(),
            working_directory=self.config.system_settings.working_directory,
            system_type="receptor",
            directory_args={
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
            },
            dirstruct="dirstruct_apo",
        )

        self.simulations.append(new_job)

    @classmethod
    def new_runner(
        cls: Type["IntermidateRunner"],
        config: Config,
        obj: dict,
    ):
        return cls(
            simulations=obj["simulations"],
            restraints=obj["restraints"],
            config=config,
            post_process_distruct=obj["post_process_distruct"],
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            adaptive=True,
            post_only=True,
            post_output=obj["post_output"],
            loaded_dataframe=obj["_loaded_dataframe"],
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

    # /nas0/ayoub/sampl9_runs/sampl9_extend_windows_diel/WP6_G2_Hmass/lambda_window/1.0/78.5/-0.2857142857142864/3.7142857142857135/WP6_G2_Hmass_state_8_0.8203353560076375_13.1253656961222_prod_traj.nc
