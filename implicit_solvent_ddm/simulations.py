"""
Classes to setup different types of simulations (i.e. remd, basic md).
"""

import os
import re
import shutil
import subprocess as sp
import sys
from datetime import datetime
from string import Template
from typing import Optional, TypedDict, Union

from toil.common import Toil
from toil.job import FileID, FunctionWrappingJob, Job

from implicit_solvent_ddm.get_dirstruct import Dirstruct
from implicit_solvent_ddm.restraints import RestraintMaker

WORKDIR = os.getcwd()
import logging
import time


class Calculation(Job):
    """Base calculation class. All other calculation classes should be inherited
    from this class.
    """

    def __init__(
        self,
        executable,
        mpi_command,
        num_cores,
        CUDA,
        prmtop,
        incrd,
        input_file,
        work_dir,
        restraint_file: Union[RestraintMaker, str],
        directory_args: TypedDict,
        debug: bool,
        dirstruct="dirstruct",
        inptraj: Union[FileID, None] = None,
        post_analysis: bool = False,
        env=None
    ):
        self.executable = executable
        self.mpi_command = mpi_command
        self.num_cores = num_cores
        self.CUDA = CUDA
        self.prmtop = prmtop
        self.incrd = incrd
        self.input_file = input_file
        self.working_directory = work_dir
        self.restraint_file = restraint_file
        self._inptraj = inptraj
        self.dirstruct = dirstruct
        self.directory_args = directory_args.copy()
        self.calc_setup = False  # This means that the setup has run successfully
        self.post_analysis = post_analysis
        self.debug = debug
        # Whether to export outputs to the (network) output_dir. Left True here so every
        # existing construction site keeps current behavior; IntermidateRunner.run()
        # overrides it from config.system_settings.export_intermediate_files before it
        # schedules each MD / post-analysis job.
        self.export_network = True
        self.exec_list = [self.mpi_command]
        # self.exec_list = []
        self.read_files = {}
        self._output_directory()
        self.env = env if env is not None else os.environ.copy() 

    @property
    def inptraj(self):
        return self._inptraj

    @inptraj.setter
    def inptraj(self, value):
        self._inptraj = value

    def _output_directory(self):
        print(self.dirstruct)
        dirs = Dirstruct("mdgb", self.directory_args, dirstruct=self.dirstruct)

        output_dir = os.path.join(
            self.working_directory, dirs.dirStruct.fromArgs(**dirs.parameters)
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir
        # self._path2dict(dirs)

    def _setLogging(self):
        file_handler = logging.FileHandler(
            os.path.join(self.output_dir, "simulations.log"), mode="w"
        )
        formatter = logging.Formatter(
            "%(asctime)s~%(levelname)s~%(message)s~module:%(module)s"
        )
        file_handler.setFormatter(formatter)
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)

        return self.logger

    def _mdin_restraint(self, fileStore, mdin):
        scratch_mdin = fileStore.getLocalTempFile()
        # scratch_restraint = fileStore.getLocalTempFile()

        restraint_file = shutil.copyfile(
            self.read_files["restraint_file"], "restraint.RST"
        )

        with open(mdin) as temp:
            template = Template(temp.read())

        if not self.post_analysis:
            final_template = template.substitute(restraint=restraint_file)
        else:
            final_template = template.substitute(
                restraint=restraint_file, extdiel=self.directory_args["extdiel"]
            )

        with open(scratch_mdin, "w") as temp_mdin:
            temp_mdin.write(final_template)

        mdin_ID = fileStore.writeGlobalFile(scratch_mdin, cleanup=True)
        # os.remove("mdin")
        return fileStore.readGlobalFile(mdin_ID)

    def export_files(self, fileStore, output_directory, parameter_files):
        restart_files = []
        traj_files = []
        fileStore.logToMaster(f"DEBUG is {self.debug}")
        for root, dirs, files in os.walk(".", topdown=False):
            for name in files:
                if name in parameter_files:
                    continue
                # output_file = fileStore.writeGlobalFile(name)
                # fileStore.export_file(output_file,"file://" + os.path.abspath(os.path.join(output_directory, os.path.basename(name))))
                if re.match(r".*\.rst7.*", name):
                    output_file = fileStore.writeGlobalFile(name, cleanup=True)
                    restart_files.append(str(output_file))
                elif re.match(r".*\.nc.*", name):
                    # Keep the raw jobStore FileID (not str()): the MD run() returns
                    # traj_files so a downstream post-analysis job can consume it as
                    # inptraj via readGlobalFile -- the Toil-promise hand-off used when
                    # export_network is off (no network trajectory copy).
                    output_file = fileStore.writeGlobalFile(name)  # cleanup=True
                    traj_files.append(output_file)

                elif not self.debug and re.match(
                    r"(rem.log)|(equilibrate)?(remd)?\.mdout\.\d*", name
                ):
                    fileStore.logToMaster(
                        f"checking the name that is not exporting {name}"
                    )
                    continue

                else:
                    output_file = fileStore.writeGlobalFile(name, cleanup=True)
                # writeGlobalFile above always lands the file in the jobStore (local
                # workDir) so the DAG/promises still work; export_file is the extra copy
                # to the (network) output_dir -- skip it entirely when export is off.
                if self.export_network:
                    fileStore.export_file(
                        output_file,
                        "file://"
                        + os.path.abspath(
                            os.path.join(output_directory, os.path.basename(name))
                        ),
                    )
        if self.export_network:
            # export parameter files
            fileStore.export_file(
                self.read_files["prmtop"],
                "file://"
                + os.path.abspath(
                    os.path.join(
                        self.output_dir, os.path.basename(self.read_files["prmtop"])
                    )
                ),
            )
            # export coordinate file
            if "incrd" in self.read_files.keys():
                fileStore.export_file(
                    self.read_files["incrd"],
                    "file://"
                    + os.path.abspath(
                        os.path.join(
                            self.output_dir, os.path.basename(self.read_files["incrd"])
                        )
                    ),
                )
            # export restraint File
            fileStore.export_file(
                self.read_files["restraint_file"],
                "file://"
                + os.path.abspath(
                    os.path.join(
                        self.output_dir,
                        os.path.basename(self.read_files["restraint_file"]),
                    )
                ),
            )
        # fileStore.logToMaster(f"the current trajectory files {traj_files}")
        # fileStore.logToMaster(f"the restart files: {restart_files}")

        return (restart_files, traj_files)

    def run(self, fileStore):
        """Runs the program. All command-line arguments must be set before
        calling this method. Command-line arguments should be set in setup()
        """
        # fileStore.logToMaster(f"the self.directory_args {self.directory_args}")
        # self._output_directory()
        start = time.perf_counter()
        self._setLogging()

        self.logger.info(f"Run Type: {self.directory_args['runtype']}\n")
        self.logger.info(f"Running: {self.directory_args['runtype']}\n")
        self.logger.info(f"Number of cores: {self.num_cores}\n")

        self.logger.info(
            f"Prior simulation datetime: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
        )

        if not self.calc_setup:
            raise RuntimeError(
                "Cannot run a calculation without calling its"
                + " its setup() function!"
            )
        stdout = sys.stdout
        stderr = sys.stderr
        own_handleo = own_handlee = False
        try:
            process_stdout = open(stdout, "w")
            own_handleo = True
        except TypeError:
            process_stdout = stdout
        try:
            process_stderr = open(stderr, "w")
            own_handlee = True
        except TypeError:
            process_stderr = stderr

        files_in_current_directory = os.listdir(
            f"{os.path.dirname(self.read_files['prmtop'])}"
        )

        # fileStore.logToMaster(f"file in current directory {files_in_current_directory}")
        fileStore.logToMaster(f"exec_list : {self.exec_list}")
        fileStore.logToMaster(f"output to -> {self.output_dir}")
        self.logger.info(
            f"The files in the current working directory: {files_in_current_directory}\n"
        )

        self.logger.info(f"exec_list: {self.exec_list}")

        # Inherit the LIVE worker environment rather than the snapshot captured
        # in ``self.env`` at construction time. Toil's accelerator scheduler sets
        # CUDA_VISIBLE_DEVICES on the worker right before this job runs to pin it
        # to its assigned GPU; self.env predates that and would mask the pin,
        # sending every window to the default device (GPU 0).
        run_env = os.environ.copy()
        if self.CUDA and not run_env.get("CUDA_VISIBLE_DEVICES"):
            # Toil left this GPU job with an EMPTY CUDA_VISIBLE_DEVICES (i.e. it did
            # not pin a device -- this happens for the endstate jobs under
            # single_machine). '' means "no GPUs visible" so pmemd.cuda dies with
            # 'no CUDA-capable device'. UNSET it instead, so CUDA falls back to the
            # GPUs the Slurm cgroup exposes (logical 0..N-1). Sequential endstate
            # jobs land on GPU 0; windows that Toil DOES pin keep a non-empty id and
            # never reach this branch (so multi-GPU pinning is unaffected).
            run_env.pop("CUDA_VISIBLE_DEVICES", None)
        if self.CUDA:
            fileStore.logToMaster(
                f"[GPU] {getattr(self, 'system_type', '?')} window "
                f"CUDA_VISIBLE_DEVICES={run_env.get('CUDA_VISIBLE_DEVICES')!r}"
            )
        # amber_output = sp.Popen(self.exec_list, stdout=sp.PIPE, stderr=sp.PIPE)
        amber_output = sp.run(self.exec_list, stdout=sp.PIPE, stderr=sp.PIPE, env=run_env)

        amber_stdout = amber_output.stdout.decode("utf-8")
        amber_stderr = amber_output.stderr.decode("utf-8")
        # amber_stdout = amber_output.stdout.read().splitlines()
        # amber_stderr = amber_output.stderr.read().splitlines()

        self.logger.error(f"AMBER stdout: {amber_stdout}\n")
        self.logger.error(f"AMBER stderr: {amber_stderr}\n")

        fileStore.logToMaster(f"amber_stdout: {amber_stdout}")
        fileStore.logToMaster(f"amber_stderr: {amber_stderr}")

        # Per-job timing record -> parsed by research/timing_report.py into a phase breakdown.
        # Phase is inferred from the output dir / runtype: the ALS pilot writes under a *_pilot tree
        # (-> "adaptive"), post-analysis jobs carry post_analysis=True (-> "post_analysis"), endstate
        # setup MD has an endstate/minimization runtype, everything else is the production intermediate MD.
        # cores is ~0.1 for GPU windows, so the parser uses the gpu flag (GPU-hours = wall; else CPU-hours
        # = wall*cores). end= is an absolute epoch so the parser can recover parallel-adjusted wall-clock.
        _wall = time.perf_counter() - start
        _rt = self.directory_args.get("runtype", "")
        # Detect the ALS pilot tree by a PATH SEGMENT ending in "_pilot" (the pilot sets
        # output_directory_name = <name> + "_pilot", e.g. "mdgb_pilot"), NOT a substring of the whole
        # path. A substring check mislabels every production job as "adaptive" when the working/scratch
        # dir merely contains "_pilot" (e.g. SCRATCH=cb7_pilot_run), which hid intermediate_md /
        # post_analysis from timing_report.py.
        if any(seg.endswith("_pilot") for seg in str(self.output_dir).split(os.sep)):
            _phase = "adaptive"
        elif self.post_analysis:
            _phase = "post_analysis"
        elif any(k in _rt for k in ("endstate", "minimization", "remd", "equil")):
            _phase = "endstate"
        else:
            _phase = "intermediate_md"
        fileStore.logToMaster(
            f"[TIMING] phase={_phase} wall_s={_wall:.2f} cores={self.num_cores} "
            f"gpu={int(bool(self.CUDA))} end={time.time():.0f} | runtype={_rt}"
        )

        # Post-analysis re-score only needs its `mdout`: the follow-on create_mdout_dataframe
        # reads {output_dir}/mdout to build simulation_mdout.parquet.gzip. The re-score's
        # trajectory / restart / parameter files are never read by anything downstream, so the
        # old full export_files() call streamed ~4188 jobs' worth of trajectories to the network
        # for nothing. Export just the mdout (restores the intent of the code below).
        if self.post_analysis:
            # Always land the mdout in the jobStore and RETURN its FileID so the
            # follow-on create_mdout_dataframe can read it via a Toil promise
            # (readGlobalFile) with no network access. The export_file copy to the
            # network output_dir is only for archival/warm-resume, so skip it when off.
            mdout_ID = fileStore.writeGlobalFile("mdout")
            if self.export_network:
                fileStore.export_file(
                    mdout_ID,
                    "file://" + os.path.abspath(os.path.join(self.output_dir, "mdout")),
                )
            return mdout_ID

        else:
            # Export all ouput files from MD simulation
            restart_ID, trajectory_ID = self.export_files(
                fileStore, self.output_dir, files_in_current_directory
            )
            # log MD simulation performance
            self.logger.info(
                f"Completed simulation datetime {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
            )
            self.logger.info(
                f"Performance runtime for current simulation Run: {time.perf_counter() - start} seconds\n"
            )
            # return trajecories from MD simulation
            return (restart_ID, trajectory_ID)


class Simulation(Calculation):
    """Setup a general sander MD simulation.

    Args:
        Calculation (_type_): _description_
    """

    def __init__(
        self,
        executable,
        mpi_command,
        num_cores,
        CUDA,
        prmtop,
        incrd,
        input_file,
        restraint_file: Union[RestraintMaker, str],
        working_directory,
        directory_args,
        system_type: Optional[str] = None,
        conformational_force=None,
        orientational_force=None,
        dirstruct="dirstruct",
        inptraj=None,
        post_analysis=False,
        restraint_key=None,
        sim_debug: bool = False,
        memory: Optional[Union[int, str]] = None,
        disk: Optional[Union[int, str]] = None,
        preemptable: Optional[Union[bool, int, str]] = None,
        unitName: Optional[str] = "",
        checkpoint: Optional[bool] = False,
        displayName: Optional[str] = "",
        descriptionClass: Optional[str] = None,
        accelerators: Optional[int] = None,
    ):
        Job.__init__(
            self,
            memory=memory,
            cores=num_cores,
            disk=disk,
            preemptible="false",
            unitName=unitName,
            checkpoint=checkpoint,
            displayName=displayName,
            accelerators=accelerators,
        )

        Calculation.__init__(
            self,
            executable,
            mpi_command,
            num_cores,
            CUDA,
            prmtop,
            incrd,
            input_file,
            working_directory,
            restraint_file,
            directory_args,
            dirstruct=dirstruct,
            inptraj=inptraj,
            post_analysis=post_analysis,
            debug=sim_debug,
        )
        self.restraint_key = restraint_key
        self.system_type = system_type

    def setup(self):
        """
        Sets up the command-line arguments. Sander requires a unique restrt file
        for the MPI version (since one is *always* written and you don't want 2
        threads fighting to write the same dumb file)
        """


        if self.CUDA and self.system_type in ["complex", "receptor"]:
            if self.mpi_command == None:
                self.exec_list.pop(0)
            self.exec_list.append(self.executable)
            # self.exec_list.append("--cuda")
            # self.exec_list.append("--gpu")
            # self.exec_list.append("--gpu_id")
            # self.exec_list.append("0")
            # self.exec_list.append("--gpu_mem")

        
        elif self.num_cores == 1 or self.mpi_command == None:
            self.exec_list.pop(0)
            self.exec_list.append(re.sub(r"\..*", "", self.executable))
        
        else:
            if self.mpi_command in ["mpiexec", "mpirun"]:

                self.exec_list.extend(("-np", str(self.num_cores)))
            # assume slurm
            else:
                self.exec_list.append("--exclusive")
                self.exec_list.extend(("-n", str(self.num_cores)))

            self.exec_list.append(self.executable)

        solu = re.sub(r"\..*", "", os.path.basename(str(self.prmtop)))
        restart_filename = f"{solu}_{self.directory_args['filename']}_restrt"
        trajector_filename = f"{solu}_{self.directory_args['filename']}_traj"
        self.exec_list.append("-O")  # overwrite flag
        self.exec_list.extend(("-i", self.read_files["mdin"]))  # input file flag
        self.exec_list.extend(("-p", self.read_files["prmtop"]))  # prmtop flag
        self.exec_list.extend(("-c", self.read_files["incrd"]))  # input coordinate flag
        self.exec_list.extend(("-r", f"{restart_filename}.rst7"))
        self.exec_list.extend(("-x", f"{trajector_filename}.nc"))
        self.exec_list.extend(("-o", "mdout"))  # output file flag
        if self._inptraj is not None:
            self.exec_list.extend(
                ("-y", self.read_files["inptraj"])
            )  # input trajectory flag

        self.calc_setup = True

    def run(self, fileStore):
        tempDir = fileStore.getLocalTempDir()
        """
        Import all command line arguments into the temporary working directory 
        """

        self.read_files["prmtop"] = fileStore.readGlobalFile(
            self.prmtop, userPath=os.path.join(tempDir, os.path.basename(self.prmtop))
        )
        fileStore.logToMaster(f"incrd file: {self.incrd}")

        # double check this -> basicMD
        if not isinstance(self.incrd, list):
            self.read_files["incrd"] = fileStore.readGlobalFile(
                self.incrd, userPath=os.path.join(tempDir, os.path.basename(self.incrd))
            )

        elif self.incrd:
            self.read_files["incrd"] = fileStore.readGlobalFile(
                self.incrd[0],
                userPath=os.path.join(tempDir, os.path.basename(self.incrd[0])),
            )

        else:
            raise RuntimeError(
                f"No input coordinate for {self.output_dir}: incrd is an empty list. "
                f"The upstream minimization/MD that produces this job's restart wrote no "
                f"output -- almost always because pmemd.cuda saw no GPU "
                f"(CUDA_VISIBLE_DEVICES=''). Fix the GPU allocation, not this job."
            )

        self.read_files["input_file"] = fileStore.readGlobalFile(
            self.input_file,
            userPath=os.path.join(tempDir, os.path.basename(self.input_file)),
        )

        if self._inptraj is not None:
            fileStore.logToMaster(f"READING inputraj: {self._inptraj}")
            self.read_files["inptraj"] = fileStore.readGlobalFile(
                self._inptraj[0],
                userPath=os.path.join(tempDir, os.path.basename(self._inptraj[0])),
            )

        if isinstance(self.restraint_file, RestraintMaker):
            # fileStore.logToMaster(f"RESTRAINT {self.restraint_file}")
            # fileStore.logToMaster(f"RESTRAINT {self.restraint_file.restraints}")
            self.read_files["restraint_file"] = fileStore.readGlobalFile(
                self.restraint_file.restraints[self.restraint_key],
                userPath=os.path.join(
                    tempDir,
                    os.path.basename(
                        self.restraint_file.restraints[self.restraint_key]
                    ),
                ),
            )
        else:
            self.read_files["restraint_file"] = fileStore.readGlobalFile(
                self.restraint_file,
                userPath=os.path.join(tempDir, os.path.basename(self.restraint_file)),
            )
        self.read_files["mdin"] = Calculation._mdin_restraint(
            self, fileStore, self.read_files["input_file"]
        )
        self.setup()

        output = Calculation.run(self, fileStore)

        return output


class REMDSimulation(Calculation):
    """This class handles Replica Exchange Molecular Dynamic calculations

    Args:
        Calculation (_type_): _description_
    """

    def __init__(
        self,
        executable,
        mpi_command,
        num_cores,
        CUDA,
        prmtop,
        incrd,
        input_file,
        working_directory,
        restraint_file,
        runtype,
        ngroups,
        directory_args,
        remd_debug: bool,
        system_type:str, # complex, receptor, ligand
        dirstruct="dirstruct",
        inptraj=None,
        memory: Optional[Union[int, str]] = None,
        disk: Optional[Union[int, str]] = None,
        accelerators: Optional[int] = None,
        preemptable: Optional[Union[bool, int, str]] = None,
        unitName: Optional[str] = "",
        checkpoint: Optional[bool] = False,
        displayName: Optional[str] = "",
        descriptionClass: Optional[str] = None,
    ):
        # accelerators is the GPU request for THIS job, which runs every replica in the chain --
        # unlike Simulation, where one job is one window on one device. AMBER binds one device per
        # MPI rank, so the natural value is ngroups; fewer means replicas share devices.
        Job.__init__(
            self,
            memory=memory,
            cores=num_cores,
            disk=disk,
            accelerators=accelerators,
            preemptible="false",
            unitName=unitName,
            checkpoint=checkpoint,
            displayName=displayName,
        )

        Calculation.__init__(
            self,
            executable=executable,
            mpi_command=mpi_command,
            num_cores=num_cores,
            prmtop=prmtop,
            incrd=incrd,
            input_file=input_file,
            work_dir=working_directory,
            restraint_file=restraint_file,
            directory_args=directory_args,
            dirstruct=dirstruct,
            inptraj=inptraj,
            debug=remd_debug,
            CUDA=CUDA,
        )   
        self.system_type = system_type
        self.runtype = runtype
        self.ng = ngroups
        self.nthreads = num_cores

    def _setup(self):
        """
        Sets up the REMD calculation. All it has to do is fill in the
        necessary command-line arguments
        """

        self.exec_list.extend(("-n", str(self.num_cores)))
        self.exec_list.append(self.executable)
        self.exec_list.extend(("-ng", str(self.ng)))
        self.exec_list.extend(("-groupfile", self.read_files["groupfile"]))

        self.calc_setup = True

    def _groupfile(self, fileStore):
        # scratch_file = fileStore.getLocalTempFile()
        # fileStore.logToMaster(f"self.input_file {self.input_file}")
        # create groupfile mdin
        with open("group.groupfile", "w") as group:
            for count, mdin in enumerate(self.input_file, start=1):
                # fileStore.logToMaster(f"mdin for remd {mdin}")
                read_mdin = fileStore.readGlobalFile(
                    mdin, userPath=os.path.join(self.tempDir, os.path.basename(mdin))
                )
                local_mdin = Calculation._mdin_restraint(self, fileStore, read_mdin)

                # fileStore.logToMaster(f"local mdin {local_mdin}")
                solu = re.sub(r"\..*", "", os.path.basename(self.prmtop))

                fileStore.logToMaster(f"read_mdin: {read_mdin}")
                fileStore.logToMaster(f"runtype: {self.runtype}")
                fileStore.logToMaster(f"mdin: {mdin}")
                fileStore.logToMaster(f"local_mdin: {local_mdin}")
                fileStore.logToMaster(f"self.input_file: {self.input_file}")

                if self.runtype == "equil":
                    group.write(
                        f"""-O -rem 0 -i {local_mdin} 
                    -p {self.read_files["prmtop"]} -c {self.read_files["incrd"]} 
                    -o equilibrate.mdout.{count:03} -inf equilibrate.mdinfo.{count:03}
                    -r {solu}_equilibrate.rst7.{count:03} -x {solu}_equilibrate.nc.{count:03}""".replace(
                            "\n", ""
                        )
                        + "\n"
                    )
                    fileStore.logToMaster(
                        f"Writing -O -rem 0 -i {local_mdin} -p {self.read_files['prmtop']} -c {self.read_files['incrd']} "
                    )

                elif self.runtype == "remd":
                    single_coordinate = [
                        coordinate
                        for coordinate in self.incrd
                        if re.search(rf".*.rst7.{count:03}", coordinate)
                    ]
                    fileStore.logToMaster(f"self.incrd: {self.incrd}")
                    fileStore.logToMaster(f"reading in remd coordiante\n")
                    fileStore.logToMaster(f"count: {count}")
                    fileStore.logToMaster(f"single_coordinate: {single_coordinate}")

                    read_coordinate = fileStore.readGlobalFile(
                        single_coordinate[0],
                        userPath=os.path.join(
                            self.tempDir, os.path.basename(single_coordinate[0])
                        ),
                    )
                    group.write(
                        f"""-O -rem 1 -remlog rem.log
                        -i {local_mdin} -p {self.read_files["prmtop"]} 
                        -c {read_coordinate} -o remd.mdout.rep.{count:03} 
                        -r remd.rst7.{count:03} -x remd.nc.{count:03} 
                        -inf remd.mdinfo.{count:03}""".replace(
                            "\n", ""
                        )
                        + "\n"
                    )

        groupfile_ID = fileStore.writeGlobalFile("group.groupfile")

        self.read_files["groupfile"] = fileStore.readGlobalFile(groupfile_ID)

        with open(self.read_files["groupfile"], "r") as output:
            data = output.readlines()
        fileStore.logToMaster(f"GROUP FILE lines: {data}")
        fileStore.logToMaster(f"GROUP FILE NAME: {groupfile_ID}")

    def run(self, fileStore):
        tempDir = self.tempDir
        fileStore.logToMaster(f"self.input files is {self.input_file}")
        # read in parameter files
        self.read_files["prmtop"] = fileStore.readGlobalFile(
            self.prmtop, userPath=os.path.join(tempDir, os.path.basename(self.prmtop))
        )
        self.read_files["restraint_file"] = fileStore.readGlobalFile(
            self.restraint_file,
            userPath=os.path.join(tempDir, os.path.basename(self.restraint_file)),
        )
        # The case in running REMD possible many equilibration restart files
        if len(self.incrd) == 1:
            self.read_files["incrd"] = fileStore.readGlobalFile(
                self.incrd[0],
                userPath=os.path.join(tempDir, os.path.basename(self.incrd[0])),
            )
        if self._inptraj is not None:
            self.read_files["inptraj"] = fileStore.readGlobalFile(
                self._inptraj,
                userPath=os.path.join(tempDir, os.path.basename(self._inptraj)),
            )

        self._groupfile(fileStore)

        self._setup()
        return Calculation.run(self, fileStore)


class ExtractTrajectories(Job):
    """
    This class handles target temperature trajectory extraction
    from Replica Exchange Molecular Dynamic Simulation

    """

    def __init__(self, solute_topology, trajectory_files, target_temp=0.0):
        Job.__init__(
            self,
            memory="2G",
            cores=1,
            disk="3G",
            preemptible="false",
        )

        self.solute_topology = solute_topology
        self.trajectory_files = trajectory_files
        self.target_temp = target_temp
        self.read_trajs = []

    def run(self, fileStore) -> tuple[FileID, FileID, FileID]:
        temp_dir = fileStore.getLocalTempDir()
        self.filestore = fileStore

        self.read_solute = fileStore.readGlobalFile(
            self.solute_topology,
            userPath=os.path.join(temp_dir, os.path.basename(self.solute_topology)),
        )
        self.tempdir = temp_dir

        # extract target temperture in REMD trajectories
        if self.target_temp != 0.0:
            # read in each trajectory replica
            for traj_file in self.trajectory_files:
                self.read_trajs.append(
                    fileStore.readGlobalFile(
                        traj_file,
                        userPath=os.path.join(temp_dir, os.path.basename(traj_file)),
                    )
                )

            # extract trajectories at target temperature
            target_trajectoryID = self.extract_target_temperature

            read_target_traj = fileStore.readGlobalFile(
                target_trajectoryID,
                userPath=os.path.join(temp_dir, os.path.basename(target_trajectoryID)),
            )

        # basic MD was performed
        elif isinstance(self.trajectory_files, list):
            fileStore.logToMaster(
                f"Extraction BASIC MD READ {os.path.basename(self.trajectory_files[0])}"
            )
            read_target_traj = fileStore.readGlobalFile(
                self.trajectory_files[0],
                userPath=os.path.join(
                    temp_dir, os.path.basename(self.trajectory_files[0])
                ),
            )
            target_trajectoryID = self.trajectory_files[0]
        # user provided there one endstate trajectory
        else:
            read_target_traj = fileStore.readGlobalFile(
                self.trajectory_files,
                userPath=os.path.join(
                    temp_dir, os.path.basename(self.trajectory_files)
                ),
            )
            # else user provied there own endstate trajectory file
            target_trajectoryID = self.trajectory_files

        solu = re.sub(r"\..*", "", os.path.basename(self.read_solute))
        lastframe_nc = f"{solu}_{self.target_temp}K_lastframe.ncrst"
        lastframe_rst7 = f"{solu}_{self.target_temp}K_lastframe.rst7"

        # get the last frame at the target temperature
        fileStore.logToMaster(f"read_target_traj {read_target_traj}")
        sp.run(
            [
                "cpptraj",
                "-p",
                self.read_solute,
                "-y",
                read_target_traj,
                "-ya",
                "lastframe",
                "-x",
                lastframe_nc,
            ]
        )
        sp.run(
            [
                "cpptraj",
                "-p",
                self.read_solute,
                "-y",
                lastframe_nc,
                "-x",
                lastframe_rst7,
            ],
            capture_output=True,
        )
        sp.run(
            [
                "cpptraj",
                "-p",
                self.read_solute,
                "-y",
                lastframe_rst7,
                "-x",
                lastframe_nc,
            ],
            capture_output=True,
        )

        lastframe_nc_ID = fileStore.writeGlobalFile(lastframe_nc)
        fileStore.logToMaster(f"write global {lastframe_nc_ID}")
        lastframe_rst7_ID = fileStore.writeGlobalFile(lastframe_rst7)

        return (target_trajectoryID, lastframe_nc_ID, lastframe_rst7_ID)

    @property
    def extract_target_temperature(self) -> FileID:
        """Run cpptraj to extract target temperature from each replica trajectory

        Args:
            bash_script (_type_): _description_
            fileStore (_type_): _description_

        Returns:
            _type_: _description_
        """
        current_files = os.listdir()
        self.filestore.logToMaster(f"tempdir directory: {self.tempdir}")
        self.filestore.logToMaster(f"current files {current_files}")
        self.filestore.logToMaster(
            f"tempdir directory: {os.listdir(os.path.abspath(self.tempdir))}"
        )
        output = sp.run(["cpptraj", self.remd_input_file], capture_output=True)
        # output = sp.run(["cpptraj", bash_script], capture_output=True)
        self.filestore.logToMaster(
            f"temperature extraction: {output.stderr}\n {output.stderr} {output}"
        )
        output_files = os.listdir()
        self.filestore.logToMaster(f"output files {output_files}")

        extract_target_temp = list(
            filter(
                lambda filename: re.match(rf".*{self.target_temp}.*\.nc", filename),
                output_files,
            )
        )[0]
        output_file = self.filestore.writeGlobalFile(extract_target_temp)
        # for file in os.listdir():
        #     if file not in current_files:
        #         output_file = self.filestore.writeGlobalFile(file)

        self.filestore.logToMaster(f"outputfile {output_file}")
        return output_file

    @property
    def remd_input_file(self) -> str:
        """Write an input file for cpptraj to extract target temperature

        Returns:
            cpptraj input file.
        """

        bash_script = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__)) + "/templates/cpptraj_remd.sh"
        )
        with open(bash_script) as t:
            template = Template(t.read())

        solu = re.sub(r"\..*", "", os.path.basename(self.solute_topology))
        output_trajectory_filename = f"{solu}_{self.target_temp}K"

        final_template = template.substitute(
            solute=self.read_solute,
            trajectory=self.initial_coordinate,
            target_temperature=self.target_temp,
            temperature_traj=output_trajectory_filename,
            rem_replicas=self.replica_names,
        )
        with open(f"{solu}_cpptraj_extract_{self.target_temp}K.x", "w") as output:
            output.write(final_template)

        return os.path.abspath(f"{solu}_cpptraj_extract_{self.target_temp}K.x")

    @property
    def initial_coordinate(self):
        return list(
            filter(
                lambda coordinate: re.match(r".*\.nc.001", coordinate), self.read_trajs
            )
        )[0]

    @property
    def replica_names(self):
        return ",".join(
            list(
                filter(
                    lambda coordinate: re.match(r".*\.nc.00[^1].*", coordinate),
                    self.read_trajs,
                )
            )
        )


def write_mdin(job, mdin_type):
    tempdir = job.fileStore.getLocalTempDir()

    mdin_path = (
        "/home/ayoub/nas0/Impicit-Solvent-DDM/implicit_solvent_ddm/templates/min.mdin"
    )

    mdin_import = job.fileStore.import_file("file://" + mdin_path)

    return mdin_import


def write_empty_restraint(job) -> str:
    scratch_file = job.fileStore.getLocalTempFile()

    with open(scratch_file, "w") as rest:
        rest.write("")

    return job.fileStore.writeGlobalFile(scratch_file)


def workflow(job: FunctionWrappingJob, parm_file, coord_files):
    output = job.addChild(
        ExtractTrajectories(parm_file, coord_files, target_temp=300.0)
    )


if __name__ == "__main__":
    options = Job.Runner.getDefaultOptions("./toilWorkflowRun")
    options.logLevel = "INFO"
    options.clean = "always"
    ligand_trajs = []
    # cb7 =os.path.abspath("structs/complex/cb7-mol01.parm7")
    # traj = pt.load("inputs/M01_000_minimization.rst7", "inputs/M01_000.parm7")
    with Toil(options) as toil:
        remd_traj = [
            toil.import_file("file://" + os.path.abspath(filename))
            for filename in [
                "/nas0/ayoub/Impicit-Solvent-DDM/ddm/remd/cb7-mol01/remd.nc.001",
                "/nas0/ayoub/Impicit-Solvent-DDM/ddm/remd/cb7-mol01/remd.nc.002",
                "/nas0/ayoub/Impicit-Solvent-DDM/ddm/remd/cb7-mol01/remd.nc.003",
                "/nas0/ayoub/Impicit-Solvent-DDM/ddm/remd/cb7-mol01/remd.nc.004",
            ]
        ]
        complex_prmtop = toil.import_file(
            "file://"
            + os.path.abspath(
                "/nas0/ayoub/Impicit-Solvent-DDM/ddm/remd/cb7-mol01/cb7-mol01.parm7"
            )
        )

        print(toil.start(Job.wrapJobFn(workflow, complex_prmtop, remd_traj)))
