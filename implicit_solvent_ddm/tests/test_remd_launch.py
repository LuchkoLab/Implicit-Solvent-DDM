"""Unit tests for how endstate jobs assemble their AMBER command line.

Covers the three faults that stopped REMD from starting: a single-replica window launched
through the MPI launcher, a dead AMBER run reported as success, and `-n` taken from the Toil
`cores` reservation instead of the replica count.

These tests do NOT touch AMBER or a real Toil workflow -- they drive setup()/run() against
fake file stores, so they run anywhere. The one test that needs real binaries skips itself
when they are absent (see test_mpirun_can_launch_a_replica_ladder). Run with::

    pytest implicit_solvent_ddm/tests/test_remd_launch.py -v
"""
import os
import shutil
import subprocess as sp
import types

import pytest

from implicit_solvent_ddm import simulations
from implicit_solvent_ddm.simulations import (
    Calculation,
    REMDSimulation,
    Simulation,
    mpi_executable,
    serial_executable,
)


class _FakeFileStore:
    """Enough of a Toil file store for these code paths: paths in, paths out."""

    def logToMaster(self, *_):
        pass

    def readGlobalFile(self, file_id, userPath=None):
        return userPath or file_id


def _make(cls, **attrs):
    """Build an instance without Toil/Dirstruct init, then set what the method reads."""
    obj = object.__new__(cls)
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


# ---------------------------------------------------------------- executable name derivation

@pytest.mark.parametrize(
    "configured,serial,mpi",
    [
        ("pmemd.cuda.MPI", "pmemd.cuda", "pmemd.cuda.MPI"),
        ("pmemd.cuda", "pmemd.cuda", "pmemd.cuda.MPI"),
        ("pmemd.MPI", "pmemd", "pmemd.MPI"),
        ("sander.MPI", "sander", "sander.MPI"),
        ("sander", "sander", "sander.MPI"),
    ],
)
def test_executable_name_derivation(configured, serial, mpi):
    """One config name serves both leg shapes: serial for a window, .MPI for a ladder."""
    assert serial_executable(configured) == serial
    assert mpi_executable(configured) == mpi


# ---------------------------------------------------------------- Simulation (single replica)

def _simulation(**overrides):
    attrs = dict(
        CUDA=True,
        system_type="complex",
        mpi_command="mpirun",
        executable="pmemd.cuda.MPI",
        num_cores=0.1,
        exec_list=["mpirun"],
        prmtop="MCL-1_ligand-1.parm7",
        directory_args={"filename": "min", "runtype": "minimization"},
        read_files={"mdin": "mdin", "prmtop": "top.parm7", "incrd": "in.rst7"},
        _inptraj=None,
    )
    attrs.update(overrides)
    return _make(Simulation, **attrs)


@pytest.mark.parametrize("system_type", ["complex", "receptor"])
def test_cuda_window_runs_the_serial_build_with_no_launcher(system_type):
    """`mpirun pmemd.cuda.MPI` with no -np is not a single-replica launch; it must not appear."""
    job = _simulation(system_type=system_type)
    job.setup()

    assert job.exec_list[0] == "pmemd.cuda"
    assert "mpirun" not in job.exec_list
    assert "pmemd.cuda.MPI" not in job.exec_list


def test_cuda_window_without_mpi_command_still_runs_serial():
    """exec_list[0] is the launcher slot even when it holds None -- it must be dropped."""
    job = _simulation(mpi_command=None, exec_list=[None])
    job.setup()

    assert job.exec_list[0] == "pmemd.cuda"
    assert None not in job.exec_list


def test_ligand_window_falls_back_to_the_plain_cpu_binary():
    """The ligand endstate runs on CPU by design: pmemd.cuda.MPI -> pmemd."""
    job = _simulation(system_type="ligand", num_cores=1)
    job.setup()

    assert job.exec_list[0] == "pmemd"


def test_non_cuda_mpi_window_keeps_its_launcher_and_np():
    job = _simulation(CUDA=False, system_type="complex", num_cores=8, executable="pmemd.MPI")
    job.setup()

    assert job.exec_list[:4] == ["mpirun", "-np", "8", "pmemd.MPI"]


# ---------------------------------------------------------------- REMDSimulation (many replicas)

def _remd(**overrides):
    attrs = dict(
        mpi_command="mpirun",
        executable="pmemd.cuda",
        nthreads=10,
        ng=10,
        exec_list=["mpirun"],
        read_files={"groupfile": "group.groupfile"},
    )
    attrs.update(overrides)
    return _make(REMDSimulation, **attrs)


def test_remd_spawns_the_mpi_build_derived_from_a_serial_config(monkeypatch):
    """A config naming the serial pmemd.cuda for its windows still gets .MPI for the ladder."""
    monkeypatch.setattr(shutil, "which", lambda exe: "/usr/bin/" + exe)
    job = _remd()
    job._setup()

    assert job.exec_list == [
        "mpirun", "-n", "10", "pmemd.cuda.MPI", "-ng", "10", "-groupfile", "group.groupfile",
    ]
    assert job.calc_setup is True


@pytest.mark.parametrize(
    "nthreads,expected_ranks",
    [
        (10, 10),    # one rank per replica
        (20, 20),    # two ranks per replica, still a whole multiple of -ng
        (0.1, 10),   # GPU-style Toil `cores` reservation says nothing about replicas
        (1, 10),     # ditto
        (5, 10),     # legacy halved value: AMBER refuses -n 5 against -ng 10
        (0, 10),
    ],
)
def test_rank_count_comes_from_nthreads_not_the_core_reservation(monkeypatch, nthreads, expected_ranks):
    monkeypatch.setattr(shutil, "which", lambda exe: "/usr/bin/" + exe)
    job = _remd(nthreads=nthreads)
    job._setup()

    assert job.exec_list[job.exec_list.index("-n") + 1] == str(expected_ranks)
    ranks = int(job.exec_list[job.exec_list.index("-n") + 1])
    assert ranks % job.ng == 0, "AMBER requires -n to be a whole multiple of -ng"


def test_remd_without_a_launcher_is_refused():
    with pytest.raises(RuntimeError, match="needs a launcher"):
        _remd(mpi_command=None, exec_list=[None])._setup()


def test_remd_names_the_missing_binary(monkeypatch):
    """Fail before mpirun does, so the error names the binary and the knob that selects it."""
    monkeypatch.setattr(shutil, "which", lambda exe: None)

    with pytest.raises(RuntimeError, match="pmemd.cuda.MPI"):
        _remd()._setup()


def test_remd_reports_an_empty_upstream_restart_list():
    """The old `len(incrd) == 1` guard turned this into `KeyError: 'incrd'` naming the wrong job."""
    job = _make(
        REMDSimulation,
        tempDir="/tmp",
        prmtop="top.parm7",
        restraint_file="empty.restraint",
        input_file=["mdin.relax.001"],
        incrd=[],
        runtype="equil",
        output_dir="/tmp/equilibration/complex",
        read_files={},
    )

    with pytest.raises(RuntimeError, match="empty restart list"):
        job.run(_FakeFileStore())


# ---------------------------------------------------------------- failed AMBER runs

def test_failed_amber_run_raises_instead_of_returning_empty_results(tmp_path, monkeypatch):
    """A dead simulation used to 'succeed' with no restart, breaking a job two steps later."""
    prmtop = tmp_path / "top.parm7"
    prmtop.write_text("")

    job = _make(
        Calculation,
        output_dir=str(tmp_path),
        directory_args={"runtype": "minimization"},
        num_cores=1,
        CUDA=False,
        calc_setup=True,
        post_analysis=False,
        exec_list=["pmemd", "-O", "-i", "mdin"],
        read_files={"prmtop": str(prmtop)},
    )
    monkeypatch.setattr(
        simulations.sp,
        "run",
        lambda *a, **k: types.SimpleNamespace(
            returncode=1, stdout=b"", stderr=b"Bad inpcrd file!"
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        job.run(_FakeFileStore())

    message = str(excinfo.value)
    assert "AMBER exited 1" in message
    assert "Bad inpcrd file!" in message, "AMBER's own diagnosis must survive"
    assert "pmemd -O -i mdin" in message, "the failing command must be reported"


# ---------------------------------------------------------------- needs a real AMBER + MPI

@pytest.mark.skipif(
    shutil.which("mpirun") is None or shutil.which("sander.MPI") is None,
    reason="needs mpirun and an AMBER MPI build; run locally with the AMBER module loaded",
)
def test_mpirun_can_launch_a_replica_ladder():
    """Guards the Slurm/OpenMPI slot limit: `--ntasks=1` gives one slot and refuses -n 2.

    Not runnable in CI (no AMBER). Run it on the cluster inside the same allocation the
    workflow uses -- it reproduces "There are not enough slots available" in seconds rather
    than after the DAG has built.
    """
    result = sp.run(
        ["mpirun", "-n", "2", "sander.MPI", "--version"],
        stdout=sp.PIPE,
        stderr=sp.PIPE,
    )

    assert b"not enough slots" not in result.stderr, (
        "MPI cannot launch 2 ranks in this allocation. Either export "
        "OMPI_MCA_rmaps_base_oversubscribe=true or raise --ntasks.\n"
        + result.stderr.decode(errors="replace")
    )
