import logging
import os
import random
import re
import shutil
import time

import parmed as pmd
import pytest
import yaml
from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.implicit_ddm_workflow import ddm_workflow
from matplotlib.pyplot import get
from toil.common import Toil
from toil.job import Job

# needs the session-wide Toil + sander CB7 run; some tests read its output files directly
pytestmark = [pytest.mark.workflow, pytest.mark.usefixtures("run_workflow")]

working_directory = os.getcwd()


# load topology without any changes
parm_file_path = os.path.join("implicit_solvent_ddm/tests/structs/")


@pytest.fixture
def load_topology():
    return pmd.load_file(os.path.join(parm_file_path, "cb7-mol01.parm7"))


# load in complex topology file with vdw/lj-potentials and no ligand charges
@pytest.fixture
def load_exclusions_no_charge():
    return pmd.load_file(
        os.path.join(parm_file_path, "exclusions_no_charge_cb7-M01.parm7")
    )


# load complex topology with Lj-pontential btw Host & Ligand but the ligand charges are set to 0
@pytest.fixture
def load_no_charge_topology():
    return pmd.load_file(os.path.join(parm_file_path, "no_charge_cb7-M01.parm7"))


# walk through all directories and collect mdout files from workflow
@pytest.fixture(scope="module", params=["mdgb/M01/", "mdgb/cb7-mol01/", "mdgb/CB7/"])
def get_mdouts(request, run_workflow):
    mdout_files = []
    for root, dirs, files in os.walk(request.param, topdown=False):
        for name in files:
            if "mdout" == name:
                mdout_files.append(os.path.join(root, name))
    yield mdout_files


@pytest.mark.parametrize(
    "test_input, expected", [("mdgb/M01/", 6), ("mdgb/CB7/", 4), ("mdgb/cb7-mol01/", 7)]
)
def test_completed_workflow(test_input, expected, run_workflow):
    print("test input:", test_input)
    mdout_files = []
    for root, dirs, files in os.walk(test_input, topdown=False):
        for name in files:
            if "mdout" == name:
                mdout_files.append(os.path.join(root, name))

    assert len(mdout_files) == expected


# read in all mdout files and check if run was completed
def test_successful_simulations(get_mdouts):
    """
    Executes the entire workflow and does a check if all output files were completed.
    """
    for mdout in get_mdouts:
        complete = False
        with open(mdout) as output:
            data = output.readlines()
            for line in data:
                if "Total time" in line:
                    complete = True

            assert complete


@pytest.fixture(scope="module", params=[0, 0.5, 1])
def test_ligand_charges(request, load_topology):
    filepath = f"mdgb/M01/electrostatics/{request.param}"

    parm = load_parm7(filepath)
    reduce_charge = [
        request.param * charge for charge in load_topology[":M01"].parm_data["CHARGE"]
    ]

    assert parm[":M01"].parm_data["CHARGE"] == reduce_charge


@pytest.fixture(scope="module", params=[0, 0.5, 1])
def test_complex_ligand_charges(request, load_topology):
    filepath = f"mdgb/cb7-mol01/electrostatics/{request.param}"
    parm = load_parm7(filepath)

    reduce_charge = [
        request.param * charge for charge in load_topology[":M01"].parm_data["CHARGE"]
    ]
    assert parm[":M01"].parm_data["CHARGE"] == reduce_charge


# check complex parameter potential that host and guest have no interactions and ligand charge = 0
def test_complex_exclusions(load_exclusions_no_charge):
    output_solute_traj = load_parm7("mdgb/cb7-mol01/no_interactions/0.0/0.0/4.0/8.0/")

    assert (
        output_solute_traj.parm_data["NUMBER_EXCLUDED_ATOMS"]
        == load_exclusions_no_charge.parm_data["NUMBER_EXCLUDED_ATOMS"]
    )


@pytest.mark.parametrize(
    "no_igb_path",
    [
        "mdgb/M01/electrostatics/",
        "mdgb/CB7/no_gb/",
        "mdgb/cb7-mol01/no_interactions/",
        "mdgb/cb7-mol01/interactions/",
    ],
)
def test_no_solvent_complex(no_igb_path):
    for root, dirs, files in os.walk(no_igb_path):
        for name in files:
            if "mdout" == name:
                with open(os.path.join(root, name)) as output:
                    data = output.read()
                assert bool(re.search(r"igb\s*=\s*6", data))


def assert_exclusions_charges(output, correct_output):
    assert (
        output.parm_data["NUMBER_EXCLUDED_ATOMS"]
        == correct_output.parm_data["NUMBER_EXCLUDED_ATOMS"]
    )

    assert (
        output[":M01"].parm_data["CHARGE"] == correct_output[":M01"].parm_data["CHARGE"]
    )


def load_parm7(output_path):
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if bool(re.match(r".*\.parm7", file)):
                solute = os.path.join(root, file)

    return pmd.load_file(solute)
