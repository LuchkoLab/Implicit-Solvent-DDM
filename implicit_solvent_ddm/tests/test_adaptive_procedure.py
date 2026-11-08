import os
import re
import shutil
import pandas as pd
import parmed as pmd
import pytest
import yaml
import numpy as np
from implicit_solvent_ddm.config import Config

pytestmark = pytest.mark.workflow

working_directory = os.getcwd()


@pytest.fixture(scope="module")
def load_complex_dataframe_fe(run_workflow):
    """_summary_

    Returns:
        _type_: _description_
    """

    return pd.read_hdf(".cache/cb7-mol01/cb7-mol01_fe.h5")


@pytest.fixture(scope="module")
def load_receptor_dataframe_fe(run_workflow):
    """_summary_

    Returns:
        _type_: _description_
    """
    return pd.read_hdf(".cache/cb7-mol01/receptor_CB7_fe.h5")


@pytest.fixture(scope="module")
def load_ligand_dataframe_fe(run_workflow):
    """_summary_

    Returns:
        _type_: _description_
    """
    return pd.read_hdf(".cache/cb7-mol01/ligand_M01_fe.h5")


# walk through all directories and collect mdout files from workflow
def test_ligand_adaptive_restraints(
    load_ligand_dataframe_fe, get_ligand_config: Config
):
    """_summary_

    Args:
        get_ligand_config (Config): _description_
    """
    completed_array = (
        load_ligand_dataframe_fe.groupby("traj_state")
        .get_group("lambda_window")
        .groupby("traj_restraints")
        .first()
        .index.astype(float)
    ).values
    adaptive_windows = np.array(
        get_ligand_config.intermediate_args.exponent_conformational_forces
    )
    completed_array.sort()
    adaptive_windows.sort()

    assert all(completed_array == adaptive_windows)


# walk through all directories and collect mdout files from workflow
def test_complex_adaptive_restraints(
    load_complex_dataframe_fe,
):
    """_summary_

    Args:
        get_ligand_config (Config): _description_
    """
    restraint_index = (
        load_complex_dataframe_fe.groupby("traj_state")
        .get_group("lambda_window")
        .groupby("traj_restraints")
        .first()
        .index
    )
    conformation_exponent = []
    for restraints in restraint_index:
        restraint_pairs = restraints.split("_")
        conformation_exponent.append(float(restraint_pairs[0]))

    conformation_exponent = np.array(conformation_exponent)
    conformation_exponent.sort()

    adaptive_conformational_windows = np.array(
        [float(x) for x in os.listdir("mdgb/cb7-mol01/lambda_window/1.0/78.5/")]
    )
    adaptive_conformational_windows.sort()

    assert all(conformation_exponent == adaptive_conformational_windows)


# walk through all directories and collect mdout files from workflow
def test_receptor_adaptive_restraints(
    load_receptor_dataframe_fe, get_receptor_config: Config
):
    """_summary_

    Args:
        get_ligand_config (Config): _description_
    """
    completed_array = (
        load_receptor_dataframe_fe.groupby("traj_state")
        .get_group("lambda_window")
        .groupby("traj_restraints")
        .first()
        .index.astype(float)
    ).values
    adaptive_windows = np.array(
        get_receptor_config.intermediate_args.exponent_conformational_forces
    )
    completed_array.sort()
    adaptive_windows.sort()

    assert all(completed_array == adaptive_windows)
