"""Unit tests for the receptor's post-processing starting coordinate.

`receptor_coordinate_filename` serves two roles: the endstate ensemble replayed through
sander `-y`, and the starting coordinate handed to `-c`. Reusing one receptor REMD run
across ligands makes that field a multi-frame trajectory, which is right for `-y` and
unusable for `-c` -- sander cannot parse it as an inpcrd and exits 1, and when the two
roles resolve to the same Toil FileID the job dies earlier still on "Cannot Overwrite".

These tests do NOT touch AMBER or a real Toil workflow. Run with::

    pytest implicit_solvent_ddm/tests/test_receptor_endstate_coordinate.py -v
"""
import os
import tempfile

import pytest
import pytraj as pt
import yaml

from implicit_solvent_ddm.config import Config, ParameterFiles

RECEPTOR_TOP = "implicit_solvent_ddm/tests/structs/CB7.parm7"
RECEPTOR_ENSEMBLE = "implicit_solvent_ddm/tests/structs/CB7_000_300K.nc"
SEED_CONFIG = "implicit_solvent_ddm/tests/input_files/config.yaml"


def _parameter_files(**attrs):
    """Build a ParameterFiles without __post_init__, which validates a real complex."""
    obj = object.__new__(ParameterFiles)
    defaults = dict(
        complex_parameter_filename=None,
        complex_coordinate_filename=None,
        ligand_parameter_filename=None,
        ligand_coordinate_filename=None,
        receptor_parameter_filename=None,
        receptor_coordinate_filename=None,
        complex_initial_coordinate=None,
        ligand_initial_coordinate=None,
        receptor_initial_coordinate=None,
    )
    defaults.update(attrs)
    for key, value in defaults.items():
        setattr(obj, key, value)
    obj.tempdir = tempfile.TemporaryDirectory()
    return obj


def test_multi_frame_receptor_gets_its_own_single_frame_restart():
    """The reused-receptor case: -c must not be the multi-frame ensemble."""
    files = _parameter_files(
        receptor_parameter_filename=RECEPTOR_TOP,
        receptor_coordinate_filename=RECEPTOR_ENSEMBLE,
    )
    assert pt.iterload(RECEPTOR_ENSEMBLE, RECEPTOR_TOP).n_frames > 1

    written = files.set_receptor_initial_coordinate()

    assert written == files.receptor_initial_coordinate
    assert os.path.exists(written)
    # The assertion that maps straight onto the production failure.
    assert pt.iterload(written, RECEPTOR_TOP).n_frames == 1


def test_ensemble_field_is_left_alone():
    """-y keeps the full ensemble, so the receptor leg loses no sampling."""
    files = _parameter_files(
        receptor_parameter_filename=RECEPTOR_TOP,
        receptor_coordinate_filename=RECEPTOR_ENSEMBLE,
    )
    files.set_receptor_initial_coordinate()

    assert files.receptor_coordinate_filename == RECEPTOR_ENSEMBLE
    assert files.receptor_initial_coordinate != RECEPTOR_ENSEMBLE


def test_explicit_config_value_is_not_overwritten():
    """A path set in the config file wins over extraction."""
    files = _parameter_files(
        receptor_parameter_filename=RECEPTOR_TOP,
        receptor_coordinate_filename=RECEPTOR_ENSEMBLE,
        receptor_initial_coordinate="/set/by/the/user.rst7",
    )

    assert files.set_receptor_initial_coordinate() == "/set/by/the/user.rst7"
    assert files.receptor_initial_coordinate == "/set/by/the/user.rst7"


@pytest.mark.parametrize(
    "attrs",
    [
        {},
        {"receptor_coordinate_filename": RECEPTOR_ENSEMBLE},
        {"receptor_parameter_filename": RECEPTOR_TOP},
    ],
    ids=["no-receptor", "coordinate-only", "topology-only"],
)
def test_incomplete_receptor_pair_extracts_nothing(attrs):
    files = _parameter_files(**attrs)

    assert files.set_receptor_initial_coordinate() is None
    assert files.receptor_initial_coordinate is None


def test_incrd_and_inptraj_do_not_share_a_basename():
    """Simulation.run reads both into tempDir by basename; equal names raise Cannot Overwrite."""
    files = _parameter_files(
        receptor_parameter_filename=RECEPTOR_TOP,
        receptor_coordinate_filename=RECEPTOR_ENSEMBLE,
    )
    files.set_receptor_initial_coordinate()

    incrd = os.path.basename(files.receptor_initial_coordinate)
    inptraj = os.path.basename(files.receptor_coordinate_filename)
    assert incrd != inptraj


def test_get_inital_coordinate_preserves_an_explicit_receptor_value():
    """The endstate_method 0 path must not clobber a path set in the config file."""
    files = _parameter_files(
        complex_parameter_filename=RECEPTOR_TOP,
        complex_coordinate_filename=RECEPTOR_ENSEMBLE,
        receptor_parameter_filename=RECEPTOR_TOP,
        receptor_coordinate_filename=RECEPTOR_ENSEMBLE,
        receptor_initial_coordinate="/set/by/the/user.rst7",
    )
    files.get_inital_coordinate()

    assert files.receptor_initial_coordinate == "/set/by/the/user.rst7"


def test_supplied_receptor_is_resolved_when_the_config_is_built():
    """Config construction, not the workflow, is what fills the field in."""
    with open(SEED_CONFIG) as yml:
        config = Config.from_config(yaml.safe_load(yml))

    receptor = config.endstate_files
    assert receptor.receptor_initial_coordinate is not None
    assert receptor.receptor_initial_coordinate != receptor.receptor_coordinate_filename
    assert pt.iterload(
        receptor.receptor_initial_coordinate, str(receptor.receptor_parameter_filename)
    ).n_frames == 1
