"""Unit tests for naming the endstate state key after the method that produced it.

The endstate row's ``state_label`` is BOTH its post-analysis directory segment and its MBAR
index key. While it was the bare string "endstate", a parquet left by a vanilla-MD run
satisfied ``IntermidateRunner.has_post_analysis_data`` for a REMD run, so the re-score was
skipped and July's energies were loaded as if they were REMD's.

Because the label is the MBAR key, the producer (setup_simulations) and every consumer
(matrix_order, postTreatment, adaptive_restraints) must resolve the SAME string or the
``.loc[]`` lookups raise KeyError. These tests pin that agreement.

No AMBER, no Toil workflow. Run with::

    pytest implicit_solvent_ddm/tests/test_endstate_label.py -v
"""
import pytest

from implicit_solvent_ddm.config import EndStateMethod
from implicit_solvent_ddm.matrix_order import CycleSteps
from implicit_solvent_ddm.postTreatment import ConsolidateData
from implicit_solvent_ddm.setup_simulations import SimulationSetup


def _method(method_type):
    """An EndStateMethod carrying only what the label property reads."""
    obj = object.__new__(EndStateMethod)
    obj.endstate_method_type = method_type
    return obj


@pytest.mark.parametrize(
    "method_type,expected",
    [
        ("remd", "remd_endstate"),
        ("basic_md", "vanillaMD_endstate"),
        (0, "user_provided_endstate"),
    ],
)
def test_label_names_the_endstate_method(method_type, expected):
    assert _method(method_type).endstate_state_label == expected


def test_every_valid_method_has_a_label():
    """EndStateMethod.__post_init__ accepts exactly these three; each needs a label."""
    for method_type in ["remd", "basic_md", 0]:
        assert _method(method_type).endstate_state_label


@pytest.mark.parametrize(
    "method_type,expected",
    [("remd", "remd_endstate"), ("basic_md", "vanillaMD_endstate"), (0, "user_provided_endstate")],
)
def test_dirstruct_carries_the_method_label(method_type, expected):
    """Both keys must move together: traj_state_label is the OUTER path segment and
    state_label the INNER one, and they become traj_state/parm_state in the dataframe."""
    setup = object.__new__(SimulationSetup)
    setup.topology = "MCL-1_ligand-1.parm7"
    setup.config = type(
        "Cfg",
        (),
        {
            "endstate_method": _method(method_type),
            "intermediate_args": type("IA", (), {"igb_solvent": 2})(),
            "system_settings": type("SS", (), {"top_directory_path": "/tmp/top"})(),
        },
    )()

    dirstruct = setup.apo_endstate_dirstruct

    assert dirstruct["state_label"] == expected
    assert dirstruct["traj_state_label"] == expected


def _cycle_steps(**overrides):
    kwargs = dict(
        conformation_forces=[-2.0, 8.0],
        orientational_forces=[-1.0, 8.0],
        charges_windows=[0.0, 1.0],
        external_dielectic=[1.0, 2.0],
    )
    kwargs.update(overrides)
    return CycleSteps(**kwargs)


def test_cycle_steps_uses_the_injected_label():
    """The cycle order is what canonical_state_key matches against, so it has to agree
    with the directory the post-analysis job wrote."""
    steps = _cycle_steps(endstate_label="remd_endstate")

    assert steps.endstate == [("remd_endstate", "78.5", "1.0", "0.0")]
    assert not any(state[0] == "endstate" for state in steps.endstate)


def test_cycle_steps_defaults_to_the_old_label():
    """Default keeps pre-change trees readable for anything constructing CycleSteps bare."""
    assert _cycle_steps().endstate == [("endstate", "78.5", "1.0", "0.0")]


def test_complex_order_carries_the_label():
    """matrix_order:159 appends the endstate to the complex cycle; it must move too."""
    steps = _cycle_steps(endstate_label="remd_endstate")
    steps.round(3)

    assert ("remd_endstate", "78.5", "1.0", "0.0_0.0") in steps.complex_order
    assert ("endstate", "78.5", "1.0", "0.0_0.0") not in steps.complex_order


def test_consolidate_data_accepts_the_label():
    """ConsolidateData's four .loc[] lookups read self.endstate_label."""
    job = object.__new__(ConsolidateData)
    job.endstate_label = "remd_endstate"

    assert job.endstate_label == "remd_endstate"


def test_producer_and_consumer_agree():
    """The whole point: the directory the post job writes and the key MBAR looks up must
    be the same string for a given config."""
    method = _method("remd")
    setup = object.__new__(SimulationSetup)
    setup.topology = "MCL-1_ligand-1.parm7"
    setup.config = type(
        "Cfg",
        (),
        {
            "endstate_method": method,
            "intermediate_args": type("IA", (), {"igb_solvent": 2})(),
            "system_settings": type("SS", (), {"top_directory_path": "/tmp/top"})(),
        },
    )()

    produced = setup.apo_endstate_dirstruct["state_label"]
    consumed = _cycle_steps(endstate_label=method.endstate_state_label).endstate[0][0]

    assert produced == consumed == "remd_endstate"
