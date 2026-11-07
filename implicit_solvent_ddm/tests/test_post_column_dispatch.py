"""Pure unit tests for the two per-column decisions in ``IntermidateRunner.only_post_analysis``.

Both have hosted a production bug, and neither had coverage:

* ``_select_post_mdin`` -- which scoring mdin a column gets. The GB-dielectric band's MD is
  written salt-free by ``generate_extdiel_mdin``; scoring it with the user's saltcon drew
  samples from one Hamiltonian and evaluated them under another (fixed in 34cff97).
* ``_column_is_required`` -- whether a banded run evaluates a cell. An unresolved target key
  used to be skipped silently, which punched holes in blocks the chained solve required and
  surfaced only as ``ValueError`` from ``chained_mbar_result``.

These tests do NOT touch Toil/AMBER: they build the runner with ``__new__`` and set only the
attributes the two methods read. Run with::

    pytest implicit_solvent_ddm/tests/test_post_column_dispatch.py -v
"""
import pytest

from implicit_solvent_ddm import block_mbar as bm
from implicit_solvent_ddm.runner import IntermidateRunner


class _Sim:
    """Just the ``directory_args`` carrier the two methods read off a Simulation."""

    def __init__(self, **directory_args):
        self.directory_args = directory_args


def _runner(saltfree="SALTFREE"):
    """An IntermidateRunner with only the mdin slots populated (no Toil Job.__init__)."""
    runner = IntermidateRunner.__new__(IntermidateRunner)
    runner.mdin = "POST"
    runner.no_solvent_mdin = "NOSOLV"
    runner.saltfree_mdin = saltfree
    return runner


# --------------------------------------------------------------------------------------------------
# _select_post_mdin
# --------------------------------------------------------------------------------------------------
def test_gb_dielectric_column_scores_saltfree():
    """The fix: the band is scored under the Hamiltonian its own MD used."""
    sim = _Sim(igb_value="igb_2", state_label="gb_dielectric", extdiel=2.0)
    assert _runner()._select_post_mdin(sim) == "SALTFREE"


def test_gb_dielectric_dispatch_survives_int_igb_value():
    """setup_gb_external_dielectric sets the string 'igb_2'; the ALS path sets the int."""
    sim = _Sim(igb_value=2, state_label="gb_dielectric", extdiel=2.0)
    assert _runner()._select_post_mdin(sim) == "SALTFREE"


def test_gas_column_scores_nosolv_even_when_labelled_gb_dielectric():
    """igb=6 wins: a gas column has no GB term to get the salt wrong."""
    sim = _Sim(igb_value=6, state_label="gb_dielectric")
    assert _runner()._select_post_mdin(sim) == "NOSOLV"


@pytest.mark.parametrize("label", ["lambda_window", "electrostatics", "endstate", "no_gb"])
def test_non_band_columns_keep_the_user_saltcon(label):
    """Only gb_dielectric is salt-free -- every other column matches its own salted MD."""
    sim = _Sim(igb_value=2, state_label=label)
    assert _runner()._select_post_mdin(sim) == "POST"


def test_missing_saltfree_mdin_falls_back_to_post_mdin():
    """Resumed jobstores predate the fifth mdin; they must keep running, not crash."""
    sim = _Sim(igb_value=2, state_label="gb_dielectric")
    assert _runner(saltfree=None)._select_post_mdin(sim) == "POST"


# --------------------------------------------------------------------------------------------------
# _column_is_required
# --------------------------------------------------------------------------------------------------
ORDER = (
    [("endstate", "78.5", "1.0", "0.0")]
    + [("lambda_window", "78.5", "1.0", f"{c}") for c in (-2.0, 0.0, 2.0, 4.0)]
    + [("electrostatics", "0.0", f"{q}", "4.0") for q in (1.0, 0.5, 0.0)]
)
PAIRS = bm.required_pairs(ORDER, 4, bm.band_boundaries(ORDER))


def test_required_cell_is_evaluated():
    lam = ("lambda_window", "78.5", "1.0", "4.0")
    ele = ("electrostatics", "0.0", "1.0", "4.0")
    assert IntermidateRunner._column_is_required(ele, lam, PAIRS)


def test_cell_outside_the_chain_is_skipped():
    far_a = ("endstate", "78.5", "1.0", "0.0")
    far_b = ("electrostatics", "0.0", "0.0", "4.0")
    assert (far_a, far_b) not in PAIRS  # the chain genuinely does not need it
    assert not IntermidateRunner._column_is_required(far_a, far_b, PAIRS)


def test_unresolved_target_is_evaluated_not_dropped():
    """The regression: a column that resolves to no cycle state must still be scored.

    Dropping it is what left block 14 with unevaluated cells in MCL-1_ligand-1_md_78746.
    """
    ele = ("electrostatics", "0.0", "1.0", "4.0")
    assert IntermidateRunner._column_is_required(ele, None, PAIRS)


def test_phantom_orientational_column_resolves_and_is_kept():
    """End-to-end of the two pieces: apo dirargs -> canonical key -> required.

    Production apo windows carry a phantom orientational_restraints (setup_apply_restraint_windows
    copies no_gb_args), so the raw key misses the cycle order entirely.
    """
    known = set(ORDER)
    apo_column = _Sim(
        state_label="lambda_window",
        extdiel=78.5,
        charge=1.0,
        conformational_restraint=4.0,
        orientational_restraints=8.0,
    )
    raw = bm.state_key_from_dirargs(apo_column.directory_args)
    assert raw not in known  # what the old code looked up, and always missed

    target = bm.canonical_state_key(apo_column.directory_args, known)
    source = ("electrostatics", "0.0", "1.0", "4.0")
    assert IntermidateRunner._column_is_required(source, target, PAIRS)
