"""Pure tests for block_mbar: partitioning, evaluation counts and state identity.

No pymbar solve and no AMBER here -- these run anywhere.
"""

import pytest

from implicit_solvent_ddm import block_mbar as bm


# --------------------------------------------------------------------------------------------------
# partition_blocks
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 5, 12, 39, 43])
def test_blocks_cover_every_transition_and_share_endpoints(n):
    for K in (2, 3, 4, 5, 7):
        blocks = bm.partition_blocks(n, K)
        assert blocks[0][0] == 0
        assert blocks[-1][-1] == n - 1
        for block in blocks:
            assert 2 <= len(block) <= K
            assert block == list(range(block[0], block[-1] + 1))
        # consecutive blocks overlap in exactly one state, so dG telescopes
        for a, b in zip(blocks, blocks[1:]):
            assert a[-1] == b[0]


def test_block_size_two_is_the_bar_chain():
    blocks = bm.partition_blocks(10, 2)
    assert blocks == [[i, i + 1] for i in range(9)]


def test_last_block_absorbs_the_remainder():
    # 8 states -> 7 transitions; K=4 chunks them 3 + 3 + 1
    assert bm.partition_blocks(8, 4) == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7]]


@pytest.mark.parametrize("bad", [0, 1, -3])
def test_block_size_below_two_rejected(bad):
    with pytest.raises(ValueError, match="block_size must be >= 2"):
        bm.partition_blocks(10, bad)


def test_too_few_states_rejected():
    with pytest.raises(ValueError, match="at least 2 states"):
        bm.partition_blocks(1, 2)


@pytest.mark.parametrize("isolated", [[3], [0], [8], [3, 6], [0, 4, 8]])
def test_no_block_straddles_an_isolated_transition(isolated):
    # The invariant that matters: an isolated transition is covered ONLY by its own 2-state
    # block, so a model switch can never sit inside a wider (and therefore disconnected) block.
    blocks = bm.partition_blocks(10, 4, boundaries=isolated)
    for t in isolated:
        covering = [b for b in blocks if b[0] <= t and t + 1 <= b[-1]]
        assert covering == [[t, t + 1]]
    assert blocks[0][0] == 0 and blocks[-1][-1] == 9


# --------------------------------------------------------------------------------------------------
# evaluation counts -- the numbers the cost case rests on
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [5, 12, 35, 39, 40, 43])
def test_bar_chain_is_the_tridiagonal(n):
    assert bm.evaluation_count(n, 2) == 3 * n - 2


def test_mcl1_complex_leg_counts():
    # N=43 (MCL-1/ligand-1 complex leg), no band alignment. 42 transitions chunk evenly for
    # K=2/3/4/7; K=5 leaves a remainder of 2, so its last block holds 3 states, not 5.
    assert bm.evaluation_count(43, 2) == 127
    assert bm.evaluation_count(43, 3) == 169
    assert bm.evaluation_count(43, 4) == 211
    assert bm.evaluation_count(43, 5) == 249
    assert bm.evaluation_count(43, 7) == 337


def test_uneven_remainder_shrinks_the_last_block():
    blocks = bm.partition_blocks(43, 5)
    assert [len(b) for b in blocks] == [5] * 10 + [3]


def test_evaluation_count_matches_required_pairs():
    order = [("lambda_window", "78.5", "1.0", f"{i}.0_{i + 4}.0") for i in range(12)]
    for K in (2, 3, 4):
        assert len(bm.required_pairs(order, K)) == bm.evaluation_count(12, K)


def test_counts_grow_with_block_size_and_stay_below_dense():
    n = 43
    counts = [bm.evaluation_count(n, K) for K in (2, 3, 4, 5, 7)]
    assert counts == sorted(counts)
    assert counts[-1] < n * n


def test_isolating_boundaries_never_costs_more():
    # Isolating a transition splits the run around it, so the neighbouring blocks get SHORTER
    # and the leg needs fewer evaluations, not more. Accuracy is what pays: the junction and
    # its neighbours are solved with less context. That is the intended trade -- a wide block
    # spanning a model switch is internally disconnected and buys nothing anyway.
    n, K = 20, 4
    assert bm.evaluation_count(n, K, boundaries=[5, 11]) <= bm.evaluation_count(n, K)


# --------------------------------------------------------------------------------------------------
# band_boundaries
# --------------------------------------------------------------------------------------------------
def test_band_boundaries_finds_sub_leg_changes():
    order = [
        ("endstate", "78.5", "1.0", "0.0"),
        ("lambda_window", "78.5", "1.0", "1.0_5.0"),
        ("lambda_window", "78.5", "1.0", "2.0_6.0"),
        ("electrostatics", "78.5", "0.5", "4.0_8.0"),
        ("gb_dielectric", "2.0", "0.0", "4.0_8.0"),
    ]
    assert bm.band_boundaries(order) == [0, 2, 3]


def test_band_boundaries_empty_for_uniform_leg():
    order = [("lambda_window", "78.5", "1.0", f"{i}.0") for i in range(6)]
    assert bm.band_boundaries(order) == []


# --------------------------------------------------------------------------------------------------
# state identity
# --------------------------------------------------------------------------------------------------
def test_state_key_matches_cyclesteps_halo_spelling():
    # CycleSteps.interations: ("interactions", "0.0", "0.0", f"{max_con}_{max_orient}")
    assert bm.state_key("interactions", 0.0, 0.0, 4.0, 8.0) == (
        "interactions",
        "0.0",
        "0.0",
        "4.0_8.0",
    )


def test_state_key_matches_cyclesteps_apo_spelling():
    # CycleSteps.no_gb: ("no_gb", "0.0", "1.0", f"{max_con}") -- conformational only
    assert bm.state_key("no_gb", 0.0, 1.0, 4.0) == ("no_gb", "0.0", "1.0", "4.0")


def test_state_key_normalises_strings_and_floats_identically():
    assert bm.state_key("endstate", "78.5", "1.0", "0.0") == bm.state_key(
        "endstate", 78.5, 1.0, 0.0
    )


def test_state_key_preserves_dielectric_precision():
    eps = 1.0204081632653061
    assert bm.state_key("gb_dielectric", eps, 0.0, 4.0, 8.0)[1] == "1.0204081632653061"


def test_state_key_from_dirargs_halo_and_apo():
    halo = {
        "state_label": "lambda_window",
        "extdiel": 78.5,
        "charge": 1.0,
        "conformational_restraint": 2.0,
        "orientational_restraints": 6.0,
    }
    assert bm.state_key_from_dirargs(halo) == ("lambda_window", "78.5", "1.0", "2.0_6.0")

    apo = {k: v for k, v in halo.items() if k != "orientational_restraints"}
    assert bm.state_key_from_dirargs(apo) == ("lambda_window", "78.5", "1.0", "2.0")


# --------------------------------------------------------------------------------------------------
# canonical_state_key -- reconciling the apo spelling
#
# Production apo legs do NOT match the `apo` case above: setup_apply_restraint_windows builds its
# args with copy(self.no_gb_args), which always sets orientational_restraints, and the
# `exponent_orientational is None` branch never removes it. So a ligand/receptor lambda_window
# carries a phantom 8.0 while CycleSteps.apply_restraints spells it with the conformational force
# alone. Scoring dropped those cells silently until the chained solve rejected the block.
# --------------------------------------------------------------------------------------------------
APO_LIGAND_ORDER = (
    [("endstate", "78.5", "1.0", "0.0")]
    + [("lambda_window", "78.5", "1.0", f"{c}") for c in (-2.0, 0.0, 2.0, 4.0)]
    + [("electrostatics", "0.0", f"{q}", "4.0") for q in (1.0, 0.5, 0.0)]
)


def _apo_dirargs(label, extdiel, charge, con, orient=8.0):
    """An apo window as production actually builds it -- phantom orientational included."""
    return {
        "state_label": label,
        "extdiel": extdiel,
        "charge": charge,
        "conformational_restraint": con,
        "orientational_restraints": orient,
    }


def test_canonical_key_resolves_phantom_orientational_to_apo_spelling():
    known = set(APO_LIGAND_ORDER)
    args = _apo_dirargs("lambda_window", 78.5, 1.0, 4.0)
    # the raw key is the one production emitted, and it is NOT in the cycle order
    assert bm.state_key_from_dirargs(args) == ("lambda_window", "78.5", "1.0", "4.0_8.0")
    assert bm.state_key_from_dirargs(args) not in known
    assert bm.canonical_state_key(args, known) == ("lambda_window", "78.5", "1.0", "4.0")


def test_canonical_key_leaves_a_genuine_halo_key_untouched():
    """The complex leg is really halo -- remove_restraints/complex_charges are compound."""
    known = {("lambda_window", "78.5", "1.0", "2.0_6.0")}
    args = _apo_dirargs("lambda_window", 78.5, 1.0, 2.0, orient=6.0)
    assert bm.canonical_state_key(args, known) == ("lambda_window", "78.5", "1.0", "2.0_6.0")


def test_canonical_key_returns_none_for_a_genuinely_absent_state():
    known = set(APO_LIGAND_ORDER)
    args = _apo_dirargs("lambda_window", 78.5, 1.0, 99.0)
    assert bm.canonical_state_key(args, known) is None


def test_canonical_key_does_not_invent_a_match_from_a_bare_conformational_key():
    """No underscore -> nothing to strip; an unknown apo key stays unknown."""
    known = set(APO_LIGAND_ORDER)
    args = {
        "state_label": "lambda_window",
        "extdiel": 78.5,
        "charge": 1.0,
        "conformational_restraint": 99.0,
    }
    assert bm.canonical_state_key(args, known) is None


def test_block14_junction_cell_resolves_in_both_directions():
    """Regression for MCL-1_ligand-1_md_78746: the isolated lambda_window<->electrostatics block.

    The electrostatics row keyed cleanly (ligand_charge_args never sets orientational) so it took
    the banded path, then dropped every lambda_window column because those keys carried the
    phantom 8.0. Both directions must now land inside required_pairs.
    """
    order = APO_LIGAND_ORDER
    known = set(order)
    pairs = bm.required_pairs(order, 4, bm.band_boundaries(order))

    lam = bm.canonical_state_key(_apo_dirargs("lambda_window", 78.5, 1.0, 4.0), known)
    ele = bm.canonical_state_key(
        {
            "state_label": "electrostatics",
            "extdiel": 0.0,
            "charge": 1.0,
            "conformational_restraint": 4.0,
        },
        known,
    )
    assert (ele, lam) in pairs  # the cell that was silently skipped
    assert (lam, ele) in pairs


def test_parm_and_traj_keys_read_their_own_sides():
    run_args = {
        "state_label": "electrostatics",
        "extdiel": 78.5,
        "charge": 0.5,
        "conformational_restraint": 4.0,
        "orientational_restraints": 8.0,
        "traj_state_label": "lambda_window",
        "traj_extdiel": 78.5,
        "traj_charge": 1.0,
        "trajectory_restraint_conrest": 2.0,
        "trajectory_restraint_orenrest": 6.0,
    }
    assert bm.parm_key(run_args) == ("electrostatics", "78.5", "0.5", "4.0_8.0")
    assert bm.traj_key(run_args) == ("lambda_window", "78.5", "1.0", "2.0_6.0")


# --------------------------------------------------------------------------------------------------
# required_pairs shape
# --------------------------------------------------------------------------------------------------
def test_required_pairs_are_symmetric_and_include_the_diagonal():
    order = [("lambda_window", "78.5", "1.0", f"{i}.0") for i in range(8)]
    pairs = bm.required_pairs(order, 3)
    for state in order:
        assert (state, state) in pairs  # every window is scored under its own Hamiltonian
    for a, b in pairs:
        assert (b, a) in pairs


def test_bar_chain_requires_only_adjacent_pairs():
    order = [("lambda_window", "78.5", "1.0", f"{i}.0") for i in range(8)]
    index = {state: i for i, state in enumerate(order)}
    for a, b in bm.required_pairs(order, 2):
        assert abs(index[a] - index[b]) <= 1
