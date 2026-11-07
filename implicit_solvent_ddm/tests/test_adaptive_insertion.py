"""Pure unit tests for the R-ADD adaptive restraint-window insertion engine.

These import the pure R-ADD helpers from ``implicit_solvent_ddm.adaptive_restraints`` and
deliberately do NOT touch the Toil/MBAR ``run_workflow`` session fixture in ``conftest.py`` (which
runs the full DDM workflow). Run with::

    pytest implicit_solvent_ddm/tests/test_adaptive_insertion.py -v
"""
import pytest

from implicit_solvent_ddm.adaptive_restraints import (
    derive_offset,
    min_direction_superdiagonal,
    find_bad_sections,
    select_section,
    worst_gap_index,
    snap_to_pool,
    insert_one,
    build_selected_block,
    restraint_band_superdiagonal,
    build_candidate_pool,
    plan_restraint_insertion,
    lambda_from_eps,
    eps_from_lambda,
    lambda_of_state,
    charge_of_state,
    plan_band_insertion,
    plan_batch_insertion,
    prune_schedule,
)

THRESH = 0.04


def _tridiag(superdiagonal):
    """Build a symmetric matrix whose min-direction superdiagonal equals ``superdiagonal``.

    Off-superdiagonal entries are 0 and the diagonal is 1, so ``min_direction_superdiagonal`` returns
    exactly the supplied adjacent-pair overlaps (used to exercise the band-slice adapter with the
    real cb7 MBAR state ordering without running MBAR).
    """
    n = len(superdiagonal) + 1
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
    for i, value in enumerate(superdiagonal):
        matrix[i][i + 1] = value
        matrix[i + 1][i] = value
    return matrix


# ---------------------------------------------------------------------------
# T8 — OFFSET derived & asserted (con/orient pairing is the atomic unit)
# ---------------------------------------------------------------------------
def test_derive_offset_constant():
    con = [-8.0, -2.0, 4.0]
    orient = [-4.0, 2.0, 8.0]
    assert derive_offset(con, orient) == 4.0


def test_derive_offset_nonconstant_raises():
    with pytest.raises(ValueError):
        derive_offset([-8.0, -2.0, 4.0], [-4.0, 2.0, 9.0])


def test_derive_offset_length_mismatch_raises():
    with pytest.raises(ValueError):
        derive_offset([-8.0, -2.0], [-4.0, 2.0, 8.0])


# ---------------------------------------------------------------------------
# T6 — min-direction superdiagonal: a one-sided weak pair registers as weak
# ---------------------------------------------------------------------------
def test_min_direction_superdiagonal_uses_min_not_average():
    # pair (0,1) is one-sided weak: fwd=0.01, rev=0.07 -> symmetric average 0.04 would PASS,
    # but the min-direction value 0.01 is correctly below threshold.
    O = [
        [0.90, 0.01, 0.00],
        [0.07, 0.85, 0.06],
        [0.00, 0.06, 0.94],
    ]
    sd = min_direction_superdiagonal(O)
    assert sd == [0.01, 0.06]
    assert sd[0] < THRESH


# ---------------------------------------------------------------------------
# T1 — bad-section finding + selection tie-breaks (total order)
# ---------------------------------------------------------------------------
def test_find_bad_sections():
    sd = [0.06, 0.02, 0.01, 0.08, 0.03]
    assert find_bad_sections(sd, THRESH) == [[1, 2], [4]]


def test_find_bad_sections_none():
    assert find_bad_sections([0.06, 0.08, 0.10], THRESH) == []


def test_select_section_longest_wins():
    sd = [0.02, 0.01, 0.08, 0.03]  # sections [[0, 1], [3]]
    assert select_section([[0, 1], [3]], sd) == [0, 1]


def test_select_section_tiebreak_min_overlap():
    # equal-length sections -> the one with the smaller MINIMUM overlap wins
    sd = [0.03, 0.08, 0.005]  # [[0]] (min 0.03) vs [[2]] (min 0.005)
    assert select_section([[0], [2]], sd) == [2]


def test_select_section_tiebreak_start_index():
    # equal length, equal minimum -> lowest start index
    sd = [0.01, 0.08, 0.01]  # [[0]] and [[2]], both min 0.01
    assert select_section([[0], [2]], sd) == [0]


# ---------------------------------------------------------------------------
# T2 — worst sub-gap within a section
# ---------------------------------------------------------------------------
def test_worst_gap_index():
    sd = [0.02, 0.005, 0.03]
    assert worst_gap_index([0, 1, 2], sd) == 1


def test_worst_gap_index_ties_low():
    sd = [0.01, 0.05, 0.01]
    assert worst_gap_index([0, 2], sd) == 0


# ---------------------------------------------------------------------------
# T3 — snap to nearest free pool candidate; ties -> lower exponent
# ---------------------------------------------------------------------------
def test_snap_to_pool_nearest():
    pool = [-7.0, -6.0, -5.0, -4.0, -3.0]
    assert snap_to_pool(-8.0, -2.0, -5.0, pool, selected=set()) == -5.0


def test_snap_to_pool_tie_breaks_low():
    pool = [-6.0, -4.0]  # both equidistant from ideal -5 -> choose the lower (-6)
    assert snap_to_pool(-8.0, -2.0, -5.0, pool, selected=set()) == -6.0


def test_snap_to_pool_skips_selected_and_outside_gap():
    pool = [-9.0, -5.0, -2.0, 5.0]  # -9, -2, 5 lie outside the open gap (-8, -2)
    assert snap_to_pool(-8.0, -2.0, -5.0, pool, selected={-5.0}) is None


# ---------------------------------------------------------------------------
# T2 (one full step) — insert at the log2 midpoint of the single worst gap
# ---------------------------------------------------------------------------
def test_insert_one_picks_worst_gap_midpoint():
    block = [-8.0, -2.0, 4.0]
    sd = [0.01, 0.06]  # pair (-8,-2) weak; pair (-2,4) ok
    pool = [-7.0, -6.0, -5.0, -4.0, -3.0]
    new_con, converged, reason = insert_one(
        block, sd, pool, THRESH, lower_bound=-8.0, upper_bound=4.0
    )
    assert converged is False
    assert new_con == -5.0  # midpoint of (-8,-2), snapped to the matching pool point


def test_insert_one_targets_the_worst_of_several_gaps():
    block = [-8.0, -4.0, 0.0, 4.0]
    sd = [0.03, 0.10, 0.005]  # weak: pair0 (0.03) and pair2 (0.005); pair2 is worse
    pool = [-6.0, -2.0, 2.0]
    new_con, converged, _ = insert_one(block, sd, pool, THRESH, -8.0, 4.0)
    assert converged is False
    assert new_con == 2.0  # midpoint of the worst gap (0,4)


# ---------------------------------------------------------------------------
# T1 — converged when every adjacent overlap is adequate
# ---------------------------------------------------------------------------
def test_insert_one_converged_when_all_overlaps_adequate():
    block = [-8.0, -2.0, 4.0]
    sd = [0.08, 0.06]
    assert insert_one(block, sd, [-5.0], THRESH, -8.0, 4.0) == (None, True, "converged")


# ---------------------------------------------------------------------------
# T4 — anchor protection: never returns con at/below endstate or at/above max
# ---------------------------------------------------------------------------
def test_insert_one_never_breaches_anchors():
    block = [-8.0, -2.0, 4.0]
    sd = [0.01, 0.01]  # both gaps weak
    # pool offers only candidates at/below the endstate or at/above the pinned max
    pool = [-9.0, -8.0, 4.0, 5.0]
    new_con, converged, reason = insert_one(block, sd, pool, THRESH, -8.0, 4.0)
    assert new_con is None
    assert converged is True
    assert reason == "pool-exhausted"


# ---------------------------------------------------------------------------
# T5 — pool exhaustion terminates (no infinite recursion) and is distinguished
#      from clean convergence
# ---------------------------------------------------------------------------
def test_insert_one_pool_exhausted_terminates():
    block = [-8.0, -2.0, 4.0]
    sd = [0.01, 0.02]  # both gaps weak
    new_con, converged, reason = insert_one(block, sd, [], THRESH, -8.0, 4.0)
    assert new_con is None and converged is True and reason == "pool-exhausted"


def test_insert_one_partial_fill_then_pool_exhausted():
    # gap0 (-8,-2) is fillable from the pool; gap1 (-2,4) is not -> first call fills gap0.
    block = [-8.0, -2.0, 4.0]
    sd = [0.01, 0.01]
    pool = [-5.0]  # only a candidate for gap0
    new_con, converged, reason = insert_one(block, sd, pool, THRESH, -8.0, 4.0)
    assert new_con == -5.0 and converged is False  # worst-tie -> lowest k=0 -> gap0 filled first


def test_insert_one_length_mismatch_raises():
    with pytest.raises(ValueError):
        insert_one([-8.0, -2.0, 4.0], [0.01], [-5.0], THRESH, -8.0, 4.0)


# ---------------------------------------------------------------------------
# T9 — band-slice adapter: leg direction (complex DESC, ligand/receptor ASC)
# ---------------------------------------------------------------------------
def test_build_selected_block_complex_is_descending():
    assert build_selected_block([-8.0, -2.0, 4.0], "complex") == [4.0, -2.0, -8.0]


def test_build_selected_block_ligand_is_ascending():
    assert build_selected_block([4.0, -8.0, -2.0], "ligand") == [-8.0, -2.0, 4.0]


def test_build_selected_block_receptor_is_ascending():
    assert build_selected_block([4.0, -8.0, -2.0], "receptor") == [-8.0, -2.0, 4.0]


# ---------------------------------------------------------------------------
# Band-slice adapter: drop the single endstate-adjacent pair, align with block
# ---------------------------------------------------------------------------
# 6-state matrix mirroring the CHARGE/GB-COLLAPSED pilot complex_order (item 9: charges=[1.0]):
# [no_int, interactions, electrostatics@max_restraint(1.0), remove_restraints(-2),
#  remove_restraints(-8), endstate]; restraint band starts at halo_restraint_matrix=2. The real cb7
# seed has 3 charge windows (halo=4); that case is pinned separately in
# test_real_cyclesteps_complex_band_alignment below.
_COMPLEX_SD = [0.50, 0.50, 0.01, 0.06, 0.90]  # pairs: (0,1)(1,2)(2,3)(3,4)(4,5)


def test_restraint_band_superdiagonal_complex_drops_endstate_pair():
    matrix = _tridiag(_COMPLEX_SD)
    # halo_restraint_matrix = 2 (collapsed cb7); band [2:] = [0.01, 0.06, 0.90]; drop the last
    # (min_restraint -> endstate) pair -> [0.01, 0.06], aligning with block pairs (4,-2),(-2,-8).
    band = restraint_band_superdiagonal(matrix, "complex", band_start=2, band_end=None)
    assert band == [0.01, 0.06]


def test_restraint_band_superdiagonal_ligand_drops_endstate_pair():
    # ligand_order [endstate, apply(-8), apply(-2), apply(4)] -> apo_end_restraint_matrix=3;
    # band [0:3] = [0.50, 0.50, 0.01]; drop the FIRST (endstate -> min_restraint) pair -> [0.50, 0.01].
    matrix = _tridiag(_COMPLEX_SD)
    band = restraint_band_superdiagonal(matrix, "ligand", band_start=0, band_end=3)
    assert band == [0.50, 0.01]


def test_restraint_band_superdiagonal_aligns_with_block_length():
    matrix = _tridiag(_COMPLEX_SD)
    block = build_selected_block([-8.0, -2.0, 4.0], "complex")
    band = restraint_band_superdiagonal(matrix, "complex", band_start=2, band_end=None)
    assert len(band) == len(block) - 1


# ---------------------------------------------------------------------------
# Candidate pool: finer than seed, strictly inside (min, max), excludes seed pts
# ---------------------------------------------------------------------------
def test_build_candidate_pool_uniform_fill_excludes_seed_and_anchors():
    # seed (-8,-2,4), step 2 -> interior {-6,-4,-2,0,2}; drop -2 (in seed) -> [-6,-4,0,2]
    assert build_candidate_pool([-8.0, -2.0, 4.0], step=2.0) == [-6.0, -4.0, 0.0, 2.0]


def test_build_candidate_pool_explicit_clamped_and_deduped():
    pool = build_candidate_pool([-8.0, -2.0, 4.0], explicit_pool=[-6.0, -5.0, -2.0, 3.0, 10.0])
    # -2 in seed, 10 at/above max -> excluded
    assert pool == [-6.0, -5.0, 3.0]


def test_build_candidate_pool_default_step_one():
    pool = build_candidate_pool([-8.0, -2.0, 4.0])
    assert pool == [-7.0, -6.0, -5.0, -4.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# plan_restraint_insertion: end-to-end overlap-matrix -> (con, orient) decision
# ---------------------------------------------------------------------------
def test_plan_restraint_insertion_complex_targets_worst_gap():
    matrix = _tridiag(_COMPLEX_SD)  # complex restraint band -> [0.01, 0.06]
    pool = build_candidate_pool([-8.0, -2.0, 4.0], step=1.0)
    new_con, new_orient, converged, reason = plan_restraint_insertion(
        overlap_matrix=matrix,
        conformational_exps=[-8.0, -2.0, 4.0],
        orientational_exps=[-4.0, 2.0, 8.0],
        system_type="complex",
        band_start=2,
        band_end=None,
        threshold=THRESH,
        pool=pool,
    )
    assert converged is False
    # weak gap is block pair (4, -2); log2 midpoint 1.0 -> nearest free pool candidate 1.0;
    # orient = con + OFFSET(=4) = 5.0
    assert new_con == 1.0
    assert new_orient == 5.0


def test_plan_restraint_insertion_converges_when_band_adequate():
    matrix = _tridiag([0.50, 0.50, 0.20, 0.30, 0.90])  # restraint band [0.20, 0.30] both >= 0.04
    new_con, new_orient, converged, reason = plan_restraint_insertion(
        overlap_matrix=matrix,
        conformational_exps=[-8.0, -2.0, 4.0],
        orientational_exps=[-4.0, 2.0, 8.0],
        system_type="complex",
        band_start=2,
        band_end=None,
        threshold=THRESH,
        pool=build_candidate_pool([-8.0, -2.0, 4.0]),
    )
    assert converged is True and reason == "converged"
    assert new_con is None and new_orient is None


# ---------------------------------------------------------------------------
# Real CycleSteps: pin the production band indices for the cb7 seed (3 charge
# windows -> complex halo_restraint_matrix=4) and prove the band aligns 1:1 with
# the block. This is the index the production wiring actually passes
# (cycle_steps.halo_restraint_matrix), which the collapsed _COMPLEX_SD fixture
# does NOT exercise. It also documents that the band's first pair is a genuine
# MAX->2nd restraint step (constant full charge), not a charge-decoupling step.
# ---------------------------------------------------------------------------
def test_real_cyclesteps_complex_band_alignment():
    from implicit_solvent_ddm.matrix_order import CycleSteps

    cs = CycleSteps([-8.0, -2.0, 4.0], [-4.0, 2.0, 8.0], [0.0, 0.5, 1.0], [])
    cs.round(3)
    # Real cb7 complex_order is 8 states with THREE electrostatics windows.
    assert len(cs.complex_order) == 8
    assert cs.halo_restraint_matrix == 4
    # The state at the band start carries the MAX restraint at full charge (the max anchor), and the
    # next state is the first released restraint -> the band's first pair is a pure restraint step.
    band_start_state = cs.complex_order[cs.halo_restraint_matrix]
    next_state = cs.complex_order[cs.halo_restraint_matrix + 1]
    assert band_start_state == ("electrostatics", "78.5", "1.0", "4.0_8.0")
    assert next_state[0] == "lambda_window" and band_start_state[2] == next_state[2]  # same charge

    # 8-state overlap: weak only at the (max -> -2) restraint pair (full index 4), ok elsewhere.
    full_sd = [0.50, 0.50, 0.50, 0.50, 0.01, 0.06, 0.90]
    matrix = _tridiag(full_sd)
    band = restraint_band_superdiagonal(
        matrix, "complex", band_start=cs.halo_restraint_matrix, band_end=None
    )
    block = build_selected_block([-8.0, -2.0, 4.0], "complex")  # [4, -2, -8]
    assert len(band) == len(block) - 1
    assert band == [0.01, 0.06]  # (max->-2) weak, (-2->-8) ok; (-8->endstate) dropped

    new_con, new_orient, converged, _ = plan_restraint_insertion(
        overlap_matrix=matrix,
        conformational_exps=[-8.0, -2.0, 4.0],
        orientational_exps=[-4.0, 2.0, 8.0],
        system_type="complex",
        band_start=cs.halo_restraint_matrix,
        band_end=None,
        threshold=THRESH,
        pool=build_candidate_pool([-8.0, -2.0, 4.0], step=1.0),
    )
    assert converged is False
    # weak gap is block pair (4, -2) -> midpoint 1.0; orient = 1.0 + OFFSET(4) = 5.0
    assert new_con == 1.0 and new_orient == 5.0


def test_real_cyclesteps_ligand_band_indices():
    from implicit_solvent_ddm.matrix_order import CycleSteps

    cs = CycleSteps([-8.0, -2.0, 4.0], [-4.0, 2.0, 8.0], [0.0, 0.5, 1.0], [])
    cs.round(3)
    # ligand restraint band is [0:apo_end_restraint_matrix]; endstate is index 0, restraints follow.
    assert cs.apo_end_restraint_matrix == 3
    assert cs.ligand_order[0][0] == "endstate"
    full_sd = [0.50, 0.01, 0.06, 0.30, 0.30]  # pair0 endstate->min dropped; band -> [0.01, 0.06]
    matrix = _tridiag(full_sd)
    band = restraint_band_superdiagonal(
        matrix, "ligand", band_start=0, band_end=cs.apo_end_restraint_matrix
    )
    assert band == [0.01, 0.06]


def test_plan_restraint_insertion_ligand_direction():
    # ligand band [0:3] -> [0.50, 0.01] after dropping endstate pair; block ASC [-8,-2,4];
    # weak gap is block pair (-2, 4) -> midpoint 1.0
    matrix = _tridiag(_COMPLEX_SD)
    new_con, new_orient, converged, _ = plan_restraint_insertion(
        overlap_matrix=matrix,
        conformational_exps=[-8.0, -2.0, 4.0],
        orientational_exps=[-4.0, 2.0, 8.0],
        system_type="ligand",
        band_start=0,
        band_end=3,
        threshold=THRESH,
        pool=build_candidate_pool([-8.0, -2.0, 4.0], step=1.0),
    )
    assert converged is False
    assert new_con == 1.0 and new_orient == 5.0


# ---------------------------------------------------------------------------
# banded=True — the ALS pilot computes MBAR over ONLY the restraint band
# (compute_mbar(restraint_band=True)), so the overlap matrix is already exactly
# the restraint windows + max anchor: there is NO endstate/LJ/charge state to
# slice off. band_start=0, band_end=None, and no element is dropped.
# ---------------------------------------------------------------------------
def test_restraint_band_superdiagonal_banded_keeps_all_pairs():
    # 3-state banded overlap: [max anchor, rst-2, rst-8]; superdiagonal = 2 restraint pairs.
    matrix = _tridiag([0.01, 0.20])  # (anchor<->rst-2)=0.01 weak, (rst-2<->rst-8)=0.20 ok
    band = restraint_band_superdiagonal(
        matrix, "complex", band_start=0, band_end=None, banded=True
    )
    assert band == [0.01, 0.20]  # both pairs kept (no drop)
    # the full-cycle rule (banded=False) would wrongly drop the last restraint pair here:
    assert restraint_band_superdiagonal(matrix, "complex", 0, None) == [0.01]


def test_plan_restraint_insertion_banded_targets_worst_pair():
    # banded restraint overlap [anchor=4, rst-2, rst-8] with the (4 <-> -2) pair weak.
    matrix = _tridiag([0.01, 0.20])
    new_con, new_orient, converged, _ = plan_restraint_insertion(
        overlap_matrix=matrix,
        conformational_exps=[-8.0, -2.0, 4.0],
        orientational_exps=[-4.0, 2.0, 8.0],
        system_type="complex",
        band_start=0,
        band_end=None,
        threshold=THRESH,
        pool=build_candidate_pool([-8.0, -2.0, 4.0], step=1.0),
        banded=True,
    )
    assert converged is False
    assert new_con == 1.0 and new_orient == 5.0  # midpoint of the weak (4,-2) gap; orient = con + 4


def test_plan_restraint_insertion_banded_converges_when_all_ok():
    matrix = _tridiag([0.20, 0.30])  # both restraint pairs >= 0.04
    new_con, new_orient, converged, reason = plan_restraint_insertion(
        overlap_matrix=matrix,
        conformational_exps=[-8.0, -2.0, 4.0],
        orientational_exps=[-4.0, 2.0, 8.0],
        system_type="complex",
        band_start=0,
        band_end=None,
        threshold=THRESH,
        pool=build_candidate_pool([-8.0, -2.0, 4.0]),
        banded=True,
    )
    assert converged is True and reason == "converged"
    assert new_con is None and new_orient is None


# ===========================================================================
# GB-dielectric (lambda) and charge bands — generic 1-D R-ADD axis
# ===========================================================================
def test_lambda_eps_round_trip():
    for eps in [1.0, 1.2, 1.6, 2.5, 5.0, 12.0, 40.0, 78.5]:
        assert eps_from_lambda(lambda_from_eps(eps)) == pytest.approx(eps)
    # gas sentinel: eps == 0 -> lambda 0 (no reaction field), and lambda 0 -> eps 1 (vacuum).
    assert lambda_from_eps(0.0) == 0.0
    assert eps_from_lambda(0.0) == pytest.approx(1.0)
    # water is near the top of the lambda range.
    assert lambda_from_eps(78.5) == pytest.approx(1.0 - 1.0 / 78.5)


def test_lambda_charge_of_state():
    # state tuple = (label, extdiel, charge, restraint)
    assert lambda_of_state(("interactions", "0.0", "0.0", "x")) == 0.0
    assert lambda_of_state(("gb_dielectric", "2.5", "0.0", "x")) == pytest.approx(0.6)
    assert lambda_of_state(("electrostatics", "78.5", "0.0", "x")) == pytest.approx(1 - 1 / 78.5)
    assert charge_of_state(("electrostatics", "78.5", "0.5", "x")) == 0.5


def test_plan_band_insertion_lambda_targets_worst_gap():
    # dielectric band coords (lambda) gas(0) .. water(0.987) with the middle pair weak.
    coords = [0.0, 0.3, 0.6, 0.987]
    matrix = _tridiag([0.50, 0.002, 0.50])  # (0.3<->0.6) is the worst pair
    pool = build_candidate_pool([0.0, 0.987], step=0.05)
    new_lam, converged, reason = plan_band_insertion(matrix, coords, THRESH, pool)
    assert converged is False
    assert 0.3 < new_lam < 0.6  # inserted strictly inside the worst gap


def test_plan_band_insertion_charge_axis():
    # charge band q=0 .. q=1 with one weak pair -> insert strictly inside [0, 1].
    coords = [0.0, 1.0]
    matrix = _tridiag([0.001])
    pool = build_candidate_pool([0.0, 1.0], step=0.1)
    new_q, converged, _ = plan_band_insertion(matrix, coords, THRESH, pool)
    assert converged is False
    assert 0.0 < new_q < 1.0


def test_plan_band_insertion_never_breaches_band_anchors():
    # Even with the endpoint pairs weak, an insertion can never land on or past either pinned anchor.
    coords = [0.0, 0.5, 0.987]
    matrix = _tridiag([0.001, 0.001])
    pool = build_candidate_pool([0.0, 0.987], step=0.1)
    new_lam, converged, _ = plan_band_insertion(matrix, coords, THRESH, pool)
    assert converged is False
    assert 0.0 < new_lam < 0.987


def test_plan_band_insertion_converges_when_all_ok():
    coords = [0.0, 0.5, 0.987]
    matrix = _tridiag([0.20, 0.30])
    pool = build_candidate_pool([0.0, 0.987], step=0.1)
    new_lam, converged, reason = plan_band_insertion(matrix, coords, THRESH, pool)
    assert converged is True and reason == "converged" and new_lam is None


def test_plan_band_insertion_pool_exhausted():
    # Weak gap but no candidate fits (empty pool) -> terminate with the pool-exhausted reason.
    new_lam, converged, reason = plan_band_insertion(_tridiag([0.001]), [0.0, 0.1], THRESH, [])
    assert converged is True and reason == "pool-exhausted" and new_lam is None


# ---------------------------------------------------------------------------
# Real CycleSteps: pin the dielectric/charge band indices + coordinates for the
# cb7 seed with a 6-window epsilon band. Mirrors the restraint-band index tests.
# ---------------------------------------------------------------------------
def test_real_cyclesteps_dielectric_charge_bands():
    from implicit_solvent_ddm.matrix_order import CycleSteps

    eps_seed = sorted({1.2, 1.6, 2.5, 5.0, 12.0, 40.0})
    cs = CycleSteps([-8.0, -2.0, 4.0], [-4.0, 2.0, 8.0], [0.0, 1.0], eps_seed)
    cs.round(3)
    g = len(eps_seed)

    # complex dielectric band = interactions(gas) -> G GB windows -> electrostatics q=0 (water anchor).
    cdiel = cs.complex_order[cs.start_gb_extdiel_matrix : cs.start_complex_charge_matrix + 1]
    assert len(cdiel) == g + 2
    assert cdiel[0][0] == "interactions"
    assert cdiel[-1] == ("electrostatics", "78.5", "0.0", "4.0_8.0")
    lam = [lambda_of_state(s) for s in cdiel]
    assert lam == sorted(lam)  # ascending gas -> water
    assert lam[0] == 0.0 and lam[-1] == pytest.approx(1 - 1 / 78.5)

    # complex charge band = electrostatics q=0 -> q=1 (constant water + max restraint).
    cchg = cs.complex_order[cs.start_complex_charge_matrix : cs.halo_restraint_matrix + 1]
    assert [charge_of_state(s) for s in cchg] == [0.0, 1.0]

    # receptor dielectric band = max-restraint water anchor -> G GB windows -> no_gb (gas).
    start = cs.start_receptor_gb_matrix
    rdiel = cs.receptor_order[start - 1 : start + g + 1]
    assert len(rdiel) == g + 2
    assert rdiel[0][0] == "lambda_window"  # the max restraint window
    assert rdiel[-1][0] == "no_gb"
    rlam = [lambda_of_state(s) for s in rdiel]
    assert rlam == sorted(rlam, reverse=True)  # descending water -> gas
    assert all(charge_of_state(s) == 1.0 for s in rdiel)  # full host charge throughout


def test_empty_dielectric_leaves_orders_static():
    # The static (restraints-only) path must be unchanged when no epsilon seed is given.
    from implicit_solvent_ddm.matrix_order import CycleSteps

    cs = CycleSteps([-8.0, -2.0, 4.0], [-4.0, 2.0, 8.0], [0.0, 1.0], [])
    cs.round(3)
    assert cs.complex_GB_exl_windows == []
    assert cs.receptor_GB_exl_windows == []
    assert cs.receptor_order == cs.endstate + cs.apply_restraints + cs.no_gb


# ---------------------------------------------------------------------------
# Close-the-loop merge: apply_converged_schedule must carry the pilot's converged
# dielectric, charge, AND restraint schedules into the production config so the
# re-decomposition's RestraintMaker materializes a file per (inserted) window.
# ---------------------------------------------------------------------------
def test_apply_converged_schedule_carries_all_bands():
    import os
    import numpy as np
    import yaml
    from implicit_solvent_ddm.config import Config
    from implicit_solvent_ddm.workflow_phases import apply_converged_schedule

    cfg_path = os.path.join("implicit_solvent_ddm", "tests", "input_files", "config.yaml")
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    production = Config.from_config(raw)
    converged = Config.from_config(raw)
    orig_prod_gb = list(production.intermediate_args.gb_extdiel_windows)
    # The seed config has no gb_extdiel_windows, so Config.__post_init__ disables the production GB gate.
    # The pilot (below) discovers GB windows starting from the gas/water anchors; the merge must turn the
    # gate back on, else setup_intermediate_simulations skips them and compute_mbar's order desyncs.
    assert production.workflow.gb_extdiel_windows is False

    # Simulate a converged pilot: dielectric + charge windows inserted, plus an R-ADD restraint window
    # (con=1.0 / orient=5.0) strictly inside the seed (-8..4) — recorded in the PAIRED _list fields.
    converged.intermediate_args.gb_extdiel_windows = [1.5, 3.0, 10.0, 40.0]
    converged.intermediate_args.charges_lambda_window = [0.0, 0.25, 0.5, 1.0]
    converged.intermediate_args.exponent_conformational_forces_list = [-8.0, -2.0, 4.0, 1.0]
    converged.intermediate_args.exponent_orientational_forces_list = [-4.0, 2.0, 8.0, 5.0]

    merged = apply_converged_schedule(production, converged)
    m = merged.intermediate_args

    # dielectric + charge window VALUES carried verbatim (production loops iterate these)
    assert m.gb_extdiel_windows == [1.5, 3.0, 10.0, 40.0]
    assert m.charges_lambda_window == [0.0, 0.25, 0.5, 1.0]
    # restraint exponents carried, index-paired
    assert m.exponent_conformational_forces == [-8.0, -2.0, 4.0, 1.0]
    assert m.exponent_orientational_forces == [-4.0, 2.0, 8.0, 5.0]
    # forces recomputed as 2**exponent (what RestraintMaker / setup iterate), index-aligned
    assert list(m.conformational_restraints_forces) == pytest.approx(list(np.exp2([-8.0, -2.0, 4.0, 1.0])))
    assert list(m.orientational_restraint_forces) == pytest.approx(list(np.exp2([-4.0, 2.0, 8.0, 5.0])))
    # anchor protection: the inserted con=1.0 is interior, so max force is unchanged
    assert max(m.exponent_conformational_forces) == 4.0
    # GB gate re-enabled: the pilot added GB windows to an empty-seed config, so the production setup
    # loop must now run (its boolean gate keys off this, while compute_mbar keys off the list).
    assert merged.workflow.gb_extdiel_windows is True
    # deep copy: the production config is not mutated (gate + windows unchanged on the original)
    assert merged is not production
    assert list(production.intermediate_args.gb_extdiel_windows) == orig_prod_gb
    assert production.workflow.gb_extdiel_windows is False


def test_apply_converged_schedule_disables_gb_gate_when_no_windows():
    """Symmetric guard: if the pilot converges to ZERO GB windows (no desolvation cliff), the merged
    config must DISABLE the GB gate so the production setup and compute_mbar order stay consistent
    (no gb_dielectric columns expected, none produced)."""
    import os
    import yaml
    from implicit_solvent_ddm.config import Config
    from implicit_solvent_ddm.workflow_phases import apply_converged_schedule

    cfg_path = os.path.join("implicit_solvent_ddm", "tests", "input_files", "config.yaml")
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    production = Config.from_config(raw)
    # Seed a production config WITH a GB band (gate True), then converge to an empty schedule.
    production.intermediate_args.gb_extdiel_windows = [2.0, 10.0]
    production.workflow.gb_extdiel_windows = True
    converged = Config.from_config(raw)
    converged.intermediate_args.gb_extdiel_windows = []

    merged = apply_converged_schedule(production, converged)
    assert merged.intermediate_args.gb_extdiel_windows == []
    assert merged.workflow.gb_extdiel_windows is False


# ---------------------------------------------------------------------------
# Batch insertion: bisect EVERY weak gap per round (parallel-friendly), vs. one-at-a-time insert_one.
# ---------------------------------------------------------------------------
def test_plan_batch_insertion_fills_every_weak_gap_in_one_call():
    # complex-style descending block; all three adjacent pairs weak; pool has each gap's midpoint.
    block = [4.0, 3.0, 2.0, 1.0]
    superdiagonal = [0.0, 0.0, 0.0]
    pool = [3.5, 2.5, 1.5]
    new_exps, converged, reason = plan_batch_insertion(
        block, superdiagonal, pool, THRESH, lower_bound=1.0, upper_bound=4.0
    )
    assert converged is False and reason == ""
    assert new_exps == [1.5, 2.5, 3.5]  # one midpoint per weak gap, all at once, sorted


def test_plan_batch_insertion_only_fills_the_weak_gaps():
    block = [4.0, 3.0, 2.0]
    superdiagonal = [0.0, 0.20]  # only the first pair is weak
    pool = [3.5, 2.5]
    new_exps, converged, reason = plan_batch_insertion(
        block, superdiagonal, pool, THRESH, lower_bound=2.0, upper_bound=4.0
    )
    assert new_exps == [3.5] and converged is False


def test_plan_batch_insertion_converged():
    new_exps, converged, reason = plan_batch_insertion(
        [4.0, 3.0], [0.20], [3.5], THRESH, lower_bound=3.0, upper_bound=4.0
    )
    assert new_exps == [] and converged is True and reason == "converged"


def test_plan_batch_insertion_pool_exhausted_when_no_candidate():
    new_exps, converged, reason = plan_batch_insertion(
        [4.0, 3.0], [0.0], pool=[], threshold=THRESH, lower_bound=3.0, upper_bound=4.0
    )
    assert new_exps == [] and converged is True and reason == "pool-exhausted"


def test_plan_batch_insertion_respects_anchor_bounds():
    # midpoint candidate exists in the pool but lies outside the (lower,upper) anchor interval -> skipped.
    new_exps, converged, reason = plan_batch_insertion(
        [4.0, 2.0], [0.0], pool=[3.5], threshold=THRESH, lower_bound=2.0, upper_bound=3.0
    )
    assert new_exps == [] and reason == "pool-exhausted"


def test_plan_batch_insertion_length_mismatch_raises():
    with pytest.raises(ValueError):
        plan_batch_insertion([4.0, 3.0, 2.0], [0.0], [2.5], THRESH, lower_bound=2.0, upper_bound=4.0)


# ---------------------------------------------------------------------------
# Prune: minimal connected subset from the FULL pairwise overlap matrix (greedy farthest jump).
# ---------------------------------------------------------------------------
def _decay_matrix(n, by_distance):
    """Symmetric overlap matrix: O[i][j] = by_distance[|i-j|] (diagonal 1.0, missing distances 0.0)."""
    return [
        [1.0 if i == j else by_distance.get(abs(i - j), 0.0) for j in range(n)]
        for i in range(n)
    ]


def test_prune_schedule_drops_redundant_windows():
    # adjacent 0.2, skip-1 0.1, farther ~0. At threshold 0.08, every-other window is reachable.
    M = _decay_matrix(5, {1: 0.2, 2: 0.1, 3: 0.03, 4: 0.01})
    assert prune_schedule(M, threshold=0.08) == [0, 2, 4]


def test_prune_schedule_keeps_anchors_and_stays_connected():
    M = _decay_matrix(5, {1: 0.2, 2: 0.1, 3: 0.03, 4: 0.01})
    kept = prune_schedule(M, threshold=0.08)
    assert kept[0] == 0 and kept[-1] == 4                      # anchors always retained
    # every consecutive kept pair clears the threshold (valid BAR ladder)
    for a, b in zip(kept, kept[1:]):
        assert min(M[a][b], M[b][a]) >= 0.08


def test_prune_schedule_higher_threshold_keeps_more():
    M = _decay_matrix(5, {1: 0.2, 2: 0.1, 3: 0.03, 4: 0.01})
    # at 0.15 even skip-1 (0.1) fails, so nothing can be dropped
    assert prune_schedule(M, threshold=0.15) == [0, 1, 2, 3, 4]


def test_prune_schedule_collapses_to_anchors_when_endpoints_overlap():
    M = _decay_matrix(5, {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2})
    assert prune_schedule(M, threshold=0.1) == [0, 4]


def test_prune_schedule_no_pruning_when_only_neighbors_connected():
    M = _decay_matrix(5, {1: 0.2, 2: 0.01, 3: 0.0, 4: 0.0})
    assert prune_schedule(M, threshold=0.08) == [0, 1, 2, 3, 4]


def test_prune_schedule_uses_min_direction():
    # 0->2 looks great one way (0.2) but is ~0 the other way; min-direction must NOT skip window 1.
    M = _decay_matrix(5, {1: 0.2, 2: 0.0, 3: 0.0, 4: 0.0})
    M[0][2] = 0.2  # asymmetric: forward strong, reverse (M[2][0]) stays 0.0
    assert prune_schedule(M, threshold=0.08) == [0, 1, 2, 3, 4]


def test_prune_schedule_small_n_is_identity():
    assert prune_schedule([[1.0]], threshold=0.08) == [0]
    assert prune_schedule([[1.0, 0.0], [0.0, 1.0]], threshold=0.08) == [0, 1]
