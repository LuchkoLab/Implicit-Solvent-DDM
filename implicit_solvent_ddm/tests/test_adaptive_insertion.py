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
