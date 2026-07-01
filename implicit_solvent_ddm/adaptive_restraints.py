"""
A collection of functions that performs simple iterative proceess to improve space phase overlap between adjecent states. 
"""

import copy
import math
import time
from typing import Optional

import numpy as np
import pandas as pd

import implicit_solvent_ddm.pandasmbar as pdmbar
from implicit_solvent_ddm.alchemical import alter_topology
from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.matrix_order import CycleSteps
from implicit_solvent_ddm.restraints import write_restraint_forces
from implicit_solvent_ddm.runner import IntermidateRunner
from implicit_solvent_ddm.mdin import generate_extdiel_mdin

AVAGADRO = 6.0221367e23
BOLTZMAN = 1.380658e-23
JOULES_PER_KCAL = 4184


# ---------------------------------------------------------------------------
# Pure R-ADD insertion helpers
# ---------------------------------------------------------------------------
# Deterministic "insert exactly one window per iteration" (R-ADD) scheduling decision used by the
# ALS pilot. These are PURE functions (no Toil job, no MD, no pymbar) so the scheduling logic can be
# unit-tested in isolation (tests/test_adaptive_insertion.py). They are CALLED synchronously by the
# Toil job functions in this module (adaptive_lambda_windows / improve_restraints_overlap), which own
# the orchestration (MBAR job -> MD/post sub-runner jobs -> recurse via addFollowOnJobFn). The math
# belongs at the tail of the MBAR job where the overlap matrix is already in memory; making it its
# own Toil job would only serialize that matrix across a promise to do arithmetic.
#
# Conventions: a leg's schedule is an ordered list of conformational-restraint *exponents* (`block`)
# with `superdiagonal[k]` the overlap of the adjacent pair (block[k], block[k+1]); orientational
# exponents are paired as ``orient = con + OFFSET``. New windows are drawn from a fixed candidate
# ``pool`` and may only land in the open interval ``(lower_bound, upper_bound)`` =
# ``(endstate_exp, pinned_max_exp)`` so the pinned-max Boresch anchor is never moved or exceeded.

ROUND_DP = 3  # restraint exponents are compared/retained at this precision (matches workflow_phases)


def derive_offset(conformational_exps, orientational_exps):
    """Return the constant ``orient - con`` exponent offset, asserting it is constant.

    The orientational exponent tracks the conformational one (``orient = con + OFFSET``), so the seed
    pairing must have a single well-defined offset. Raises ``ValueError`` otherwise rather than
    silently scheduling an inconsistent ``(con, orient)`` pair.
    """
    if len(conformational_exps) != len(orientational_exps):
        raise ValueError(
            f"con/orient length mismatch: {len(conformational_exps)} vs "
            f"{len(orientational_exps)}"
        )
    offsets = {
        round(o - c, ROUND_DP)
        for c, o in zip(conformational_exps, orientational_exps)
    }
    if len(offsets) != 1:
        raise ValueError(f"non-constant orient-con offset: {sorted(offsets)}")
    return offsets.pop()


def min_direction_superdiagonal(overlap_matrix):
    """Return ``[min(O[i][i+1], O[i+1][i]) for i in range(N-1)]``.

    Using the *minimum* of the two directions (not the symmetric average) means a one-sided weak
    transition (e.g. fwd=0.01, rev=0.07, whose average 0.04 would pass) still registers as weak.
    ``overlap_matrix`` may be a numpy array or a list of lists — only ``[i][j]`` indexing is used.
    No rounding is applied, so the threshold comparison downstream is not quantized.
    """
    n = len(overlap_matrix)
    return [
        min(float(overlap_matrix[i][i + 1]), float(overlap_matrix[i + 1][i]))
        for i in range(n - 1)
    ]


def find_bad_sections(superdiagonal, threshold):
    """Return contiguous runs (each a list of indices) where ``superdiagonal[k] < threshold``."""
    sections = []
    current = []
    for k, value in enumerate(superdiagonal):
        if value < threshold:
            current.append(k)
        elif current:
            sections.append(current)
            current = []
    if current:
        sections.append(current)
    return sections


def select_section(sections, superdiagonal):
    """Pick the winning bad section by a deterministic total order:

    1. longest section (most consecutive weak transitions),
    2. then the section whose *minimum* superdiagonal value is smallest (the paper's tie-break),
    3. then the lowest start index (final deterministic tie-break).
    """
    return min(
        sections,
        key=lambda section: (
            -len(section),
            min(superdiagonal[k] for k in section),
            section[0],
        ),
    )


def worst_gap_index(section, superdiagonal):
    """Return the index ``k`` within ``section`` with the smallest overlap (ties -> lowest ``k``)."""
    return min(section, key=lambda k: (superdiagonal[k], k))


def snap_to_pool(lo_exp, hi_exp, ideal_exp, pool, selected):
    """Return the nearest unused pool candidate strictly inside ``(lo_exp, hi_exp)``.

    ``selected`` is the set of already-selected exponents (3-dp rounded). Distance is measured to
    ``ideal_exp`` (the log2-space midpoint); ties break toward the lower exponent. Returns ``None``
    when no free candidate lies strictly inside the gap.
    """
    free_inside = [
        p for p in pool
        if lo_exp < p < hi_exp and round(p, ROUND_DP) not in selected
    ]
    if not free_inside:
        return None
    return min(free_inside, key=lambda p: (abs(p - ideal_exp), p))


def insert_one(block, superdiagonal, pool, threshold, lower_bound, upper_bound):
    """Perform a single R-ADD step for one leg.

    Parameters
    ----------
    block : list of float
        Conformational exponents of the currently-selected restraint windows, ordered so that
        ``superdiagonal[k]`` is the overlap of the pair ``(block[k], block[k+1])``.
    superdiagonal : list of float
        Min-direction adjacent overlaps; must satisfy ``len == len(block) - 1``.
    pool : list of float
        Sorted candidate conformational exponents (finer than the seed).
    threshold : float
        Insert where the superdiagonal overlap is below this (e.g. 0.04).
    lower_bound, upper_bound : float
        Exclusive interval ``(endstate exponent, pinned-max exponent)``. A new window must satisfy
        ``lower_bound < new < upper_bound`` so the pinned-max Boresch anchor is never touched.

    Returns
    -------
    (new_con, converged, reason) : (float | None, bool, str)
        ``new_con`` is the single exponent to insert this iteration, or ``None`` when no insertion is
        made. ``converged`` is True when the loop should stop. ``reason`` is one of ``"converged"``
        (all adjacent overlaps adequate) or ``"pool-exhausted"`` (weak gaps remain but no free
        candidate can fill them — emit the best schedule with a warning). The pool-exhausted exit is
        mandatory: without it a coarse pool that cannot reach ``threshold`` would recurse forever,
        growing the Toil graph unbounded.
    """
    if len(superdiagonal) != len(block) - 1:
        raise ValueError(
            f"superdiagonal length {len(superdiagonal)} != len(block)-1 {len(block) - 1}"
        )
    selected = {round(c, ROUND_DP) for c in block}
    sd = list(superdiagonal)  # local copy; unfillable pairs get masked to +inf
    masked_any = False
    while True:
        sections = find_bad_sections(sd, threshold)
        if not sections:
            # No weak pair remains. If we only got here by masking unfillable pairs, the pool was too
            # coarse to satisfy the threshold -> report exhaustion rather than clean convergence.
            return (None, True, "pool-exhausted" if masked_any else "converged")

        section = select_section(sections, sd)
        k = worst_gap_index(section, sd)
        lo_exp, hi_exp = sorted((block[k], block[k + 1]))
        ideal_exp = (lo_exp + hi_exp) / 2.0
        new_con = snap_to_pool(lo_exp, hi_exp, ideal_exp, pool, selected)

        if (
            new_con is None                                  # no free pool candidate in this gap
            or not (lower_bound < new_con < upper_bound)     # anchor protection (never reach max/endstate)
            or round(new_con, ROUND_DP) in selected          # no-progress guard (defensive)
        ):
            sd[k] = math.inf  # this weak pair is unfillable; reconsider the next-worst pair
            masked_any = True
            continue

        return (new_con, False, "")


def plan_batch_insertion(block, superdiagonal, pool, threshold, lower_bound, upper_bound):
    """Batch R-ADD: return the window to insert for EVERY weak adjacent pair this round (not just one).

    The batch analogue of :func:`insert_one`. Instead of inserting a single window in the globally-worst
    gap and re-piloting, this returns one new exponent per *fillable* weak pair so the caller can run them
    all in parallel — one pilot round per bisection *generation* instead of one round per window (the
    decisive win when a leg needs many windows, e.g. flexible-protein restraints). Each new point is the
    gap midpoint snapped to the candidate ``pool`` (identical placement rule to ``insert_one``), confined
    to the open anchor interval ``(lower_bound, upper_bound)``. A weak pair already at pool spacing (no
    free candidate inside) is skipped — it gets bisected on a later round once the pool refines, or is the
    min-gap floor.

    Returns ``(new_exps, converged, reason)``:
    * ``new_exps`` — sorted exponents to insert this round (empty when no insertion is made).
    * ``converged`` — True when this should be the final round (no fillable weak pair remains).
    * ``reason`` — ``"converged"`` (no weak pair at all) | ``"pool-exhausted"`` (weak pairs remain but
      none fillable — emit best + warn) | ``""`` (more rounds to do).
    """
    if len(superdiagonal) != len(block) - 1:
        raise ValueError(
            f"superdiagonal length {len(superdiagonal)} != len(block)-1 {len(block) - 1}"
        )
    selected = {round(c, ROUND_DP) for c in block}
    chosen = set()
    new_exps = []
    weak = [k for k, value in enumerate(superdiagonal) if value < threshold]
    if not weak:
        return ([], True, "converged")
    for k in weak:
        lo_exp, hi_exp = sorted((block[k], block[k + 1]))
        ideal_exp = (lo_exp + hi_exp) / 2.0
        new_con = snap_to_pool(lo_exp, hi_exp, ideal_exp, pool, selected | chosen)
        if (
            new_con is None                                  # no free pool candidate in this gap
            or not (lower_bound < new_con < upper_bound)     # anchor protection
            or round(new_con, ROUND_DP) in (selected | chosen)
        ):
            continue
        chosen.add(round(new_con, ROUND_DP))
        new_exps.append(round(new_con, ROUND_DP))
    if not new_exps:
        # weak pairs remain but none fillable (all already at pool spacing) -> floor reached
        return ([], True, "pool-exhausted")
    return (sorted(new_exps), False, "")


def prune_schedule(overlap_matrix, threshold):
    """Select the MINIMAL subset of an ordered, fully-connected window ladder whose CONSECUTIVE members
    still clear ``threshold`` — from the FULL pairwise overlap matrix (greedy farthest-reachable jump).

    Precondition: call this only after the dense pilot has converged so every *adjacent* pair already
    passes (``min_direction_superdiagonal(overlap_matrix)`` all >= the adjacency threshold). The
    off-diagonal entries then reveal redundancy: if a far window is still reachable (overlap >= threshold)
    from the current one, the windows in between can be dropped. Greedy "jump as far as still-connected"
    gives the minimum window count for a ladder whose overlap is monotone in index-distance (the restraint
    case); for a non-monotone leg it still returns a valid connected subset (BFS/min-nodes is the robust
    optimum, a future refinement). The two end windows (indices 0 and N-1, the pinned anchors) are always
    kept.

    Pass a MARGINED ``threshold`` here (e.g. ~2x the production 0.04): the pilot (~50 ps) over-estimates
    overlap vs. production (~10 ns), so prune conservatively, then verify at production length. The edge
    test is the min-direction overlap ``min(O[i][j], O[j][i])`` (consistent with the insertion engine).

    Returns the sorted list of kept indices (always includes 0 and N-1).
    """
    n = len(overlap_matrix)
    if n <= 2:
        return list(range(n))

    def edge(i, j):
        return min(float(overlap_matrix[i][j]), float(overlap_matrix[j][i]))

    kept = [0]
    i = 0
    while i < n - 1:
        # Farthest j > i still connected to i; fall back to i+1 (never drop a window we can't bridge —
        # guaranteed reachable by the adjacency precondition, so the walk always advances and terminates).
        farthest = i + 1
        for j in range(n - 1, i, -1):
            if edge(i, j) >= threshold:
                farthest = j
                break
        kept.append(farthest)
        i = farthest
    return kept


def prune_restraint_exponents(
    overlap_matrix, conformational_exps, orientational_exps, system_type, threshold
):
    """Prune a converged dense restraint ladder to its minimal connected subset, returning the kept
    ``(conformational, orientational)`` exponents (both sorted ascending).

    Uses the FULL pairwise band overlap matrix the pilot already computed (``compute_mbar(band=
    "restraint")`` -> ``compute_overlap()``). Its rows/cols are in the same order as
    ``build_selected_block(conformational_exps, system_type)`` — the alignment the insertion engine
    already relies on (pinned in ``tests/test_real_cyclesteps_complex_band_alignment``) — so
    ``prune_schedule``'s kept indices map straight onto the block exponents. The two anchors (pinned max
    + weakest, ``block[0]``/``block[-1]``) are always retained; orientational exponents track
    conformational by the constant seed offset. Pass a MARGINED ``threshold`` (e.g. ``min_degree_overlap
    * prune_margin``) since the short pilot over-estimates overlap vs. production. Raises ``ValueError``
    on a block/matrix size mismatch (fail loud rather than silently corrupt the schedule).
    """
    block = build_selected_block(conformational_exps, system_type)
    if len(block) != len(overlap_matrix):
        raise ValueError(
            f"prune: block length {len(block)} != overlap-matrix dim {len(overlap_matrix)} "
            f"({system_type}) — refusing to prune on a misaligned matrix"
        )
    offset = derive_offset(conformational_exps, orientational_exps)
    kept_idx = prune_schedule(overlap_matrix, threshold)
    kept_con = sorted(round(float(block[i]), ROUND_DP) for i in kept_idx)
    kept_orient = [round(c + offset, ROUND_DP) for c in kept_con]
    return kept_con, kept_orient


# ---------------------------------------------------------------------------
# Band-slice adapter: pymbar overlap matrix -> R-ADD decision
# ---------------------------------------------------------------------------
# These pure functions translate a leg's full pymbar overlap matrix into the (block, superdiagonal,
# bounds, pool) that ``insert_one`` consumes. They encode the MBAR state ordering from
# ``matrix_order.CycleSteps`` (see tests/test_adaptive_insertion.py, which pins the indices against a
# REAL CycleSteps built from the cb7 seed con=[-8,-2,4]/orient=[-4,2,8]):
#
#   complex_order = [no_interactions, interactions, GB.., electrostatics(charges ASC)..,
#                    remove_restraints(DESC, max DROPPED via [1:]), endstate].
#     The restraint band starts at ``halo_restraint_matrix`` = the index of the LAST pre-restraint
#     state, which is the electrostatics(charge=1.0) window CARRYING THE MAX RESTRAINT (every
#     no_interactions/interactions/GB/electrostatics state sits at max_con_max_orient). So the band's
#     FIRST pair (electrostatics@maxrst -> remove_restraints[0]) is a genuine MAX->2nd restraint step
#     at constant full charge+solvent — NOT a charge-decoupling step. The dropped-max in
#     remove_restraints is therefore still represented (as that electrostatics anchor), so the
#     (max -> 2nd) restraint overlap IS visible. The band's LAST pair is (min_restraint -> endstate).
#     NB: ``halo_restraint_matrix`` is independent of the charge-window count (it always lands on the
#     max-restraint anchor), so the alignment holds for 1 collapsed charge (pilot, item 9; halo=2 for
#     cb7) or the full 3 charges (halo=4 for cb7) alike.
#   ligand/receptor = [endstate, apply_restraints(ASC, max INCLUDED), charges/no_gb]
#                    -> restraint band is [0:apo_end_restraint_matrix]; the band's FIRST adjacent
#                    pair is (endstate -> min_restraint).
#
# PILOT MODEL (Step 5a, FULL-SHORT-PILOT — supersedes the earlier charge/GB-collapse idea): the pilot
# runs the FULL intermediate cycle but at SHORT MD length (pilot_nstlim), NOT a restraints-only collapse.
# Rationale: compute_mbar's _ordered() validates the *full* complex_order (anchors + charges + restraints
# + endstate), and halo_restraint_matrix is charge-count-independent (it always lands on the last charge
# window = the max-restraint anchor), so the band-slice is already correct for full charges. Running the
# whole cycle short satisfies _ordered() with ZERO config surgery — every column is produced. The old
# "collapse charges to [1.0]" approach is also impossible via Config (config.py force-injects {0.0,1.0}).
# See workflow_phases.adaptive_restraint_pilot (Phase 4.5) and merry-floating-kahn plan.
#
# v1 scope: the pinned max and the weakest seed window are FIXED boundaries; insertions land strictly
# between them, so the single endstate-adjacent band pair is dropped (the weakest restraint force is
# ~2^-8 ~ 0, i.e. near-unrestrained, so its overlap with the true endstate is physically near-perfect
# and not a productive insertion target). That dropped pair is instead surfaced by the advisory
# production-overlap check (Phase 6), not auto-filled here.


def build_selected_block(conformational_exps, system_type):
    """Order the restraint conformational exponents to match the leg's MBAR overlap band.

    ``insert_one`` requires ``block[k], block[k+1]`` to be adjacent restraint states. The MBAR matrix
    lists complex restraint states in DESCENDING force (``remove_restraints``: max -> min) and
    ligand/receptor states in ASCENDING force (``apply_restraints``: min -> max). Returns ONLY the
    restraint-window exponents in that direction (the endstate is not a restraint window and is
    handled as the fixed lower anchor).
    """
    ordered = sorted(float(e) for e in conformational_exps)
    if system_type == "complex":
        ordered.reverse()
    return ordered


def restraint_band_superdiagonal(
    overlap_matrix, system_type, band_start, band_end, banded=False
):
    """Min-direction adjacent overlaps for the restraint-window pairs ONLY.

    Slices the full min-direction superdiagonal (``min_direction_superdiagonal``) to the leg's
    restraint band so the result aligns 1:1 with ``build_selected_block`` (``len == len(block) - 1``).

    ``banded=False`` (full-cycle overlap): the band still contains the single endstate-adjacent pair,
    which is dropped:
    * ``complex``           band ``[halo_restraint_matrix:]`` ends with ``(min_restraint -> endstate)``
                            -> drop the LAST element.
    * ``ligand``/``receptor`` band ``[0:apo_end_restraint_matrix]`` starts with
                            ``(endstate -> min_restraint)`` -> drop the FIRST element.

    ``banded=True`` (overlap already computed over ONLY the restraint band — the ALS pilot path, where
    ``compute_mbar(restraint_band=True)`` excludes the endstate, LJ, and lower charges): there is no
    endstate-adjacent pair, so the band IS exactly the restraint↔restraint pairs — return it unchanged
    (dropping an element would discard a real restraint pair). Here ``band_start=0, band_end=None``.

    ``band_start``/``band_end`` come from ``CycleSteps`` (full-cycle complex: ``halo_restraint_matrix``,
    ``None``; ligand/receptor: ``0``, ``apo_end_restraint_matrix``).
    """
    full = min_direction_superdiagonal(overlap_matrix)
    band = full[band_start:] if band_end is None else full[band_start:band_end]
    if not band:
        return band
    if banded:
        return band
    if system_type == "complex":
        return band[:-1]
    return band[1:]


def build_candidate_pool(conformational_exps, explicit_pool=None, step=None):
    """Return the sorted fixed candidate exponent pool, strictly inside the seed ``(min, max)``.

    R-ADD draws each new window from a pool finer than the seed. If ``explicit_pool`` is provided it is
    used verbatim (rounded, de-duplicated, clamped to the open interval). Otherwise a uniform fill at
    ``step`` (default 1.0 exponent) between the seed min and max is generated. The pinned max and the
    weakest seed window are fixed boundaries, so candidates equal to or outside them, or already in the
    seed, are excluded.
    """
    seed = sorted(round(float(e), ROUND_DP) for e in conformational_exps)
    if len(seed) < 2:
        return []
    lo, hi = seed[0], seed[-1]
    seed_set = set(seed)
    if explicit_pool:
        candidates = [round(float(p), ROUND_DP) for p in explicit_pool]
    else:
        s = step if step else 1.0
        # number of whole steps spanning (lo, hi); interior points only (1 .. n-1).
        n = int(round((hi - lo) / s))
        candidates = [round(lo + i * s, ROUND_DP) for i in range(1, n)]
    return sorted({c for c in candidates if lo < c < hi and c not in seed_set})


def plan_restraint_insertion(
    overlap_matrix,
    conformational_exps,
    orientational_exps,
    system_type,
    band_start,
    band_end,
    threshold,
    pool,
    banded=False,
):
    """Decide the single R-ADD restraint window to insert this iteration (pure).

    Ties together the band-slice adapter and ``insert_one`` so the whole "overlap matrix ->
    (con, orient) to insert OR converged" decision is unit-testable without Toil/MBAR. Returns
    ``(new_con, new_orient, converged, reason)`` where ``new_con``/``new_orient`` are ``None`` when no
    insertion is made (``converged is True``). ``new_orient = new_con + OFFSET`` with the offset
    derived (and asserted constant) from the seed pairing. ``banded`` is forwarded to the band slice
    (True when ``overlap_matrix`` was computed over only the restraint band — the ALS pilot).
    """
    offset = derive_offset(conformational_exps, orientational_exps)
    block = build_selected_block(conformational_exps, system_type)
    superdiagonal = restraint_band_superdiagonal(
        overlap_matrix, system_type, band_start, band_end, banded
    )
    new_con, converged, reason = insert_one(
        block,
        superdiagonal,
        pool,
        threshold,
        lower_bound=min(block),
        upper_bound=max(block),
    )
    new_orient = round(new_con + offset, ROUND_DP) if new_con is not None else None
    return new_con, new_orient, converged, reason


# ---------------------------------------------------------------------------
# GB-dielectric / charge bands: generic 1-D R-ADD axes
# ---------------------------------------------------------------------------
# The dielectric band schedules in lambda = (1 - 1/eps): GB polar solvation is exactly linear in lambda
# (intdiel=1), so uniform lambda spacing ~ uniform energy change ~ uniform overlap. We schedule/insert in
# lambda and map back to epsilon for the mdin. The gas anchor (igb=6, no reaction field) is lambda=0; full
# water (eps=78.5) is lambda = 1 - 1/78.5 ~ 0.987. The charge band schedules directly in the charge
# fraction q in [0, 1]. Both reuse the SAME generic engine (insert_one) as restraints; only the coordinate
# axis and the (no orientational) pairing differ — so plan_band_insertion is a sibling of
# plan_restraint_insertion, not a wrapper (the restraint path keeps its build_selected_block/orient logic).


def lambda_from_eps(eps):
    """Map external dielectric ``eps`` -> GB solvation coordinate ``lambda = 1 - 1/eps``.

    ``eps == 0`` is the AMBER gas sentinel (igb=6, no reaction field) and maps to ``lambda = 0``.
    """
    eps = float(eps)
    if eps == 0.0:
        return 0.0
    return 1.0 - 1.0 / eps


def eps_from_lambda(lam):
    """Inverse of :func:`lambda_from_eps`: ``eps = 1 / (1 - lambda)`` (``lambda=0 -> eps=1``, vacuum)."""
    lam = float(lam)
    return 1.0 / (1.0 - lam)


def lambda_of_state(state):
    """Dielectric-band coordinate (lambda) of a CycleSteps state tuple ``(label, extdiel, charge, rst)``."""
    return lambda_from_eps(float(state[1]))


def charge_of_state(state):
    """Charge-band coordinate of a CycleSteps state tuple ``(label, extdiel, charge, rst)``."""
    return float(state[2])


def plan_band_insertion(overlap_matrix, coords, threshold, pool):
    """Decide the single R-ADD window to insert on a generic 1-D axis (dielectric lambda or charge).

    ``coords`` are the band states' axis values IN MATRIX ORDER, so ``coords[k], coords[k+1]`` are the
    adjacent pair whose overlap is ``min_direction_superdiagonal(overlap_matrix)[k]``. The overlap matrix
    here is ALWAYS the banded sub-grid (``compute_mbar(band=...)``), so the superdiagonal aligns 1:1 with
    the adjacent ``coords`` pairs and there is no endstate-adjacent pair to drop. The two band endpoints
    (``min``/``max`` coord) are the pinned anchors; insertions land strictly between them.

    Returns ``(new_coord, converged, reason)`` with ``new_coord`` ``None`` when no insertion is made.
    """
    block = [float(c) for c in coords]
    superdiagonal = min_direction_superdiagonal(overlap_matrix)
    return insert_one(
        block,
        superdiagonal,
        pool,
        threshold,
        lower_bound=min(block),
        upper_bound=max(block),
    )


def compute_mbar(
    simulation_data: list[pd.DataFrame],
    temperature: float,
    matrix_order: Optional[CycleSteps],
    system: str,
    memory="2G",
    cores=1,
    disk="3G",
    restraint_band: bool = False,
    band: Optional[str] = None,
    log=print,
):
    """Execute MBAR analysis.

    Arrange and structure DataFrames to perform MBAR analysis.

    Parameters
    ----------
    simulation_data: list[pd.DataFrame]
        A completed mdout output that contains system information including timestep energies, and temperature.
    temperature: float
        Specified thermostat temperature used in MD simulations.
    matrix_order: CycleSteps
        Arranges the MBAR matrix in chronological order depending on the system.
    system: str
        Denoting the chronogical order of the matrix (i.e. complex, receptor or ligand).

    Returns
    -------
    pdmbar.mbar(df_subsampled): tuple[DataFrame, DataFrame, MBAR]
        DataFrames for the free energies differences (Deltaf_ij), error estimates in free energy difference (dDeltaf_ij), and the pyMBAR object.
    df_mbar: pd.DataFrame
        An formated and chronological arrange DataFrame before any MBAR analysis was performed. (Which can be used to create pdfs of MBAR matrix).
    """

    def create_mbar_format():
        df = pd.concat(simulation_data, axis=0, ignore_index=True)

        # df = df["solute"].iloc[0]
        df = df.set_index(
            [
                "solute",
                "parm_state",
                "extdiel",
                "charge",
                "parm_restraints",
                "traj_state",
                "traj_extdiel",
                "traj_charge",
                "traj_restraints",
                "Frames",
            ],
            drop=True,
        )
        df = df[["ENERGY"]]
        df = df.unstack(["parm_state", "extdiel", "charge", "parm_restraints"])  # type: ignore
        df = df.reset_index(["Frames", "solute"], drop=True)
        states = [_ for _ in zip(*df.columns)][1]
        extdiels = [_ for _ in zip(*df.columns)][2]
        charges = [_ for _ in zip(*df.columns)][3]
        restraints = [_ for _ in zip(*df.columns)][4]

        column_names = [
            (state, extdiel, charge, restraint)
            for state, extdiel, charge, restraint in zip(
                states, extdiels, charges, restraints
            )
        ]

        df.columns = column_names  # type: ignore

        # divide by Kcal per Kt
        # kcals_per_Kt = ((BOLTZMAN * (AVAGADRO)) / JOULES_PER_KCAL) * temperature
        print(f"Created Unique MBAR dataframe {df.index.unique()}\n")

        return df / kcals_per_Kt

    kcals_per_Kt = ((BOLTZMAN * (AVAGADRO)) / JOULES_PER_KCAL) * temperature

    df_mbar = create_mbar_format()

    print(f"df.index.unique :\n {df_mbar.index.unique()}\n")
    print(f"df.index.values :\n {df_mbar.index.values}\n")
    print(f"df.columns.unique :\n {df_mbar.columns.unique()}\n")
    print(f"df.columns.values :\n {df_mbar.columns.values}\n")
    def _ordered(order):
        # Invariant (ALS R1 safety): every CycleSteps state must exist as an MBAR-dataframe column.
        # A mismatch means the schedule (built from the canonical exponent_*_forces_list) and the
        # post_output trajectories have drifted apart — fail LOUD with the diff instead of the bare
        # pandas KeyError the reindex would otherwise raise. Thin pilot data that fails to produce a
        # window's parquet would also surface here.
        missing = [column for column in order if column not in df_mbar.columns]
        if missing:
            raise ValueError(
                f"compute_mbar[{system}]: {len(missing)} CycleSteps state(s) are absent from the "
                f"MBAR dataframe columns — schedule/data mismatch.\n"
                f"  missing (CycleSteps order, not in df): {missing}\n"
                f"  present df columns: {list(df_mbar.columns)}"
            )
        return df_mbar[order]

    def _band_states(order, system, band_name):
        # Select the contiguous interpolatable sub-band of ``order`` that the ALS scheduler reads. Each
        # band runs anchor -> windows -> anchor at constant everything-else, so its adjacent overlaps are
        # not contaminated by the other coordinates. Indices come from CycleSteps.
        #
        #   restraint  : restraint windows + max-restraint anchor, EXCLUDING the endstate. complex starts
        #                at halo_restraint_matrix (the max-restraint = last full-charge electrostatics,
        #                since remove_restraints drops the max lambda_window); ligand/receptor is
        #                apply_restraints (max included), endstate (index 0) excluded.
        #   dielectric : complex = gas `interactions` anchor -> GB windows -> water-q0 `electrostatics`
        #                anchor [start_gb_extdiel_matrix : start_complex_charge_matrix + 1]; receptor =
        #                max-restraint water anchor -> GB windows -> gas `no_gb` anchor.
        #   charge     : complex = electrostatics q=0 -> q=1 [start_complex_charge_matrix :
        #                halo_restraint_matrix + 1]; ligand = ligand_charges.
        if band_name == "restraint":
            if system == "complex":
                states = order[matrix_order.halo_restraint_matrix:]
            else:
                states = order[1 : matrix_order.apo_end_restraint_matrix + 1]
            return [state for state in states if state[0] != "endstate"]
        if band_name == "dielectric":
            if system == "complex":
                return order[
                    matrix_order.start_gb_extdiel_matrix
                    : matrix_order.start_complex_charge_matrix + 1
                ]
            if system == "receptor":
                start = matrix_order.start_receptor_gb_matrix
                g = len(matrix_order.external_dielectic)
                return order[start - 1 : start + g + 1]
            raise ValueError(f"dielectric band undefined for system '{system}'")
        if band_name == "charge":
            if system == "complex":
                return order[
                    matrix_order.start_complex_charge_matrix
                    : matrix_order.halo_restraint_matrix + 1
                ]
            if system == "ligand":
                return order[matrix_order.start_ligand_charge_matrix:]
            raise ValueError(f"charge band undefined for system '{system}'")
        raise ValueError(f"unknown band '{band_name}'")

    def _ordered_band(order, system, band_name):
        # MBAR over ONLY the band sub-grid (band trajectories x band states) of the full N x N the pilot
        # ran. The ROW filter is required: keeping non-band trajectory frames while dropping their columns
        # would break pymbar's sum(N_k)==n_samples invariant.
        states = _band_states(order, system, band_name)
        banded = _ordered(states)  # band columns (validated present by the invariant above)
        return banded.loc[banded.index.isin(states)]  # band trajectory rows only

    # Back-compat: restraint_band=True is the legacy spelling of band="restraint".
    if band is None and restraint_band:
        band = "restraint"

    # flat bottom to no flat bottom -> EXP()
    if matrix_order is None:
        pass

    elif system == "complex":
        order = matrix_order.complex_order
        df_mbar = _ordered_band(order, "complex", band) if band else _ordered(order)

    elif system == "ligand":
        order = matrix_order.ligand_order
        df_mbar = _ordered_band(order, "ligand", band) if band else _ordered(order)

    else:
        order = matrix_order.receptor_order
        df_mbar = _ordered_band(order, "receptor", band) if band else _ordered(order)

    equil_info = pdmbar.detect_equilibration(df_mbar)

    df_subsampled = pdmbar.subsample_correlated_data(df_mbar, equil_info=equil_info)

    log("performing MBAR")
    # MBAR timing record (parsed by timing_report.py). A band-sliced solve is a pilot pass
    # (-> "adaptive"); a full-cycle solve (band is None) is the production Phase-7 MBAR (-> "mbar").
    # `log` defaults to print() but every Toil call site passes job.log so the record reaches the
    # LEADER log -- the file timing_report.py parses. A raw print() goes only to the worker's stdout,
    # which Toil does not surface for successful jobs, so the mbar phase was invisible to the report
    # even on a cold run.
    _mbar_t0 = time.perf_counter()
    _mbar_result = pdmbar.mbar(df_subsampled)
    log(
        f"[TIMING] phase={'adaptive' if band else 'mbar'} "
        f"wall_s={time.perf_counter() - _mbar_t0:.2f} cores=1 gpu=0 end={time.time():.0f} "
        f"| mbar system={system} band={band}"
    )
    return _mbar_result, df_mbar

    # return pdmbar.mbar(df_subsampled), df_mbar

def run_compute_mbar(
    job,
    system_runner: IntermidateRunner,
    config: Config,
    system_type: str,
    memory="2G",
    cores=1,
    disk="3G",
    accelerators=None,
):
    """
    Run the compute_mbar function.
    """
    job.log(f"Running MBAR for {system_type}")
    cycle_steps = CycleSteps(
        conformation_forces=config.intermediate_args.exponent_conformational_forces_list,
        orientational_forces=config.intermediate_args.exponent_orientational_forces_list,
        charges_windows=config.intermediate_args.charges_lambda_window,
        external_dielectic=config.intermediate_args.gb_extdiel_windows,
    )
    cycle_steps.round(3)
    
    return compute_mbar(
        simulation_data=system_runner.post_output,
        temperature=config.intermediate_args.temperature,
        matrix_order=cycle_steps,
        system=system_type,
        log=job.log,
    )

def adaptive_lambda_windows(
    job,
    system_runner: IntermidateRunner,
    config: Config,
    system_type: str,
    restraints_scaling: bool = False,
    charge_scaling: bool = False,
    gb_scaling: bool = False,
    max_iterations: Optional[int] = None,
):
    """
    Simple iterative process to improve poor space phase overlap between restraints and/or ligand charge windows.

    Parameters
    ----------
    job: Toil.job
        The atomic unit of work in a Toil workflow is a Job.
    system_runner: IntermidateRunner
        An system specific runner object to create and inital any new MD runs needed.
    config: Config
        User specified configuration file containing necesssary input information.
    system_type: str
        System type to denote the specific system_runner (i.e. complex, receptor or ligand)
    restraints_scaling: bool
        Whether to perfom adaptive process for restraint windows.
    charge_scaling: bool
        Whether to perform adaptive process for ligand charge scaling.
    gb_scaling: bool
        Whether to perform adaptive procces for scaling GB external dielectric.

    Returns
    -------
    results: tuple[DataFrame, DataFrame, MBAR], pd.DataFrame
        Return values from compute_mbar(*args) function.
    updated_config: Config
        An updated config object with newly added restraint or ligand charge windows.
    """

    updated_config = copy.deepcopy(config)
    # Sort all thermodynamic cycle steps in chronological order
    job.log(f"THE SYSTEM PASSED {system_type}")
    job.log(
        f"conformational exponets: {updated_config.intermediate_args.exponent_conformational_forces_list}"
    )
    job.log(
        f"orientational exponets: {updated_config.intermediate_args.exponent_orientational_forces_list}"
    )
    cycle_steps = CycleSteps(
        conformation_forces=updated_config.intermediate_args.exponent_conformational_forces_list,
        orientational_forces=updated_config.intermediate_args.exponent_orientational_forces_list,
        charges_windows=updated_config.intermediate_args.charges_lambda_window,
        external_dielectic=updated_config.intermediate_args.gb_extdiel_windows,
    )
    # round all restraint forces values to 3 sig. figs for readable dataframes.
    cycle_steps.round(3)
    job.log(f"THE SYSTEM PASSED {system_type}")
    job.log(f"complex ordered steps: {cycle_steps.complex_order}")
    # Select the interpolatable band this pass schedules within, then compute MBAR over ONLY that band
    # (anchor(s) + windows). The other coordinates' states are still RUN by the pilot but are excluded
    # here so they do not contaminate the conditioning of the overlaps the scheduler reads. The overlap
    # matrix returned IS the band sub-grid, so its min-direction superdiagonal aligns 1:1 with the band's
    # ordered states.
    if restraints_scaling:
        band = "restraint"
    elif charge_scaling:
        band = "charge"
    elif gb_scaling:
        band = "dielectric"
    else:
        raise ValueError(
            "adaptive_lambda_windows requires exactly one of restraints_scaling / charge_scaling / "
            "gb_scaling to be True"
        )

    results = compute_mbar(
        simulation_data=system_runner.post_output,
        temperature=updated_config.intermediate_args.temperature,
        matrix_order=cycle_steps,
        system=system_type,
        band=band,
        log=job.log,
    )
    job.log(f"SYSTEM TYPE {system_type}; ALS band={band}")
    overlap_matrix = results[0][-1].compute_overlap()["matrix"]

    # ------------------------------------------------------------------
    # Restraint windows: deterministic R-ADD (insert exactly one window in the worst gap; never move
    # or drop an already-run window). This replaces the legacy bisect-every-bad-pair loop and reads
    # the canonical exponent_*_forces_list (R1), so the schedule the adapter sees is the schedule
    # CycleSteps/compute_mbar use.
    # ------------------------------------------------------------------
    if restraints_scaling:
        if max_iterations is None:
            max_iterations = updated_config.intermediate_args.max_adaptive_iterations

        con_list = updated_config.intermediate_args.exponent_conformational_forces_list
        orient_list = updated_config.intermediate_args.exponent_orientational_forces_list

        # The overlap matrix is ALREADY the restraint band (compute_mbar(band="restraint") above),
        # so the band is the whole matrix: start at 0, no end slice, and no endstate-adjacent pair to
        # drop (banded=True). The min-direction superdiagonal then aligns 1:1 with build_selected_block.
        band_start, band_end = 0, None

        pool = build_candidate_pool(
            con_list,
            explicit_pool=updated_config.intermediate_args.candidate_conformational_pool,
            step=updated_config.intermediate_args.candidate_pool_step,
        )

        # Observability (Step 5a): log the restraint-band overlap that drives the R-ADD decision, so a
        # convergence/insertion can be read directly from the run log instead of inferred. Logs each
        # adjacent restraint pair's min-direction overlap and flags those below the threshold. A
        # degenerate band (thin pilot data -> ~1.0 or NaN everywhere) is visible here.
        threshold = updated_config.intermediate_args.min_degree_overlap
        block = build_selected_block(con_list, system_type)
        band_sd = restraint_band_superdiagonal(
            overlap_matrix, system_type, band_start, band_end, banded=True
        )
        job.log(
            f"[ALS][{system_type}] restraint band ({len(block)} windows, {len(band_sd)} adjacent "
            f"pairs), min_degree_overlap threshold={threshold}; conformational exps (band order)="
            f"{block}"
        )
        for k in range(len(band_sd)):
            flag = "WEAK -> insert" if band_sd[k] < threshold else "ok"
            job.log(
                f"[ALS][{system_type}]   pair (con {block[k]} <-> {block[k + 1]}): "
                f"overlap={band_sd[k]:.4f}  [{flag}]"
            )

        # BATCH mode (flag-gated): insert a window in EVERY weak gap this round and run them in parallel,
        # converging a many-window leg in ~log rounds instead of one round per window. Default (flag off)
        # falls through to the one-at-a-time R-ADD below — byte-identical to today.
        if updated_config.intermediate_args.batch_insertion:
            new_cons, converged, reason = plan_batch_insertion(
                block, band_sd, pool, threshold,
                lower_bound=min(block), upper_bound=max(block),
            )
            if converged or max_iterations <= 0:
                if converged and reason == "pool-exhausted":
                    job.log(
                        f"[ALS][{system_type}] WARNING: candidate pool exhausted with weak gaps "
                        f"remaining — emitting best schedule {sorted(con_list)}"
                    )
                elif converged:
                    job.log(
                        f"[ALS][{system_type}] restraint schedule converged ({len(con_list)} "
                        f"windows): {sorted(con_list)}"
                    )
                else:
                    job.log(
                        f"[ALS][{system_type}] round cap reached; stopping with {len(con_list)} "
                        f"windows: {sorted(con_list)}"
                    )
                return (results, updated_config, system_runner)
            offset = derive_offset(con_list, orient_list)
            pairs = [(c, round(c + offset, ROUND_DP)) for c in new_cons]
            job.log(
                f"[ALS][{system_type}] BATCH inserting {len(pairs)} restraint windows "
                f"con={[p[0] for p in pairs]} ({max_iterations - 1} rounds left)"
            )
            improve_job = job.addChildJobFn(
                improve_restraints_overlap_batch,
                system_runner,
                pairs,
                updated_config,
                system_type,
            )
            return improve_job.addFollowOnJobFn(
                adaptive_lambda_windows,
                improve_job.rv(0),
                improve_job.rv(1),
                system_type=system_type,
                restraints_scaling=True,
                max_iterations=max_iterations - 1,
            ).rv()

        new_con, new_orient, converged, reason = plan_restraint_insertion(
            overlap_matrix=overlap_matrix,
            conformational_exps=con_list,
            orientational_exps=orient_list,
            system_type=system_type,
            band_start=band_start,
            band_end=band_end,
            threshold=updated_config.intermediate_args.min_degree_overlap,
            pool=pool,
            banded=True,
        )

        if converged or max_iterations <= 0:
            if converged and reason == "pool-exhausted":
                job.log(
                    f"[ALS][{system_type}] WARNING: candidate pool exhausted with weak gaps "
                    f"remaining — emitting best schedule {sorted(con_list)}"
                )
            elif converged:
                job.log(
                    f"[ALS][{system_type}] restraint schedule converged ({len(con_list)} windows): "
                    f"{sorted(con_list)}"
                )
            else:
                job.log(
                    f"[ALS][{system_type}] iteration cap reached; stopping with "
                    f"{len(con_list)} windows: {sorted(con_list)}"
                )
            return (
                results,
                updated_config,
                system_runner,
            )

        job.log(
            f"[ALS][{system_type}] inserting restraint window con={new_con} orient={new_orient} "
            f"({max_iterations - 1} iterations left)"
        )
        improve_job = job.addChildJobFn(
            improve_restraints_overlap,
            system_runner,
            new_con,
            new_orient,
            updated_config,
            system_type,
        )
        return improve_job.addFollowOnJobFn(
            adaptive_lambda_windows,
            improve_job.rv(0),
            improve_job.rv(1),
            system_type=system_type,
            restraints_scaling=True,
            max_iterations=max_iterations - 1,
        ).rv()

    # ------------------------------------------------------------------
    # GB-dielectric / charge bands: the SAME deterministic R-ADD engine as restraints, on a generic 1-D
    # axis (dielectric lambda = 1 - 1/eps, or charge fraction q). The band overlap matrix above is already
    # the band sub-grid, so its min-direction superdiagonal aligns 1:1 with the ordered band coordinates
    # (results[1].columns). The two band endpoints (gas/water for dielectric; q=0/q=1 for charge) are the
    # pinned anchors; insertions land strictly between them (insert_one's lower/upper bound = min/max).
    # ------------------------------------------------------------------
    if max_iterations is None:
        max_iterations = updated_config.intermediate_args.max_adaptive_iterations

    threshold = updated_config.intermediate_args.min_degree_overlap
    band_states = list(results[1].columns)  # ordered band state tuples (aligned with overlap_matrix)

    if gb_scaling:
        axis = "dielectric(lambda)"
        coords = [lambda_of_state(s) for s in band_states]
        pool = build_candidate_pool(
            coords,
            explicit_pool=updated_config.intermediate_args.candidate_dielectric_pool,
            step=updated_config.intermediate_args.dielectric_pool_step or 0.02,
        )
    else:  # charge_scaling
        axis = "charge(q)"
        coords = [charge_of_state(s) for s in band_states]
        pool = build_candidate_pool(
            coords,
            explicit_pool=updated_config.intermediate_args.candidate_charge_pool,
            step=updated_config.intermediate_args.charge_pool_step or 0.05,
        )

    band_sd = min_direction_superdiagonal(overlap_matrix)
    job.log(
        f"[ALS][{system_type}] {axis} band ({len(coords)} states, {len(band_sd)} adjacent pairs), "
        f"min_degree_overlap threshold={threshold}; coords (matrix order)="
        f"{[round(c, 4) for c in coords]}"
    )
    for k in range(len(band_sd)):
        flag = "WEAK -> insert" if band_sd[k] < threshold else "ok"
        job.log(
            f"[ALS][{system_type}]   pair ({round(coords[k], 4)} <-> {round(coords[k + 1], 4)}): "
            f"overlap={band_sd[k]:.4f}  [{flag}]"
        )

    new_coord, converged, reason = plan_band_insertion(overlap_matrix, coords, threshold, pool)

    if converged or max_iterations <= 0:
        if converged and reason == "pool-exhausted":
            job.log(
                f"[ALS][{system_type}] WARNING: {axis} candidate pool exhausted with weak gaps "
                f"remaining — emitting best schedule"
            )
        elif converged:
            job.log(f"[ALS][{system_type}] {axis} schedule converged ({len(coords)} states)")
        else:
            job.log(f"[ALS][{system_type}] iteration cap reached for {axis} band")
        return (results, updated_config, system_runner)

    if gb_scaling:
        new_eps = float(eps_from_lambda(new_coord))
        job.log(
            f"[ALS][{system_type}] inserting GB window eps={new_eps:.4f} (lambda={new_coord:.4f}) "
            f"({max_iterations - 1} iterations left)"
        )
        improve_job = job.addChildJobFn(
            improve_dielectric_overlap,
            system_runner,
            new_eps,
            updated_config,
            system_type,
        )
    else:
        new_charge = float(round(new_coord, ROUND_DP))
        job.log(
            f"[ALS][{system_type}] inserting charge window q={new_charge} "
            f"({max_iterations - 1} iterations left)"
        )
        improve_job = job.addChildJobFn(
            improve_charge_overlap,
            system_runner,
            new_charge,
            updated_config,
            system_type,
        )

    return improve_job.addFollowOnJobFn(
        adaptive_lambda_windows,
        improve_job.rv(0),
        improve_job.rv(1),
        system_type=system_type,
        charge_scaling=charge_scaling,
        gb_scaling=gb_scaling,
        max_iterations=max_iterations - 1,
    ).rv()


def improve_restraints_overlap(
    job,
    runner: IntermidateRunner,
    new_con: float,
    new_orient: float,
    config: Config,
    system_type: str,
):
    """Insert exactly ONE restraint window (R-ADD) and run it via a two-phase sub-runner.

    The single ``(new_con, new_orient)`` exponent pair to insert is chosen upstream by
    ``adaptive_lambda_windows`` (``plan_restraint_insertion`` -> ``insert_one``), so this wrapper is a
    thin Toil orchestrator: it writes one restraint file, appends one MD window, records the pair in
    the canonical ``exponent_*_forces_list`` schedule, and runs the window in two phases.

    Two-phase sub-runner (R4): the legacy tail scheduled ``new_runner(post_only=True)``, which SKIPS
    any window whose MD is not already on disk (``runner.run`` post-only branch) — so a freshly
    inserted window would get neither MD nor post-analysis. Instead we run, over ONLY the new
    window(s): (1) ``new_runner(post_only=False)`` to produce the short-pilot MD, then
    ``addFollowOn`` (2) ``new_runner(post_only=True)`` to re-score it. Both share ``post_output`` /
    ``_loaded_dataframe`` by reference with ``runner``, so the recursive ``adaptive_lambda_windows``
    MBAR (which reads ``system_runner.post_output``) sees the new window's dataframe.

    Parameters
    ----------
    new_con, new_orient : float
        Conformational / orientational restraint exponents (log2 of the force constant) to insert.
    config : Config
        The (deep-copied) config whose ``exponent_*_forces_list`` are extended in place and returned.

    Returns
    -------
    (post_runner_promise, config)
        ``post_runner_promise`` is the ``.rv()`` of the post-analysis sub-runner (consumed by the
        recursion as the next ``system_runner``); ``config`` carries the extended schedule.
    """
    new_con = float(round(new_con, ROUND_DP))
    new_orient = float(round(new_orient, ROUND_DP))
    new_con_force = float(np.exp2(new_con))
    new_orient_force = float(np.exp2(new_orient))

    job.log(
        f"[ALS][{system_type}] writing restraint window con_exp={new_con} "
        f"(force={new_con_force:.5g}) orient_exp={new_orient} (force={new_orient_force:.5g})"
    )

    # Short-pilot MD length (item 7). Fall back to the production mdin if the pilot mdin was not
    # emitted (e.g. flag-gated wiring not yet active), so this stays callable in isolation.
    pilot_mdin = config.inputs.get("pilot_mdin") or config.inputs["default_mdin"]

    restraints_job = job.addChildJobFn(initilized_jobs)
    before = len(runner.simulations)

    if system_type == "complex":
        runner._add_complex_simulation(
            conformational=new_con,
            orientational=new_orient,
            mdin=pilot_mdin,
            restraint_file=restraints_job.addChildJobFn(
                write_restraint_forces,
                conformational_template=runner.restraints.complex_conformational_restraints,
                orientational_template=runner.restraints.boresch.boresch_template,
                conformational_force=new_con_force,
                orientational_force=new_orient_force,
            ),
        )
    elif system_type == "ligand":
        runner._add_ligand_simulation(
            conformational=new_con,
            mdin=pilot_mdin,
            restraint_file=restraints_job.addChildJobFn(
                write_restraint_forces,
                conformational_template=runner.restraints.ligand_conformational_restraints,
                conformational_force=new_con_force,
            ),
        )
    else:
        runner._add_receptor_simulation(
            conformational=new_con,
            mdin=pilot_mdin,
            restraint_file=restraints_job.addChildJobFn(
                write_restraint_forces,
                conformational_template=runner.restraints.receptor_conformational_restraints,
                conformational_force=new_con_force,
            ),
        )

    # The MD pass runs ONLY the just-added window (the old windows' trajectories are already on disk);
    # runner.simulations keeps the full accumulated list for the post-analysis cross-evaluation below.
    new_sims = runner.simulations[before:]

    # Canonical schedule (R1): the adapter and CycleSteps read these lists, so record the pair here.
    config.intermediate_args.exponent_conformational_forces_list.append(new_con)
    config.intermediate_args.exponent_orientational_forces_list.append(new_orient)

    # Gate the MD pass on the restraint-file write (a child of restraints_job) completing.
    restraints_done = restraints_job.addFollowOnJobFn(initilized_jobs)
    md_runner = restraints_done.addChild(
        runner.new_runner(
            config, runner.__dict__, post_only=False, simulations=new_sims
        )
    )
    # The POST pass must run over the FULL window list (not just the new one): MBAR needs the complete
    # N×N grid, so the new trajectory has to be re-scored under EVERY state and EVERY existing
    # trajectory re-scored under the new state. only_post_analysis loops simulations×simulations, and
    # the shared _loaded_dataframe cache skips the already-computed cells, so this adds exactly the
    # ~2N missing cross terms. Passing only new_sims here yields a ragged matrix (one new×new cell) and
    # pymbar fails with "sum of all N_k must equal the total number of samples".
    post_runner = md_runner.addFollowOn(
        runner.new_runner(config, runner.__dict__, post_only=True)
    )

    return (
        post_runner.rv(),
        config,
    )


def improve_restraints_overlap_batch(
    job,
    runner: IntermidateRunner,
    con_orient_pairs: list,
    config: Config,
    system_type: str,
):
    """Insert a BATCH of restraint windows (one per weak gap) and run them via ONE two-phase sub-runner.

    The batch analogue of :func:`improve_restraints_overlap` (gated by
    ``intermediate_args.batch_insertion``). ``con_orient_pairs`` is the full set of ``(con, orient)``
    exponent pairs chosen this round by ``plan_batch_insertion`` (the midpoint of every weak adjacent
    pair). All new windows are added, then their MD runs in a SINGLE ``new_runner(post_only=False)``
    sub-runner so Toil schedules them across GPUs/cores in parallel — converging a many-window leg in
    ~log rounds instead of one round per window. A single post-analysis pass then re-scores the full
    N×N grid (same cross-evaluation contract as the one-at-a-time path). Returns
    ``(post_runner_promise, config)`` exactly like :func:`improve_restraints_overlap`.
    """
    pilot_mdin = config.inputs.get("pilot_mdin") or config.inputs["default_mdin"]
    restraints_job = job.addChildJobFn(initilized_jobs)
    before = len(runner.simulations)

    for new_con, new_orient in con_orient_pairs:
        new_con = float(round(new_con, ROUND_DP))
        new_orient = float(round(new_orient, ROUND_DP))
        new_con_force = float(np.exp2(new_con))
        new_orient_force = float(np.exp2(new_orient))
        job.log(
            f"[ALS][{system_type}] writing restraint window con_exp={new_con} "
            f"(force={new_con_force:.5g}) orient_exp={new_orient} (force={new_orient_force:.5g})"
        )
        if system_type == "complex":
            runner._add_complex_simulation(
                conformational=new_con,
                orientational=new_orient,
                mdin=pilot_mdin,
                restraint_file=restraints_job.addChildJobFn(
                    write_restraint_forces,
                    conformational_template=runner.restraints.complex_conformational_restraints,
                    orientational_template=runner.restraints.boresch.boresch_template,
                    conformational_force=new_con_force,
                    orientational_force=new_orient_force,
                ),
            )
        elif system_type == "ligand":
            runner._add_ligand_simulation(
                conformational=new_con,
                mdin=pilot_mdin,
                restraint_file=restraints_job.addChildJobFn(
                    write_restraint_forces,
                    conformational_template=runner.restraints.ligand_conformational_restraints,
                    conformational_force=new_con_force,
                ),
            )
        else:
            runner._add_receptor_simulation(
                conformational=new_con,
                mdin=pilot_mdin,
                restraint_file=restraints_job.addChildJobFn(
                    write_restraint_forces,
                    conformational_template=runner.restraints.receptor_conformational_restraints,
                    conformational_force=new_con_force,
                ),
            )
        # Canonical schedule (R1): the adapter and CycleSteps read these lists, so record each pair here.
        config.intermediate_args.exponent_conformational_forces_list.append(new_con)
        config.intermediate_args.exponent_orientational_forces_list.append(new_orient)

    # ALL newly-added windows this round; their MD runs together in one sub-runner (parallel via Toil).
    new_sims = runner.simulations[before:]

    restraints_done = restraints_job.addFollowOnJobFn(initilized_jobs)
    md_runner = restraints_done.addChild(
        runner.new_runner(config, runner.__dict__, post_only=False, simulations=new_sims)
    )
    # POST pass over the FULL window list (MBAR needs the complete N×N grid; the shared _loaded_dataframe
    # cache skips already-computed cells, so this adds only the new cross terms). Same contract as the
    # one-at-a-time path — passing only new_sims here would give pymbar a ragged matrix.
    post_runner = md_runner.addFollowOn(
        runner.new_runner(config, runner.__dict__, post_only=True)
    )

    return (
        post_runner.rv(),
        config,
    )


def improve_dielectric_overlap(
    job,
    runner: IntermidateRunner,
    new_eps: float,
    config: Config,
    system_type: str,
):
    """Insert exactly ONE GB-dielectric window (R-ADD) and run it via a two-phase sub-runner.

    Mirrors :func:`improve_restraints_overlap` but on the dielectric axis: the single ``new_eps`` to
    insert is chosen upstream by ``adaptive_lambda_windows`` (``plan_band_insertion`` -> ``insert_one``
    in lambda, mapped back to epsilon). It writes one short-pilot extdiel mdin (saltcon=0), appends one MD
    window at the MAX restraint, records the epsilon in the canonical ``gb_extdiel_windows`` schedule, and
    runs the window in two phases (MD then re-score) sharing ``post_output`` by reference with ``runner``.

    The complex band decharges the ligand (q=0 topology via ``alter_topology``) under dirstruct_halo; the
    receptor band reuses the full-charge apo topology (no ligand to decharge) under dirstruct_apo.
    """
    new_eps = float(new_eps)
    job.log(f"[ALS][{system_type}] writing GB-dielectric window extdiel={new_eps:.5g}")

    setup_job = job.addChildJobFn(initilized_jobs)
    before = len(runner.simulations)

    # Short-pilot extdiel mdin (saltcon=0 keeps the band linear in lambda). nstlim/ntwx fall back to the
    # production length (None) if the pilot inputs were not emitted, so this stays callable in isolation.
    extdiel_mdin = setup_job.addChildJobFn(
        generate_extdiel_mdin,
        user_mdin_ID=config.intermediate_args.mdin_intermediate_file,
        gb_extdiel=new_eps,
        nstlim=config.inputs.get("pilot_nstlim_steps"),
        ntwx=config.inputs.get("pilot_ntwx"),
    ).rv()

    max_con = max(config.intermediate_args.exponent_conformational_forces_list)

    if system_type == "complex":
        runner._add_complex_simulation(
            conformational=max_con,
            orientational=max(config.intermediate_args.exponent_orientational_forces_list),
            mdin=extdiel_mdin,
            restraint_file=runner.restraints.max_complex_restraint,
            charge=0.0,
            charge_parm=setup_job.addChildJobFn(
                alter_topology,
                solute_amber_parm=config.endstate_files.complex_parameter_filename,
                solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                ligand_mask=config.amber_masks.ligand_mask,
                receptor_mask=config.amber_masks.receptor_mask,
                set_charge=0.0,
            ),
            gb_extdiel=new_eps,
        )
    else:  # receptor: full host charge, default apo topology
        runner._add_receptor_simulation(
            conformational=max_con,
            mdin=extdiel_mdin,
            restraint_file=runner.restraints.max_receptor_conformational_restraint,
            gb_extdiel=new_eps,
        )

    new_sims = runner.simulations[before:]

    # Canonical schedule (R1): CycleSteps + compute_mbar read this list, so record the new epsilon here.
    config.intermediate_args.gb_extdiel_windows.append(new_eps)

    setup_done = setup_job.addFollowOnJobFn(initilized_jobs)
    md_runner = setup_done.addChild(
        runner.new_runner(config, runner.__dict__, post_only=False, simulations=new_sims)
    )
    post_runner = md_runner.addFollowOn(
        runner.new_runner(config, runner.__dict__, post_only=True)
    )
    return (post_runner.rv(), config)


# ---------------------------------------------------------------------------
# Matched dual-leg GB-dielectric scheduler (pilot)
# ---------------------------------------------------------------------------
# The dielectric schedule (config.gb_extdiel_windows) is SHARED by the complex and receptor legs so the
# host desolvation cancels between them. The pilot therefore schedules the dielectric band on BOTH legs in
# lockstep: every inserted epsilon window is added to BOTH runners (so each leg's MBAR grid stays
# consistent with the shared schedule), and the insertion decision uses the MIN of the two legs' adjacent
# overlaps (so the worse-overlapping leg governs — the receptor cliff is actually worse than the complex
# one). Charge + restraint bands stay single-leg (complex) via adaptive_lambda_windows.


def _dielectric_band_ascending(runner, config, system_type, log=print):
    """Return ``(coords, superdiagonal)`` for one leg's dielectric band, normalized to ASCENDING lambda.

    Both legs span the SAME lambda interval [0, ~0.987] with the SAME interior windows, so normalizing to
    ascending lambda makes the two legs' superdiagonals align index-for-index (the receptor band is stored
    water->gas, i.e. descending, so it is reversed here).
    """
    cycle_steps = CycleSteps(
        conformation_forces=config.intermediate_args.exponent_conformational_forces_list,
        orientational_forces=config.intermediate_args.exponent_orientational_forces_list,
        charges_windows=config.intermediate_args.charges_lambda_window,
        external_dielectic=config.intermediate_args.gb_extdiel_windows,
    )
    cycle_steps.round(3)
    results = compute_mbar(
        simulation_data=runner.post_output,
        temperature=config.intermediate_args.temperature,
        matrix_order=cycle_steps,
        system=system_type,
        band="dielectric",
        log=log,
    )
    overlap = results[0][-1].compute_overlap()["matrix"]
    coords = [lambda_of_state(s) for s in results[1].columns]
    sd = min_direction_superdiagonal(overlap)
    if coords and coords[0] > coords[-1]:  # descending (receptor) -> flip to ascending
        coords = list(reversed(coords))
        sd = list(reversed(sd))
    return coords, sd


def pilot_dielectric_scheduler(
    job,
    complex_runner: IntermidateRunner,
    receptor_runner: IntermidateRunner,
    config: Config,
    max_iterations: Optional[int] = None,
):
    """Drive the matched complex+receptor GB-dielectric R-ADD to convergence (pilot).

    Returns ``(complex_runner, receptor_runner, converged_config)``. ``converged_config`` carries the
    expanded shared ``gb_extdiel_windows`` consumed by the production rebuild (Stage D / close the loop).
    """
    updated_config = copy.deepcopy(config)
    if max_iterations is None:
        max_iterations = updated_config.intermediate_args.max_adaptive_iterations
    threshold = updated_config.intermediate_args.min_degree_overlap

    c_coords, c_sd = _dielectric_band_ascending(complex_runner, updated_config, "complex", log=job.log)
    r_coords, r_sd = _dielectric_band_ascending(receptor_runner, updated_config, "receptor", log=job.log)

    # Both legs share the lambda axis; the combined superdiagonal is the per-pair min (worse leg governs).
    n = min(len(c_sd), len(r_sd))
    combined = [min(c_sd[k], r_sd[k]) for k in range(n)]
    coords = c_coords  # ascending lambda, identical interior to r_coords

    job.log(
        f"[ALS][dielectric] band ({len(coords)} states); threshold={threshold}; "
        f"lambda coords={[round(c, 4) for c in coords]}"
    )
    for k in range(n):
        flag = "WEAK -> insert" if combined[k] < threshold else "ok"
        job.log(
            f"[ALS][dielectric]   pair ({round(coords[k], 4)} <-> {round(coords[k + 1], 4)}): "
            f"complex={c_sd[k]:.4f} receptor={r_sd[k]:.4f} min={combined[k]:.4f} [{flag}]"
        )

    pool = build_candidate_pool(
        coords,
        explicit_pool=updated_config.intermediate_args.candidate_dielectric_pool,
        step=updated_config.intermediate_args.dielectric_pool_step or 0.02,
    )
    new_lam, converged, reason = insert_one(
        list(coords),
        combined,
        pool,
        threshold,
        lower_bound=min(coords),
        upper_bound=max(coords),
    )

    if converged or max_iterations <= 0:
        if converged and reason == "pool-exhausted":
            job.log("[ALS][dielectric] WARNING: candidate pool exhausted with weak gaps remaining")
        elif converged:
            job.log(
                f"[ALS][dielectric] schedule converged: "
                f"{sorted(updated_config.intermediate_args.gb_extdiel_windows)}"
            )
        else:
            job.log("[ALS][dielectric] iteration cap reached")
        return (complex_runner, receptor_runner, updated_config)

    new_eps = float(eps_from_lambda(new_lam))
    job.log(
        f"[ALS][dielectric] inserting matched GB window eps={new_eps:.4f} (lambda={new_lam:.4f}) "
        f"into BOTH legs ({max_iterations - 1} iterations left)"
    )
    improve_job = job.addChildJobFn(
        improve_dielectric_overlap_both,
        complex_runner,
        receptor_runner,
        new_eps,
        updated_config,
    )
    return improve_job.addFollowOnJobFn(
        pilot_dielectric_scheduler,
        improve_job.rv(0),
        improve_job.rv(1),
        improve_job.rv(2),
        max_iterations=max_iterations - 1,
    ).rv()


def improve_dielectric_overlap_both(
    job,
    complex_runner: IntermidateRunner,
    receptor_runner: IntermidateRunner,
    new_eps: float,
    config: Config,
):
    """Insert ONE matched GB-dielectric window into BOTH legs and run each via a two-phase sub-runner.

    The single epsilon is appended to the shared ``gb_extdiel_windows`` ONCE (not once per leg). Returns
    ``(complex_post_runner, receptor_post_runner, config)``.
    """
    new_eps = float(new_eps)
    job.log(f"[ALS][dielectric] writing matched GB window extdiel={new_eps:.5g} (complex + receptor)")

    setup_job = job.addChildJobFn(initilized_jobs)
    c_before = len(complex_runner.simulations)
    r_before = len(receptor_runner.simulations)

    # One short-pilot extdiel mdin (saltcon=0) reused by both legs (same epsilon, same pilot length).
    extdiel_mdin = setup_job.addChildJobFn(
        generate_extdiel_mdin,
        user_mdin_ID=config.intermediate_args.mdin_intermediate_file,
        gb_extdiel=new_eps,
        nstlim=config.inputs.get("pilot_nstlim_steps"),
        ntwx=config.inputs.get("pilot_ntwx"),
    ).rv()

    max_con = max(config.intermediate_args.exponent_conformational_forces_list)
    max_orient = max(config.intermediate_args.exponent_orientational_forces_list)

    complex_runner._add_complex_simulation(
        conformational=max_con,
        orientational=max_orient,
        mdin=extdiel_mdin,
        restraint_file=complex_runner.restraints.max_complex_restraint,
        charge=0.0,
        charge_parm=setup_job.addChildJobFn(
            alter_topology,
            solute_amber_parm=config.endstate_files.complex_parameter_filename,
            solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
            ligand_mask=config.amber_masks.ligand_mask,
            receptor_mask=config.amber_masks.receptor_mask,
            set_charge=0.0,
        ),
        gb_extdiel=new_eps,
    )
    receptor_runner._add_receptor_simulation(
        conformational=max_con,
        mdin=extdiel_mdin,
        restraint_file=receptor_runner.restraints.max_receptor_conformational_restraint,
        gb_extdiel=new_eps,
    )

    c_sims = complex_runner.simulations[c_before:]
    r_sims = receptor_runner.simulations[r_before:]

    # Canonical shared schedule: append the epsilon ONCE.
    config.intermediate_args.gb_extdiel_windows.append(new_eps)

    setup_done = setup_job.addFollowOnJobFn(initilized_jobs)
    c_md = setup_done.addChild(
        complex_runner.new_runner(config, complex_runner.__dict__, post_only=False, simulations=c_sims)
    )
    c_post = c_md.addFollowOn(
        complex_runner.new_runner(config, complex_runner.__dict__, post_only=True)
    )
    r_md = setup_done.addChild(
        receptor_runner.new_runner(config, receptor_runner.__dict__, post_only=False, simulations=r_sims)
    )
    r_post = r_md.addFollowOn(
        receptor_runner.new_runner(config, receptor_runner.__dict__, post_only=True)
    )
    return (c_post.rv(), r_post.rv(), config)


def improve_charge_overlap(
    job,
    runner: IntermidateRunner,
    new_charge: float,
    config: Config,
    system_type: str,
):
    """Insert exactly ONE charge window (R-ADD) and run it via a two-phase sub-runner.

    Mirrors :func:`improve_restraints_overlap` on the charge axis: the single ``new_charge`` to insert is
    chosen upstream by ``adaptive_lambda_windows``. It builds the scaled-charge topology via
    ``alter_topology``, appends one MD window at the MAX restraint and full water (eps=78.5), records the
    charge in the canonical ``charges_lambda_window`` schedule, and runs the window in two phases sharing
    ``post_output`` by reference with ``runner``.
    """
    new_charge = float(round(new_charge, ROUND_DP))
    job.log(f"[ALS][{system_type}] writing charge window q={new_charge}")

    # Short-pilot MD length; fall back to the production mdin if the pilot mdin was not emitted.
    pilot_mdin = config.inputs.get("pilot_mdin") or config.inputs["default_mdin"]
    pilot_no_solv = config.inputs.get("pilot_no_solvent_mdin") or config.inputs["no_solvent_mdin"]

    setup_job = job.addChildJobFn(initilized_jobs)
    before = len(runner.simulations)
    max_con = max(config.intermediate_args.exponent_conformational_forces_list)

    if system_type == "complex":
        runner._add_complex_simulation(
            conformational=max_con,
            orientational=max(config.intermediate_args.exponent_orientational_forces_list),
            mdin=pilot_mdin,
            restraint_file=runner.restraints.max_complex_restraint,
            charge=new_charge,
            charge_parm=setup_job.addChildJobFn(
                alter_topology,
                solute_amber_parm=config.endstate_files.complex_parameter_filename,
                solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                ligand_mask=config.amber_masks.ligand_mask,
                receptor_mask=config.amber_masks.receptor_mask,
                set_charge=new_charge,
            ),
        )
    else:  # ligand-only charge scaling (igb=6 gas) — uses the no-solvent pilot mdin
        runner._add_ligand_simulation(
            conformational=max_con,
            mdin=pilot_no_solv,
            restraint_file=runner.restraints.max_ligand_conformational_restraint,
            charge=new_charge,
            charge_parm=setup_job.addChildJobFn(
                alter_topology,
                solute_amber_parm=config.endstate_files.ligand_parameter_filename,
                solute_amber_coordinate=config.endstate_files.ligand_coordinate_filename,
                ligand_mask=config.amber_masks.ligand_mask,
                receptor_mask=config.amber_masks.receptor_mask,
                set_charge=new_charge,
            ),
        )

    new_sims = runner.simulations[before:]

    # Canonical schedule (R1): CycleSteps + compute_mbar read this list, so record the new charge here.
    config.intermediate_args.charges_lambda_window.append(new_charge)

    setup_done = setup_job.addFollowOnJobFn(initilized_jobs)
    md_runner = setup_done.addChild(
        runner.new_runner(config, runner.__dict__, post_only=False, simulations=new_sims)
    )
    post_runner = md_runner.addFollowOn(
        runner.new_runner(config, runner.__dict__, post_only=True)
    )
    return (post_runner.rv(), config)



def run_exponential_averaging(
    job,
    system_runner: IntermidateRunner,
    temperature: float,
):
    """Execute exponential averaging

    Parameters
    ----------
    system_runner: IntermidateRunner
        A runner class that handles system specific simulations.
    temperature: float
        Specified simulation temperature

    Returns:
    --------
    pdmbar.mbar(df_subsampled): tuple[DataFrame, DataFrame, MBAR]
        DataFrames for the free energies differences (Deltaf_ij), error estimates in free energy difference (dDeltaf_ij), and the pyMBAR object.
    df_mbar: pd.DataFrame
        An formated and chronological arrange DataFrame before any MBAR analysis was performed. (Which can be used to create pdfs of MBAR matrix).
    """
    job.fileStore.logToMaster(f"Running exponential averaging")
    job.fileStore.logToMaster(f"Running a total of {len(system_runner.post_output)} simulations")
    job.fileStore.logToMaster(f"Post output: {system_runner.post_output}")
    return compute_mbar(
        simulation_data=system_runner.post_output,
        temperature=temperature,
        matrix_order=None,
        system="free_flat_bottom",
        log=job.log,
    )


def initilized_jobs(job):
    "Place holder to schedule jobs for MD and post-processing"
    return
