"""
A collection of functions that performs simple iterative proceess to improve space phase overlap between adjecent states. 
"""

import copy
import math
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
        mandatory: without it a coarse pool that cannot reach ``threshold`` would recurse forever
        (``good_enough`` never flips), growing the Toil graph unbounded.
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


def compute_mbar(
    simulation_data: list[pd.DataFrame],
    temperature: float,
    matrix_order: Optional[CycleSteps],
    system: str,
    memory="2G",
    cores=1,
    disk="3G",
    restraint_band: bool = False,
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

    def _restraint_band_states(order, system):
        # The RESTRAINT band = the restraint windows + their max-restraint anchor, EXCLUDING the
        # endstate. ALS schedules WITHIN this interpolatable band; the LJ on/off single step, the lower
        # charge windows, and the fixed endstate are NOT part of it (they would only contaminate the
        # conditioning of the restraint overlaps the scheduler reads). For the complex the band starts
        # at the max-restraint anchor = the last (full-charge) electrostatics state
        # (``halo_restraint_matrix``); ``remove_restraints`` drops the max ``lambda_window``, so that
        # anchor IS the max restraint. For ligand/receptor the band is ``apply_restraints`` (max
        # included), endstate (index 0) excluded.
        if system == "complex":
            band = order[matrix_order.halo_restraint_matrix:]
        else:
            band = order[1 : matrix_order.apo_end_restraint_matrix + 1]
        return [state for state in band if state[0] != "endstate"]

    def _ordered_band(order, system):
        # MBAR over ONLY the restraint-band sub-grid (band trajectories x band states) of the full N x N
        # the pilot ran. The ROW filter is required: keeping non-band trajectory frames while dropping
        # their columns would break pymbar's sum(N_k)==n_samples invariant.
        band = _restraint_band_states(order, system)
        banded = _ordered(band)  # band columns (validated present by the invariant above)
        return banded.loc[banded.index.isin(band)]  # band trajectory rows only

    # flat bottom to no flat bottom -> EXP()
    if matrix_order is None:
        pass

    elif system == "complex":
        order = matrix_order.complex_order
        df_mbar = _ordered_band(order, "complex") if restraint_band else _ordered(order)

    elif system == "ligand":
        order = matrix_order.ligand_order
        df_mbar = _ordered_band(order, "ligand") if restraint_band else _ordered(order)

    else:
        order = matrix_order.receptor_order
        df_mbar = _ordered_band(order, "receptor") if restraint_band else _ordered(order)

    equil_info = pdmbar.detect_equilibration(df_mbar)

    df_subsampled = pdmbar.subsample_correlated_data(df_mbar, equil_info=equil_info)

    print("performing MBAR")
    return pdmbar.mbar(df_subsampled), df_mbar

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
    # Compute MBAR over ONLY the restraint band — the restraint windows + their max-restraint anchor.
    # ALS schedules within this interpolatable band; the LJ on/off single step, the lower charge
    # windows, and the fixed endstate are excluded from this solve (they would only contaminate the
    # conditioning of the restraint overlaps the scheduler reads). The pilot still RUNS those states
    # (for the complex leg's charge scaling + the future charge/GB bands); they are just not in the
    # restraint overlap. So the overlap matrix here IS the restraint band (anchor + windows).
    results = compute_mbar(
        simulation_data=system_runner.post_output,
        temperature=updated_config.intermediate_args.temperature,
        matrix_order=cycle_steps,
        system=system_type,
        restraint_band=True,
    )
    job.log(f"SYSTEM TYPE {system_type}")
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

        # The overlap matrix is ALREADY the restraint band (compute_mbar(restraint_band=True) above),
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
    # Ligand-charge / GB-dielectric scaling: legacy symmetric-average path (deferred scope; the
    # R-ADD revival targets restraints first). Unchanged behaviour.
    # ------------------------------------------------------------------
    if charge_scaling:
        job.log("Attempting to improve ligand charge windows space phase overlap")
        func = improve_charge_scaling
        matrix_start = cycle_steps.start_complex_charge_matrix
        matrix_end = cycle_steps.halo_restraint_matrix
        if system_type == "ligand":
            matrix_start = cycle_steps.start_ligand_charge_matrix
            matrix_end = None
    elif gb_scaling:
        job.log("Attempting to improve GB external dielectric windows space phase overlap")
        func = improve_gb_dielectric
        matrix_start = cycle_steps.start_gb_extdiel_matrix
        matrix_end = cycle_steps.start_complex_charge_matrix
        updated_config.intermediate_args.gb_extdiel_windows.sort()
    else:
        raise ValueError(
            "adaptive_lambda_windows requires exactly one of restraints_scaling / charge_scaling / "
            "gb_scaling to be True"
        )

    averages = overlap_average(overlap_matrix, matrix_start, end=matrix_end)
    job.log(f"The current windows averages: {averages}")

    if good_enough(
        space_phase_overlaps=averages,
        min=updated_config.intermediate_args.min_degree_overlap,
    ):
        return (
            results,
            updated_config,
            system_runner,
        )

    improve_job = job.addChildJobFn(
        func,
        system_runner,
        averages,
        updated_config,
        system_type,
    )
    return improve_job.addFollowOnJobFn(
        adaptive_lambda_windows,
        improve_job.rv(0),
        improve_job.rv(1),
        system_type=system_type,
        restraints_scaling=restraints_scaling,
        charge_scaling=charge_scaling,
        gb_scaling=gb_scaling,
    ).rv()


def overlap_average(overlap_matrix, start, end=None):
    """
    Compute the average of the degree of space phase overlap between a slice adjacent states.

    Parameters
    ----------
    overlap_matrix: list
        Overlap matrix between the states.
    start: int
        Where the matrix should start reading degree of phase space.
    end: int
        The position to end the matrix.

    Returns
    ------
    A list of averages of the degree of phase space overlap between adjacent states.
    """

    overlap_neighbors = group_overlap_neighbors(overlap_matrix)
    print(f"OVERLAP NEIGH: {overlap_neighbors}")
    restraints_overlap = overlap_neighbors[start:]

    if end is not None:
        restraints_overlap = overlap_neighbors[start:end]

    print(f"RESTRAINT OVERLAPS NUMBERS: {restraints_overlap}")
    return [(x[0] + x[1]) / 2 for x in restraints_overlap]


def good_enough(space_phase_overlaps, min=0.03):
    """Check that all averge degree of overlap are about the minimum criteria.

    Parameters
    ----------
    averages: list
        A list of averages degree of overlap between adjacent stats.
    min: float
        Minimum criteria of degree of overlap percent. Default value = 0.03 (3% overlap)

    Returns
       A boolean wheter all overlap are above the minimum criteria.
    """

    return all([x >= min for x in space_phase_overlaps])


def group_overlap_neighbors(matrix):
    """
    Retireve both the foward and reverse degree of overlap between adjacent states.

    Parameters
    ----------
    matrix: np.ndarray
        Estimated state overlap matrix : O[i,j] is an estimate of the probability of observing a sample from state i in state j

    Returns
    -------
    overlap_neighbors: List[tuple[float, float]]
        Returns a list of tuples of estimated probability of both forward and reverse degree of overlap.
    """
    size = matrix.shape[0] - 1

    def get_overlap_neighbors(n=0, new=[]):
        if n == size:
            return new

        else:
            a = round(matrix[n, n + 1], 2)
            b = round(matrix[n + 1, n], 2)
            new.append((a, b))

            return get_overlap_neighbors(n + 1, new=new)

    return get_overlap_neighbors()


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


def improve_charge_scaling(
    job,
    runner: IntermidateRunner,
    avg_overlap,
    config: Config,
    system_type: str,
):
    """Improve poor space overlap of two adjacent states via bisection.

    If the user specifed an upper and lower bound (within the configuration file) an biscetion
    will be attempted to improve the overlap between the upper and lower bounds. If only provided
    an upper bound limit than a subtraction of 1 will be computed and biscetion when needed.
    This adaptive procedure will be applied towards ligand net charge bisceting between
    1 (ligand fully charge) to 0 (ligand's net charge = 0).
    Args:
        job (_type_): _description_
        runner (IntermidateRunner): _description_
        avg_overlap (_type_): _description_
        config (Config): _description_
        system_type (str): _description_

    Parameters
    ----------
    job: Toil.job
        The atomic unit of work in a Toil workflow is a Job.
    runner: IntermidateRunner
        An system specific runner object to create and inital any new MD runs needed.
    avg_overlap: list
        A list of averages degree of overlap between adjacent stats.
    config: Config
        User specified configuration file containing necesssary input information.
    system_type: str
        System type to denote the specific system_runner (i.e. complex, receptor or ligand)

    Returns
    -------
    A runner job promise (toil.job.Promise) is essentially a pointer to for the return value that is replaced by the actual return value once it has been evaluated. If any windows were created MD simulation and post-process analysis will be performed and returned.
    config: Config
        An updated configuration file with new inserted states.
    """
    # init job scaling job
    charge_job = job.addChildJobFn(initilized_jobs)
    charges = config.intermediate_args.charges_lambda_window.copy()

    job.log(f"Interating over windows {avg_overlap}")

    for index, overlap in enumerate(avg_overlap):
        # if sufficient overlap continue iterating
        if overlap > config.intermediate_args.min_degree_overlap:
            continue
        # biscet between adjacent windows with poor overlap
        new_charge = bisect_between(charges[index], charges[index + 1])
        # append to list
        job.log(
            f"Biscent between charges {charges[index], charges[index + 1]} = {new_charge}"
        )
        config.intermediate_args.charges_lambda_window.append(new_charge)
        # check to see if the system passed is an complex
        if system_type == "complex":
            runner._add_complex_simulation(
                conformational=max(
                    config.intermediate_args.exponent_conformational_forces
                ),
                orientational=max(
                    config.intermediate_args.exponent_orientational_forces
                ),
                mdin=config.inputs["default_mdin"],
                restraint_file=runner.restraints.max_complex_restraint,
                charge=new_charge,
                charge_parm=charge_job.addChildJobFn(
                    alter_topology,
                    solute_amber_parm=config.endstate_files.complex_parameter_filename,
                    solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                    ligand_mask=config.amber_masks.ligand_mask,
                    receptor_mask=config.amber_masks.receptor_mask,
                    set_charge=new_charge,
                ),
            )
        else:
            # if not a complex then it must be a ligand only system
            runner._add_ligand_simulation(
                conformational=max(
                    config.intermediate_args.exponent_conformational_forces
                ),
                mdin=config.inputs["no_solvent_mdin"],
                restraint_file=runner.restraints.max_ligand_conformational_restraint,
                charge=new_charge,
                charge_parm=charge_job.addChildJobFn(
                    alter_topology,
                    solute_amber_parm=config.endstate_files.ligand_parameter_filename,
                    solute_amber_coordinate=config.endstate_files.ligand_coordinate_filename,
                    ligand_mask=config.amber_masks.ligand_mask,
                    receptor_mask=config.amber_masks.receptor_mask,
                    set_charge=new_charge,
                ),
            )

    charges_done = charge_job.addFollowOnJobFn(initilized_jobs)
    return (
        charges_done.addChild(runner.new_runner(config, runner.__dict__)).rv(),
        config,
    )


def improve_gb_dielectric(
    job,
    runner: IntermidateRunner,
    avg_overlap,
    config: Config,
    system_type: str,
):
    """_summary_

    Args:
        job (_type_): _description_
        runner (IntermidateRunner): _description_
        avg_overlap (_type_): _description_
        config (Config): _description_
        system_type (str): _description_
    """
    # init gb external dielectric
    gb_dielectric_job = job.addChildJobFn(initilized_jobs)

    lower_upper_bound = [0.0, 78.5]

    gb_dielectric = config.intermediate_args.gb_extdiel_windows.copy()
    # add in lower & upper bounds then sort
    gb_dielectric = sorted(list(set(gb_dielectric + lower_upper_bound)))

    job.log(f"Interating over GB windows {avg_overlap}")

    for index, overlap in enumerate(avg_overlap):
        # if sufficient overlap continue iterating
        if overlap > config.intermediate_args.min_degree_overlap:
            continue

        # poor overlap
        else:
            # biscet between upper and lower bound
            new_gb_dielectric = bisect_between(
                gb_dielectric[index], gb_dielectric[index + 1]
            )
            job.log(
                f"bisecting between {gb_dielectric[index]} & {gb_dielectric[index + 1]} = {new_gb_dielectric}"
            )
        # append new dielectric
        config.intermediate_args.gb_extdiel_windows.append(new_gb_dielectric)

        # create new runner simulation
        runner._add_complex_simulation(
            conformational=max(config.intermediate_args.exponent_conformational_forces),
            orientational=max(config.intermediate_args.exponent_orientational_forces),
            mdin=gb_dielectric_job.addChildJobFn(
                generate_extdiel_mdin,
                user_mdin_ID=config.intermediate_args.mdin_intermediate_file,
                gb_extdiel=new_gb_dielectric,
            ).rv(),
            restraint_file=runner.restraints.max_complex_restraint,
            charge_parm=gb_dielectric_job.addChildJobFn(
                alter_topology,
                solute_amber_parm=config.endstate_files.complex_parameter_filename,
                solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                ligand_mask=config.amber_masks.ligand_mask,
                receptor_mask=config.amber_masks.receptor_mask,
                set_charge=0.0,
            ),
            charge=0.0,
            gb_extdiel=new_gb_dielectric,
        )

    # sort the new added windows
    config.intermediate_args.gb_extdiel_windows.sort()

    # gb scaling done
    gb_scaling_done = gb_dielectric_job.addFollowOnJobFn(initilized_jobs)

    return (
        gb_scaling_done.addChild(runner.new_runner(config, runner.__dict__)).rv(),
        config,
    )


def bisect_between(start, end):
    """
    Perform a bisection search between two numbers.

    Parameters
    ----------
        start (float): The start of the interval.
        end (float): The end of the interval.

    Returns
    -------
        float: The midpoint between start and end.
    """
    return (start + end) / 2


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
    )


def initilized_jobs(job):
    "Place holder to schedule jobs for MD and post-processing"
    return
