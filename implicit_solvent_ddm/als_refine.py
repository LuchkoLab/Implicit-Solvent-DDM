"""Post-production adaptive lambda scheduling (ALS).

After a run's production windows are scored, every adjacent pair of states in each leg gets a
two-state overlap. Pairs below a threshold get a new window in their gap, the window lists are
extended, and the workflow re-runs setup, MD and post-analysis with warm resume, so only the new
windows and the new energy cells are computed.

The trigger is the two-state overlap ``min(O_ab, O_ba)`` of each adjacent pair, solved from those
two states' samples only. The full-N MBAR overlap matrix cannot be used: in a dense band of
near-identical states each row spreads over every state, so its superdiagonal is about 1/N even when
neighbours overlap almost perfectly.

Everything here is pure (no Toil, no MD) so it can run on finished runs as well as inside the DAG.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import implicit_solvent_ddm.pandasmbar as pdmbar

AVAGADRO = 6.0221367e23
BOLTZMAN = 1.380658e-23
JOULES_PER_KCAL = 4184

ROUND_DP = 3  # window values are labelled, and directories named, at this precision

BANDS = ("restraint", "charge", "lj", "screening", "dielectric")
DEFAULT_BANDS = ("restraint", "charge", "lj", "screening")
REPORT_ONLY = ("endstate", "desolvation", "unclassified")

# Which legs share each band's window list. A gap's decision overlap is the minimum over them.
SHARED_LEGS = {
    "restraint": ("complex", "ligand", "receptor"),
    "charge": ("complex", "ligand"),
    "lj": ("complex",),
    "screening": ("complex",),
    "dielectric": ("complex", "receptor"),
}

# MD windows one insertion adds (one per leg that shares the list), for the window budget.
WINDOWS_PER_INSERTION = {"restraint": 3, "charge": 2, "lj": 1, "screening": 1, "dielectric": 2}

# Smallest half-gap a split may create, per band (log2 units for restraints, linear otherwise).
DEFAULT_MIN_STEP = {"restraint": 0.05, "charge": 0.02, "lj": 0.005, "screening": 0.02, "dielectric": 0.01}


def kcal_per_kt(temperature: float) -> float:
    """kcal/mol per kT at ``temperature``, with the constants ``compute_mbar`` uses."""
    return ((BOLTZMAN * AVAGADRO) / JOULES_PER_KCAL) * float(temperature)


# --------------------------------------------------------------------------------------------------
# Pairwise overlaps
# --------------------------------------------------------------------------------------------------
def subsample_like_production(df_mbar: pd.DataFrame) -> pd.DataFrame:
    """Subsample a cycle-ordered reduced-energy matrix exactly as Phase 7 does.

    Parameters
    ----------
    df_mbar : pandas.DataFrame
        Rows indexed by sampled state, columns the cycle-ordered states, values in kT.

    Returns
    -------
    pandas.DataFrame
        The decorrelated samples ``compute_mbar`` would pass to pymbar.
    """
    equil_info = pdmbar.detect_equilibration(df_mbar)
    return pdmbar.subsample_correlated_data(df_mbar, equil_info=equil_info)


def adjacent_pair_overlaps(df_sub: pd.DataFrame, order: Optional[Sequence[tuple]] = None) -> pd.DataFrame:
    """Two-state overlap of every adjacent pair of states.

    Subsampling is per sampled state and uses only that state's own column, so slicing a pair out of
    the subsampled full matrix gives the same samples a two-state analysis would use.

    Parameters
    ----------
    df_sub : pandas.DataFrame
        Subsampled reduced energies (kT); rows indexed by sampled state, columns the states.
    order : sequence of tuple, optional
        State order; defaults to the column order.

    Returns
    -------
    pandas.DataFrame
        One row per adjacent pair: ``i, a, b, o_ab, o_ba, o_pair, dG, ddG, n_a, n_b, status``.
        ``o_pair = min(o_ab, o_ba)``; ``dG`` is ``f_b - f_a`` in kT. ``status`` is ``ok``,
        ``unscored`` (a state has no samples or a cell is NaN) or ``solve_failed`` (``o_pair = 0``).
    """
    if isinstance(df_sub.columns, pd.MultiIndex):
        # pandasmbar matches sampled states to columns as flat tuples, as the workflow builds them
        df_sub = df_sub.set_axis(pd.Index(list(df_sub.columns), tupleize_cols=False), axis=1)
    states = list(order) if order is not None else list(df_sub.columns)
    index = df_sub.index
    rows = []
    for i in range(len(states) - 1):
        a, b = states[i], states[i + 1]
        row = dict(i=i, a=a, b=b, o_ab=np.nan, o_ba=np.nan, o_pair=np.nan, dG=np.nan, ddG=np.nan)
        in_a = index.isin([a])
        in_b = index.isin([b])
        row["n_a"], row["n_b"] = int(in_a.sum()), int(in_b.sum())
        if row["n_a"] == 0 or row["n_b"] == 0 or a not in df_sub.columns or b not in df_sub.columns:
            rows.append({**row, "status": "unscored"})
            continue
        sub = df_sub.loc[in_a | in_b, [a, b]]
        if sub.isna().to_numpy().any():
            rows.append({**row, "status": "unscored"})
            continue
        try:
            fe, err, mbar = pdmbar.mbar(sub)
            overlap = np.asarray(mbar.compute_overlap()["matrix"])
            o_ab, o_ba = float(overlap[0][1]), float(overlap[1][0])
            row.update(o_ab=o_ab, o_ba=o_ba, o_pair=min(o_ab, o_ba),
                       dG=float(fe.iloc[0, 1]), ddG=float(err.iloc[0, 1]))
            status = "ok" if math.isfinite(row["o_pair"]) else "solve_failed"
        except Exception:  # a pair pymbar cannot solve is as weak as a pair can be
            status = "solve_failed"
        if status == "solve_failed":
            row["o_pair"] = 0.0
        rows.append({**row, "status": status})
    return pd.DataFrame(rows, columns=["i", "a", "b", "o_ab", "o_ba", "o_pair", "dG", "ddG", "n_a", "n_b", "status"])


# --------------------------------------------------------------------------------------------------
# Classifying pairs into bands
# --------------------------------------------------------------------------------------------------
def _is_lj_state(state) -> bool:
    return state[0] in ("no_interactions", "interactions") or state[0].startswith("lj_")


def _is_screening_side(state) -> bool:
    return state[0] == "interactions" or state[0].startswith("screening_")


def _in_water(state) -> bool:
    return float(state[1]) != 0.0


def classify_pair(a: tuple, b: tuple, endstate_label: str) -> str:
    """Which window band an adjacent pair belongs to.

    Works on state labels, not ladder indices, so it applies to any cycle order and to the columns
    of a stored ``*_formatted.h5``.

    Parameters
    ----------
    a, b : tuple
        Adjacent ``(label, extdiel, charge, restraint)`` states.
    endstate_label : str
        Label of the unrestrained endstate.

    Returns
    -------
    str
        One of ``restraint``, ``charge``, ``lj``, ``screening``, ``dielectric`` (fillable), or
        ``endstate``, ``desolvation``, ``unclassified`` (reported only).
    """
    la, lb = a[0], b[0]
    pair = {la, lb}
    if endstate_label in pair:
        return "endstate"
    if _is_lj_state(a) and _is_lj_state(b) and pair != {"interactions"}:
        return "lj"
    # screening runs interactions -> screening_* -> uncharged electrostatics, all in water
    if (_is_screening_side(a) or _is_screening_side(b)) and _in_water(a) and _in_water(b):
        other = b if _is_screening_side(a) else a
        if _is_screening_side(other) or (other[0] == "electrostatics" and float(other[2]) == 0.0):
            return "screening"
    if "gb_dielectric" in pair or "no_gb" in pair:
        return "dielectric"
    if pair == {"interactions", "electrostatics"}:
        return "dielectric"  # gas cycle with no GB windows: gas interactions -> water q=0
    if pair == {"electrostatics"}:
        return "charge" if a[1] == b[1] else "desolvation"
    if pair == {"lambda_window", "electrostatics"}:
        el, lw = (a, b) if la == "electrostatics" else (b, a)
        if el[1] != lw[1]:
            return "desolvation"  # gas-cycle ligand: water restraint anchor -> gas q=1
        # complex: q=1 at the max restraint -> restraint band; ligand in water: q=1 anchor -> q<1
        return "restraint" if float(el[2]) == 1.0 else "charge"
    if pair == {"lambda_window"}:
        return "restraint"
    return "unclassified"


def band_coordinate(state: tuple, band: str) -> float:
    """A state's position on its band's axis.

    Parameters
    ----------
    state : tuple
        ``(label, extdiel, charge, restraint)``.
    band : str
        Band from :func:`classify_pair`.

    Returns
    -------
    float
        restraint: the conformational log2 exponent (first field of the restraint label, so the
        complex ``con_orient`` and apo ``con`` labels share one axis); charge: the charge fraction;
        lj: 0 for ``no_interactions``, 1 for ``interactions``, else the ``lj_x`` value; screening: 0
        for ``interactions``, 1 for uncharged ``electrostatics``, else the ``screening_x`` value;
        dielectric: ``1 - 1/eps`` (0 in gas).
    """
    label = state[0]
    if band == "restraint":
        return round(float(str(state[3]).split("_")[0]), ROUND_DP)
    if band == "charge":
        return round(float(state[2]), ROUND_DP)
    if band == "lj":
        if label == "no_interactions":
            return 0.0
        if label == "interactions":
            return 1.0
        return round(float(label[len("lj_"):]), ROUND_DP)
    if band == "screening":
        if label == "interactions":
            return 0.0
        if label == "electrostatics":
            return 1.0
        return round(float(label[len("screening_"):]), ROUND_DP)
    if band == "dielectric":
        return lambda_of_eps(float(state[1]))
    raise ValueError(f"band {band!r} has no coordinate")


def lambda_of_eps(eps: float) -> float:
    """GB-dielectric coordinate ``1 - 1/eps`` at 6 dp; the gas sentinel ``eps = 0`` maps to 0."""
    eps = float(eps)
    return 0.0 if eps == 0.0 else round(1.0 - 1.0 / eps, 6)


def eps_from_lambda(lam: float) -> float:
    """External dielectric for GB-dielectric coordinate ``lam = 1 - 1/eps``."""
    return 1.0 / (1.0 - float(lam))


# --------------------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------------------
@dataclass
class Schedule:
    """The window lists ALS can extend.

    Attributes
    ----------
    restraint : list of (float, float)
        ``(conformational, orientational)`` exponent pairs at 3 dp, as they label states.
    charges, lj, screening : list of float
        Window values (charges include the 0 and 1 anchors, as the config stores them).
    dielectric : list of float
        GB external dielectrics (epsilon).
    endstate_label : str
        Label of the unrestrained endstate.
    """

    restraint: list
    charges: list
    lj: list
    screening: list
    dielectric: list
    endstate_label: str

    @classmethod
    def from_config(cls, config) -> "Schedule":
        """Read the schedule off a workflow ``Config``."""
        args = config.intermediate_args
        return cls(
            restraint=list(zip(args.exponent_conformational_forces_list, args.exponent_orientational_forces_list)),
            charges=[float(q) for q in args.charges_lambda_window],
            lj=[float(x) for x in args.lj_scale_windows],
            screening=[float(x) for x in args.screening_windows],
            dielectric=[float(e) for e in args.gb_extdiel_windows],
            endstate_label=config.endstate_method.endstate_state_label,
        )

    def coordinates(self, band: str) -> set:
        """Every coordinate the band's states sit at, anchors included, rounded like the labels."""
        if band == "restraint":
            return {round(float(c), ROUND_DP) for c, _ in self.restraint}
        if band == "charge":
            return {round(float(q), ROUND_DP) for q in {0.0, 1.0, *self.charges}}
        if band == "lj":
            return {0.0, 1.0, *(round(float(x), ROUND_DP) for x in self.lj)}
        if band == "screening":
            return {0.0, 1.0, *(round(float(x), ROUND_DP) for x in self.screening)}
        if band == "dielectric":
            return {0.0, lambda_of_eps(78.5), *(lambda_of_eps(e) for e in self.dielectric)}
        raise ValueError(f"unknown band {band!r}")

    def orientational_partner(self, new_con: float) -> float:
        """Orientational exponent for a new conformational one: the constant offset when there is one,
        otherwise interpolated at the same fractional position."""
        con = [c for c, _ in self.restraint]
        orient = [o for _, o in self.restraint]
        offsets = {round(o - c, ROUND_DP) for c, o in self.restraint}
        if len(offsets) == 1:
            return round(new_con + offsets.pop(), ROUND_DP)
        order = np.argsort(con)
        return round(float(np.interp(new_con, np.asarray(con)[order], np.asarray(orient)[order])), ROUND_DP)


@dataclass
class RefinementPlan:
    """What one round of refinement decided.

    Attributes
    ----------
    insertions : dict
        ``{band: [new values]}``; restraint values are ``(con, orient)`` pairs, dielectric values are
        epsilons.
    records : pandas.DataFrame
        One row per (leg, adjacent pair) with its band, overlap and decision (``status``).
    endstate_gaps : pandas.DataFrame
        Weak endstate-adjacent pairs, reported only.
    n_insertions, n_md_windows : int
        New window values, and the MD windows they add across legs.
    reason : str
        ``converged`` (no weak fillable pair), ``pool-exhausted`` (weak pairs remain, none fillable)
        or empty (insertions were made).
    threshold : float
        The two-state overlap threshold used.
    band_reasons : dict
        Per band: ``inserted``, ``pool-exhausted``, ``deferred``, ``disabled`` or ``converged``.
    """

    insertions: dict
    records: pd.DataFrame
    endstate_gaps: pd.DataFrame
    n_insertions: int
    n_md_windows: int
    reason: str
    threshold: float = 0.0
    band_reasons: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """JSON-ready summary for ``plan.json`` and the log."""
        return dict(
            threshold=self.threshold,
            reason=self.reason,
            n_insertions=self.n_insertions,
            n_md_windows=self.n_md_windows,
            insertions={band: [list(v) if isinstance(v, tuple) else v for v in values]
                        for band, values in self.insertions.items()},
            band_reasons=self.band_reasons,
            weak_endstate_gaps=int(len(self.endstate_gaps)),
        )


def _split_value(lo, hi, band, existing, pool, min_step):
    """The value to insert in gap ``(lo, hi)``, or None when the gap cannot be split.

    Parameters
    ----------
    lo, hi : float
        Gap ends on the band axis.
    band : str
        Band name; restraint values are log2 exponents.
    existing : set of float
        Coordinates already in use, including values chosen earlier this round.
    pool : sequence of float, optional
        Explicit candidates; the one nearest the midpoint strictly inside the gap wins.
    min_step : float
        Smallest half-gap a split may create.

    Returns
    -------
    float or None
    """
    if (hi - lo) / 2.0 < min_step:
        return None
    ideal = (lo + hi) / 2.0
    if pool:
        inside = [float(p) for p in pool if lo < float(p) < hi]
        if not inside:
            return None
        ideal = min(inside, key=lambda p: (abs(p - ideal), p))
    value = round(ideal, 6 if band == "dielectric" else ROUND_DP)
    if not lo < value < hi or value in existing:
        return None  # rounding collapsed the split onto an end, or onto a label already in use
    return value


def plan_refinement(
    pair_tables: dict,
    schedule: Schedule,
    threshold: float,
    bands: Sequence[str] = DEFAULT_BANDS,
    pools: Optional[dict] = None,
    min_steps: Optional[dict] = None,
    max_new_windows: Optional[int] = None,
) -> RefinementPlan:
    """Decide which windows to add this round.

    A gap shared by several legs (one window list feeds them all) is weak when its two-state overlap
    is below ``threshold`` in any of them, and one insertion fills it in every leg. Pairs that touch
    the endstate are reported and never filled; the band ends (the max-restraint anchor, the weakest
    restraint window, q = 0 / 1, LJ 0 / 1, screening 0 / 1) are never moved, because insertions land
    strictly inside a gap.

    Parameters
    ----------
    pair_tables : dict
        ``{leg: DataFrame}`` from :func:`adjacent_pair_overlaps`, for any of complex/ligand/receptor.
    schedule : Schedule
        The current window lists.
    threshold : float
        A pair is weak when its two-state overlap is below this (two-state overlap tops out at 0.5).
    bands : sequence of str
        Bands that may be filled. Weak pairs in other bands are recorded as ``weak_band_disabled``.
    pools : dict, optional
        ``{band: [candidates]}``; a band without a pool splits at the midpoint (3 dp).
    min_steps : dict, optional
        ``{band: smallest half-gap}``, defaults in ``DEFAULT_MIN_STEP``.
    max_new_windows : int, optional
        Cap on MD windows added this round; the weakest gaps are filled first.

    Returns
    -------
    RefinementPlan

    Raises
    ------
    ValueError
        If the coordinates found in the data do not match the schedule, i.e. the window lists and
        the scored data have drifted apart.
    """
    pools = pools or {}
    steps = {**DEFAULT_MIN_STEP, **(min_steps or {})}
    records = []
    gaps: dict = {}  # band -> {(lo, hi): {leg: o_pair}}
    seen: dict = {}  # band -> coordinates found in the data
    for leg, table in pair_tables.items():
        for row in table.itertuples(index=False):
            band = classify_pair(row.a, row.b, schedule.endstate_label)
            rec = dict(leg=leg, i=int(row.i), a="|".join(map(str, row.a)), b="|".join(map(str, row.b)),
                       band=band, o_pair=float(row.o_pair), pair_status=row.status, lo=np.nan, hi=np.nan)
            if band not in REPORT_ONLY:
                ca, cb = band_coordinate(row.a, band), band_coordinate(row.b, band)
                lo, hi = sorted((ca, cb))
                rec.update(lo=lo, hi=hi)
                seen.setdefault(band, set()).update((ca, cb))
                if row.status != "unscored":
                    gaps.setdefault(band, {}).setdefault((lo, hi), {})[leg] = float(row.o_pair)
            records.append(rec)

    for band, coords in seen.items():
        expected = schedule.coordinates(band)
        if coords != expected:
            raise ValueError(
                f"ALS: the {band} states in the data {sorted(coords)} do not match the schedule "
                f"{sorted(expected)}; the window lists and the scored data have drifted apart"
            )

    weak = []  # (overlap, band, lo, hi)
    decision: dict = {}  # (band, lo, hi) -> status
    for band, by_gap in gaps.items():
        for (lo, hi), per_leg in by_gap.items():
            merged = min(per_leg.values())
            if merged >= threshold:
                decision[(band, lo, hi)] = "ok"
            elif band in bands:
                weak.append((merged, band, lo, hi))
            else:
                decision[(band, lo, hi)] = "weak_band_disabled"

    existing = {band: set(schedule.coordinates(band)) for band in BANDS}
    insertions = {band: [] for band in BANDS}
    n_windows = 0
    for _, band, lo, hi in sorted(weak):
        value = _split_value(lo, hi, band, existing[band], pools.get(band), steps[band])
        if value is None:
            decision[(band, lo, hi)] = "weak_pool_exhausted"
            continue
        cost = WINDOWS_PER_INSERTION[band]
        if max_new_windows is not None and n_windows + cost > max_new_windows:
            decision[(band, lo, hi)] = "weak_budget_deferred"
            continue
        existing[band].add(value)
        if band == "restraint":
            insertions[band].append((value, schedule.orientational_partner(value)))
        elif band == "dielectric":
            insertions[band].append(round(eps_from_lambda(value), 4))
        else:
            insertions[band].append(value)
        decision[(band, lo, hi)] = "weak_fill"
        n_windows += cost

    band_reasons = {}
    for band in BANDS:
        statuses = {s for (b, _, _), s in decision.items() if b == band}
        if statuses:
            band_reasons[band] = next(
                reason for status, reason in (
                    ("weak_fill", "inserted"), ("weak_pool_exhausted", "pool-exhausted"),
                    ("weak_budget_deferred", "deferred"), ("weak_band_disabled", "disabled"),
                    ("ok", "converged"),
                ) if status in statuses
            )

    records = pd.DataFrame(records, columns=["leg", "i", "a", "b", "band", "o_pair", "pair_status", "lo", "hi"])
    status = []
    for rec in records.itertuples(index=False):
        if rec.band == "endstate":
            status.append("endstate_report")
        elif rec.band in REPORT_ONLY:
            status.append("report_only")
        elif rec.pair_status == "unscored":
            status.append("unscored")
        else:
            status.append(decision[(rec.band, rec.lo, rec.hi)])
    records["status"] = status
    records["weak"] = records.o_pair < threshold
    endstate_gaps = records[(records.band == "endstate") & records.weak].reset_index(drop=True)

    insertions = {band: sorted(values) for band, values in insertions.items() if values}
    n_insertions = sum(len(v) for v in insertions.values())
    if n_insertions:
        reason = ""
    elif not weak:
        reason = "converged"
    else:
        reason = "pool-exhausted"
    return RefinementPlan(
        insertions=insertions, records=records, endstate_gaps=endstate_gaps,
        n_insertions=n_insertions, n_md_windows=n_windows, reason=reason,
        threshold=float(threshold), band_reasons=band_reasons,
    )


# --------------------------------------------------------------------------------------------------
# Extending the schedule
# --------------------------------------------------------------------------------------------------
_ANCHORS = ("strongest_conformational_exponent", "boresch_Kr", "long_restraint_window_exponent",
            "max_conformational_restraint", "max_orientational_restraint")


def extend_schedule(config, insertions: dict):
    """Return a copy of ``config`` with the planned windows added to its lists.

    Only the window lists change. Old values keep their exact floats, so the force constants of the
    windows that already ran are bit-identical, and the anchors (strongest and weakest restraint, the
    Boresch distance constant) are asserted unchanged.

    Parameters
    ----------
    config : Config
        The current round's config. Not modified.
    insertions : dict
        ``RefinementPlan.insertions``.

    Returns
    -------
    Config
        Shallow copy with a new ``intermediate_args`` and its own ``inputs`` dict.

    Raises
    ------
    ValueError
        On a restraint insertion in rmsd mode, a label collision, or a moved anchor.
    """
    old = config.intermediate_args
    restraint = insertions.get("restraint", [])
    if restraint:
        if old.rmsd_mode:
            raise ValueError("ALS restraint insertion is not supported with rmsd_tolerance_windows")
        con = [float(c) for c in old.exponent_conformational_forces]
        orient = [float(o) for o in old.exponent_orientational_forces]
        distance = (None if old.exponent_boresch_distance_forces is None
                    else [float(d) for d in old.exponent_boresch_distance_forces])
        labels = set(old.exponent_conformational_forces_list)
        for new_con, new_orient in restraint:
            new_con = round(float(new_con), ROUND_DP)
            if new_con in labels:
                raise ValueError(f"restraint exponent {new_con} collides with an existing window label")
            if distance is not None:
                order = np.argsort(con)
                distance.append(round(float(np.interp(
                    new_con, np.asarray(con)[order], np.asarray(distance)[order])), ROUND_DP))
            con.append(new_con)
            orient.append(round(float(new_orient), ROUND_DP))
            labels.add(new_con)
        order = sorted(range(len(con)), key=lambda k: con[k])
        # replace() re-runs __post_init__, which re-derives every force ladder from the exponents
        args = dataclasses.replace(
            old,
            exponent_conformational_forces=[con[k] for k in order],
            exponent_orientational_forces=[orient[k] for k in order],
            exponent_boresch_distance_forces=None if distance is None else [distance[k] for k in order],
        )
        for name in _ANCHORS:
            if getattr(args, name) != getattr(old, name):
                raise ValueError(
                    f"ALS insertion changed the anchor {name}: {getattr(old, name)} -> {getattr(args, name)}"
                )
        if min(args.exponent_conformational_forces_list) != min(old.exponent_conformational_forces_list):
            raise ValueError("ALS insertion moved the weakest restraint window")
    else:
        args = copy.copy(old)

    def union(current, new):
        return sorted({*map(float, current), *map(float, new)})

    if insertions.get("charge"):
        args.charges_lambda_window = union(old.charges_lambda_window, insertions["charge"])
    if insertions.get("lj"):
        args.lj_scale_windows = union(old.lj_scale_windows, insertions["lj"])
    if insertions.get("screening"):
        args.screening_windows = union(old.screening_windows, insertions["screening"])
    if insertions.get("dielectric"):
        args.gb_extdiel_windows = union(old.gb_extdiel_windows, insertions["dielectric"])

    new = copy.copy(config)
    new.intermediate_args = args
    new.inputs = dict(config.inputs)
    if insertions.get("dielectric"):
        new.workflow = copy.copy(config.workflow)
        new.workflow.gb_extdiel_windows = bool(args.gb_extdiel_windows)
    return new
