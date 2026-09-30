#!/usr/bin/env python
"""bar_vs_mbar.py -- how much DeltaG accuracy is lost by replacing full MBAR with a chain of
2-state BAR calculations between ADJACENT windows only?

Why
---
`only_post_analysis` (runner.py) re-scores every trajectory under every window in a leg -- a full
N^2 of `sander imin=5` evaluations -- because MBAR needs every frame's energy at every state. That
CPU tail, not the GPU MD, dominates the DDM budget (see PERFORMANCE_TIMING.md). If a BAR chain over
adjacent windows is accurate enough, post-analysis only needs the TRIDIAGONAL: 3N-2 blocks instead
of N^2 (cb7 complex N=39: 115 vs 1521, ~13x; a 74-window protein-ligand leg: 220 vs 5476, ~25x).

This script answers "how much accuracy would that cost, and where does the loss concentrate?"
using ONLY the already-computed `.cache/<system>/*_formatted.h5` files. No MD, no sander.

Method
------
Per (system, replica, leg):

  1. read `*_formatted.h5` (cycle-ordered energies in kcal/mol) and divide by kcals_per_Kt,
  2. run the production front-end ONCE -- `pdmbar.detect_equilibration` then
     `pdmbar.subsample_correlated_data` (verbatim from `compute_mbar`, adaptive_restraints.py:694),
  3. REFERENCE: full MBAR over all N states -> DeltaG = f(last) - f(first), plus the overlap matrix,
  4. TEST: for every adjacent column pair, slice the 2-column / 2-state sub-block and run
     `pdmbar.mbar()` on it. A 2-state MBAR IS Bennett acceptance ratio -- same self-consistent
     equation, same analytic error -- so no separate BAR implementation is needed (the same trick
     `gb_ddG_bar.py` uses). DeltaG_BAR = sum of the per-pair values; it telescopes to f(last)-f(first).

BOTH estimators consume the IDENTICAL subsampled frames, so the only difference is the estimator.
(`detect_equilibration`/`subsample_correlated_data` read only each state's own diagonal column, so
slicing first would keep the same frames anyway -- the tridiagonal data really is sufficient.)

Adjacency comes from `df.columns`, which `_ordered()` (adaptive_restraints.py:608) already put in
canonical thermodynamic-cycle order before the file was written. Do NOT rebuild `CycleSteps` to get
it: today's `matrix_order.py` includes `receptor_GB_exl_windows` (commit 5e7d0ef, 2026-06-24) which
post-dates the Dec-2025 data, so a re-derived `receptor_order` would not match those files.

Correctness check
-----------------
The formatted file is the UN-subsampled matrix (MBAR was solved on `df_subsampled`, but `df_mbar`
is what got written), so step 2 has to be redone here. That gives a free, strong check: the
re-computed full MBAR must reproduce the stored `*_fe.h5` to within `--fe-tol`. If it does not, the
temperature is wrong -- use `--calibrate` to find the one that does.

Usage
-----
  conda activate isddm_env                       # pymbar + pytables + pandas

  python bar_vs_mbar.py --selftest               # verify the math, no data needed
  python bar_vs_mbar.py --root <run_dir> --calibrate            # find the right temperature
  python bar_vs_mbar.py --root <run_dir> --systems cb7_adn      # stage 1: one system, all reps
  python bar_vs_mbar.py --root <run_dir> --nproc 8              # stage 2: everything

where <run_dir> holds rep1..rep5, each with `.cache/<system>/*_formatted.h5`, e.g.
/storage/.../different_gbs/igb_2/cb7_re_parameterization

Outputs (written next to --out-prefix, default ./bar_vs_mbar):
  *_pairs.csv    one row per ADJACENT PAIR -- the diagnostic showing WHERE the chain breaks
  *_summary.csv  one row per (system, rep, leg) + a per-(system, rep) 3-leg total

Per-task results are cached under --scratch, so a dropped mount or a partial run resumes.
"""
import argparse
import concurrent.futures
import glob
import math
import os
import re
import sys
import traceback

import numpy as np
import pandas as pd

# Same constants the package uses (postTreatment.py:21-23 / adaptive_restraints.py:21-23).
AVAGADRO = 6.0221367e23
BOLTZMAN = 1.380658e-23
JOULES_PER_KCAL = 4184

LEGS = ("complex", "ligand", "receptor")
CANDIDATE_TEMPERATURES = (298.0, 298.15, 300.0)


def kcals_per_kt(temperature):
    """kcal/mol per kT -- identical to `ConsolidateData.kcals_per_Kt` (postTreatment.py:182)."""
    return ((BOLTZMAN * AVAGADRO) / JOULES_PER_KCAL) * temperature


def _import_package(repo):
    """Import the package's own MBAR wrapper so this script runs the SAME code as production."""
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    from implicit_solvent_ddm import pandasmbar as pdmbar

    return pdmbar


# --------------------------------------------------------------------------------------------------
# Locating the cached matrices
# --------------------------------------------------------------------------------------------------
def find_tasks(root, systems=None, reps=None, rep_glob="rep*"):
    """Yield (system, rep, leg, formatted_h5, fe_h5) for every cached leg under `root`.

    Layout: <root>/<rep>/.cache/<system>/{<system>,ligand_*,receptor_*}_formatted.h5
    The complex file is named after the system directory; the other two carry a leg prefix.

    `rep_glob` selects the replicate directories (default `rep*`). Run trees that name them
    differently need it set, e.g. `--rep-glob 'mcl1_rep*'` for
    fragment-opt-abfe-benchmark/isdmm_runs/mcl1_rep{1,2,3}.

    If the glob matches nothing and `<root>` itself holds a `.cache/`, then `<root>` IS the
    replicate -- it is used directly and named after its own basename. That makes
    `--root .../mcl1_rep1` work without a symlink shim. The glob is tried first, so a tree that
    has both a root-level `.cache/` and replicate dirs still resolves to the replicates.
    """
    tasks = []
    rep_dirs = sorted(d for d in glob.glob(os.path.join(root, rep_glob)) if os.path.isdir(d))
    if not rep_dirs and os.path.isdir(os.path.join(root, ".cache")):
        rep_dirs = [os.path.normpath(root)]
    for rep_dir in rep_dirs:
        rep = os.path.basename(rep_dir)
        if reps and rep not in reps:
            continue
        for sys_dir in sorted(glob.glob(os.path.join(rep_dir, ".cache", "*"))):
            if not os.path.isdir(sys_dir):
                continue
            system = os.path.basename(sys_dir)
            # substring match, so `--systems cb7_adn` selects `cb7_adn_ligand_1_Hmass`
            if systems and not any(pat in system for pat in systems):
                continue
            for leg in LEGS:
                if leg == "complex":
                    hits = [os.path.join(sys_dir, f"{system}_formatted.h5")]
                else:
                    hits = sorted(glob.glob(os.path.join(sys_dir, f"{leg}_*_formatted.h5")))
                hits = [h for h in hits if os.path.exists(h)]
                if not hits:
                    continue
                if len(hits) > 1:
                    print(f"[warn] {system}/{rep}/{leg}: {len(hits)} candidates, using {hits[0]}")
                formatted = hits[0]
                fe = formatted.replace("_formatted.h5", "_fe.h5")
                tasks.append((system, rep, leg, formatted, fe if os.path.exists(fe) else None))
    return tasks


# --------------------------------------------------------------------------------------------------
# The two estimators
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# Solver protocols
# --------------------------------------------------------------------------------------------------
# pymbar's DEFAULT_SOLVER_PROTOCOL (mbar_solvers.py:97) is (hybr, adaptive). On this data `hybr`
# fails on essentially every solve and falls through to `adaptive`, which then runs its full 10,000
# iterations and STILL reports `max_delta = 0.0` -- a stalled solver, not a slow one. That is both
# the runtime blowup (10,000 sweeps over a 39 x 390,000 matrix) and the reason the reported gradient
# norms reach ~3.5e4. Skipping `hybr` and capping the iteration budget fixes both.
BAR_PROTOCOL = (dict(method="adaptive", options=dict(min_sc_iter=0, maxiter=200)),)
MBAR_PROTOCOL = (
    dict(method="adaptive", options=dict(min_sc_iter=0, maxiter=1000)),
    dict(method="L-BFGS-B", options=dict(maxiter=1000)),
)


def _N_k(df):
    """Samples per state, aligned to `df.columns` — identical to `pandasmbar._mbar`
    (pandasmbar.py:251-271), so the solve stays faithful to production apart from the protocol."""
    N_k = df.groupby(df.index.names).count().iloc[:, [0]]
    N_k.index = N_k.index.tolist()
    N_k = pd.merge(
        pd.DataFrame(index=df.columns), N_k, left_index=True, right_index=True, how="outer"
    )
    return N_k.fillna(0).reindex(df.columns).values.flatten()


def _solve(df, protocol, initial_f_k=None):
    """`pandasmbar._mbar` with an explicit solver protocol and optional warm start.

    Needed because `pandasmbar.mbar()` accepts `solver_protocol` but never passes it on
    (pandasmbar.py:82 — a dead parameter). Also returns the final gradient norm: MBAR's solution is
    unique (the objective is strictly convex), so a warm start cannot bias a CONVERGED solve — but it
    absolutely can leak into an unconverged one, so the caller must be able to check.

    Returns (free_energies, errors, MBAR, max_abs_gradient), free energies in kT.
    """
    import pymbar
    from pymbar import mbar_solvers

    u_kn = df.values.T
    N_k = _N_k(df)
    mbar = pymbar.MBAR(u_kn, N_k, initial_f_k=initial_f_k, solver_protocol=protocol)
    results = mbar.compute_free_energy_differences()

    try:
        index = pd.MultiIndex.from_tuples(df.columns, names=df.index.names)
    except Exception:
        index = pd.Index(df.columns, name=df.index.name)
    free_energies = pd.DataFrame(results["Delta_f"], columns=df.columns.values, index=index)
    errors = pd.DataFrame(results["dDelta_f"], columns=df.columns.values, index=index)

    try:
        grad = float(np.max(np.abs(mbar_solvers.mbar_gradient(u_kn, N_k, mbar.f_k))))
    except Exception:
        grad = float("nan")
    return free_energies, errors, mbar, grad


def two_state_bar(df_subsampled, state_i, state_j, pdmbar, solver="fast"):
    """BAR between two adjacent states, as a 2-column / 2-state MBAR.

    A 2-state MBAR solves the Bennett acceptance-ratio equation exactly, so this IS BAR -- both the
    estimate and its analytic error. Slicing follows `compute_mbar._ordered_band`
    (adaptive_restraints.py:666): take the two COLUMNS *and* filter the ROWS to frames sampled at
    those two states. The row filter is mandatory -- keeping other trajectories' frames while
    dropping their columns breaks pymbar's `sum(N_k) == n_samples` invariant, and it is what makes
    each BAR a (2 x n_frames) x 2 problem instead of the full (K x n_frames) x K one.

    Returns (dG, err, gradient_norm) in kT, for state_i -> state_j.
    """
    pair = [state_i, state_j]
    block = df_subsampled[pair]
    block = block.loc[block.index.isin(pair)]
    if solver == "production":
        free_energies, errors, _ = pdmbar.mbar(block)
        grad = float("nan")
    else:
        free_energies, errors, _, grad = _solve(block, BAR_PROTOCOL)
    # rows and columns are both `pair`, in order, so [0, 1] is f_j - f_i
    return float(free_energies.values[0, 1]), float(errors.values[0, 1]), grad


def analyze_leg(formatted_h5, fe_h5, temperature, repo, fe_tol=0.01, verbose=True,
                equil_nskip=1, stride=1, solver="fast", save_overlap=(None, None)):
    """Full MBAR vs adjacent-pair BAR chain for one leg. Returns (summary_dict, pairs_DataFrame).

    `equil_nskip` and `stride` are the two speed knobs. BOTH change which frames enter the
    calculation, but they change them IDENTICALLY for the two estimators (the BAR slices are taken
    from the same `df_sub` the full MBAR is solved on), so neither can bias the BAR-vs-MBAR
    difference this script measures. They DO move the absolute DeltaG away from the production
    value, so the `*_fe.h5` self-check is only meaningful at the defaults (1, 1).
    """
    pdmbar = _import_package(repo)
    kT = kcals_per_kt(temperature)

    df = pd.read_hdf(formatted_h5) / kT  # stored in kcal/mol -> reduced units
    states = list(df.columns)
    n = len(states)
    if n < 2:
        raise RuntimeError(f"{formatted_h5}: only {n} state(s)")

    if stride > 1:
        # thin every state's trajectory by the same factor, preserving the per-state row blocks
        df = df.groupby(df.index, sort=False, group_keys=False).apply(lambda g: g.iloc[::stride])

    # --- production front-end (adaptive_restraints.py:694-706); nskip=1 is the production value ---
    equil_info = pdmbar.detect_equilibration(df, nskip=equil_nskip)
    df_sub = pdmbar.subsample_correlated_data(df, equil_info=equil_info)
    n_frames_raw, n_frames_used = len(df), len(df_sub)
    del df

    # --- TEST first: the BAR chain is cheap (each solve is a (2 x frames) x 2 problem, ~380x less
    # data than the full matrix) AND its cumulative sum is an excellent warm start for MBAR below ---
    rows = []
    for i in range(n - 1):
        dG_kt, e_kt, g = two_state_bar(df_sub, states[i], states[i + 1], pdmbar, solver)
        rows.append(
            {
                "i": i,
                "state_i": "|".join(map(str, states[i])),
                "state_j": "|".join(map(str, states[i + 1])),
                "dG_bar_kt": dG_kt,
                "dG_bar": dG_kt * kT,
                "err_bar": e_kt * kT,
                "bar_grad": g,
            }
        )
    pairs = pd.DataFrame(rows)
    dG_bar = float(pairs["dG_bar"].sum())
    err_bar = float(np.sqrt((pairs["err_bar"] ** 2).sum()))  # independent-pair assumption

    # --- REFERENCE: full MBAR over all N states ---
    # Warm-started from the BAR chain (gauge f_0 = 0), which starts the solver near the answer
    # instead of several hundred kT away. Safe ONLY because the gradient norm is checked below:
    # MBAR's objective is strictly convex, so a converged solve is the unique optimum regardless of
    # where it started -- but an UNCONVERGED one would inherit the BAR chain and bias this
    # comparison toward agreement. `mbar_grad_norm` is reported so that can never pass silently.
    f_init = np.concatenate([[0.0], np.cumsum(pairs["dG_bar_kt"].to_numpy())])
    if solver == "production":
        fe, err, mbar_obj = pdmbar.mbar(df_sub)
        mbar_grad = float("nan")
    else:
        fe, err, mbar_obj, mbar_grad = _solve(df_sub, MBAR_PROTOCOL, initial_f_k=f_init)
    fe_v, err_v = fe.values, err.values
    dG_mbar = float(fe_v[0, -1]) * kT
    err_mbar = float(err_v[0, -1]) * kT
    overlap = np.asarray(mbar_obj.compute_overlap()["matrix"])

    # optionally persist the FULL N x N overlap matrix (the CSVs only carry the superdiagonal).
    # Rows/cols are in the same cycle order as `states`, per pandasmbar._mbar feeding df.values.T
    # straight to pymbar (postTreatment.py:249-255 documents the same guarantee).
    ov_dir, ov_tag = save_overlap
    if ov_dir:
        os.makedirs(ov_dir, exist_ok=True)
        np.savez_compressed(os.path.join(ov_dir, f"{ov_tag}.overlap.npz"), overlap=overlap,
                            states=np.array(["|".join(map(str, s)) for s in states]))

    # --- self-check against the stored MBAR result ---
    dG_stored, fe_check = float("nan"), float("nan")
    if fe_h5:
        stored = pd.read_hdf(fe_h5)
        dG_stored = float(stored.values[0, -1])
        fe_check = abs(dG_mbar - dG_stored)
        if verbose and not (fe_check < fe_tol) and equil_nskip == 1 and stride == 1:
            note = ("production's stored value came from a NON-converged solve, so a mismatch here "
                    "is expected and is itself the finding" if solver != "production"
                    else "wrong --temperature? (try --calibrate)")
            print(f"[note] {os.path.basename(formatted_h5)}: recomputed MBAR {dG_mbar:+.4f} vs "
                  f"stored _fe.h5 {dG_stored:+.4f} (diff {fe_check:.4f} > {fe_tol}) -- {note}")

    pairs["dG_mbar_pair"] = [float(fe_v[i, i + 1]) * kT for i in range(n - 1)]
    pairs["err_mbar_pair"] = [float(err_v[i, i + 1]) * kT for i in range(n - 1)]
    pairs["overlap_fwd"] = [float(overlap[i][i + 1]) for i in range(n - 1)]
    pairs["overlap_rev"] = [float(overlap[i + 1][i]) for i in range(n - 1)]
    pairs["overlap_min"] = pairs[["overlap_fwd", "overlap_rev"]].min(axis=1)
    pairs["abs_diff"] = (pairs["dG_bar"] - pairs["dG_mbar_pair"]).abs()
    pairs = pairs.drop(columns=["dG_bar_kt"])

    summary = {
        "n_states": n,
        "n_frames_raw": n_frames_raw,
        "n_frames_used": n_frames_used,
        "dG_mbar": dG_mbar,
        "err_mbar": err_mbar,
        "dG_bar": dG_bar,
        "err_bar": err_bar,
        "diff": dG_bar - dG_mbar,
        "abs_diff": abs(dG_bar - dG_mbar),
        "dG_stored_fe_h5": dG_stored,
        "fe_selfcheck_diff": fe_check,
        "mbar_grad_norm": mbar_grad,          # must be small or the warm start is not trustworthy
        "bar_grad_norm_max": float(pairs["bar_grad"].max()),
        "min_overlap": float(pairs["overlap_min"].min()),
        "worst_pair_i": int(pairs["abs_diff"].idxmax()),
        "worst_pair_abs_diff": float(pairs["abs_diff"].max()),
        # cost: MBAR needs the full N^2 of evaluated blocks, adjacent-BAR only the tridiagonal
        "n_evals_mbar": n * n,
        "n_evals_bar": 3 * n - 2,
        "eval_speedup": (n * n) / (3 * n - 2),
    }
    return summary, pairs


# --------------------------------------------------------------------------------------------------
# Task driver (cached, resumable)
# --------------------------------------------------------------------------------------------------
def _task_paths(scratch, system, rep, leg):
    tag = f"{system}__{rep}__{leg}"
    return (os.path.join(scratch, f"{tag}.summary.csv"), os.path.join(scratch, f"{tag}.pairs.csv"))


def run_task(task, temperature, repo, scratch, fe_tol, force, equil_nskip=1, stride=1,
             solver="fast", save_overlap=None):
    system, rep, leg, formatted, fe_h5 = task
    s_path, p_path = _task_paths(scratch, system, rep, leg)
    if not force and os.path.exists(s_path) and os.path.exists(p_path):
        return system, rep, leg, "cached", None

    try:
        summary, pairs = analyze_leg(formatted, fe_h5, temperature, repo, fe_tol=fe_tol,
                                     equil_nskip=equil_nskip, stride=stride, solver=solver,
                                     save_overlap=(save_overlap, f"{system}__{rep}__{leg}"))
    except Exception:
        return system, rep, leg, "FAILED", traceback.format_exc(limit=6)

    keys = {"system": system, "rep": rep, "leg": leg,
            "equil_nskip": equil_nskip, "stride": stride, "solver": solver}
    for frame, path in ((pd.DataFrame([{**keys, **summary}]), s_path),
                        (pairs.assign(**keys), p_path)):
        tmp = f"{path}.{os.getpid()}.tmp"
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)  # atomic, so a killed run never leaves a half-written cache entry
    return system, rep, leg, f"dG_mbar={summary['dG_mbar']:+.3f} dG_bar={summary['dG_bar']:+.3f} " \
                            f"diff={summary['diff']:+.3f}", None


def consolidate(scratch, out_prefix):
    """Concatenate the per-task caches; add a per-(system, rep) 3-leg total row."""
    summaries = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(scratch, "*.summary.csv")))]
    pairs = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(scratch, "*.pairs.csv")))]
    if not summaries:
        print("[bar_vs_mbar] nothing to consolidate")
        return None, None
    summary = pd.concat(summaries, ignore_index=True)
    pairs = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()

    # 3-leg total. Boresch + flat-bottom are identical under either estimator (analytic / separate
    # leg with no _formatted.h5), so they cancel in `diff` and are simply omitted here.
    totals = []
    for (system, rep), grp in summary.groupby(["system", "rep"]):
        if set(grp["leg"]) != set(LEGS):
            continue
        totals.append(
            {
                "system": system,
                "rep": rep,
                "leg": "TOTAL_3LEG",
                "n_states": int(grp["n_states"].sum()),
                "dG_mbar": grp["dG_mbar"].sum(),
                "err_mbar": math.sqrt((grp["err_mbar"] ** 2).sum()),
                "dG_bar": grp["dG_bar"].sum(),
                "err_bar": math.sqrt((grp["err_bar"] ** 2).sum()),
                "diff": grp["dG_bar"].sum() - grp["dG_mbar"].sum(),
                "abs_diff": abs(grp["dG_bar"].sum() - grp["dG_mbar"].sum()),
                "min_overlap": grp["min_overlap"].min(),
                "n_evals_mbar": int(grp["n_evals_mbar"].sum()),
                "n_evals_bar": int(grp["n_evals_bar"].sum()),
                "eval_speedup": grp["n_evals_mbar"].sum() / grp["n_evals_bar"].sum(),
            }
        )
    if totals:
        summary = pd.concat([summary, pd.DataFrame(totals)], ignore_index=True)

    s_out, p_out = f"{out_prefix}_summary.csv", f"{out_prefix}_pairs.csv"
    summary.to_csv(s_out, index=False)
    if len(pairs):
        pairs.to_csv(p_out, index=False)
    print(f"[bar_vs_mbar] wrote {s_out} ({len(summary)} rows) and {p_out} ({len(pairs)} rows)")
    return summary, pairs


def report(summary):
    """Headline numbers: accuracy cost per leg and the evaluation-count saving."""
    if summary is None or not len(summary):
        return
    print("\n=== BAR(adjacent) - MBAR(all windows), kcal/mol ===")
    print(f"{'leg':<12} {'n':>4} {'mean diff':>10} {'RMSE':>8} {'max |diff|':>11} "
          f"{'min overlap':>12} {'evals MBAR->BAR':>18}")
    for leg in list(LEGS) + ["TOTAL_3LEG"]:
        grp = summary[summary["leg"] == leg]
        if not len(grp):
            continue
        rmse = float(np.sqrt((grp["diff"] ** 2).mean()))
        print(f"{leg:<12} {len(grp):>4} {grp['diff'].mean():>+10.3f} {rmse:>8.3f} "
              f"{grp['abs_diff'].max():>11.3f} {grp['min_overlap'].min():>12.4f} "
              f"{int(grp['n_evals_mbar'].mean()):>8d} -> {int(grp['n_evals_bar'].mean()):<6d}")

    bad = summary[summary["fe_selfcheck_diff"] > 0.01] if "fe_selfcheck_diff" in summary else []
    if len(bad):
        print(f"\n[warn] {len(bad)} leg(s) did NOT reproduce the stored _fe.h5 within 0.01 kcal/mol "
              f"-- the temperature is probably wrong; run --calibrate.")


# --------------------------------------------------------------------------------------------------
# Temperature calibration
# --------------------------------------------------------------------------------------------------
def calibrate(tasks, repo, candidates=CANDIDATE_TEMPERATURES):
    """Find the temperature that reproduces the stored `*_fe.h5`, using the smallest leg available.

    kcals_per_Kt is not a free scale factor -- MBAR is solved on u = E/kT, so the wrong T gives a
    different answer, not just a rescaled one. This finds the T production actually used.
    """
    with_fe = [t for t in tasks if t[4]]
    if not with_fe:
        raise SystemExit("[calibrate] no *_fe.h5 found to calibrate against")
    # ligand legs are the smallest/fastest
    task = sorted(with_fe, key=lambda t: (t[2] != "ligand", os.path.getsize(t[3])))[0]
    system, rep, leg, formatted, fe_h5 = task
    print(f"[calibrate] using {system}/{rep}/{leg}: {os.path.basename(formatted)}")
    stored = float(pd.read_hdf(fe_h5).values[0, -1])
    print(f"[calibrate] stored _fe.h5 DeltaG = {stored:+.4f} kcal/mol")

    best = None
    for T in candidates:
        try:
            summary, _ = analyze_leg(formatted, fe_h5, T, repo, verbose=False)
        except Exception as exc:
            print(f"[calibrate]   T={T:<7} FAILED: {exc}")
            continue
        diff = summary["fe_selfcheck_diff"]
        print(f"[calibrate]   T={T:<7} recomputed {summary['dG_mbar']:+.4f}  |diff| {diff:.5f}")
        if best is None or diff < best[1]:
            best = (T, diff)
    if best:
        print(f"[calibrate] BEST: --temperature {best[0]}  (|diff| {best[1]:.5f} kcal/mol)")
    return best


# --------------------------------------------------------------------------------------------------
# Self-test: verify the estimators WITHOUT any data
# --------------------------------------------------------------------------------------------------
def selftest(repo):
    """Three harmonic states with a common partition function -- every DeltaG is exactly 0.

    u_k(x) = (x - m_k)^2 / 2, sampled x ~ N(m_k, 1). Shifting a harmonic well does not change Z, so
    f_k is identical for all k. That makes the true answer 0 for BOTH estimators, and the chain must
    telescope. Also checks the trivial rigid-shift case where the answer is a known constant.
    """
    pdmbar = _import_package(repo)
    rng = np.random.default_rng(0)
    names = ["traj_state", "traj_extdiel", "traj_charge", "traj_restraints"]

    def build(means, n_per_state):
        states = [(f"s{k}", "78.5", "1.0", f"{m}") for k, m in enumerate(means)]
        idx, block = [], []
        for k, m in enumerate(means):
            x = rng.normal(m, 1.0, n_per_state)
            block.append(np.column_stack([0.5 * (x - mj) ** 2 for mj in means]))
            idx.extend([states[k]] * n_per_state)
        df = pd.DataFrame(np.vstack(block), columns=states)
        df.index = pd.MultiIndex.from_tuples(idx, names=names)
        return df, states

    # (a) 3 overlapping harmonic states -- every free-energy difference is exactly 0
    df, states = build([0.0, 0.6, 1.2], 20000)
    fe, _, mbar_obj = pdmbar.mbar(df)
    dG_mbar = float(fe.values[0, -1])
    chain = sum(two_state_bar(df, states[i], states[i + 1], pdmbar)[0] for i in range(len(states) - 1))
    print(f"[selftest] harmonic 3-state: MBAR f2-f0 = {dG_mbar:+.4f} kT, "
          f"BAR chain = {chain:+.4f} kT (both should be 0)")
    assert abs(dG_mbar) < 0.02, f"MBAR {dG_mbar}"
    assert abs(chain) < 0.02, f"BAR chain {chain}"
    assert abs(chain - dG_mbar) < 0.02, "chain does not telescope to the MBAR endpoint difference"

    # (b) rigid shift: u_1 = u_0 + c => dG = c exactly, and 2-state MBAR must recover it
    c = 1.7
    n = 20000
    u0 = rng.normal(0, 1, 2 * n)
    two = [("a", "78.5", "1.0", "0"), ("b", "78.5", "1.0", "0")]
    # build exactly like (a) -- a dict keyed by tuples yields a different column-index type
    df2 = pd.DataFrame(np.column_stack([u0, u0 + c]), columns=two)
    df2.index = pd.MultiIndex.from_tuples([two[0]] * n + [two[1]] * n, names=names)
    dG, _, _ = two_state_bar(df2, two[0], two[1], pdmbar)
    print(f"[selftest] rigid shift: 2-state MBAR (== BAR) = {dG:+.4f} kT (expect {c})")
    assert abs(dG - c) < 0.02, f"{dG} != {c}"

    # (c) unit conversion matches the package
    assert abs(kcals_per_kt(298.0) - 0.5921) < 1e-3, kcals_per_kt(298.0)
    print(f"[selftest] kcals_per_kt(298) = {kcals_per_kt(298.0):.4f} kcal/mol")
    print("[selftest] all checks passed")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", help="run directory holding rep1..repN with .cache/<system>/")
    ap.add_argument("--systems", nargs="*", help="restrict to these system names (default: all)")
    ap.add_argument("--reps", nargs="*", help="restrict to these rep dirs, e.g. rep1 rep2")
    ap.add_argument("--rep-glob", default="rep*",
                    help="glob for replicate dirs under --root (default: rep*). Ignored when "
                         "--root itself holds a .cache/")
    ap.add_argument("--temperature", type=float, default=298.0,
                    help="temperature used by the production run (default 298.0); "
                         "verify with --calibrate")
    ap.add_argument("--repo",
                    default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="path to the implicit_solvent_ddm checkout (default: this script's PARENT "
                         "dir, since the script lives in scripts/)")
    ap.add_argument("--out-prefix", default="bar_vs_mbar", help="output CSV prefix")
    ap.add_argument("--scratch", default=None,
                    help="per-task result cache (default <out-prefix>_scratch); reused on re-run")
    ap.add_argument("--nproc", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--fe-tol", type=float, default=0.01,
                    help="tolerance (kcal/mol) for reproducing the stored _fe.h5")
    ap.add_argument("--save-overlap", default=None, metavar="DIR",
                    help="also write each leg's FULL N x N MBAR overlap matrix to DIR as .npz "
                         "(the CSVs only carry the adjacent-pair superdiagonal)")
    ap.add_argument("--solver", choices=("fast", "production"), default="fast",
                    help="'fast' (default): skip pymbar's failing `hybr` stage, cap the adaptive "
                         "iterations, and warm-start the full MBAR from the BAR chain, with a "
                         "gradient-norm check. 'production': go through pandasmbar.mbar() exactly as "
                         "the workflow does -- needed to reproduce the stored _fe.h5, but it stalls "
                         "for 10,000 iterations per solve.")
    ap.add_argument("--equil-nskip", type=int, default=1,
                    help="stride for pymbar's equilibration-start scan (production uses 1). This is "
                         "the run's bottleneck: nskip=1 tests EVERY frame as a candidate t0, per "
                         "state. 10 is ~10x cheaper and affects both estimators identically.")
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth frame of each trajectory before subsampling. Cuts BOTH the "
                         "equilibration scan and the MBAR solve. Affects both estimators identically, "
                         "so it cannot bias the BAR-vs-MBAR difference, but it does move the absolute "
                         "DeltaG off the production value (the _fe.h5 self-check only holds at 1).")
    ap.add_argument("--force", action="store_true", help="recompute, ignoring the cache")
    ap.add_argument("--calibrate", action="store_true",
                    help="find the temperature reproducing the stored _fe.h5, then exit")
    ap.add_argument("--consolidate-only", action="store_true",
                    help="just rebuild the CSVs from the existing cache")
    ap.add_argument("--selftest", action="store_true", help="verify the estimators, no data needed")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.repo)
        return
    if not args.root and not args.consolidate_only:
        ap.error("provide --root (or --selftest / --consolidate-only)")

    scratch = args.scratch or f"{args.out_prefix}_scratch"
    os.makedirs(scratch, exist_ok=True)

    if args.consolidate_only:
        report(consolidate(scratch, args.out_prefix)[0])
        return

    tasks = find_tasks(args.root, args.systems, args.reps, args.rep_glob)
    if not tasks:
        raise SystemExit(f"[bar_vs_mbar] no cached legs found under {args.root}")

    if args.calibrate:
        calibrate(tasks, args.repo)
        return

    print(f"[bar_vs_mbar] {len(tasks)} legs, T={args.temperature} K, {args.nproc} worker(s), "
          f"solver={args.solver}, equil_nskip={args.equil_nskip}, stride={args.stride}, "
          f"scratch {scratch}")
    if args.equil_nskip != 1 or args.stride != 1:
        print("[bar_vs_mbar] NOTE: non-default equil-nskip/stride -- the BAR-vs-MBAR difference is "
              "still fair (both estimators see the same frames), but absolute DeltaG will drift off "
              "the production value and the _fe.h5 self-check will not hold.")

    # biggest legs first: the complex leg dominates, so starting it early avoids a long tail
    tasks = sorted(tasks, key=lambda t: -os.path.getsize(t[3]))
    args_common = (args.temperature, args.repo, scratch, args.fe_tol, args.force,
                   args.equil_nskip, args.stride, args.solver, args.save_overlap)
    if args.nproc > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.nproc) as ex:
            futures = [ex.submit(run_task, t, *args_common) for t in tasks]
            for fut in concurrent.futures.as_completed(futures):
                system, rep, leg, status, err = fut.result()
                print(f"[bar_vs_mbar] {system}/{rep}/{leg}: {status}", flush=True)
                if err:
                    print(err)
    else:
        for t in tasks:
            system, rep, leg, status, err = run_task(t, *args_common)
            print(f"[bar_vs_mbar] {system}/{rep}/{leg}: {status}", flush=True)
            if err:
                print(err)

    report(consolidate(scratch, args.out_prefix)[0])


if __name__ == "__main__":
    main()
