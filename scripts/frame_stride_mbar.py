#!/usr/bin/env python
"""frame_stride_mbar.py -- how much does DeltaG move when post-analysis scores FEWER FRAMES?

Why
---
Each `sander imin=5` post-analysis job re-scores EVERY frame of its source trajectory: production
writes `ntwx=250` over 2.5M steps, so that is 10,000 frames per job, and the full N^2 matrix pays
that cost 10,000 times over. Measured on MCL-1/ligand-1: 4,344 s per job, 4,208 CPU-h per replicate,
23.4 h on 180 cores (see research/STATUS.md).

But MBAR does not USE 10,000 frames. `subsample_correlated_data` runs AFTER the expensive pass and
throws away everything inside the correlation time, and standard practice targets only ~100-500
UNCORRELATED samples per state. If striding the trajectory to 500 frames leaves DeltaG unchanged,
the post-analysis cost drops ~20x -- multiplicatively with any N^2 pruning (see bar_vs_mbar.py).

This script measures that directly, using ONLY the already-computed `.cache/<system>/*_formatted.h5`
matrices. No MD, no sander.

Method
------
This is an MBAR-vs-MBAR comparison. The ONLY difference between arms is how many frames enter.

Per (system, replica, leg), the `*_formatted.h5` is read ONCE (they are ~110 MB each and often sit
on a network mount, so re-reading per target would dominate the runtime), then for each target frame
count:

  1. STRIDE the raw matrix to ~`target` frames per state -- `df.iloc[::stride]` within each sampled
     state, so every state is thinned by the same factor and the per-state row blocks are preserved.
     This is what changing production `ntwx` (or a cpptraj pre-stride) would actually produce.
  2. run the FULL production front-end on the thinned data -- `pdmbar.detect_equilibration` then
     `pdmbar.subsample_correlated_data`, verbatim from `compute_mbar` (adaptive_restraints.py:694).
     This must be re-run per arm: equilibration detection and the statistical inefficiency `g` are
     properties of the thinned series, not of the original.
  3. solve the FULL MBAR over all N states -> DeltaG = f(last) - f(first).

The `target=<n_frames_raw>` arm (default 10000) is the reference: it is the full data set, so it
reproduces production. `consolidate` reports every other arm as a delta against it.

Because `detect_equilibration` also returns the statistical inefficiency, this answers the "what
stride is actually justified?" question as a side effect: `mean_g` is the correlation length in
frames and `mean_Neff` is how many uncorrelated samples MBAR really had. If the full-data arm shows
mean_Neff ~ 500, then striding to 500 frames is provably free.

Solver
------
Uses `bar_vs_mbar.MBAR_PROTOCOL`, NOT pymbar's default. pymbar's `DEFAULT_SOLVER_PROTOCOL` is
(hybr, adaptive); `hybr` fails on this data and `adaptive` then stalls for its full 10,000 iterations
while reporting `max_delta = 0.0`, yielding gradient norms ~3.5e4 (see research/STATUS.md). Every row
carries `mbar_grad_norm`; check it before believing any DeltaG. `--warm-start` initializes from the
arm's OWN adjacent-pair BAR chain -- identical treatment across arms, and MBAR's objective is
strictly convex so a CONVERGED solve is the unique optimum regardless of where it started. The
gradient norm is what makes that safe; do not silence it.

Usage
-----
  conda activate isddm_env                       # pymbar + pytables + pandas

  # the headline run: 500 frames vs all 10,000, every replica, one system
  python frame_stride_mbar.py --root <run_dir> --systems cb7_adn --frames 10000 500

  # full sweep
  python frame_stride_mbar.py --root <run_dir> --frames 10000 5000 1000 500 --nproc 8

where <run_dir> holds rep1..rep5, each with `.cache/<system>/*_formatted.h5`, e.g.
  ~/titan_ssh/different_gbs/igb_2/cb7_re_parameterization

Outputs (written next to --out-prefix, default ./frame_stride):
  *_legs.csv      one row per (system, rep, leg, target_frames)
  *_totals.csv    3-leg total per (system, rep, target_frames), with the delta vs the reference arm
  *_pivot.csv     systems x frame-target table of total DeltaG, for eyeballing

Per-task results are cached under --scratch, so a dropped mount or a partial run resumes.

NOTE on the 3-leg total: Boresch + flat-bottom contributions are analytic / carried in separate files
with no `_formatted.h5`, and neither depends on the frame count, so they cancel in every delta this
script reports and are omitted -- exactly as `bar_vs_mbar.consolidate` omits them.
"""
import argparse
import concurrent.futures
import glob
import math
import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_vs_mbar as bvm  # noqa: E402  -- reuse discovery, solver protocol and constants

DEFAULT_TARGETS = (10000, 5000, 1000, 500)


# --------------------------------------------------------------------------------------------------
# Striding
# --------------------------------------------------------------------------------------------------
def stride_for(n_per_state, target):
    """Stride that leaves >= `target` frames per state (floor, so we never undershoot).

    n=10000 -> target 5000/1000/500 gives stride 2/10/20 and exactly 5000/1000/500 frames.
    """
    if not target or target >= n_per_state:
        return 1
    return max(1, int(n_per_state // target))


def thin(df, stride):
    """Thin every sampled state's trajectory by the same factor, preserving per-state row blocks.

    Identical to the `--stride` knob in `bar_vs_mbar.analyze_leg`. Grouping by `df.index` (not a
    positional slice of the whole frame) is mandatory: the rows are stacked per sampled state, so a
    global `[::stride]` would thin states unevenly and break pymbar's `sum(N_k) == n_samples`.
    """
    if stride <= 1:
        return df
    return df.groupby(df.index, sort=False, group_keys=False).apply(lambda g: g.iloc[::stride])


# --------------------------------------------------------------------------------------------------
# One leg, all frame targets
# --------------------------------------------------------------------------------------------------
def analyze_leg_frames(formatted_h5, temperature, repo, targets, warm_start=False, verbose=True):
    """Read one `*_formatted.h5` once; return one summary row per frame target."""
    pdmbar = bvm._import_package(repo)
    kT = bvm.kcals_per_kt(temperature)

    df_full = pd.read_hdf(formatted_h5) / kT  # stored kcal/mol -> reduced units
    states = list(df_full.columns)
    n_states = len(states)
    if n_states < 2:
        raise RuntimeError(f"{formatted_h5}: only {n_states} state(s)")
    per_state = df_full.groupby(df_full.index).size()
    n_per_state = int(per_state.min())

    rows = []
    for target in targets:
        stride = stride_for(n_per_state, target)
        df = thin(df_full, stride)

        # --- production front-end, re-run per arm (g is a property of the THINNED series) ---
        equil = pdmbar.detect_equilibration(df, nskip=1)
        df_sub = pdmbar.subsample_correlated_data(df, equil_info=equil)
        n_raw, n_used = len(df), len(df_sub)

        f_init = None
        if warm_start:
            chain = [0.0]
            for i in range(n_states - 1):
                pair = [states[i], states[i + 1]]
                block = df_sub[pair]
                block = block.loc[block.index.isin(pair)]
                fe_p, _, _, _ = bvm._solve(block, bvm.BAR_PROTOCOL)
                chain.append(chain[-1] + float(fe_p.values[0, 1]))
            f_init = np.asarray(chain)

        fe, err, _mbar, grad = bvm._solve(df_sub, bvm.MBAR_PROTOCOL, initial_f_k=f_init)
        dG = float(fe.values[0, -1]) * kT
        dG_err = float(err.values[0, -1]) * kT

        rows.append(
            {
                "target_frames": int(target),
                "stride": int(stride),
                "n_per_state_raw": n_per_state,
                "n_frames_raw": int(n_raw),
                "n_frames_used": int(n_used),
                "n_states": n_states,
                # the diagnostic that says what stride is justified
                "mean_g": float(equil["g"].mean()),
                "median_g": float(equil["g"].median()),
                "max_g": float(equil["g"].max()),
                "mean_Neff": float(equil["Neff_max"].mean()),
                "min_Neff": float(equil["Neff_max"].min()),
                "mean_t0": float(equil["t"].mean()),
                "dG": dG,
                "err": dG_err,
                "mbar_grad_norm": grad,
            }
        )
        if verbose:
            print(
                f"    [{os.path.basename(formatted_h5)}] target={target:>6} stride={stride:>3} "
                f"raw={n_raw:>7} used={n_used:>6} g={rows[-1]['mean_g']:6.1f} "
                f"Neff={rows[-1]['mean_Neff']:8.1f} dG={dG:+9.4f} grad={grad:.1e}",
                flush=True,
            )
    return rows


def _task_tag(system, rep, leg):
    return f"{system}__{rep}__{leg}"


def run_task(args_tuple):
    """Worker: one (system, rep, leg). Cached so a dropped mount resumes."""
    system, rep, leg, formatted, _fe_h5, opts = args_tuple
    tag = _task_tag(system, rep, leg)
    cache = os.path.join(opts["scratch"], f"{tag}.legs.csv")
    if os.path.exists(cache) and not opts["force"]:
        return tag, "cached", None
    try:
        rows = analyze_leg_frames(
            formatted,
            opts["temperature"],
            opts["repo"],
            opts["targets"],
            warm_start=opts["warm_start"],
            verbose=opts["verbose"],
        )
        out = pd.DataFrame(rows)
        out.insert(0, "leg", leg)
        out.insert(0, "rep", rep)
        out.insert(0, "system", system)
        os.makedirs(opts["scratch"], exist_ok=True)
        out.to_csv(cache, index=False)
        return tag, "ok", None
    except Exception:
        return tag, "FAILED", traceback.format_exc()


# --------------------------------------------------------------------------------------------------
# Consolidation
# --------------------------------------------------------------------------------------------------
def consolidate(scratch, out_prefix, reference):
    """Concatenate per-leg caches; build 3-leg totals and deltas vs the reference frame count."""
    files = sorted(glob.glob(os.path.join(scratch, "*.legs.csv")))
    if not files:
        print("[frame_stride] nothing to consolidate")
        return None, None
    legs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    totals = []
    for (system, rep, target), grp in legs.groupby(["system", "rep", "target_frames"]):
        if set(grp["leg"]) != set(bvm.LEGS):
            continue  # incomplete: do not report a partial 3-leg sum
        totals.append(
            {
                "system": system,
                "rep": rep,
                "target_frames": int(target),
                "stride": int(grp["stride"].max()),
                "n_frames_used": int(grp["n_frames_used"].sum()),
                "mean_g": float(grp["mean_g"].mean()),
                "mean_Neff": float(grp["mean_Neff"].mean()),
                "dG_total": float(grp["dG"].sum()),
                "err_total": math.sqrt(float((grp["err"] ** 2).sum())),
                "max_grad": float(grp["mbar_grad_norm"].max()),
            }
        )
    totals = pd.DataFrame(totals)

    if not totals.empty:
        ref = (
            totals[totals["target_frames"] == reference]
            .set_index(["system", "rep"])["dG_total"]
            .to_dict()
        )
        totals["dG_reference"] = [ref.get((s, r), np.nan) for s, r in
                                  zip(totals["system"], totals["rep"])]
        totals["delta_vs_reference"] = totals["dG_total"] - totals["dG_reference"]
        totals["abs_delta"] = totals["delta_vs_reference"].abs()
        totals = totals.sort_values(["system", "rep", "target_frames"], ascending=[1, 1, 0])

    l_out, t_out = f"{out_prefix}_legs.csv", f"{out_prefix}_totals.csv"
    legs.to_csv(l_out, index=False)
    totals.to_csv(t_out, index=False)

    pivot = None
    if not totals.empty:
        pivot = totals.pivot_table(
            index=["system", "rep"], columns="target_frames", values="dG_total"
        )
        pivot.to_csv(f"{out_prefix}_pivot.csv")

    # ---- report ----
    print(f"\n[frame_stride] wrote {l_out}, {t_out}, {out_prefix}_pivot.csv")
    if not totals.empty:
        print(f"\n=== 3-leg total DeltaG (kcal/mol) vs the {reference}-frame reference ===")
        for target, grp in totals.groupby("target_frames", sort=False):
            if target == reference:
                continue
            d = grp["delta_vs_reference"].dropna()
            if d.empty:
                continue
            print(
                f"  {int(target):>6} frames (stride {int(grp['stride'].max()):>3}, "
                f"{len(d)} system-reps): "
                f"mean {d.mean():+.4f}  RMSE {np.sqrt((d ** 2).mean()):.4f}  "
                f"MAE {d.abs().mean():.4f}  max|d| {d.abs().max():.4f}"
            )
        ref_rows = totals[totals["target_frames"] == reference]
        if not ref_rows.empty:
            print(
                f"\n  reference arm: mean g = {ref_rows['mean_g'].mean():.1f} frames, "
                f"mean Neff = {ref_rows['mean_Neff'].mean():.0f} uncorrelated samples/state"
            )
        bad = totals[totals["max_grad"] > 1e-4]
        if not bad.empty:
            print(f"\n  [WARN] {len(bad)} total(s) have MBAR gradient norm > 1e-4 -- NOT converged:")
            for _, r in bad.iterrows():
                print(f"     {r['system']} {r['rep']} target={int(r['target_frames'])} "
                      f"grad={r['max_grad']:.2e}")
    return legs, totals


# --------------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="MBAR vs MBAR: how much does DeltaG move when you stride the frames?"
    )
    p.add_argument("--root", required=True,
                   help="dir holding rep1..repN, each with .cache/<system>/*_formatted.h5")
    p.add_argument("--frames", type=int, nargs="+", default=list(DEFAULT_TARGETS),
                   help=f"target frames per state (default {' '.join(map(str, DEFAULT_TARGETS))})")
    p.add_argument("--reference", type=int, default=None,
                   help="which frame count is the reference arm (default: the largest in --frames)")
    p.add_argument("--systems", nargs="*", default=None, help="substring filters, e.g. cb7_adn")
    p.add_argument("--reps", nargs="*", default=None, help="e.g. rep1 rep2")
    p.add_argument("--rep-glob", default="rep*",
                   help="glob for replicate dirs under --root (default: rep*). Ignored when "
                        "--root itself holds a .cache/")
    p.add_argument("--legs", nargs="*", default=None, choices=list(bvm.LEGS),
                   help="restrict to these legs (default all three)")
    p.add_argument("--temperature", type=float, default=298.0)
    p.add_argument("--repo", default=None,
                   help="path to the repo root so `implicit_solvent_ddm` imports (default: parent)")
    p.add_argument("--warm-start", action="store_true",
                   help="initialize MBAR from the arm's own adjacent-pair BAR chain (faster; safe "
                        "only because mbar_grad_norm is reported -- check it)")
    p.add_argument("--nproc", type=int, default=1, help="parallel tasks (I/O bound on a network mount)")
    p.add_argument("--scratch", default="./.frame_stride_cache")
    p.add_argument("--out-prefix", default="./frame_stride")
    p.add_argument("--force", action="store_true", help="ignore cached per-task results")
    p.add_argument("--consolidate-only", action="store_true")
    a = p.parse_args()

    targets = sorted(set(a.frames), reverse=True)
    reference = a.reference if a.reference is not None else max(targets)
    if reference not in targets:
        p.error(f"--reference {reference} is not in --frames {targets}")

    repo = a.repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(a.scratch, exist_ok=True)

    if not a.consolidate_only:
        tasks = bvm.find_tasks(a.root, a.systems, a.reps, a.rep_glob)
        if a.legs:
            tasks = [t for t in tasks if t[2] in a.legs]
        if not tasks:
            print(f"[frame_stride] no tasks found under {a.root}")
            return 1
        opts = {
            "temperature": a.temperature,
            "repo": repo,
            "targets": targets,
            "warm_start": a.warm_start,
            "scratch": a.scratch,
            "force": a.force,
            "verbose": a.nproc == 1,
        }
        payload = [(s, r, leg, fmt, fe, opts) for (s, r, leg, fmt, fe) in tasks]
        print(f"[frame_stride] {len(payload)} leg-tasks x {len(targets)} frame targets "
              f"{targets} (reference {reference}), nproc={a.nproc}")

        n_ok = n_cached = n_fail = 0
        if a.nproc > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=a.nproc) as ex:
                for tag, status, tb in ex.map(run_task, payload):
                    n_ok += status == "ok"
                    n_cached += status == "cached"
                    n_fail += status == "FAILED"
                    print(f"  [{status:>6}] {tag}" + (f"\n{tb}" if tb else ""), flush=True)
        else:
            for item in payload:
                tag, status, tb = run_task(item)
                n_ok += status == "ok"
                n_cached += status == "cached"
                n_fail += status == "FAILED"
                print(f"  [{status:>6}] {tag}" + (f"\n{tb}" if tb else ""), flush=True)
        print(f"[frame_stride] {n_ok} ok, {n_cached} cached, {n_fail} failed")

    consolidate(a.scratch, a.out_prefix, reference)
    return 0


if __name__ == "__main__":
    sys.exit(main())
