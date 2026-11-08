#!/usr/bin/env python
"""block_mbar_sweep.py -- how much DeltaG accuracy does a block-chained MBAR cost, per block size?

Runs ``implicit_solvent_ddm.block_mbar.sweep_block_sizes`` over the cached
``.cache/<system>/*_formatted.h5`` matrices: full dense MBAR (the reference) versus a chain of
K-state MBAR blocks, for each K. No MD, no sander.

K=2 is the adjacent-window BAR chain and must reproduce ``bar_vs_mbar.py``.

Usage
-----
  conda activate isddm_env

  python block_mbar_sweep.py --root <run_dir> --systems cb7_adn --blocks 2 3 4
  python block_mbar_sweep.py --root <run_dir> --blocks 2 3 4 5 7 --nproc 8

where <run_dir> holds rep1..repN, each with `.cache/<system>/*_formatted.h5`.

Outputs (next to --out-prefix, default ./block_mbar):
  *_legs.csv    one row per (system, rep, leg, block_size)
  *_totals.csv  3-leg total per (system, rep, block_size) + delta vs the dense reference

Per-task results are cached under --scratch so a dropped mount resumes.
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
import bar_vs_mbar as bvm  # noqa: E402  -- reuse task discovery and the kT constant

DEFAULT_BLOCKS = (2, 3, 4, 5, 7)


def analyze_leg_blocks(formatted_h5, temperature, repo, blocks, align_bands=True, verbose=True):
    """Sweep block sizes for one leg.

    Parameters
    ----------
    formatted_h5 : str
        Path to the cached MBAR matrix.
    temperature : float
        Simulation temperature in Kelvin.
    repo : str
        Repo root, so ``implicit_solvent_ddm`` imports.
    blocks : sequence of int
        Block sizes to evaluate.
    align_bands : bool
        Isolate sub-leg transitions into their own blocks.
    verbose : bool
        Print a line per block size.

    Returns
    -------
    pandas.DataFrame
        One row per block size plus the dense reference, in kcal/mol.
    """
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    from implicit_solvent_ddm import block_mbar as bm
    from implicit_solvent_ddm import pandasmbar as pdmbar

    kT = bvm.kcals_per_kt(temperature)
    df = pd.read_hdf(formatted_h5) / kT
    order = list(df.columns)  # written in canonical cycle order

    equil = pdmbar.detect_equilibration(df, nskip=1)
    sub = pdmbar.subsample_correlated_data(df, equil_info=equil)

    out = bm.sweep_block_sizes(sub, order, list(blocks), align_bands=align_bands)
    out["dG"] = out["dG"] * kT
    out["err"] = out["err"] * kT
    out["delta_vs_full"] = out["delta_vs_full"] * kT
    out["n_frames_used"] = len(sub)
    if verbose:
        for _, r in out.iterrows():
            print(
                f"    [{os.path.basename(formatted_h5)}] {r['label']:>10} "
                f"blocks={int(r['n_blocks']):>3} evals={int(r['n_evaluations']):>5} "
                f"speedup={r['speedup']:5.1f}x dG={r['dG']:+9.4f} d={r['delta_vs_full']:+7.4f}",
                flush=True,
            )
    return out


def run_task(args_tuple):
    """Worker for one (system, rep, leg); caches its result CSV."""
    system, rep, leg, formatted, _fe, opts = args_tuple
    tag = f"{system}__{rep}__{leg}"
    cache = os.path.join(opts["scratch"], f"{tag}.blocks.csv")
    if os.path.exists(cache) and not opts["force"]:
        return tag, "cached", None
    try:
        out = analyze_leg_blocks(
            formatted,
            opts["temperature"],
            opts["repo"],
            opts["blocks"],
            opts["align_bands"],
            opts["verbose"],
        )
        out.insert(0, "leg", leg)
        out.insert(0, "rep", rep)
        out.insert(0, "system", system)
        os.makedirs(opts["scratch"], exist_ok=True)
        out.to_csv(cache, index=False)
        return tag, "ok", None
    except Exception:
        return tag, "FAILED", traceback.format_exc()


def consolidate(scratch, out_prefix):
    """Concatenate per-leg caches, build 3-leg totals and report RMSE per block size."""
    files = sorted(glob.glob(os.path.join(scratch, "*.blocks.csv")))
    if not files:
        print("[block_mbar_sweep] nothing to consolidate")
        return None, None
    legs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    totals = []
    for (system, rep, K), grp in legs.groupby(["system", "rep", "block_size"]):
        if set(grp["leg"]) != set(bvm.LEGS):
            continue
        totals.append(
            {
                "system": system,
                "rep": rep,
                "block_size": int(K),
                "label": grp["label"].iloc[0],
                "n_evaluations": int(grp["n_evaluations"].sum()),
                "dG_total": float(grp["dG"].sum()),
                "err_total": math.sqrt(float((grp["err"] ** 2).sum())),
                "max_grad": float(grp["max_grad"].max()),
            }
        )
    totals = pd.DataFrame(totals)

    if not totals.empty:
        ref = (
            totals[totals["block_size"] == 0]
            .set_index(["system", "rep"])[["dG_total", "n_evaluations"]]
            .to_dict("index")
        )
        totals["dG_reference"] = [
            ref.get((s, r), {}).get("dG_total", np.nan)
            for s, r in zip(totals["system"], totals["rep"])
        ]
        totals["evals_reference"] = [
            ref.get((s, r), {}).get("n_evaluations", np.nan)
            for s, r in zip(totals["system"], totals["rep"])
        ]
        totals["delta_vs_full"] = totals["dG_total"] - totals["dG_reference"]
        totals["speedup"] = totals["evals_reference"] / totals["n_evaluations"]

    legs.to_csv(f"{out_prefix}_legs.csv", index=False)
    totals.to_csv(f"{out_prefix}_totals.csv", index=False)
    print(f"\n[block_mbar_sweep] wrote {out_prefix}_legs.csv, {out_prefix}_totals.csv")

    if not totals.empty:
        print("\n=== 3-leg total DeltaG vs the dense-MBAR reference (kcal/mol) ===")
        for K, grp in totals[totals["block_size"] > 0].groupby("block_size"):
            d = grp["delta_vs_full"].dropna()
            if d.empty:
                continue
            print(
                f"  K={int(K)} ({grp['speedup'].mean():5.1f}x fewer evals, {len(d)} system-reps): "
                f"mean {d.mean():+.4f}  RMSE {np.sqrt((d ** 2).mean()):.4f}  "
                f"MAE {d.abs().mean():.4f}  max|d| {d.abs().max():.4f}"
            )
        bad = totals[totals["max_grad"] > 1e-4]
        if not bad.empty:
            print(f"\n  [WARN] {len(bad)} total(s) with gradient norm > 1e-4 (not converged)")
    return legs, totals


def main():
    p = argparse.ArgumentParser(description="Block-chained MBAR accuracy vs block size")
    p.add_argument("--root", required=True)
    p.add_argument("--blocks", type=int, nargs="+", default=list(DEFAULT_BLOCKS))
    p.add_argument("--systems", nargs="*", default=None)
    p.add_argument("--rep-glob", default="rep*",
                   help="glob for replicate dirs under --root (default: rep*). Ignored when "
                        "--root itself holds a .cache/")
    p.add_argument("--reps", nargs="*", default=None)
    p.add_argument("--legs", nargs="*", default=None, choices=list(bvm.LEGS))
    p.add_argument("--temperature", type=float, default=298.0)
    p.add_argument("--repo", default=None)
    p.add_argument(
        "--no-align-bands",
        action="store_true",
        help="do not isolate sub-leg transitions into their own blocks",
    )
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--scratch", default="./.block_mbar_cache")
    p.add_argument("--out-prefix", default="./block_mbar")
    p.add_argument("--force", action="store_true")
    p.add_argument("--consolidate-only", action="store_true")
    a = p.parse_args()

    repo = a.repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(a.scratch, exist_ok=True)

    if not a.consolidate_only:
        tasks = bvm.find_tasks(a.root, a.systems, a.reps, a.rep_glob)
        if a.legs:
            tasks = [t for t in tasks if t[2] in a.legs]
        if not tasks:
            print(f"[block_mbar_sweep] no tasks found under {a.root}")
            return 1
        opts = {
            "temperature": a.temperature,
            "repo": repo,
            "blocks": sorted(set(a.blocks)),
            "align_bands": not a.no_align_bands,
            "scratch": a.scratch,
            "force": a.force,
            "verbose": a.nproc == 1,
        }
        payload = [(s, r, leg, fmt, fe, opts) for (s, r, leg, fmt, fe) in tasks]
        print(
            f"[block_mbar_sweep] {len(payload)} leg-tasks x block sizes {opts['blocks']} "
            f"(align_bands={opts['align_bands']}), nproc={a.nproc}"
        )
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
        print(f"[block_mbar_sweep] {n_ok} ok, {n_cached} cached, {n_fail} failed")

    consolidate(a.scratch, a.out_prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
