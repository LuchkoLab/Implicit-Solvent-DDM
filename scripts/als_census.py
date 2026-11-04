#!/usr/bin/env python
"""Pairwise-overlap census of finished runs, and the windows post-production ALS would add.

Read-only on the runs. ``scan`` solves the two-state overlap of every adjacent pair of every leg
from the stored ``*_formatted.h5`` (the same subsampling as Phase 7) and caches one table per leg.
``plan`` applies ``als_refine.plan_refinement`` to the cached tables at several thresholds and
writes the would-be insertions with a cost estimate.

    python scripts/als_census.py scan --out <dir> [--workers 32]
    python scripts/als_census.py plan --out <dir> [--thresholds 0.01 0.02 0.03 0.04 0.05]
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml

from implicit_solvent_ddm import als_refine as ar
from implicit_solvent_ddm.matrix_order import CycleSteps

ROOT = "/storage/storage1/sayoub/different_gbs"
SYSTEMS = ("acd", "bcd", "cb7", "oah", "sampl9")
LEGS = ("complex", "ligand", "receptor")
ENDSTATE_LABELS = {"remd": "remd_endstate", "basic_md": "vanillaMD_endstate", 0: "user_provided_endstate"}

# Cost model (single CPU core, 10 ns window; one post-analysis cell = one trajectory under one
# Hamiltonian). Measured on sampl9 G4 igb5 rep1; override on the command line.
MD_SECONDS = {"complex": 3335.0, "ligand": 124.0, "receptor": 3000.0}
CELL_SECONDS = 19.0


# --------------------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------------------
def discover(root, gbs, systems, reps):
    """Every finished-or-not run under ``root``, one dict per config file."""
    runs = []
    for gb in gbs:
        for system in systems:
            for rep in reps:
                d = f"{root}/igb_{gb}/{system}/in_solvent/rep{rep}"
                for path in sorted(glob.glob(f"{d}/config_files/*.yaml")):
                    if os.path.basename(path).startswith("."):
                        continue
                    with open(path) as fh:
                        cfg = yaml.safe_load(fh)
                    complex_name = os.path.splitext(os.path.basename(
                        cfg["endstate_parameter_files"]["complex_parameter_filename"]))[0]
                    cache = os.path.join(cfg["hardware_parameters"]["cache_directory_output"], ".cache", complex_name)
                    runs.append(dict(
                        gb=gb, system=system, rep=rep,
                        guest=os.path.splitext(os.path.basename(path))[0].split("-", 1)[1].lower(),
                        config=path, complex=complex_name, cache=cache,
                    ))
    return runs


def schedule_from_yaml(path):
    """``(Schedule, CycleSteps)`` from a run's YAML, without building a full ``Config``."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    args = cfg["workflow"]["intermediate_states_arguments"]
    method = cfg["workflow"].get("endstate_method", "remd")
    label = ENDSTATE_LABELS[method]
    con = [round(float(np.log2(2.0 ** float(c))), 3) for c in args["exponent_conformational_forces"]]
    orient = [round(float(np.log2(2.0 ** float(o))), 3) for o in args["exponent_orientational_forces"]]
    charges = sorted({0.0, 1.0, *map(float, args.get("charges_lambda_window", []))})
    lj = sorted({float(x) for x in args.get("lj_scale_windows", [])})
    screening = sorted({float(x) for x in args.get("screening_windows", [])})
    dielectric = sorted({float(x) for x in args.get("gb_extdiel_windows", [])})
    schedule = ar.Schedule(restraint=list(zip(con, orient)), charges=charges, lj=lj, screening=screening,
                           dielectric=dielectric, endstate_label=label)
    cycle = CycleSteps(conformation_forces=con, orientational_forces=orient, charges_windows=charges,
                       external_dielectic=dielectric, endstate_label=label, lj_windows=lj,
                       screening_windows=screening, in_solvent=bool(args.get("decouple_in_solvent", False)))
    cycle.round(3)
    temperature = float(args.get("temperature", 298))
    return schedule, cycle, temperature


def leg_files(cache, complex_name, leg):
    """``(formatted.h5, O_MBAR.h5)`` of one leg, or ``(None, None)`` when the leg has no result."""
    if leg == "complex":
        f = f"{cache}/{complex_name}_formatted.h5"
        return (f, f"{cache}/{complex_name}_O_MBAR.h5") if os.path.exists(f) else (None, None)
    found = [f for f in glob.glob(f"{cache}/{leg}_*_formatted.h5") if not os.path.basename(f).startswith(".")]
    if len(found) != 1:
        return None, None
    return found[0], found[0].replace("_formatted.h5", "_O_MBAR.h5")


def leg_key(run, leg):
    return f"igb{run['gb']}_{run['system']}_rep{run['rep']}_{run['guest']}_{leg}"


# --------------------------------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------------------------------
def scan_leg(task):
    """Solve and cache one leg's pairwise table. Returns a status dict."""
    run, leg, out_dir, force = task
    key = leg_key(run, leg)
    out = f"{out_dir}/legs/{key}.pkl"
    formatted, overlap_file = leg_files(run["cache"], run["complex"], leg)
    status = dict(key=key, **{k: run[k] for k in ("gb", "system", "rep", "guest")}, leg=leg)
    if formatted is None:
        return {**status, "status": "no_result"}
    if not force and os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(formatted):
        return {**status, "status": "cached"}
    t0 = time.time()
    try:
        _, cycle, temperature = schedule_from_yaml(run["config"])
        order = [tuple(map(str, s)) for s in getattr(cycle, f"{leg}_order")]
        df = pd.read_hdf(formatted)
        columns = [tuple(map(str, c)) for c in df.columns]
        if columns != order:
            return {**status, "status": "order_mismatch",
                    "detail": f"{len(columns)} columns vs {len(order)} cycle states"}
        n_frames = df.groupby(level=list(range(df.index.nlevels))).size()
        df = df / ar.kcal_per_kt(temperature)
        pairs = ar.adjacent_pair_overlaps(ar.subsample_like_production(df))
        if os.path.exists(overlap_file):
            full = pd.read_hdf(overlap_file).to_numpy()
            pairs["o_fullN"] = [min(full[i][i + 1], full[i + 1][i]) for i in pairs.i]
        pairs.to_pickle(out)
        return {**status, "status": "ok", "seconds": round(time.time() - t0, 1),
                "frames_min": int(n_frames.min()), "frames_max": int(n_frames.max())}
    except Exception as err:  # keep going; the summary lists failures
        return {**status, "status": "error", "detail": f"{type(err).__name__}: {err}",
                "trace": traceback.format_exc(limit=3)}


def cmd_scan(args):
    os.makedirs(f"{args.out}/legs", exist_ok=True)
    runs = discover(args.root, args.gbs, args.systems, args.reps)
    pd.DataFrame(runs).to_csv(f"{args.out}/runs.csv", index=False)
    tasks = [(run, leg, args.out, args.force) for run in runs for leg in LEGS]
    print(f"{len(runs)} runs, {len(tasks)} legs, {args.workers} workers", flush=True)
    rows = []
    with Pool(args.workers) as pool:
        for k, row in enumerate(pool.imap_unordered(scan_leg, tasks), 1):
            rows.append(row)
            if row["status"] not in ("ok", "cached") or k % 50 == 0:
                print(f"[{k}/{len(tasks)}] {row['key']}: {row['status']} {row.get('detail', '')}", flush=True)
    status = pd.DataFrame(rows).sort_values("key")
    status.to_csv(f"{args.out}/scan_status.csv", index=False)
    print(status.status.value_counts().to_string())


# --------------------------------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------------------------------
def new_cells(n_states, k):
    """Post-analysis cells a leg gains when ``k`` states are added to ``n_states``."""
    return (n_states + k) ** 2 - n_states ** 2


def cmd_plan(args):
    runs = pd.read_csv(f"{args.out}/runs.csv").to_dict("records")
    pair_rows, plan_rows, seam_rows = [], [], []
    for run in runs:
        tables = {}
        for leg in LEGS:
            f = f"{args.out}/legs/{leg_key(run, leg)}.pkl"
            if os.path.exists(f):
                tables[leg] = pd.read_pickle(f)
        ident = {k: run[k] for k in ("gb", "system", "rep", "guest")}
        if set(tables) != set(LEGS):
            plan_rows.append({**ident, "threshold": np.nan, "status": "incomplete",
                              "legs": ",".join(sorted(tables))})
            continue
        schedule, _, _ = schedule_from_yaml(run["config"])
        n_states = {leg: len(t) + 1 for leg, t in tables.items()}
        for thr in args.thresholds:
            try:
                plan = ar.plan_refinement(tables, schedule, thr, bands=args.bands)
            except ValueError as err:
                plan_rows.append({**ident, "threshold": thr, "status": "error", "detail": str(err)})
                continue
            per_leg = {leg: 0 for leg in LEGS}
            for band, values in plan.insertions.items():
                for leg in ar.SHARED_LEGS[band]:
                    per_leg[leg] += len(values)
            cells = sum(new_cells(n_states[leg], per_leg[leg]) for leg in LEGS)
            md_hours = sum(per_leg[leg] * MD_SECONDS[leg] for leg in LEGS) / 3600
            plan_rows.append({
                **ident, "threshold": thr, "status": "ok", "reason": plan.reason or "inserted",
                "n_insertions": plan.n_insertions, "n_md_windows": plan.n_md_windows,
                "new_cells": cells, "md_core_h": round(md_hours, 2),
                "cell_core_h": round(cells * CELL_SECONDS / 3600, 2),
                "insertions": json.dumps(plan.summary()["insertions"]),
                "weak_endstate_gaps": len(plan.endstate_gaps),
                **{f"bands_{band}": reason for band, reason in plan.band_reasons.items()},
            })
            if thr == args.thresholds[0]:
                recs = plan.records.assign(**ident)
                pair_rows.append(recs)
                for leg, t in tables.items():
                    if "o_fullN" in t:
                        pair_rows[-1].loc[pair_rows[-1].leg == leg, "o_fullN"] = t.o_fullN.values
            seams = plan.endstate_gaps.assign(**ident, threshold=thr)
            seam_rows.append(seams)

    plans = pd.DataFrame(plan_rows)
    plans.to_csv(f"{args.out}/census_plan.csv", index=False)
    if pair_rows:
        pd.concat(pair_rows).to_csv(f"{args.out}/census_pairs.csv", index=False)
    if seam_rows:
        pd.concat(seam_rows).to_csv(f"{args.out}/census_endstate_seams.csv", index=False)

    ok = plans[plans.status == "ok"].copy()
    ok["affected"] = ok.n_insertions > 0
    summary = ok.groupby(["threshold", "gb", "system"]).agg(
        runs=("guest", "size"), affected=("affected", "sum"), insertions=("n_insertions", "sum"),
        md_windows=("n_md_windows", "sum"), new_cells=("new_cells", "sum"),
        md_core_h=("md_core_h", "sum"), cell_core_h=("cell_core_h", "sum"),
    ).reset_index()
    total = ok.groupby("threshold").agg(
        runs=("guest", "size"), affected=("affected", "sum"), insertions=("n_insertions", "sum"),
        md_windows=("n_md_windows", "sum"), new_cells=("new_cells", "sum"),
        md_core_h=("md_core_h", "sum"), cell_core_h=("cell_core_h", "sum"),
    ).reset_index().assign(gb="all", system="all")
    summary = pd.concat([summary, total])
    summary.to_csv(f"{args.out}/census_summary.csv", index=False)
    for thr in args.thresholds:
        targets = ok[(ok.threshold == thr) & ok.affected]
        cfgs = pd.DataFrame(runs).merge(targets[["gb", "system", "rep", "guest"]])
        cfgs.config.to_csv(f"{args.out}/targets_t{thr}.txt", index=False, header=False)
    pd.set_option("display.width", 220)
    print(total.round({"md_core_h": 1, "cell_core_h": 1}).to_string(index=False))
    bad = plans[plans.status != "ok"]
    if len(bad):
        print(f"\n{len(bad)} run/threshold rows not planned:")
        print(bad.drop_duplicates(["gb", "system", "rep", "guest"]).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scan", "plan"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="census directory")
        p.add_argument("--root", default=ROOT)
        p.add_argument("--gbs", nargs="+", type=int, default=[1, 5])
        p.add_argument("--systems", nargs="+", default=list(SYSTEMS))
        p.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    sub.choices["scan"].add_argument("--workers", type=int, default=8)
    sub.choices["scan"].add_argument("--force", action="store_true", help="re-solve cached legs")
    sub.choices["plan"].add_argument("--thresholds", nargs="+", type=float, default=[0.01, 0.02, 0.03, 0.04, 0.05])
    sub.choices["plan"].add_argument("--bands", nargs="+", default=list(ar.DEFAULT_BANDS))
    args = ap.parse_args()
    {"scan": cmd_scan, "plan": cmd_plan}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
