#!/usr/bin/env python
"""gb_endpoint_analyze.py -- build the two MBAR problems from gb_endpoint_probe.py output.

  PROD : GB ladder scored at saltcon=0.3 (what post_mdin actually did) + vacuum (igb=6)
  FIX  : GB ladder scored at saltcon=0.0 (what the MD used)          + vacuum (igb=6)

The two differ ONLY in the scoring salt concentration. Reports the linearity fit, both overlap
matrices, both superdiagonals and both dG chains.

Usage:  python scripts/gb_endpoint_analyze.py --workdir /tmp/gb_endpoint_probe
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import gb_ddG_bar as gbb  # noqa: E402
from implicit_solvent_ddm import block_mbar as bm  # noqa: E402

LADDER = ["L78.5", "L2.0", "L1.3158", "L1.1364", "L1.0638", "L1.0204"]
EPS = [78.5, 2.0, 1.3157894736842106, 1.1363636363636365, 1.0638297872340425, 1.0204081632653061]
TRAJS = LADDER + ["T_vac"]


def build(df, suffix, kT):
    """Return (u_kn, N_k, labels) for the 7-state problem using ladder columns tagged `suffix`."""
    hams = [f"{n}{suffix}" for n in LADDER] + ["VAC"]
    piv = {(t, h): g["E"].to_numpy(float)
           for (t, h), g in df.groupby(["traj", "ham"], sort=False)}
    N_k = np.array([len(piv[(t, hams[0])]) for t in TRAJS])
    u_kn = np.vstack([np.concatenate([piv[(t, h)] for t in TRAJS]) for h in hams]) / kT
    return u_kn, N_k, hams


def solve(u_kn, N_k):
    import pymbar

    mb = pymbar.MBAR(u_kn, N_k, solver_protocol=bm.BLOCK_SOLVER_PROTOCOL)
    res = mb.compute_free_energy_differences()
    return res["Delta_f"], res["dDelta_f"], np.asarray(mb.compute_overlap()["matrix"])


def show(name, dF, dE, ov, kT):
    n = len(TRAJS)
    print(f"\n{'='*78}\n{name}\n{'='*78}")
    print("adjacent transitions (superdiagonal):")
    print(f"  {'transition':<26} {'dG kcal/mol':>13} {'err':>10} {'overlap':>11}")
    for i in range(n - 1):
        lab = f"{TRAJS[i]} -> {TRAJS[i+1]}"
        print(f"  {lab:<26} {dF[i, i+1]*kT:>13.4f} {dE[i, i+1]*kT:>10.4f} {ov[i, i+1]:>11.4e}")
    print(f"  {'TOTAL (state0 -> vac)':<26} {dF[0, -1]*kT:>13.4f} {dE[0, -1]*kT:>10.4f}")
    print("\noverlap matrix:")
    print("        " + "".join(f"{t:>10}" for t in TRAJS))
    for i, t in enumerate(TRAJS):
        print(f"  {t:>6}" + "".join(f"{ov[i, j]:>10.3e}" for j in range(n)))
    return ov[n - 2, n - 1], dF[n - 2, n - 1] * kT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default="/tmp/gb_endpoint_probe")
    p.add_argument("--temperature", type=float, default=298.0)
    a = p.parse_args()

    kT = gbb.kcals_per_kt(a.temperature)
    df = pd.read_csv(os.path.join(a.workdir, "results.csv"))
    print(f"kT = {kT:.7f} kcal/mol   rows={len(df)}")

    # ------------------------------------------------------------------ P1: linearity + intercept
    print(f"\n{'='*78}\nP1  EGB linearity in x = 1 - 1/eps, and the intercept at x = 0\n{'='*78}")
    for suffix, salt in (("_s03", 0.3), ("_s00", 0.0)):
        sub = df[(df.traj == "L1.0204") & (df.ham.str.endswith(suffix))]
        xs = np.array([1 - 1 / e for e in EPS])
        means = np.array([sub[sub.ham == f"{n}{suffix}"]["E"].to_numpy(float).mean()
                          for n in LADDER])
        vac = df[(df.traj == "L1.0204") & (df.ham == "VAC")]["E"].to_numpy(float).mean()
        slope, icept = np.polyfit(xs, means, 1)
        resid = means - (slope * xs + icept)
        print(f"\n  saltcon={salt}:  slope={slope:+10.3f}  max|resid|={np.abs(resid).max():.5f} kcal/mol")
        print(f"    <E> extrapolated to x=0 : {icept:+12.4f}")
        print(f"    <E> actual vacuum (igb=6): {vac:+12.4f}")
        print(f"    GAP (the discontinuity)  : {vac - icept:+12.4f} kcal/mol")
        print(f"    implied x_eff of vacuum  : {(vac - icept)/slope:+.4f}   (0 == true vacuum)")

    # ------------------------------------------------------------------ the two MBAR problems
    results = {}
    for name, suffix in (("PROD  (ladder scored at saltcon=0.3 -- what production did)", "_s03"),
                         ("FIX   (ladder scored at saltcon=0.0 -- matching the MD)", "_s00")):
        u_kn, N_k, hams = build(df, suffix, kT)
        dF, dE, ov = solve(u_kn, N_k)
        results[suffix] = show(name, dF, dE, ov, kT)

    o_prod, g_prod = results["_s03"]
    o_fix, g_fix = results["_s00"]
    print(f"\n{'='*78}\nVERDICT -- the eps=1.0204 -> vacuum junction\n{'='*78}")
    print(f"  PROD (saltcon=0.3): overlap {o_prod:.4e}   dG {g_prod:+9.3f} kcal/mol")
    print(f"  FIX  (saltcon=0.0): overlap {o_fix:.4e}   dG {g_fix:+9.3f} kcal/mol")
    print(f"  overlap improvement: {o_fix/max(o_prod, 1e-300):.3e}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
