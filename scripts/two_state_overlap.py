#!/usr/bin/env python
"""two_state_overlap.py -- MBAR overlap between the endstate and one restraint window.

WHY
---
The receptor leg's endstate seam (``user_provided_endstate -> lambda_window -14.0``) has an MBAR
overlap of 0.0157 at 2 ns. Measured against the real restraint file, the two ensembles' restraint
energies do not share support at ALL -- the window's highest dU (4.73 kcal/mol) sits 19 kcal/mol
below the endstate's lowest (23.68) -- and 76% of that 31.3 kcal/mol gap comes from residues 22-32,
which sit 22.6 A from the ligand.

No sander needed. The two states differ ONLY by the conformational restraint (same igb, saltcon,
extdiel, cut), so the reduced potential difference IS the restraint energy, computable directly
from coordinates and the DISANG file. MBAR is invariant to a per-sample constant, so setting
u(endstate) = 0 and u(window) = U_restraint/kT is exact, not an approximation.

That makes the whole check local and instant, and it means you can ask "what would the overlap be
under a different restraint selection" WITHOUT rerunning anything.

WHAT THIS DOES
--------------
Computes the 2-state MBAR overlap twice -- once with the restraint file as-is, once with every
restraint touching ``--free`` removed -- so the comparison comes out of a single run.

HOW TO READ IT
--------------
``min_degree_overlap`` defaults to 0.04 in this workflow.

* Both arms low -> the restraint selection is not the problem; look at the sampling instead.
* Filtered arm clears the threshold -> the freed region carries the seam, and re-running the leg
  with ``intermediate_states_arguments.unrestrained_receptor_mask`` should fix it.

IMPORTANT: run against the CURRENT window trajectory and the filtered number is a LOWER BOUND. Those
frames are still trapped in the holo basin the window was seeded from; re-run the MD with the freed
selection and the loop can actually relax, closing the gap further. To measure the real thing, point
``--window-traj`` at the re-run trajectory.

USAGE
-----
  cd .../lambda_window/1.0/78.5/-14.0

  python <repo>/scripts/two_state_overlap.py \\
      --restraint     restraint.RST \\
      --parm          <receptor>.parm7 \\
      --window-traj   <receptor>_state_2_-14.0_prod_traj.nc \\
      --endstate-traj <run>/remd/<receptor>/<receptor>_298.0K.nc \\
      --free "resid 22:32" --stride 10

To re-run the window MD against the filtered restraints first:

  python <repo>/scripts/filter_restraints.py --restraint restraint.RST \\
      --parm <receptor>.parm7 --free "resid 22:32" --out restraint_free.RST
  sed 's/DISANG = restraint.RST/DISANG = restraint_free.RST/' tmp*.tmp > mdin_free
  $AMBERHOME/bin/pmemd.cuda -O -i mdin_free -p <receptor>.parm7 \\
      -c split_receptor_system.ncrst.1 -o mdout_free -x traj_free.nc -r restrt_free.rst7

NOTES
-----
* ``--free`` is an MDAnalysis selection against the receptor topology, matching filter_restraints.py.
* Error bars assume independent samples. Pass ``--subsample`` to thin by the statistical
  inefficiency of the restraint energy first, which is the honest version for a reported number.
* Cost is O(n_restraints x n_frames); ``--stride`` is the knob. 82,522 restraints x 2,000 frames is
  a few minutes.
"""
import argparse
import re
import sys

import numpy as np

KB = 0.0019872041  # kcal/mol/K


def parse_restraints(path):
    """(atom pairs 0-based, r2, r3, rk) from an AMBER DISANG file of two-atom distance restraints."""
    text = open(path).read()
    pat = re.compile(
        r"&rst\s+iat\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*"
        r"r1\s*=\s*[-\d.eE+]+\s*,\s*r2\s*=\s*([-\d.eE+]+)\s*,\s*"
        r"r3\s*=\s*([-\d.eE+]+)\s*,\s*r4\s*=\s*[-\d.eE+]+\s*"
        r"rk2\s*=\s*([-\d.eE+]+)\s*,\s*rk3\s*=\s*([-\d.eE+]+)"
    )
    rows = pat.findall(text)
    declared = text.count("&rst")
    if len(rows) != declared:
        sys.exit(f"parsed {len(rows)} restraints but the file declares {declared}")
    arr = np.array([[float(x) for x in r] for r in rows])
    if not np.allclose(arr[:, 4], arr[:, 5]):
        sys.exit("rk2 != rk3 somewhere; this script assumes a symmetric well")
    return arr[:, :2].astype(int) - 1, arr[:, 2], arr[:, 3], arr[:, 4]


def restraint_energy(parm, traj, stride, pairs, r2, r3, rk, label):
    """Per-frame AMBER flat-bottom restraint energy: rk*(d-r2)^2 below, 0 inside, rk*(d-r3)^2 above."""
    import MDAnalysis as mda

    universe = mda.Universe(parm, traj)
    i, j = pairs[:, 0], pairs[:, 1]
    out = []
    for _ in universe.trajectory[::stride]:
        xyz = universe.atoms.positions
        d = np.linalg.norm(xyz[i] - xyz[j], axis=1)
        out.append(np.where(d < r2, rk * (d - r2) ** 2, np.where(d > r3, rk * (d - r3) ** 2, 0.0)))
    energies = np.array(out)
    print(f"  {label}: {len(energies)} frames, {universe.atoms.n_atoms} atoms")
    return energies


def overlap(window_e, endstate_e, kt, subsample, label):
    """2-state MBAR with u(endstate)=0, u(window)=U_restraint/kT."""
    import pymbar

    we, ee = window_e.sum(1), endstate_e.sum(1)
    if subsample:
        from pymbar import timeseries

        we = we[timeseries.subsample_correlated_data(we)]
        ee = ee[timeseries.subsample_correlated_data(ee)]
    n_e, n_w = len(ee), len(we)
    # sample order must match N_k: endstate block first, then window
    u = np.zeros((2, n_e + n_w))
    u[1] = np.concatenate([ee, we]) / kt
    mbar = pymbar.MBAR(u, [n_e, n_w])
    res = mbar.compute_free_energy_differences()
    ov = mbar.compute_overlap()["matrix"]
    dg, err = res["Delta_f"][0, 1] * kt, res["dDelta_f"][0, 1] * kt

    gap = ee.mean() - we.mean()
    print(f"\n{label}")
    print(f"  restraints used   : {window_e.shape[1]}")
    print(f"  dU window frames  : {we.mean():8.3f} +/- {we.std():6.3f}  [{we.min():.2f}, {we.max():.2f}]")
    print(f"  dU endstate frames: {ee.mean():8.3f} +/- {ee.std():6.3f}  [{ee.min():.2f}, {ee.max():.2f}]")
    print(f"  gap               : {gap:8.3f} kcal/mol ({gap / kt:.1f} kT)   ranges overlap: {ee.min() < we.max()}")
    print(f"  samples (e/w)     : {n_e} / {n_w}")
    print(f"  MBAR dG(end->win) : {dg:8.3f} +/- {err:.3f} kcal/mol")
    print(f"  MBAR overlap      : {ov[0, 1]:8.4f}  (reverse {ov[1, 0]:.4f}, diag {ov[0, 0]:.4f})")
    return ov[0, 1]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--restraint", required=True)
    ap.add_argument("--parm", required=True, help="receptor topology that iat indexes")
    ap.add_argument("--window-traj", required=True)
    ap.add_argument("--endstate-traj", required=True)
    ap.add_argument("--free", default=None, help="MDAnalysis selection to leave UNrestrained")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=298.0)
    ap.add_argument("--subsample", action="store_true", help="thin by statistical inefficiency")
    args = ap.parse_args()

    kt = KB * args.temperature
    pairs, r2, r3, rk = parse_restraints(args.restraint)
    print(f"restraints: {len(pairs)}   rk = {rk[0]:.6g}   flat bottom {np.median(r3 - r2):.3f} A")
    print(f"kT at {args.temperature:.1f} K = {kt:.4f} kcal/mol\n")

    print("reading trajectories (this is the slow part)")
    we = restraint_energy(args.parm, args.window_traj, args.stride, pairs, r2, r3, rk, "window  ")
    ee = restraint_energy(args.parm, args.endstate_traj, args.stride, pairs, r2, r3, rk, "endstate")

    base = overlap(we, ee, kt, args.subsample, "=== AS-IS (all restraints) ===")

    if args.free:
        import MDAnalysis as mda

        universe = mda.Universe(args.parm)
        freed = universe.select_atoms(args.free)
        if freed.n_atoms == 0:
            sys.exit(f"selection {args.free!r} matched no atoms")
        fset = set(freed.indices.tolist())
        keep = np.array([p[0] not in fset and p[1] not in fset for p in pairs])
        print(f"\nfreeing {args.free!r}: {freed.n_atoms} atoms, "
              f"dropping {(~keep).sum()} of {len(pairs)} restraints ({(~keep).sum() / len(pairs):.1%})")
        filt = overlap(we[:, keep], ee[:, keep], kt, args.subsample, f"=== FREED {args.free!r} ===")
        print(f"\noverlap {base:.4f} -> {filt:.4f}   (min_degree_overlap = 0.04)")
        print("This is a LOWER BOUND: the window frames are still trapped in the holo basin they")
        print("were seeded from. Re-run the MD with the freed selection to measure the real value.")


if __name__ == "__main__":
    main()
