#!/usr/bin/env python
"""bound_vs_apo_receptor_rmsd.py -- how far does the receptor reorganize on binding?

WHY
---
The receptor leg turns on conformational restraints defined at the BOUND geometry and ends at the
free APO receptor, so it must traverse the receptor's binding reorganization. Its endstate seam
(endstate -> lambda_window(-14.0)) has an MBAR overlap of 0.0039 against the complex leg's 0.0798,
with dG +5.96 kT vs +3.55 kT. The restraint networks are near-identical (82,077 vs 82,202 restraints,
mean applied energy 2.398 vs 2.482 kcal/mol), so the restraints are not the difference -- the size of
the conformational gap is.

``endstate_window_rmsd.py`` measures the apo endstate against a restraint WINDOW, which conflates two
things: the real reorganization, and the fact that a weakly-restrained window may not have finished
relaxing in its 10 ns. At exponent -14 the effective per-atom force constant is 4.1e-3 kcal/mol/A^2
(67 restraints/atom x 2^-14), a ~12 A thermal scale on top of a 1 A flat bottom -- so that window is
effectively free and its 1.75 A displacement is unfinished relaxation, not restraint.

This script removes that confound. It compares the two ENDSTATE trajectories directly -- the apo
receptor against the receptor as it exists in the complex, with the ligand stripped. Both are long
unrestrained MD, so what is left is the reorganization itself: the conformational distance the
receptor leg is charged with crossing, measured independently of any window's convergence.

WHAT THIS DOES
--------------
No MD, no MBAR. cpptraj only:

1. strip the ligand from the complex endstate trajectory (``strip <ligand_mask>``) and write a
   matching stripped topology, then check its atom count against the receptor topology;
2. average structure of the bound (stripped) and apo ensembles;
3. per-frame RMSD of each to its own average (within-ensemble spread) and to the other's average
   (cross);
4. the separation: average-to-average.

HOW TO READ IT
--------------
* separation smaller than either spread -> the receptor barely reorganizes; the poor receptor-leg
  overlap is then NOT reorganization and points back at window convergence or the schedule.
* separation comparable to or larger than the spreads -> a real reorganization the leg must cross.
  No amount of window insertion removes it; the leg has to sample the transition, which argues for
  Hamiltonian REMD across the restraint ladder or much longer MD in the weak-k tail.

Compare the separation here against ``endstate_window_rmsd.py``'s receptor number (2.493 A). If this
one is much SMALLER, most of that 2.493 A was the window failing to relax rather than true
reorganization -- a sampling problem. If they are similar, the reorganization is real.

USAGE
-----
  module load amber                      # or: export CPPTRAJ=$AMBERHOME/bin/cpptraj

  python scripts/bound_vs_apo_receptor_rmsd.py \\
      --complex-parm  <...>/MCL-1_ligand-1.parm7 \\
      --complex-traj  <...>/basicMD_endstate/MCL-1_ligand-1/MCL-1_ligand-1_basicMD_traj.nc \\
      --receptor-parm <...>/MCL-1_receptor-ligand-1_with_mcl1_uxU_uxU.parm7 \\
      --receptor-traj <...>/basicMD_endstate/MCL-1_receptor-.../MCL-1_receptor-..._basicMD_traj.nc

  # cheaper first look
  ... --stride 10

NOTES
-----
* ``--ligand-mask`` defaults to ``:LIG``, matching ``AMBER_masks.ligand_mask`` in the run yaml.
* The receptor topology was itself produced by ``split_complex_system`` with the same masks
  (``workflow_phases.py:178``), so stripping the complex should reproduce it atom-for-atom. The
  script checks the atom counts and refuses to continue if they disagree, because a mismatch would
  silently compare different atom orderings.
* RMSD is mass-weighted and best-fit on the same mask it measures, so this is conformational
  difference, not rigid-body drift.
* ``--reference`` optionally adds each ensemble's RMSD to a common structure (e.g. the window's
  ``split_receptor_system.ncrst.1``, which IS the complex endstate last frame with the ligand
  removed) for continuity with the other script's rows.
"""
import argparse
import csv
import os
import re
import subprocess
import sys

import numpy as np

CPPTRAJ = os.environ.get("CPPTRAJ", "cpptraj")


def run_cpptraj(script: str, workdir: str, tag: str, dry: bool) -> str:
    """Write and execute one cpptraj input; return its stdout."""
    path = os.path.abspath(os.path.join(workdir, f"{tag}.cpptraj"))
    with open(path, "w") as handle:
        handle.write(script)
    if dry:
        print(f"--- {tag} ---\n{script}")
        return ""
    proc = subprocess.run([CPPTRAJ, "-i", path], capture_output=True, text=True, cwd=workdir)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"cpptraj failed on {tag} (exit {proc.returncode})")
    return proc.stdout


def natoms(parm: str, workdir: str, tag: str) -> int:
    """Atom count of a topology, read back from cpptraj rather than guessed."""
    out = run_cpptraj(f"parm {parm}\nparminfo\n", workdir, tag, dry=False)
    match = re.search(r"(\d+)\s+atoms", out)
    if not match:
        raise SystemExit(f"could not read atom count for {parm}")
    return int(match.group(1))


def read_dat(path: str) -> np.ndarray:
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                rows.append(float(parts[1]))
    return np.asarray(rows)


def describe(name: str, values: np.ndarray) -> dict:
    return {
        "set": name,
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "p50": float(np.median(values)),
        "max": float(values.max()),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--complex-parm", required=True)
    ap.add_argument("--complex-traj", required=True, help="complex endstate trajectory")
    ap.add_argument("--receptor-parm", required=True)
    ap.add_argument("--receptor-traj", required=True, help="apo receptor endstate trajectory")
    ap.add_argument("--ligand-mask", default=":LIG", help="AMBER_masks.ligand_mask (default :LIG)")
    ap.add_argument("--mask", default=None, help="RMSD mask (default: all atoms of the receptor)")
    ap.add_argument("--reference", default=None, help="optional common reference structure")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--workdir", default="./bound_vs_apo_work")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    complex_parm = os.path.abspath(args.complex_parm)
    complex_traj = os.path.abspath(args.complex_traj)
    receptor_parm = os.path.abspath(args.receptor_parm)
    receptor_traj = os.path.abspath(args.receptor_traj)
    reference = os.path.abspath(args.reference) if args.reference else None
    # Absolute: run_cpptraj executes with cwd=W, so any relative path built from W would be
    # resolved a second time against W itself (./work/./work/bound_receptor.parm7).
    W = os.path.abspath(args.workdir)
    os.makedirs(W, exist_ok=True)
    trajin = f"1 last {args.stride}" if args.stride > 1 else ""

    # ----------------------------------------------------------------------------------------------
    # 1. Strip the ligand out of the complex endstate -> the receptor in its BOUND conformation.
    # ----------------------------------------------------------------------------------------------
    run_cpptraj(
        f"parm {complex_parm}\n"
        f"parmstrip {args.ligand_mask}\n"
        f"parmwrite out bound_receptor.parm7\n"
        "run\n",
        W,
        "strip_parm",
        args.dry_run,
    )
    run_cpptraj(
        f"parm {complex_parm}\n"
        f"trajin {complex_traj} {trajin}\n"
        f"strip {args.ligand_mask}\n"
        f"trajout bound_receptor.nc netcdf\n"
        "run\n",
        W,
        "strip_traj",
        args.dry_run,
    )

    n_receptor = None
    if not args.dry_run:
        n_stripped = natoms(os.path.join(W, "bound_receptor.parm7"), W, "info_stripped")
        n_receptor = natoms(receptor_parm, W, "info_receptor")
        print(f"[check] complex stripped of {args.ligand_mask}: {n_stripped} atoms")
        print(f"[check] receptor topology            : {n_receptor} atoms")
        if n_stripped != n_receptor:
            raise SystemExit(
                f"atom count mismatch ({n_stripped} vs {n_receptor}). Stripping the complex did not\n"
                "reproduce the receptor topology, so the two trajectories are not comparable\n"
                "atom-for-atom. Check --ligand-mask against AMBER_masks.ligand_mask in the run yaml."
            )

    # Default to every receptor atom, matching endstate_window_rmsd.py's restraint-derived @1-2444.
    mask = args.mask or (f"@1-{n_receptor}" if n_receptor else "@*")
    bound_parm = os.path.join(W, "bound_receptor.parm7")
    bound_traj = os.path.join(W, "bound_receptor.nc")
    print(f"[mask] {mask}\n")

    # ----------------------------------------------------------------------------------------------
    # 2-4. Averages, cross RMSD, separation. Both ensembles now share the stripped topology.
    # ----------------------------------------------------------------------------------------------
    ref_line = f"reference {reference} [ref]\n" if reference else ""
    ref_rms = f"rms toref ref [ref] {mask} mass out {{tag}}_to_reference.dat\n" if reference else ""

    for tag, traj in (("bound", bound_traj), ("apo", receptor_traj)):
        run_cpptraj(
            f"parm {bound_parm}\n"
            f"trajin {traj} {'' if tag == 'bound' else trajin}\n"
            + ref_line
            # Superimpose BEFORE averaging. cpptraj's `average` accumulates coordinates as they
            # arrive, so without a fit the rotational diffusion of two independent MD runs averages
            # the structure into a collapsed centroid -- every frame then sits ~10 A from its "own"
            # average, further than from the other ensemble's, and the effect size goes negative.
            # Fitting is not optional here; `rms toref` below only supplied it by accident, and only
            # when --reference happened to be given.
            + f"rms fitfirst first {mask} mass\n"
            + ref_rms.format(tag=tag)
            + f"average {tag}_avg.rst7 restart\n"
            "run\n",
            W,
            f"avg_{tag}",
            args.dry_run,
        )

    for tag, traj, other in (("bound", bound_traj, "apo"), ("apo", receptor_traj, "bound")):
        run_cpptraj(
            f"parm {bound_parm}\n"
            f"trajin {traj} {'' if tag == 'bound' else trajin}\n"
            f"reference {tag}_avg.rst7 [own]\n"
            f"reference {other}_avg.rst7 [other]\n"
            f"rms own ref [own] {mask} mass out {tag}_to_own_avg.dat\n"
            f"rms cross ref [other] {mask} mass out {tag}_to_{other}_avg.dat\n"
            "run\n",
            W,
            f"cross_{tag}",
            args.dry_run,
        )

    run_cpptraj(
        f"parm {bound_parm}\n"
        f"trajin bound_avg.rst7\n"
        f"reference apo_avg.rst7 [a]\n"
        f"rms sep ref [a] {mask} mass out avg_to_avg.dat\n"
        "run\n",
        W,
        "separation",
        args.dry_run,
    )

    if args.dry_run:
        return

    # ----------------------------------------------------------------------------------------------
    bound_own = read_dat(os.path.join(W, "bound_to_own_avg.dat"))
    apo_own = read_dat(os.path.join(W, "apo_to_own_avg.dat"))
    bound_cross = read_dat(os.path.join(W, "bound_to_apo_avg.dat"))
    apo_cross = read_dat(os.path.join(W, "apo_to_bound_avg.dat"))
    separation = float(read_dat(os.path.join(W, "avg_to_avg.dat"))[0])

    rows = [
        describe("bound (complex, ligand stripped) -> own avg", bound_own),
        describe("apo (receptor endstate)          -> own avg", apo_own),
        describe("bound -> apo average   (cross)", bound_cross),
        describe("apo   -> bound average (cross)", apo_cross),
    ]
    if reference:
        rows = [
            describe("bound -> common reference", read_dat(os.path.join(W, "bound_to_reference.dat"))),
            describe("apo   -> common reference", read_dat(os.path.join(W, "apo_to_reference.dat"))),
        ] + rows

    print(f"{'set':46s} {'n':>6s} {'mean':>7s} {'sd':>6s} {'min':>7s} {'p50':>7s} {'max':>7s}")
    for row in rows:
        print(
            f"{row['set']:46s} {row['n']:6d} {row['mean']:7.3f} {row['sd']:6.3f} "
            f"{row['min']:7.3f} {row['p50']:7.3f} {row['max']:7.3f}"
        )

    def effect(own, cross):
        sd = own.std(ddof=1)
        return (cross.mean() - own.mean()) / sd if sd > 0 else float("inf")

    eff_bound, eff_apo = effect(bound_own, bound_cross), effect(apo_own, apo_cross)
    spread = max(bound_own.mean(), apo_own.mean())
    print(f"\nseparation (bound avg -> apo avg) : {separation:.3f} A")
    print(f"within-ensemble spread (larger)   : {spread:.3f} A")
    print(f"effect size, bound frames         : {eff_bound:.1f} sd")
    print(f"effect size, apo frames           : {eff_apo:.1f} sd")

    # A frame must be closer to its own ensemble's average than to a foreign one. If it is not,
    # the average is not a structure (unfitted frames average to a collapsed centroid) and every
    # number above is an artifact -- refuse to emit a verdict rather than dress it up.
    if eff_bound < 0 or eff_apo < 0:
        raise SystemExit(
            f"\nABORT: negative effect size (bound {eff_bound:.1f} sd, apo {eff_apo:.1f} sd).\n"
            "Each ensemble sits FURTHER from its own average than from the other's, which is\n"
            "geometrically impossible for a real average structure. The average collapsed because\n"
            "the frames were not superimposed before averaging. No verdict is meaningful."
        )

    print(
        "\nCompare against endstate_window_rmsd.py on the receptor leg (separation 2.493 A,\n"
        "apo-vs-window). Interpretation:"
    )
    if separation < 0.6 * 2.493:
        print(
            "  This separation is MUCH SMALLER than the apo-vs-window number, so most of that gap\n"
            "  was the -14.0 window failing to relax rather than true reorganization. That is a\n"
            "  SAMPLING problem: longer MD in the weak-k tail, or Hamiltonian REMD across the\n"
            "  ladder so windows can exchange out of the bound basin."
        )
    elif separation > 1.4 * 2.493:
        print(
            "  This separation is LARGER than the apo-vs-window number -- the window sits partway\n"
            "  between bound and apo. The reorganization is real and the ladder is only partly\n"
            "  crossing it."
        )
    else:
        print(
            "  This separation is COMPARABLE to the apo-vs-window number, so the gap is genuine\n"
            "  reorganization, not window under-convergence. Window insertion will not remove it;\n"
            "  the leg has to sample the transition."
        )

    summary = os.path.join(W, "bound_vs_apo_summary.csv")
    with open(summary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {summary}")


if __name__ == "__main__":
    main()
