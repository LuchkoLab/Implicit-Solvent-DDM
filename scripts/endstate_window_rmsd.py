#!/usr/bin/env python
"""endstate_window_rmsd.py -- is the endstate in the same basin as the weakest restraint window?

WHY
---
On MCL-1 rep1 the receptor leg's endstate -> lambda_window(-14.0) transition has an MBAR overlap of
0.0039 -- the worst in the whole cycle, ~60x below the median -- and carries dG = +5.96 kT with an
error of 0.352 kT, 60x the error of the very next step (0.006).

That should not happen. Restraint force constants are ``2**exponent`` (workflow_phases.py:920), so
the -14.0 window runs at k = 6.1e-5 kcal/mol/A^2: the restraint is off. In the linear-response
regime dA is proportional to dk, and the ladder obeys that (dG roughly doubles as k doubles:
2.39 -> 4.23 -> 7.09 -> 11.84 ...). But the endstate seam has the SMALLEST change in k of any step
(0 -> 6.1e-5, half the increment of -14 -> -13) and costs MORE than double. The endstate is
therefore not the k->0 limit of the ladder; it is a different ensemble.

Near-zero overlap + a real free-energy offset + an error spike is the signature of two trajectories
sampling different basins, not of a stiffness gap. This script tests that directly. It matters
because the two diagnoses have opposite fixes: a stiffness gap is solved by inserting windows (ALS),
while a basin mismatch is not -- you can subdivide 0 -> 6.1e-5 forever and every sub-window will
overlap just as badly with the endstate.

The receptor endstate is produced by a separate phase (``user_defined_endstate`` / ``run_endstate``,
trajectory ``*_basicMD_traj.nc``); ``apo_endstate_dirstruct`` sets ``runtype: "remd"``. A replica-
exchange or user-supplied apo ensemble can easily relax away from the structure the restraint file
targets, which is exactly what this measures.

WHAT THIS DOES
--------------
No MD, no MBAR. Four cpptraj passes over two existing trajectories:

1. average structure of the endstate trajectory
2. average structure of the window trajectory
3. per-frame RMSD of BOTH trajectories to the restraint reference (the coordinates the restraint
   actually pulls toward) -- shows whether either has drifted off it
4. cross RMSD -- endstate frames to the window average, and window frames to the endstate average

The verdict compares spread to separation:

* separation (average-to-average) <= within-basin spread  -> same basin; the overlap problem is
  NOT structural, look at the restraint definition or the scoring instead
* separation >> spread, and both cross distributions sit well outside their own within
  distributions -> different basins; inserting windows will not help

By default the mask is the atoms the restraint file actually restrains, parsed out of the ``iat=``
records in ``restraint.RST``. That is the coordinate the free energy is being computed along, so it
is the mask that matters; ``--mask`` overrides it.

USAGE
-----
  module load amber                       # or: export CPPTRAJ=$AMBERHOME/bin/cpptraj

  # receptor leg, default -14.0 window, endstate trajectory found automatically
  python scripts/endstate_window_rmsd.py \\
      --leg-root <run>/mcl1_rep1/MCL-1_receptor-ligand-1_with_mcl1_uxU_uxU \\
      --endstate-traj <path>/MCL-1_receptor-ligand-1_with_mcl1_uxU_uxU_basicMD_traj.nc

  # any other window, e.g. sanity-check a healthy seam
  python scripts/endstate_window_rmsd.py --leg-root <...> --endstate-traj <...> --window -13.0

  # cheaper on a long trajectory
  python scripts/endstate_window_rmsd.py --leg-root <...> --endstate-traj <...> --stride 10

Writes ``endstate_window_rmsd.csv`` (per-frame) and ``endstate_window_rmsd_summary.csv`` to
``--workdir``, and prints the verdict.

NOTES
-----
* The endstate trajectory usually is NOT under the leg directory -- the endstate phase writes it
  elsewhere, and with ``export_intermediate_files`` off it may only exist inside the Toil jobstore.
  If ``--endstate-traj`` is omitted the script searches a few likely places and, failing that, tells
  you what to look for rather than guessing.
* RMSD is mass-weighted and best-fit on the same mask it measures, so this reports conformational
  difference, not rigid-body drift.
* Frame counts need not match between the two trajectories.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import numpy as np

CPPTRAJ = os.environ.get("CPPTRAJ", "cpptraj")


# --------------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------------
def find_window_dir(leg_root: str, window: str) -> str:
    """Return the MD directory for one restraint window.

    Layout is ``<leg_root>/lambda_window/<charge>/<extdiel>/<exponent>/``, with the trajectory
    directly inside -- the ``igb_*`` level exists only under ``post_processing/``. The complex leg
    adds an orientational level, and the number of leading levels varies by leg, so glob both depths
    rather than hardcode and keep whichever actually holds a trajectory.
    """
    hits = []
    for pattern in (
        os.path.join(leg_root, "lambda_window", "*", "*", window),
        os.path.join(leg_root, "lambda_window", "*", "*", window, "*"),
        os.path.join(leg_root, "lambda_window", "*", "*", "*", window),
        os.path.join(leg_root, "lambda_window", "*", "*", "*", window, "*"),
    ):
        hits += [
            d
            for d in glob.glob(pattern)
            if os.path.isdir(d) and glob.glob(os.path.join(d, "*prod_traj.nc"))
        ]
    if not hits:
        raise SystemExit(
            f"no lambda_window/{window} directory with a *prod_traj.nc under {leg_root}\n"
            f"available windows: "
            + ", ".join(
                sorted(
                    os.path.basename(p)
                    for p in glob.glob(os.path.join(leg_root, "lambda_window", "*", "*", "*"))
                    if os.path.isdir(p)
                )
            )
        )
    if len(hits) > 1:
        print(f"[warn] {len(hits)} candidate dirs for window {window}; using {hits[0]}")
    return hits[0]


def one(pattern: str, what: str) -> str:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"no {what} matching {pattern}")
    return hits[0]


def guess_endstate_traj(leg_root: str) -> str:
    """Look for ``*_basicMD_traj.nc`` in the usual places above the leg directory."""
    run_root = os.path.dirname(os.path.abspath(leg_root))
    for depth in ("", "*/", "*/*/"):
        hits = sorted(glob.glob(os.path.join(run_root, depth, "*basicMD_traj.nc")))
        if hits:
            return hits[0]
    raise SystemExit(
        "could not find the endstate trajectory (*_basicMD_traj.nc).\n"
        "It is written by the endstate phase, not the leg directory, and with\n"
        "export_intermediate_files=False it may exist only inside the Toil jobstore.\n"
        "Pass it explicitly with --endstate-traj, or re-run the endstate phase with export on."
    )


def restraint_mask(rst_path: str) -> str:
    """Build an AMBER mask from the atoms a restraint file actually restrains.

    Parses every ``iat=`` record and unions the atom numbers. AMBER uses negative iat entries to
    flag group-defined restraints (with the real atoms in following igr records); those are skipped,
    so a purely group-based restraint file yields no mask and the caller falls back.
    """
    text = open(rst_path).read()
    atoms = set()
    for record in re.findall(r"iat\s*=\s*([-\d,\s]+)", text):
        for token in record.replace("\n", " ").split(","):
            token = token.strip()
            if token and re.fullmatch(r"-?\d+", token):
                value = int(token)
                if value > 0:
                    atoms.add(value)
    if not atoms:
        return "", 0
    # Collapse consecutive atoms into ranges. These restraints often cover every atom in the
    # system (2444 of 2444 on the MCL-1 receptor), and a literal comma list is ~15 kB -- long
    # enough to run into cpptraj's input line handling. "@1-2444" is the same selection.
    ordered = sorted(atoms)
    spans, start, prev = [], ordered[0], ordered[0]
    for atom in ordered[1:]:
        if atom == prev + 1:
            prev = atom
            continue
        spans.append((start, prev))
        start = prev = atom
    spans.append((start, prev))
    mask = "@" + ",".join(f"{a}-{b}" if b > a else f"{a}" for a, b in spans)
    return mask, len(ordered)


# --------------------------------------------------------------------------------------------------
# cpptraj
# --------------------------------------------------------------------------------------------------
def run_cpptraj(script: str, workdir: str, tag: str, dry: bool) -> None:
    path = os.path.abspath(os.path.join(workdir, f"{tag}.cpptraj"))
    with open(path, "w") as handle:
        handle.write(script)
    if dry:
        print(f"--- {tag} ---\n{script}")
        return
    # cwd=workdir so the .dat/.rst7 outputs land there by bare name. The -i path and every input
    # path in the script must therefore be absolute -- they were resolved against the caller's CWD.
    proc = subprocess.run(
        [CPPTRAJ, "-i", path], capture_output=True, text=True, cwd=workdir
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"cpptraj failed on {tag} (exit {proc.returncode})")


def read_dat(path: str) -> np.ndarray:
    """Read a one-column cpptraj .dat (frame value) and return the values."""
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


# --------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--leg-root", required=True, help="leg dir (contains lambda_window/)")
    ap.add_argument("--window", default="-14.0", help="restraint exponent (default -14.0)")
    ap.add_argument("--endstate-traj", default=None, help="endstate *_basicMD_traj.nc")
    ap.add_argument("--parm", default=None, help="parm7 (default: the window's own)")
    ap.add_argument("--reference", default=None, help="restraint reference (default: window .ncrst)")
    ap.add_argument("--mask", default=None, help="atom mask (default: atoms in restraint.RST)")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame (default 1)")
    ap.add_argument("--workdir", default="./endstate_window_rmsd_work")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    window_dir = find_window_dir(args.leg_root, args.window)
    # Absolute: cpptraj runs with cwd=workdir, so anything relative to the caller's CWD would break.
    parm = os.path.abspath(args.parm or one(os.path.join(window_dir, "*.parm7"), "parm7"))
    window_traj = os.path.abspath(
        one(os.path.join(window_dir, "*prod_traj.nc"), "window trajectory")
    )
    endstate_traj = os.path.abspath(args.endstate_traj or guess_endstate_traj(args.leg_root))
    reference = os.path.abspath(
        args.reference or one(os.path.join(window_dir, "*.ncrst*"), "reference coordinates")
    )

    mask = args.mask
    if mask is None:
        rst = os.path.join(window_dir, "restraint.RST")
        mask, n_atoms = restraint_mask(rst) if os.path.exists(rst) else ("", 0)
        if mask:
            print(f"[mask] {n_atoms} restrained atoms parsed from {rst}")
        else:
            mask = "@CA"
            print(f"[mask] no atom-indexed restraints found; falling back to {mask}")

    os.makedirs(args.workdir, exist_ok=True)
    print(f"[in]  endstate : {endstate_traj}")
    print(f"[in]  window   : {window_traj}")
    print(f"[in]  parm     : {parm}")
    print(f"[in]  reference: {reference}")
    print(f"[in]  mask     : {mask if len(mask) < 70 else mask[:67] + '...'}")
    print(f"[in]  stride   : {args.stride}\n")

    trajin = f"1 last {args.stride}" if args.stride > 1 else ""

    # 1+2: average structure of each ensemble, and drift from the restraint reference.
    for tag, traj in (("endstate", endstate_traj), ("window", window_traj)):
        run_cpptraj(
            f"parm {parm}\n"
            f"trajin {traj} {trajin}\n"
            f"reference {reference} [ref]\n"
            f"rms toref ref [ref] {mask} mass out {tag}_to_reference.dat\n"
            f"average {tag}_avg.rst7 restart\n"
            "run\n",
            args.workdir,
            f"avg_{tag}",
            args.dry_run,
        )

    # 3: cross RMSD -- each ensemble against the OTHER's average structure.
    for tag, traj, other in (
        ("endstate", endstate_traj, "window"),
        ("window", window_traj, "endstate"),
    ):
        run_cpptraj(
            f"parm {parm}\n"
            f"trajin {traj} {trajin}\n"
            f"reference {tag}_avg.rst7 [own]\n"
            f"reference {other}_avg.rst7 [other]\n"
            f"rms own ref [own] {mask} mass out {tag}_to_own_avg.dat\n"
            f"rms cross ref [other] {mask} mass out {tag}_to_{other}_avg.dat\n"
            "run\n",
            args.workdir,
            f"cross_{tag}",
            args.dry_run,
        )

    # 4: the single separation number -- average structure to average structure.
    run_cpptraj(
        f"parm {parm}\n"
        f"trajin endstate_avg.rst7\n"
        f"reference window_avg.rst7 [w]\n"
        f"rms sep ref [w] {mask} mass out avg_to_avg.dat\n"
        "run\n",
        args.workdir,
        "separation",
        args.dry_run,
    )

    if args.dry_run:
        return

    # ----------------------------------------------------------------------------------------------
    # Verdict
    # ----------------------------------------------------------------------------------------------
    W = args.workdir
    endstate_own = read_dat(os.path.join(W, "endstate_to_own_avg.dat"))
    window_own = read_dat(os.path.join(W, "window_to_own_avg.dat"))
    endstate_cross = read_dat(os.path.join(W, "endstate_to_window_avg.dat"))
    window_cross = read_dat(os.path.join(W, "window_to_endstate_avg.dat"))
    endstate_ref = read_dat(os.path.join(W, "endstate_to_reference.dat"))
    window_ref = read_dat(os.path.join(W, "window_to_reference.dat"))
    separation = float(read_dat(os.path.join(W, "avg_to_avg.dat"))[0])

    rows = [
        describe("endstate -> restraint reference", endstate_ref),
        describe("window   -> restraint reference", window_ref),
        describe("endstate -> own average (spread)", endstate_own),
        describe("window   -> own average (spread)", window_own),
        describe("endstate -> window average (cross)", endstate_cross),
        describe("window   -> endstate average (cross)", window_cross),
    ]

    print(f"{'set':38s} {'n':>6s} {'mean':>7s} {'sd':>6s} {'min':>7s} {'p50':>7s} {'max':>7s}")
    for row in rows:
        print(
            f"{row['set']:38s} {row['n']:6d} {row['mean']:7.3f} {row['sd']:6.3f} "
            f"{row['min']:7.3f} {row['p50']:7.3f} {row['max']:7.3f}"
        )

    spread = max(endstate_own.mean(), window_own.mean())
    ratio = separation / spread if spread > 0 else float("inf")

    # separation/spread alone is too crude: on a 2444-atom mask an average displacement of only a
    # couple of Angstrom already destroys phase-space overlap, so a "modest" ratio can still mean
    # two ensembles that never visit each other's territory. Judge instead by how far each cross
    # distribution sits from that ensemble's OWN spread, in units of its own width.
    def effect(own, cross):
        sd = own.std(ddof=1)
        return (cross.mean() - own.mean()) / sd if sd > 0 else float("inf")

    eff_window = effect(window_own, window_cross)
    eff_endstate = effect(endstate_own, endstate_cross)
    weakest = min(eff_window, eff_endstate)
    disjoint = (
        window_cross.min() > window_own.max() or endstate_cross.min() > endstate_own.max()
    )

    print(f"\nseparation (average-to-average) : {separation:.3f} A")
    print(f"within-basin spread (larger of) : {spread:.3f} A")
    print(f"separation / spread             : {ratio:.2f}")
    print(f"effect size, window frames      : {eff_window:.1f} sd from their own spread")
    print(f"effect size, endstate frames    : {eff_endstate:.1f} sd from their own spread")
    print(f"ensembles disjoint on one side  : {disjoint}")
    print(f"drift from restraint reference  : endstate {endstate_ref.mean():.2f} A vs "
          f"window {window_ref.mean():.2f} A ({endstate_ref.mean()/window_ref.mean():.2f}x)")

    if weakest >= 3.0 or disjoint:
        print(
            "\nVERDICT: DIFFERENT BASINS.\n"
            "  The two ensembles are further apart than either is wide, so the poor overlap is\n"
            "  structural. Inserting windows between the endstate and this one will NOT help --\n"
            "  every sub-window inherits the same mismatch. Fix the sampling instead: equilibrate\n"
            "  the endstate to the restraint reference, or apply a weak restraint to the endstate\n"
            "  so both ends of the seam sit in the same basin."
        )
    elif weakest >= 1.5:
        print(
            "\nVERDICT: MARGINAL.\n"
            "  The ensembles are displaced but still visit each other's territory. Structural drift\n"
            "  is contributing but may not be the whole story; check whether the endstate drift from\n"
            "  the restraint reference (row 1) is much larger than the window's (row 2)."
        )
    else:
        print(
            "\nVERDICT: SAME BASIN.\n"
            "  The ensembles sit on top of each other, so the overlap problem is NOT structural.\n"
            "  Look instead at the restraint definition at this window and at what the endstate\n"
            "  Hamiltonian actually is -- the endstate carries conformational_restraint=0.0, which\n"
            "  under the 2**exponent convention reads as k=1.0 rather than k=0."
        )

    import csv

    summary = os.path.join(W, "endstate_window_rmsd_summary.csv")
    with open(summary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_frame = os.path.join(W, "endstate_window_rmsd.csv")
    with open(per_frame, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ensemble", "frame", "to_restraint_reference", "to_own_avg", "to_other_avg"])
        for name, ref, own, cross in (
            ("endstate", endstate_ref, endstate_own, endstate_cross),
            ("window", window_ref, window_own, window_cross),
        ):
            for i in range(min(len(ref), len(own), len(cross))):
                writer.writerow([name, i + 1, f"{ref[i]:.4f}", f"{own[i]:.4f}", f"{cross[i]:.4f}"])

    print(f"\nwrote {summary}\nwrote {per_frame}")


if __name__ == "__main__":
    main()
