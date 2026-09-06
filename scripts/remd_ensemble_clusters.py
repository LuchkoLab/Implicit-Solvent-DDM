#!/usr/bin/env python
"""remd_ensemble_clusters.py -- do the complex and apo receptor REMD ensembles share basins?

WHY
---
The receptor leg's ``-14.0`` conformational window is seeded from the complex REMD last frame with
the ligand stripped -- a HOLO receptor structure -- while its MBAR neighbour, the endstate, is the
APO receptor REMD ensemble. At exponent -14 the restraint is effectively off
(``bound_vs_apo_receptor_rmsd.py`` puts the per-atom scale at 4.1e-3 kcal/mol/A^2, a ~12 A thermal
width), so that window's equilibrium distribution IS the apo distribution. It should relax
holo -> apo and overlap the endstate almost perfectly. Measured overlap is 0.0590 at 10 ns and
0.0157 at 2 ns with the endstate diagonal at 0.9231: the window has not relaxed, it is trapped in
the basin it was seeded into.

``bound_vs_apo_receptor_rmsd.py`` already measures HOW FAR apart the two ensembles sit. This script
asks the question that decides what to do about it: do they share any basins at all?

  * If they do, the holo <-> apo barrier is crossable at the temperatures the REMD ladder already
    runs, which argues for making the floor window itself REMD.
  * If they do not, no amount of run length on a 298 K window will cross it, and the fix is to seed
    the floor window from the apo ensemble instead -- which this script also emits.

WHAT THIS DOES
--------------
No MD, no MBAR.

1. cpptraj strips the ligand from the complex REMD trajectory, and the run aborts unless the result
   matches the receptor topology atom-for-atom;
2. MDAnalysis loads both ensembles under that one stripped topology, superposes every frame on a
   common reference, and clustering runs over BOTH at once so cluster identity is shared;
3. cluster membership is cross-tabulated by ensemble of origin;
4. with ``--seeds N``, N apo frames are written as restarts, allocated across clusters in
   proportion to apo population -- the seed set for a multi-seeded floor window.

HOW TO READ IT
--------------
The headline is the fraction of each ensemble sitting in MIXED clusters (both ensembles present,
minority at least ``--mix-fraction`` of the cluster):

  * both fractions near zero -> disjoint basins. The complex REMD never reaches apo-like
    conformations even at its top rung. Seed the floor window from the apo ensemble, and do not
    expect either longer 298 K MD or a REMD floor window to cross it.
  * both fractions substantial -> shared basins. The transition is reachable at REMD temperatures,
    so a REMD floor window can cross it and the complex ensemble does contain apo-like structures.
  * one high, one low -> one ensemble is a subset of the other. If the apo fraction is the low one,
    apo is the broader ensemble and the holo basin is a sub-population of it.

USAGE
-----
  module load amber                      # or: export CPPTRAJ=$AMBERHOME/bin/cpptraj

  python scripts/remd_ensemble_clusters.py \\
      --complex-parm  <run>/remd/MCL-1_ligand-1/MCL-1_ligand-1.parm7 \\
      --complex-traj  <run>/remd/MCL-1_ligand-1/MCL-1_ligand-1_298.0K.nc \\
      --receptor-parm <run>/remd/MCL-1_receptor-...IRN/MCL-1_receptor-...IRN.parm7 \\
      --receptor-traj <run>/remd/MCL-1_receptor-...IRN/MCL-1_receptor-...IRN_298.0K.nc \\
      --stride 10 --seeds 10

NOTES
-----
* Clustering is average-linkage hierarchical on the pairwise RMSD matrix, cut at ``--epsilon``
  Angstrom -- the same scipy approach ``analyze_clusters.py`` uses. cpptraj's own ``cluster``
  command segfaults on these REMD trajectories (SIGSEGV in the pairwise distance stage), and an
  RMSD cutoff in Angstrom is in any case the knob that means "same basin". MDAnalysis ENCORE
  (``encore.ces``) answers a related question with one scalar, but it is deprecated in MDAnalysis
  2.8+ and removed in 3.0, so it is deliberately not a dependency here.
* Frames are superposed on a common reference (first complex frame) and the RMSD is then computed
  without further fitting -- the rationale ``analyze_clusters.py`` gives for its pose clustering.
* ``--mask`` defaults to ``@CA``: the question is fold-scale reorganization, and an all-atom RMSD
  over 2,444 atoms is both slower and dominated by sidechain noise. Set it to the restrained
  selection if you want clusters defined by what actually enters the restraint energy -- that is
  the metric MBAR overlap responds to, at a much higher clustering cost.
* Memory is O(n_frames^2). At ``--stride 10`` on a 10,000-frame pair that is 2,000 frames and a few
  tens of MB; at ``--stride 1`` it is 20,000 frames and ~1.6 GB.
* Seeds come from the APO ensemble, because the floor window's equilibrium is the apo distribution.
  Seeding it from complex-derived structures is the bug this script exists to diagnose.
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
    out = run_cpptraj(f"parm {parm}\nparminfo\n", workdir, tag, dry=False)
    match = re.search(r"(\d+)\s+atoms", out)
    if not match:
        raise SystemExit(f"could not read atom count for {parm}")
    return int(match.group(1))


def load_frames(parm, traj, mask, stride):
    """(0-based frame indices, coordinates) for `mask`, taking every `stride`-th frame."""
    import MDAnalysis as mda

    universe = mda.Universe(parm, traj)
    selection = universe.select_atoms(mask)
    if selection.n_atoms == 0:
        raise SystemExit(f"mask {mask!r} selected no atoms in {parm}")
    frames, coords = [], []
    for step in universe.trajectory[::stride]:
        frames.append(step.frame)
        coords.append(selection.positions.copy())
    return np.asarray(frames), np.asarray(coords, dtype=np.float64)


def superpose(coords, reference):
    """Kabsch-superpose every frame onto `reference`; both are centred on their centroid."""
    ref = reference - reference.mean(axis=0)
    out = np.empty_like(coords)
    for index, frame in enumerate(coords):
        centred = frame - frame.mean(axis=0)
        u_mat, _, vt = np.linalg.svd(centred.T @ ref)
        sign = np.sign(np.linalg.det(vt.T @ u_mat.T))
        rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u_mat.T
        out[index] = centred @ rotation.T
    return out


def ligand_distance(complex_parm, complex_traj, ligand_resname):
    """Per-CA minimum distance to the ligand, from the complex's first frame."""
    import MDAnalysis as mda

    universe = mda.Universe(complex_parm, complex_traj)
    universe.trajectory[0]
    ligand = universe.select_atoms(f"resname {ligand_resname}")
    alphas = universe.select_atoms("protein and name CA")
    if ligand.n_atoms == 0:
        raise SystemExit(f"no atoms matched 'resname {ligand_resname}' in the complex topology")
    return alphas.resids, np.array(
        [np.linalg.norm(ligand.positions - p, axis=1).min() for p in alphas.positions]
    )


def make_figure(coords, n_bound, rows, resids, lig_dist, path, max_frames=3000):
    """Four panels. A/B show the ensembles are disjoint; C/D show WHERE the difference sits.

    The PCA panel is descriptive only -- projecting two groups onto their joint principal axes
    separates them whenever they differ at all, so it illustrates rather than tests. The evidence
    is B (cross-ensemble RMSD lying outside both within-ensemble distributions) and the mixed
    count in the caption of D. C and D are the interpretation: whether the holo/apo difference is
    induced fit at the site or motion somewhere irrelevant to it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist, squareform

    bound_c, apo_c = "#D55E00", "#0072B2"
    n_apo = len(coords) - n_bound
    take = max(1, max_frames // 2)
    bi = np.linspace(0, n_bound - 1, min(take, n_bound)).round().astype(int)
    ai = np.linspace(0, n_apo - 1, min(take, n_apo)).round().astype(int) + n_bound
    sub = coords[np.concatenate([bi, ai])].reshape(len(bi) + len(ai), -1)
    nb = len(bi)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_pca, ax_hist, ax_res, ax_shell = axes.ravel()

    centred = sub - sub.mean(axis=0)
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    proj = centred @ vt[:2].T
    var = sv**2 / (sv**2).sum()
    ax_pca.scatter(proj[:nb, 0], proj[:nb, 1], s=6, alpha=0.35, c=bound_c, label="complex (holo)")
    ax_pca.scatter(proj[nb:, 0], proj[nb:, 1], s=6, alpha=0.35, c=apo_c, label="apo receptor")
    ax_pca.set_xlabel(f"PC1 ({var[0]:.0%})")
    ax_pca.set_ylabel(f"PC2 ({var[1]:.0%})")
    ax_pca.set_title("A. CA conformational space")
    ax_pca.legend(frameon=False, markerscale=2)

    dist = squareform(pdist(sub) / np.sqrt(coords.shape[1]))
    within_b = squareform(dist[:nb, :nb], checks=False)
    within_a = squareform(dist[nb:, nb:], checks=False)
    cross = dist[:nb, nb:].ravel()
    for values, colour, label in (
        (within_b, bound_c, "within complex"),
        (within_a, apo_c, "within apo"),
        (cross, "0.25", "complex vs apo"),
    ):
        ax_hist.hist(values, bins=90, density=True, histtype="step", lw=1.8, color=colour, label=label)
        ax_hist.axvline(values.mean(), color=colour, ls=":", lw=1.2)
    ax_hist.set_xlabel("CA RMSD (\u00c5)")
    ax_hist.set_ylabel("density")
    ax_hist.set_title(f"B. pairwise RMSD (closest cross pair {cross.min():.2f} \u00c5)")
    ax_hist.legend(frameon=False, fontsize=9)

    disp = np.linalg.norm(coords[:n_bound].mean(0) - coords[n_bound:].mean(0), axis=1)
    site = lig_dist < 8.0
    ax_res.bar(resids[~site], disp[~site], color="0.6", width=1.0, label="> 8 \u00c5 from ligand")
    ax_res.bar(resids[site], disp[site], color="#009E73", width=1.0, label="binding site (< 8 \u00c5)")
    ax_res.set_xlabel("residue")
    ax_res.set_ylabel("holo \u2192 apo CA shift (\u00c5)")
    ax_res.set_title(f"C. where the difference is (global RMSD {np.sqrt((disp**2).mean()):.2f} \u00c5)")
    ax_res.legend(frameon=False, fontsize=9)

    msd = disp**2
    edges = [(0, 8), (8, 15), (15, 25), (25, 999)]
    labels, fracs = [], []
    for lo, hi in edges:
        sel = (lig_dist >= lo) & (lig_dist < hi)
        if sel.any():
            labels.append(f"{lo}-{hi} \u00c5" if hi < 999 else f"> {lo} \u00c5")
            fracs.append(msd[sel].sum() / msd.sum())
    colours = ["#009E73" if lab.startswith("0-") else "0.6" for lab in labels]
    ax_shell.bar(labels, fracs, color=colours)
    for i, f in enumerate(fracs):
        ax_shell.text(i, f, f" {f:.0%}", ha="center", va="bottom", fontsize=9)
    ax_shell.set_xlabel("CA distance from ligand")
    ax_shell.set_ylabel("fraction of total MSD")
    ax_shell.set_ylim(0, max(fracs) * 1.2)
    far = lig_dist >= 15
    ax_shell.set_title(f"D. {msd[far].sum() / msd.sum():.0%} of the difference sits > 15 \u00c5 away")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nwrote {path}")
    return {
        "within_complex_mean": float(within_b.mean()),
        "within_apo_mean": float(within_a.mean()),
        "cross_mean": float(cross.mean()),
        "cross_min": float(cross.min()),
        "global_shift_rmsd": float(np.sqrt(msd.mean())),
        "frac_msd_beyond_15A": float(msd[far].sum() / msd.sum()),
        "frac_msd_within_8A": float(msd[lig_dist < 8.0].sum() / msd.sum()),
    }


def allocate(counts: dict, total: int) -> dict:
    """Largest-remainder allocation of `total` seeds across clusters by population."""
    pool = sum(counts.values())
    if pool == 0:
        return {}
    exact = {c: total * n / pool for c, n in counts.items()}
    alloc = {c: int(v) for c, v in exact.items()}
    for cluster in sorted(exact, key=lambda c: exact[c] - alloc[c], reverse=True):
        if sum(alloc.values()) >= total:
            break
        alloc[cluster] += 1
    return {c: n for c, n in alloc.items() if n > 0}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--complex-parm", required=True)
    ap.add_argument("--complex-traj", required=True, help="complex REMD ensemble (demuxed 298 K)")
    ap.add_argument("--receptor-parm", required=True)
    ap.add_argument("--receptor-traj", required=True, help="apo receptor REMD ensemble (demuxed 298 K)")
    ap.add_argument("--ligand-mask", default=":LIG", help="AMBER_masks.ligand_mask (default :LIG)")
    ap.add_argument("--mask", default="name CA", help="MDAnalysis selection (default 'name CA')")
    ap.add_argument("--epsilon", type=float, default=2.0, help="RMSD cutoff, A (default 2.0)")
    ap.add_argument("--clusters", type=int, default=None, help="fixed cluster count; overrides --epsilon")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument(
        "--mix-fraction",
        type=float,
        default=0.10,
        help="a cluster is MIXED when the minority ensemble is at least this fraction of it",
    )
    ap.add_argument("--seeds", type=int, default=0, help="write N apo frames as restarts for seeding")
    ap.add_argument("--plot", action="store_true", help="write remd_cluster_overlap.png")
    ap.add_argument("--plot-frames", type=int, default=3000, help="frames subsampled for the figure")
    ap.add_argument("--workdir", default="./remd_clusters_work")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    complex_parm = os.path.abspath(args.complex_parm)
    complex_traj = os.path.abspath(args.complex_traj)
    receptor_parm = os.path.abspath(args.receptor_parm)
    receptor_traj = os.path.abspath(args.receptor_traj)
    W = os.path.abspath(args.workdir)
    os.makedirs(W, exist_ok=True)
    trajin = f"1 last {args.stride}" if args.stride > 1 else ""

    # 1. Strip the ligand -> the receptor in its BOUND conformation, on the receptor topology.
    run_cpptraj(
        f"parm {complex_parm}\n"
        f"parmstrip {args.ligand_mask}\n"
        "parmwrite out bound_receptor.parm7\n"
        "run\n",
        W,
        "strip_parm",
        args.dry_run,
    )
    run_cpptraj(
        f"parm {complex_parm}\n"
        f"trajin {complex_traj} {trajin}\n"
        f"strip {args.ligand_mask}\n"
        "trajout bound_receptor.nc netcdf\n"
        "run\n",
        W,
        "strip_traj",
        args.dry_run,
    )

    bound_parm = os.path.join(W, "bound_receptor.parm7")
    bound_traj = os.path.join(W, "bound_receptor.nc")

    if args.dry_run:
        return

    n_stripped = natoms(bound_parm, W, "info_stripped")
    n_receptor = natoms(receptor_parm, W, "info_receptor")
    print(f"[check] complex stripped of {args.ligand_mask}: {n_stripped} atoms")
    print(f"[check] receptor topology            : {n_receptor} atoms")
    if n_stripped != n_receptor:
        raise SystemExit(
            f"atom count mismatch ({n_stripped} vs {n_receptor}). Stripping the complex did not\n"
            "reproduce the receptor topology, so the two ensembles are not comparable\n"
            "atom-for-atom. Check --ligand-mask against AMBER_masks.ligand_mask in the run yaml."
        )

    # 2. Both ensembles under the one stripped topology, superposed on a common reference.
    #    bound_receptor.nc was already strided at the strip step; only apo is strided here.
    _, bound_xyz = load_frames(bound_parm, bound_traj, args.mask, 1)
    apo_frames, apo_xyz = load_frames(bound_parm, receptor_traj, args.mask, args.stride)
    n_bound, n_apo = len(bound_xyz), len(apo_xyz)
    n_sel = bound_xyz.shape[1]
    print(f"[frames] bound (complex, stripped): {n_bound}")
    print(f"[frames] apo   (receptor endstate): {n_apo}")
    print(f"[mask]   {args.mask!r} -> {n_sel} atoms\n")

    coords = superpose(np.concatenate([bound_xyz, apo_xyz]), bound_xyz[0])

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    condensed = pdist(coords.reshape(len(coords), -1)) / np.sqrt(n_sel)
    print(f"pairwise RMSD: mean {condensed.mean():.2f} A, max {condensed.max():.2f} A")
    tree = linkage(condensed, method="average")
    if args.clusters:
        assign = fcluster(tree, t=args.clusters, criterion="maxclust")
    else:
        assign = fcluster(tree, t=args.epsilon, criterion="distance")
    bound_ids, apo_ids = assign[:n_bound], assign[n_bound:]

    # 3. Cross-tabulate.
    rows = []
    for cluster in sorted(set(assign.tolist())):
        nb = int((bound_ids == cluster).sum())
        na = int((apo_ids == cluster).sum())
        total = nb + na
        minority = min(nb, na) / total if total else 0.0
        rows.append(
            {
                "cluster": int(cluster),
                "n_total": total,
                "n_bound": nb,
                "n_apo": na,
                "frac_bound": nb / total if total else 0.0,
                "minority_frac": minority,
                "mixed": minority >= args.mix_fraction,
            }
        )

    print(f"\n{'cluster':>7s} {'total':>7s} {'bound':>7s} {'apo':>7s} {'frac_bound':>11s} {'minority':>9s}  mixed")
    for row in rows:
        print(
            f"{row['cluster']:7d} {row['n_total']:7d} {row['n_bound']:7d} {row['n_apo']:7d} "
            f"{row['frac_bound']:11.3f} {row['minority_frac']:9.3f}  {'yes' if row['mixed'] else ''}"
        )

    n_mixed = sum(1 for r in rows if r["mixed"])
    f_bound = sum(r["n_bound"] for r in rows if r["mixed"]) / n_bound
    f_apo = sum(r["n_apo"] for r in rows if r["mixed"]) / n_apo
    print(f"\nclusters: {len(rows)} total, {n_mixed} mixed (minority >= {args.mix_fraction:.2f})")
    print(f"bound frames in mixed clusters : {f_bound:.3f}")
    print(f"apo   frames in mixed clusters : {f_apo:.3f}")

    print("\nverdict:")
    if f_bound < 0.05 and f_apo < 0.05:
        print(
            "  DISJOINT. The two ensembles occupy separate basins -- the complex REMD never reaches\n"
            "  apo-like conformations, even at its top rung. A 298 K floor window seeded from the\n"
            "  complex cannot relax across this, and neither can a REMD floor window, because the\n"
            "  ladder that produced the complex ensemble already failed to cross it. Seed the floor\n"
            "  window from the apo ensemble (--seeds), and treat the endstate seam as a\n"
            "  reorganization the leg must be given a path across, not a spacing problem."
        )
    elif f_bound >= 0.30 and f_apo >= 0.30:
        print(
            "  SHARED. Both ensembles populate common basins, so the holo <-> apo transition IS\n"
            "  reachable at REMD temperatures. A REMD floor window is then the principled fix, and\n"
            "  the trapping is a property of plain 298 K MD rather than of the barrier itself."
        )
    else:
        print(
            "  PARTIAL / NESTED. One ensemble is largely a sub-population of the other\n"
            f"  (bound {f_bound:.3f} vs apo {f_apo:.3f}). The lower fraction names the broader\n"
            "  ensemble. Seeding from the broader one covers the narrower; the reverse does not."
        )

    summary = os.path.join(W, "remd_cluster_contingency.csv")
    with open(summary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {summary}")

    if args.plot:
        resids, lig_dist = ligand_distance(
            complex_parm, complex_traj, args.ligand_mask.lstrip(":")
        )
        if len(resids) != coords.shape[1]:
            raise SystemExit(
                f"complex has {len(resids)} CA atoms but the clustering mask selected "
                f"{coords.shape[1]}. Panel C needs one displacement per CA; use --mask 'name CA'."
            )
        stats = make_figure(
            coords, n_bound, rows, resids, lig_dist,
            os.path.join(W, "remd_cluster_overlap.png"), args.plot_frames,
        )
        print(
            f"  within complex {stats['within_complex_mean']:.2f} A | within apo "
            f"{stats['within_apo_mean']:.2f} A | cross {stats['cross_mean']:.2f} A "
            f"(min {stats['cross_min']:.2f} A)"
        )
        print(
            f"  holo->apo shift {stats['global_shift_rmsd']:.2f} A | "
            f"{stats['frac_msd_beyond_15A']:.0%} of it > 15 A from the ligand, "
            f"{stats['frac_msd_within_8A']:.0%} within 8 A"
        )

    # 4. Seed set, drawn from apo and spread across clusters by apo population.
    if args.seeds:
        members = {
            int(c): np.flatnonzero(apo_ids == c).tolist()
            for c in sorted(set(apo_ids.tolist()))
        }
        alloc = allocate({c: len(v) for c, v in members.items()}, args.seeds)
        seeds = []
        for cluster, count in sorted(alloc.items()):
            picks = np.linspace(0, len(members[cluster]) - 1, count).round().astype(int)
            for k in dict.fromkeys(picks.tolist()):
                # apo_frames holds MDAnalysis 0-based indices; cpptraj counts from 1.
                seeds.append((cluster, int(apo_frames[members[cluster][k]]) + 1))

        os.makedirs(os.path.join(W, "seeds"), exist_ok=True)
        for index, (cluster, frame) in enumerate(seeds):
            run_cpptraj(
                f"parm {receptor_parm}\n"
                f"trajin {receptor_traj} {frame} {frame} 1\n"
                f"trajout seeds/seed_{index:02d}_c{cluster}_f{frame}.rst7 restart\n"
                "run\n",
                W,
                f"seed_{index:02d}",
                False,
            )
        print(f"\nwrote {len(seeds)} apo seeds to {os.path.join(W, 'seeds')}")
        for index, (cluster, frame) in enumerate(seeds):
            print(f"  seed_{index:02d}  cluster {cluster:3d}  apo frame {frame}")
        print(
            "\nThese are original-numbering frames of --receptor-traj. Use them as the starting\n"
            "coordinates for independent segments of the lowest conformational window, each run to\n"
            "total_frames / len(seeds), and equilibrium-detect each segment BEFORE concatenating --\n"
            "detect_equilibration and subsample_correlated_data assume one continuous time series."
        )


if __name__ == "__main__":
    main()
