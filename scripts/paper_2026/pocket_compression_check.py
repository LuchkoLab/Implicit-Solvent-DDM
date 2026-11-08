#!/usr/bin/env python
"""Pocket restraint deviations and ligand pose in the LJ-off and LJ-on windows."""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import pytraj as pt

WINDOWS = ("no_interactions", "interactions")
KT = 0.0019872041 * 298.0


def kabsch(mobile, target):
    """Rotation and centroids superposing ``mobile`` onto ``target``.

    Parameters
    ----------
    mobile, target : numpy.ndarray
        ``(n, 3)`` coordinates.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        Rotation, mobile centroid, target centroid: fitted = (x - mc) @ rot + tc.
    """
    mc, tc = mobile.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (target - tc))
    return u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt, mc, tc


def boresch_restraints(path, ligand_set):
    """The ligand-receptor (Boresch) restraints of a DISANG file.

    Parameters
    ----------
    path : str
        ``restraint.RST``.
    ligand_set : set of int
        0-based ligand atom indices.

    Returns
    -------
    list of (list of int, float, float)
        0-based atoms, target (A or degrees), force constant.
    """
    out = []
    for block in re.findall(r"&rst(.*?)/", open(path).read(), re.S):
        iat = [int(v) - 1 for v in re.findall(r"-?\d+", re.search(r"iat\s*=\s*([-\d,\s]+)", block).group(1)) if int(v) > 0]
        if any(a in ligand_set for a in iat) and any(a not in ligand_set for a in iat):
            get = lambda key: float(re.search(key + r"\s*=\s*([-\d.eE+]+)", block).group(1))
            out.append((iat, get("r2"), get("rk2")))
    return out


def geometry(x, atoms):
    """Distance (A), angle or dihedral (degrees) over 2, 3 or 4 atoms.

    Parameters
    ----------
    x : numpy.ndarray
        ``(n_atoms, 3)`` coordinates.
    atoms : list of int
        0-based atom indices.

    Returns
    -------
    float
        The internal coordinate.
    """
    p = x[atoms]
    if len(p) == 2:
        return float(np.linalg.norm(p[0] - p[1]))
    if len(p) == 3:
        a, b = p[0] - p[1], p[2] - p[1]
        return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - (b0 @ b1) * b1, b2 - (b2 @ b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


def window_files(run, system):
    """Trajectory, topology and restraint file of each LJ window.

    Parameters
    ----------
    run : str
        Run directory holding ``.cache/`` and the window tree.
    system : str
        Complex system name.

    Returns
    -------
    dict
        ``{window: (traj, parm, restraint.RST, window dir)}``.
    """
    found = {}
    for window in WINDOWS:
        trajs = glob.glob(os.path.join(run, "*", system, window, "**", "*_prod_traj.nc"), recursive=True)
        if len(trajs) != 1:
            sys.exit(f"expected one {window} trajectory under {run}/*/{system}, found {len(trajs)}")
        d = os.path.dirname(trajs[0])
        found[window] = (trajs[0], glob.glob(os.path.join(d, "*.parm7"))[0], os.path.join(d, "restraint.RST"), d)
    return found


def distance_restraints(path):
    """Two-atom distance restraints of a DISANG file.

    Parameters
    ----------
    path : str
        ``restraint.RST``.

    Returns
    -------
    tuple of numpy.ndarray
        0-based atom pairs ``(n, 2)``, reference distance d0, flat-bottom half-width, force constant.
    """
    pairs, d0, half, k = [], [], [], []
    for block in re.findall(r"&rst(.*?)/", open(path).read(), re.S):
        iat = [int(v) for v in re.findall(r"-?\d+", re.search(r"iat\s*=\s*([-\d,\s]+)", block).group(1)) if int(v) != 0]
        if len(iat) != 2 or min(iat) < 1:
            continue
        get = lambda key: float(re.search(key + r"\s*=\s*([-\d.eE+]+)", block).group(1))
        r2, r3 = get("r2"), get("r3")
        pairs.append((iat[0] - 1, iat[1] - 1))
        d0.append((r2 + r3) / 2)
        half.append((r3 - r2) / 2)
        k.append(get("rk2"))
    return np.array(pairs), np.array(d0), np.array(half), np.array(k)


def reference_file(window_dir, given):
    """The restraint reference the windows started from.

    Parameters
    ----------
    window_dir : str
        LJ-off window directory, where the workflow copies its input coordinates.
    given : str or None
        Explicit path.

    Returns
    -------
    str
        Reference coordinate file.
    """
    if given:
        return given
    candidates = [f for f in glob.glob(os.path.join(window_dir, "*"))
                  if (f.endswith(".rst7") or f.endswith(".ncrst")) and "_prod_restrt" not in f]
    if len(candidates) != 1:
        sys.exit(f"cannot identify the reference in {window_dir} ({candidates}); pass --reference")
    return candidates[0]


def summarise(dev, half, near):
    """Median deviation and where pairs sit in the flat-bottom well.

    Parameters
    ----------
    dev : numpy.ndarray
        ``(frames, pairs)`` deviations d - d0, A.
    half : numpy.ndarray
        Flat-bottom half-width per pair.
    near : float
        Width of the band just inside the compressed wall counted as "near the edge".

    Returns
    -------
    dict
        median, fraction near the compressed edge, past it, and past the stretched wall.
    """
    h = half[None, :]
    return dict(median=float(np.median(dev)),
                near_edge=float(np.mean((dev < -h + near) & (dev >= -h))),
                past_wall=float(np.mean(dev < -h)),
                stretched=float(np.mean(dev > h)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run directory holding .cache/ and the window tree")
    parser.add_argument("--system", required=True)
    parser.add_argument("--ligand-mask", default=":LIG")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--contact", type=float, default=4.5, help="pocket-contact cutoff, A")
    parser.add_argument("--far", type=float, default=15.0, help="control pairs beyond this, A")
    parser.add_argument("--clash", type=float, default=2.5, help="clashing-atom cutoff, A")
    parser.add_argument("--near-edge", type=float, default=0.2)
    parser.add_argument("--narrow", type=float, default=0.1, help="hypothetical flat-bottom half-width, A")
    parser.add_argument("--lig-contact", type=float, default=2.2, help="ligand-atom contact cutoff for section 7, A")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("-o", "--out", help="CSV prefix")
    args = parser.parse_args()

    files = window_files(args.run, args.system)
    pairs, d0, half, k = distance_restraints(files["no_interactions"][2])
    other = distance_restraints(files["interactions"][2])
    if len(other[0]) != len(pairs) or not np.array_equal(other[0], pairs):
        print("WARNING: the two windows' distance restraints differ; the comparison is not controlled")

    parm = files["no_interactions"][1]
    top = pt.load_topology(parm)
    ref = pt.load(reference_file(files["no_interactions"][3], args.reference), parm).xyz[0]
    ligand = pt.select_atoms(args.ligand_mask, top)
    ligand_set = set(ligand.tolist())
    is_h = np.array([a.name.startswith("H") for a in top.atoms])
    dist_to_ligand = np.linalg.norm(ref[:, None] - ref[ligand][None], axis=2).min(axis=1)
    pocket = np.where((dist_to_ligand <= args.contact) & ~np.isin(np.arange(top.n_atoms), ligand))[0]
    pocket_set = set(pocket.tolist())

    receptor_pair = np.array([a not in ligand_set and b not in ligand_set for a, b in pairs])
    touches_pocket = np.array([a in pocket_set or b in pocket_set for a, b in pairs]) & receptor_pair
    far = receptor_pair & (dist_to_ligand[pairs[:, 0]] > args.far) & (dist_to_ligand[pairs[:, 1]] > args.far)
    heavy = ~is_h[pairs[:, 0]] & ~is_h[pairs[:, 1]]
    groups = {"pocket heavy-heavy": touches_pocket & heavy, "pocket with H": touches_pocket & ~heavy,
              "far control heavy-heavy": far & heavy, "far control with H": far & ~heavy}

    bor = boresch_restraints(files["no_interactions"][2], ligand_set)
    ligand_heavy = np.array([i for i in ligand if not is_h[i]])
    pocket_ca = pt.select_atoms(":" + ",".join(str(r) for r in sorted({top.atom(int(i)).resid + 1 for i in pocket})) + "@CA", top)
    l1 = next(a for iat, _, _ in bor if len(iat) == 2 for a in iat if a in ligand_set) if bor else ligand_heavy[0]
    lever = np.linalg.norm(ref[ligand] - ref[l1], axis=1)
    near_counts = np.zeros(ligand.size)
    rec_all = np.setdiff1d(np.arange(top.n_atoms), ligand)
    pose, bvals = {}, {}

    print(f"{args.system} in {args.run}")
    print(f"  {len(pairs)} distance restraints, flat-bottom half-width {np.median(half):.2f} A, k {np.median(k):g}")
    print(f"  {pocket.size} pocket-contact atoms (<= {args.contact} A of the ligand); "
          + ", ".join(f"{name}: {mask.sum()} pairs" for name, mask in groups.items()))

    rows, per_atom, clash_dev, energy = [], {}, [], {}
    for window in WINDOWS:
        traj, wparm = files[window][0], files[window][1]
        dev_frames, closing, n = [], [], 0
        pose[window], bvals[window] = [], []
        for frame in pt.iterload(traj, wparm, stride=args.stride):
            x = frame.xyz
            d = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1)
            dev_frames.append(d - d0)
            # pocket atoms' closest approach to the ligand, versus the reference
            closing.append(np.linalg.norm(x[pocket][:, None] - x[ligand][None], axis=2).min(axis=1)
                           - dist_to_ligand[pocket])
            if window == "no_interactions":
                near = np.linalg.norm(x[pocket][:, None] - x[ligand][None], axis=2).min(axis=1) < args.clash
                clashing = pocket[near]
                sel = (np.isin(pairs[:, 0], clashing) | np.isin(pairs[:, 1], clashing)) & receptor_pair
                if sel.any():
                    clash_dev.append((d - d0)[sel] + half[sel])  # distance from the compressed wall
            # ligand rigid-body pose relative to the pocket
            rot, mc, tc = kabsch(x[pocket_ca], ref[pocket_ca])
            y = (x - mc) @ rot + tc
            lrot, lmc, ltc = kabsch(y[ligand_heavy], ref[ligand_heavy])
            angle = np.degrees(np.arccos(np.clip((np.trace(lrot) - 1) / 2, -1, 1)))
            pose[window].append((np.sqrt(((y[ligand_heavy] - ref[ligand_heavy]) ** 2).sum(1).mean()),
                                 np.linalg.norm(y[ligand_heavy].mean(0) - ref[ligand_heavy].mean(0)), angle))
            bvals[window].append([geometry(x, atoms) for atoms, _, _ in bor])
            if window == "no_interactions":
                closest = np.linalg.norm(x[ligand][:, None] - x[rec_all][None], axis=2).min(axis=1)
                near_counts += closest < args.lig_contact
            n += 1
        dev = np.array(dev_frames)
        per_atom[window] = np.array(closing)
        for name, mask in groups.items():
            rows.append(dict(window=window, group=name, **summarise(dev[:, mask], half[mask], args.near_edge)))
        if window == "no_interactions":
            m = groups["pocket heavy-heavy"] | groups["pocket with H"]
            over = lambda w: (k[m][None, :] * np.clip(np.abs(dev[:, m]) - w, 0, None) ** 2).sum(axis=1)
            energy = dict(current=np.median(over(half[m][None, :])), narrow=np.median(over(args.narrow)))
        print(f"  {window}: {n} frames")

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n== 1. network pairs: deviation d - d0 (A) and where they sit in the flat-bottom well")
    print(table.pivot(index="group", columns="window",
                      values=["median", "near_edge", "past_wall", "stretched"]).round(3).to_string())
    shift = {g: table[(table.group == g) & (table.window == "no_interactions")]["median"].iloc[0]
             - table[(table.group == g) & (table.window == "interactions")]["median"].iloc[0] for g in groups}
    print("   median shift, LJ off minus LJ on: " + ", ".join(f"{g} {v:+.3f}" for g, v in shift.items()))

    print("\n== 2. pocket atoms' closest approach to the ligand, change from the reference (A; negative = closer)")
    for window in WINDOWS:
        c = per_atom[window]
        print(f"   {window:16s} median {np.median(c):+.3f}, 10th pct {np.percentile(c, 10):+.3f}, "
              f"atoms closer by > 0.3 A: {np.mean(c < -0.3) * 100:.1f}%")

    print(f"\n== 3. restraint pairs of atoms clashing (< {args.clash} A) with the LJ-off ligand: distance from the compressed wall (A)")
    if clash_dev:
        c = np.concatenate(clash_dev)
        print(f"   {c.size} pair-frames: median {np.median(c):+.3f}, inside the free zone (> 0): {np.mean(c > 0) * 100:.1f}%, "
              f"past the wall: {np.mean(c < 0) * 100:.1f}%")
    else:
        print("   no clashing atoms")

    print(f"\n== 4. pocket restraint energy of LJ-off frames (median, kcal/mol): current well {energy['current']:.2f}, "
          f"with a +/-{args.narrow} A bottom {energy['narrow']:.2f}")
    fmt = os.path.join(args.run, ".cache", args.system, f"{args.system}_formatted.h5")
    if os.path.exists(fmt):
        f = pd.read_hdf(fmt)
        c0 = [c for c in f.columns if c[0] == "no_interactions"][0]
        c1 = [c for c in f.columns if c[0] == "interactions"][0]
        off = f[f.index.get_level_values(0) == "no_interactions"]
        print(f"   LJ penalty these frames carry now: median dU(LJ on - off) {np.median(off[c1] - off[c0]):.1f} kcal/mol")

    print("\n== 5. ligand pose relative to the pocket (fit on pocket C-alphas), vs the reference")
    spread = {}
    for window in WINDOWS:
        a = np.array(pose[window])
        spread[window] = a
        print(f"   {window:16s} heavy RMSD median {np.median(a[:, 0]):.2f} / 90th pct {np.percentile(a[:, 0], 90):.2f} A | "
              f"centroid shift median {np.median(a[:, 1]):.2f} A | rotation median {np.median(a[:, 2]):.1f} / 90th pct {np.percentile(a[:, 2], 90):.1f} deg")

    print("\n== 6. Boresch coordinates: spread (SD) in each window vs thermal prediction sqrt(kT/2k)")
    wobble = []
    for i, (atoms, target, kf) in enumerate(bor):
        kind = {2: "distance", 3: "angle", 4: "dihedral"}[len(atoms)]
        thermal = np.sqrt(KT / (2 * kf)) * (1.0 if kind == "distance" else 180.0 / np.pi)
        sds = []
        for window in WINDOWS:
            v = np.array(bvals[window])[:, i]
            if kind == "dihedral":
                v = (v - target + 180.0) % 360.0 - 180.0 + target
            sds.append(v.std())
        wobble.append(sds[0] / thermal)
        unit = "A" if kind == "distance" else "deg"
        print(f"   {kind:8s} k={kf:g}: SD LJ off {sds[0]:.3f} / LJ on {sds[1]:.3f} {unit}; thermal {thermal:.3f} {unit}")

    print(f"\n== 7. ligand atoms within {args.lig_contact} A of the protein with LJ off, by lever arm from anchor L1")
    frac = near_counts / max(1, len(pose["no_interactions"]))
    order = np.argsort(-frac)[:8]
    print("   " + ", ".join(f"{top.atom(int(ligand[i])).name} {frac[i] * 100:.0f}% ({lever[i]:.1f} A)" for i in order))
    weighted = (frac * lever).sum() / frac.sum() if frac.sum() else float("nan")
    print(f"   contact-weighted lever arm {weighted:.1f} A vs ligand mean {lever.mean():.1f} A (max {lever.max():.1f} A)")

    off = table[(table.window == "no_interactions") & (table.group == "pocket heavy-heavy")].iloc[0]
    far_shift = abs(shift["far control heavy-heavy"])
    if shift["pocket heavy-heavy"] < -0.05 and off.past_wall < 0.10 and far_shift < 0.03:
        verdict = "A: the pocket closes inside the flat-bottom free play"
    elif shift["pocket heavy-heavy"] < -0.05 and off.past_wall >= 0.10:
        verdict = "B: pocket pairs pushed past the wall; the walls are engaged but soft"
    elif abs(shift["pocket heavy-heavy"]) <= 0.05 and (
            np.mean(per_atom["no_interactions"] < -0.3) - np.mean(per_atom["interactions"] < -0.3)) > 0.05:
        verdict = "C: contacts close but the network pair distances do not"
    else:
        verdict = "unclear; read the tables"
    print(f"\nprovisional verdict: {verdict}")
    wider = np.percentile(spread["no_interactions"][:, 0], 90) > 1.2 * np.percentile(spread["interactions"][:, 0], 90)
    thermal_like = bool(wobble) and np.median(wobble) > 0.7
    print(f"Boresch wobble: ligand pose {'broader' if wider else 'NOT broader'} with LJ off; "
          f"Boresch SDs {'near' if thermal_like else 'well below'} thermal (median ratio {np.median(wobble):.2f})"
          if wobble else "Boresch wobble: no Boresch restraints found")
    if args.out:
        table.to_csv(f"{args.out}_pairs.csv", index=False)


if __name__ == "__main__":
    main()
