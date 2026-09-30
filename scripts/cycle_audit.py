#!/usr/bin/env python
"""Per-step cancellation between the complex and receptor legs across replicates."""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

from two_state_overlap import parse_restraints

ACID = {"ASP": "@OD1,OD2", "GLU": "@OE1,OE2"}
BASE = {"LYS": "@NZ", "ARG": "@NE,NH1,NH2", "HIP": "@ND1,NE2"}
KT = 0.0019872041 * 298.0

# (complex step, receptor step) that should cancel; bands named by _band()
PAIRED = [
    ("gas -> gb", "gb -> gas"),
    ("gb", "gb"),
    ("gb -> charge", "restraint -> gb"),
    ("restraint", "restraint"),
    ("restraint -> endstate", "endstate -> restraint"),
]
COMPLEX_ONLY = ["decoupled -> gas", "charge", "charge -> restraint"]


# ---------------------------------------------------------------- discovery

def run_paths(run, system):
    """Cache directory, output tree and receptor system name for one run.

    Parameters
    ----------
    run : str
        Run directory holding ``.cache/``.
    system : str
        Complex system name.

    Returns
    -------
    dict
        ``label``, ``cache``, ``tree`` (parent of the per-system window trees) and ``receptor``.
    """
    cache = os.path.join(run, ".cache", system)
    rec = glob.glob(os.path.join(cache, "receptor_*_fe.h5"))
    if len(rec) != 1:
        sys.exit(f"expected one receptor_*_fe.h5 in {cache}, found {len(rec)}")
    receptor = os.path.basename(rec[0])[len("receptor_"):-len("_fe.h5")]
    trees = [os.path.dirname(p) for p in glob.glob(os.path.join(run, "*", system))
             if os.path.isdir(os.path.join(p, "gb_dielectric"))]
    return dict(label=os.path.basename(os.path.normpath(run)), cache=cache,
                tree=trees[0] if trees else None, receptor=receptor)


def load_leg(cache, stem):
    """Free energies, errors and overlap of one leg, all in cycle order.

    Parameters
    ----------
    cache : str
        Per-system cache directory.
    stem : str
        File stem of the leg's cache files.

    Returns
    -------
    tuple of (list of tuple, numpy.ndarray, numpy.ndarray, numpy.ndarray)
        States, f relative to the first state, error matrix, overlap matrix.
    """
    fe = pd.read_hdf(os.path.join(cache, f"{stem}_fe.h5"))
    err = pd.read_hdf(os.path.join(cache, f"{stem}_error.h5")).values
    ov = pd.read_hdf(os.path.join(cache, f"{stem}_O_MBAR.h5")).values
    states = [tuple(s) for s in fe.columns]
    if err.shape[0] != len(states) or ov.shape[0] != len(states):
        sys.exit(f"{stem}: fe/error/overlap sizes differ")
    return states, fe.values[0], err, ov


def _band(label):
    if label.endswith("_endstate"):
        return "endstate"
    return {"lambda_window": "restraint", "electrostatics": "charge", "gb_dielectric": "gb",
            "no_gb": "gas", "interactions": "gas", "no_interactions": "decoupled"}.get(label, label)


def segments(states, f):
    """Sum each leg's adjacent steps into bands and the junctions between them.

    Parameters
    ----------
    states : list of tuple
        Cycle-ordered states.
    f : numpy.ndarray
        Free energy of each state relative to the first.

    Returns
    -------
    dict
        ``{"gb": dG, "gb -> charge": dG, ...}``.
    """
    out = {}
    for i in range(1, len(states)):
        a, b = _band(states[i - 1][0]), _band(states[i][0])
        key = a if a == b else f"{a} -> {b}"
        out[key] = out.get(key, 0.0) + f[i] - f[i - 1]
    return out


def gb_ladder(states, f, err, ov):
    """Map each GB-band point (dielectric, plus the water and gas anchors) to its state index.

    Parameters
    ----------
    states : list of tuple
        Cycle-ordered states.
    f, err, ov : numpy.ndarray
        As returned by ``load_leg``.

    Returns
    -------
    dict
        ``{78.5: i, 8.0: j, ..., "gas": k}``.
    """
    gb = [i for i, s in enumerate(states) if s[0] == "gb_dielectric"]
    if not gb:
        return {}
    ladder = {float(states[i][1]): i for i in gb}
    lo, hi = min(gb) - 1, max(gb) + 1
    # the gas anchor is whichever neighbour is not a water-dielectric state
    for i in (lo, hi):
        ladder["gas" if _band(states[i][0]) == "gas" else 78.5] = i
    return ladder


# ---------------------------------------------------------------- cancel

def cmd_cancel(args):
    """Print where the complex and receptor legs fail to cancel, per run."""
    band_rows, step_rows = [], []
    for run in args.runs:
        p = run_paths(run, args.system)
        cs, cf, ce, co = load_leg(p["cache"], args.system)
        rs, rf, re_, ro = load_leg(p["cache"], f"receptor_{p['receptor']}")
        cseg, rseg = segments(cs, cf), segments(rs, rf)
        for c, r in PAIRED:
            band_rows.append(dict(run=p["label"], step=c, complex=cseg.get(c, np.nan),
                                  receptor=rseg.get(r, np.nan)))
        for c in COMPLEX_ONLY:
            band_rows.append(dict(run=p["label"], step=c, complex=cseg.get(c, np.nan), receptor=np.nan))

        cl, rl = gb_ladder(cs, cf, ce, co), gb_ladder(rs, rf, re_, ro)
        keys = sorted((k for k in cl if k != "gas"), reverse=True) + ["gas"]
        for a, b in zip(keys[:-1], keys[1:]):
            if a not in rl or b not in rl:
                continue
            ci, cj, ri, rj = cl[a], cl[b], rl[a], rl[b]
            step_rows.append(dict(
                run=p["label"], step=f"{a:g} -> {b}" if b == "gas" else f"{a:g} -> {b:g}",
                complex=cf[cj] - cf[ci], receptor=rf[rj] - rf[ri],
                complex_err=ce[min(ci, cj), max(ci, cj)], receptor_err=re_[min(ri, rj), max(ri, rj)],
                complex_overlap=min(co[ci, cj], co[cj, ci]), receptor_overlap=min(ro[ri, rj], ro[rj, ri])))

    bands = pd.DataFrame(band_rows)
    bands["sum"] = bands.complex + bands.receptor.fillna(0.0)
    steps = pd.DataFrame(step_rows)
    # the legs run the band in opposite directions, so matching ensembles give equal step dG
    steps["mismatch"] = steps.complex - steps.receptor

    pd.set_option("display.width", 200)
    print("== complex + receptor per band (paired rows should be small and equal across runs;"
          " complex-only rows carry binding)")
    print(bands.pivot(index="step", columns="run", values="sum")
          .reindex([c for c, _ in PAIRED] + COMPLEX_ONLY).round(2).to_string())
    print("\n== GB band, per step: complex - receptor (0 when both legs hold the same protein)")
    print(steps.pivot(index="step", columns="run", values="mismatch")
          .reindex(steps.step.drop_duplicates()).round(2).to_string())
    worst = steps.loc[steps.mismatch.abs().idxmax()]
    print(f"\nlargest step mismatch {worst.mismatch:.2f} kcal/mol ({worst.run}, {worst.step}); "
          f"MBAR error on that step {max(worst.complex_err, worst.receptor_err):.2f}, overlap "
          f"{min(worst.complex_overlap, worst.receptor_overlap):.3f}")
    if args.out:
        bands.to_csv(f"{args.out}_bands.csv", index=False)
        steps.to_csv(f"{args.out}_gb_steps.csv", index=False)


# ---------------------------------------------------------------- trajectory helpers

def kabsch(mobile, target):
    """Rotation and translation that superpose ``mobile`` onto ``target``.

    Parameters
    ----------
    mobile, target : numpy.ndarray
        ``(n, 3)`` coordinates of the fitting atoms.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        Rotation, mobile centroid, target centroid: fitted = (x - mc) @ rot + tc.
    """
    mc, tc = mobile.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (target - tc))
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt, mc, tc


def gb_windows(tree, system, receptor):
    """GB-dielectric window trajectories of both legs.

    Parameters
    ----------
    tree : str
        Run output tree holding the per-system window directories.
    system, receptor : str
        Complex and receptor system names.

    Returns
    -------
    list of (str, float, str, str)
        ``(leg, dielectric, trajectory, topology)``.
    """
    out = []
    for leg, name, charge in (("complex", system, "0.0"), ("receptor", receptor, "1.0")):
        base = os.path.join(tree, name, "gb_dielectric", charge)
        for nc in sorted(glob.glob(os.path.join(base, "*", "**", "*_prod_traj.nc"), recursive=True)):
            eps = float(os.path.relpath(nc, base).split(os.sep)[0])
            parm = glob.glob(os.path.join(os.path.dirname(nc), "*.parm7"))
            out.append((leg, eps, nc, parm[0]))
    return out


def charged_groups(top, table, mask=None):
    """Charged-group atoms per residue.

    Parameters
    ----------
    top : pytraj.Topology
        Topology.
    table : dict
        Residue name -> atom mask of its charged group (``ACID`` or ``BASE``).
    mask : str, optional
        Restrict to residues inside this AMBER mask.

    Returns
    -------
    dict
        ``{"ASP25": atom indices, ...}``.
    """
    import pytraj as pt

    allowed = None if mask is None else {top.atom(int(i)).resid for i in pt.select_atoms(mask, top)}
    found = {}
    for res in top.residues:
        if res.name in table and (allowed is None or res.index in allowed):
            idx = pt.select_atoms(f":{res.index + 1}{table[res.name]}", top)
            if idx.size:
                found[f"{res.name}{res.index + 1}"] = idx
    return found


# ---------------------------------------------------------------- saltbridge

def cmd_saltbridge(args):
    """Salt-bridge states of the freed selection's charged residues, per GB window and leg."""
    import pytraj as pt

    frames, devs = [], []
    for run in args.runs:
        p = run_paths(run, args.system)
        means = {}
        for leg, eps, nc, parm in gb_windows(p["tree"], args.system, p["receptor"]):
            traj = pt.iterload(nc, parm, stride=args.stride)
            top = traj.top
            acids, bases = charged_groups(top, ACID), charged_groups(top, BASE)
            own = {**charged_groups(top, ACID, args.free), **charged_groups(top, BASE, args.free)}
            protein = pt.select_atoms(f"!{args.ligand_mask}", top)
            core = pt.select_atoms(f"@CA&!({args.free})&!{args.ligand_mask}", top)
            ref, total, n = None, np.zeros((protein.size, 3)), 0
            for f, frame in enumerate(traj):
                x = frame.xyz
                for name, idx in own.items():
                    partners = bases if name[:3] in ACID else acids
                    best, dist = "none", np.inf
                    for pname, pidx in partners.items():
                        d = np.linalg.norm(x[idx][:, None] - x[pidx][None], axis=2).min()
                        if d < dist:
                            best, dist = pname, d
                    frames.append(dict(run=p["label"], leg=leg, eps=eps, frame=f, residue=name,
                                       partner=best if dist < args.cutoff else "none", dist=dist))
                # running mean structure, fitted on the restrained core
                ref = x[core] if ref is None else ref
                rot, mc, tc = kabsch(x[core], ref)
                total += (x[protein] - mc) @ rot + tc
                n += 1
            # keep plain labels, not the Topology: it dangles once the trajectory is freed
            labels = [(top.atom(int(a)).name, f"{top.atom(int(a)).resname}{top.atom(int(a)).resid + 1}")
                      for a in protein]
            means[(leg, eps)] = (total / n, labels, protein, core)
            print(f"{p['label']} {leg} eps={eps:g}: {n} frames", flush=True)

        for eps in sorted({e for _, e in means}):
            if ("complex", eps) not in means or ("receptor", eps) not in means:
                continue
            xc, labels, protein, core = means[("complex", eps)]
            xr = means[("receptor", eps)][0]
            core_pos = np.searchsorted(protein, core)
            rot, mc, tc = kabsch(xr[core_pos], xc[core_pos])
            d = np.linalg.norm(xc - ((xr - mc) @ rot + tc), axis=1)
            per_res = {}
            for k, (atom_name, key) in enumerate(labels):
                if not atom_name.startswith("H"):
                    per_res[key] = max(per_res.get(key, 0.0), d[k])
            for name, val in per_res.items():
                devs.append(dict(run=p["label"], eps=eps, residue=name, max_heavy_dev=val))

    frames, devs = pd.DataFrame(frames), pd.DataFrame(devs)
    # occupancy of each partner state per leg; total-variation distance 0 = same, 1 = disjoint
    occ = (frames.groupby(["residue", "run", "eps", "leg"]).partner
           .value_counts(normalize=True).unstack(fill_value=0.0))
    rows = []
    for (res, run, eps), g in occ.groupby(level=[0, 1, 2]):
        legs = g.droplevel([0, 1, 2])
        if {"complex", "receptor"} <= set(legs.index):
            tvd = 0.5 * (legs.loc["complex"] - legs.loc["receptor"]).abs().sum()
            top_c, top_r = legs.loc["complex"].idxmax(), legs.loc["receptor"].idxmax()
            rows.append(dict(residue=res, run=run, eps=eps, mismatch=tvd,
                             complex=f"{top_c} {100 * legs.loc['complex', top_c]:.0f}%",
                             receptor=f"{top_r} {100 * legs.loc['receptor', top_r]:.0f}%"))
    summary = pd.DataFrame(rows)

    pd.set_option("display.width", 200)
    for res, g in summary.groupby("residue"):
        print(f"\n== {res}: dominant partner per leg, and state mismatch between legs (0 same, 1 disjoint)")
        print(g.set_index(["run", "eps"])[["complex", "receptor", "mismatch"]].round(2).to_string())
    print("\n== residues differing most between legs (max heavy-atom deviation of mean structures, A)")
    for (run, eps), g in devs.groupby(["run", "eps"]):
        top = g.nlargest(4, "max_heavy_dev")
        print(f"{run} eps={eps:<7g}", ", ".join(f"{r} {v:.1f}" for r, v in zip(top.residue, top.max_heavy_dev)))
    if args.out:
        frames.to_csv(f"{args.out}_frames.csv", index=False)
        summary.to_csv(f"{args.out}_summary.csv", index=False)
        devs.to_csv(f"{args.out}_resdev.csv", index=False)


# ---------------------------------------------------------------- pose

def cmd_pose(args):
    """Ligand pose held in one complex GB window per run, against the input structure."""
    import pytraj as pt

    from implicit_solvent_ddm.pose_clusters import build_pocket_atoms

    ref = pt.load(args.input_rst7, args.input_parm)
    pocket, ligand, _ = build_pocket_atoms(ref.top, ref.xyz[0], args.ligand_mask,
                                           f"!{args.ligand_mask}", radius=args.pocket_radius)
    x0 = ref.xyz[0]
    rows, poses = [], {}
    for run in args.runs:
        p = run_paths(run, args.system)
        found = [w for w in gb_windows(p["tree"], args.system, p["receptor"])
                 if w[0] == "complex" and np.isclose(w[1], args.eps)]
        if not found:
            sys.exit(f"{p['label']}: no complex GB window at eps={args.eps}")
        _, _, nc, parm = found[0]
        rmsd, com, held = [], [], []
        for frame in pt.iterload(nc, parm, stride=args.stride):
            x = frame.xyz
            rot, mc, tc = kabsch(x[pocket], x0[pocket])
            lig = (x[ligand] - mc) @ rot + tc
            rmsd.append(np.sqrt(((lig - x0[ligand]) ** 2).sum(1).mean()))
            com.append(np.linalg.norm(lig.mean(0) - x0[ligand].mean(0)))
            held.append(lig)
        poses[p["label"]] = np.mean(held, axis=0)
        rows.append(dict(run=p["label"], ligand_rmsd=np.mean(rmsd), rmsd_sd=np.std(rmsd),
                         com_shift=np.mean(com), frames=len(rmsd)))
    table = pd.DataFrame(rows)
    print(f"== ligand heavy atoms vs the input pose, pocket-superposed (complex GB window eps={args.eps:g})")
    print(table.round(2).to_string(index=False))
    labels = list(poses)
    print("\n== mean pose RMSD between runs (A)")
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            print(f"{a} vs {b}: {np.sqrt(((poses[a] - poses[b]) ** 2).sum(1).mean()):.2f}")
    if args.out:
        table.to_csv(f"{args.out}_pose.csv", index=False)


# ---------------------------------------------------------------- seam

def receptor_endstate(run, override):
    """Receptor endstate trajectory and topology for one run.

    Parameters
    ----------
    run : str
        Run directory.
    override : str or None
        Explicit trajectory path; its topology is the ``.parm7`` beside it.

    Returns
    -------
    tuple of (str, str)
        Trajectory and topology.
    """
    if override:
        nc = override
    else:
        found = glob.glob(os.path.join(run, "*", "remd", "*receptor*", "*K.nc"))
        if len(found) != 1:
            sys.exit(f"{run}: found {len(found)} receptor endstate trajectories; pass --receptor-endstate")
        nc = found[0]
    parm = glob.glob(os.path.join(os.path.dirname(nc), "*.parm7"))
    if len(parm) != 1:
        sys.exit(f"no unique .parm7 beside {nc}")
    return nc, parm[0]


def weakest_window(tree, receptor):
    """The lowest-exponent receptor restraint window.

    Parameters
    ----------
    tree : str
        Run output tree.
    receptor : str
        Receptor system name.

    Returns
    -------
    tuple of (str, str, str)
        restraint.RST, trajectory, topology.
    """
    rsts = glob.glob(os.path.join(tree, receptor, "lambda_window", "**", "restraint.RST"), recursive=True)
    if not rsts:
        sys.exit(f"no receptor lambda_window restraint.RST under {tree}")
    rst = min(rsts, key=lambda p: float(os.path.basename(os.path.dirname(p))))
    d = os.path.dirname(rst)
    return rst, glob.glob(os.path.join(d, "*_prod_traj.nc"))[0], glob.glob(os.path.join(d, "*.parm7"))[0]


def restraint_by_residue(traj, parm, stride, restraints, n_res, resid):
    """Per-frame restraint energy split half to each atom's residue.

    Parameters
    ----------
    traj, parm : str
        Trajectory and topology.
    stride : int
        Frame stride.
    restraints : tuple
        ``(pairs, r2, r3, rk)`` from ``parse_restraints``.
    n_res : int
        Residue count.
    resid : numpy.ndarray
        0-based residue of every atom.

    Returns
    -------
    numpy.ndarray
        ``(n_frames, n_res)`` energies in kcal/mol.
    """
    import pytraj as pt

    pairs, r2, r3, rk = restraints
    ri, rj = resid[pairs[:, 0]], resid[pairs[:, 1]]
    rows = []
    for frame in pt.iterload(traj, parm, stride=stride):
        x = frame.xyz
        d = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1)
        e = np.where(d < r2, rk * (d - r2) ** 2, np.where(d > r3, rk * (d - r3) ** 2, 0.0))
        rows.append(np.bincount(ri, e / 2, n_res) + np.bincount(rj, e / 2, n_res))
    return np.array(rows)


def two_state(endstate_u, window_u):
    """Endstate -> window dG and overlap; u(endstate)=0, u(window)=dU/kT exactly.

    Parameters
    ----------
    endstate_u, window_u : numpy.ndarray
        Per-frame restraint energy of each ensemble, kcal/mol.

    Returns
    -------
    tuple of (float, float, float)
        dG, its error (kcal/mol) and the min-direction overlap.
    """
    import pymbar

    u = np.zeros((2, endstate_u.size + window_u.size))
    u[1] = np.concatenate([endstate_u, window_u]) / KT
    mbar = pymbar.MBAR(u, [endstate_u.size, window_u.size])
    res = mbar.compute_free_energy_differences()
    ov = mbar.compute_overlap()["matrix"]
    return res["Delta_f"][0, 1] * KT, res["dDelta_f"][0, 1] * KT, min(ov[0, 1], ov[1, 0])


def cmd_seam(args):
    """Receptor seam gap by region per run, and the endstate x reference cross check."""
    import pytraj as pt

    ref = pt.load(args.input_rst7, args.input_parm)
    lig = ref.xyz[0][pt.select_atoms(args.ligand_mask, ref.top)]
    rec_top = None
    runs, rows, per_res_rows = [], [], []
    for i, run in enumerate(args.runs):
        p = run_paths(run, args.system)
        rst, win_nc, win_parm = weakest_window(p["tree"], p["receptor"])
        end_nc, end_parm = receptor_endstate(run, args.receptor_endstate[i] if args.receptor_endstate else None)
        runs.append(dict(p, rst=rst, restraints=parse_restraints(rst), win=(win_nc, win_parm),
                         end=(end_nc, end_parm)))
        if rec_top is None:
            rec_top = pt.load_topology(win_parm)

    # residue numbering is the complex's with the ligand removed; check before using its distances
    names = [r.name for r in rec_top.residues]
    cnames = [r.name for r in ref.top.residues if r.index not in
              {ref.top.atom(int(a)).resid for a in pt.select_atoms(args.ligand_mask, ref.top)}]
    if names != cnames:
        sys.exit("receptor residues do not match the input complex minus the ligand")
    resid = np.array([a.resid for a in rec_top.atoms])
    n_res = rec_top.n_residues
    region = {}
    for mask_name, mask in [("freed", args.free)] + [tuple(r.split("=", 1)) for r in args.region]:
        if mask:
            for a in pt.select_atoms(mask, rec_top):
                region.setdefault(int(resid[a]), mask_name)
    for r in range(n_res):
        if r not in region:
            d = np.linalg.norm(ref.xyz[0][pt.select_atoms(f":{r + 1}", ref.top)][:, None] - lig[None], axis=2).min()
            region[r] = "pocket <=8A" if d <= 8 else ("8-15A" if d <= 15 else ">15A")

    for r in runs:
        end = restraint_by_residue(*r["end"], args.endstate_stride, r["restraints"], n_res, resid)
        win = restraint_by_residue(*r["win"], args.window_stride, r["restraints"], n_res, resid)
        eu, wu = end.sum(1), win.sum(1)
        dg, dg_err, ov = two_state(eu, wu)
        gap = end.mean(0) - win.mean(0)
        split = pd.Series(gap).groupby(pd.Series(region)).sum()
        rows.append(dict(run=r["label"], endstate_dU=eu.mean(), endstate_min=eu.min(), window_dU=wu.mean(),
                         window_max=wu.max(), gap=eu.mean() - wu.mean(), disjoint=eu.min() >= wu.max(),
                         dG=dg, dG_err=dg_err, overlap=ov, **{f"gap[{k}]": v for k, v in split.items()}))
        for k in range(n_res):
            per_res_rows.append(dict(run=r["label"], residue=f"{names[k]}{k + 1}", region=region[k], gap=gap[k]))
        print(f"{r['label']}: seam scored", flush=True)

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n== receptor seam: endstate -> weakest restraint window (restraint energy, kcal/mol)")
    print(table.round(3).to_string(index=False))
    per_res = pd.DataFrame(per_res_rows)
    for label, g in per_res.groupby("run"):
        top = g.nlargest(6, "gap")
        print(f"{label} top residues:", ", ".join(f"{a} {b:.2f} ({c})" for a, b, c in zip(top.residue, top.gap, top.region)))

    # every endstate under every run's reference: a bad endstate is a high row, a bad reference a high column
    cross = pd.DataFrame(index=[f"endstate {r['label']}" for r in runs],
                         columns=[f"ref {r['label']}" for r in runs], dtype=float)
    for e in runs:
        for f in runs:
            u = restraint_by_residue(*e["end"], args.cross_stride, f["restraints"], n_res, resid).sum(1)
            cross.loc[f"endstate {e['label']}", f"ref {f['label']}"] = u.mean()
    print("\n== mean restraint dU: rows = endstate ensemble, columns = restraint reference")
    print(cross.round(2).to_string())
    if args.out:
        table.to_csv(f"{args.out}_seam.csv", index=False)
        per_res.to_csv(f"{args.out}_seam_residues.csv", index=False)
        cross.to_csv(f"{args.out}_crossref.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("runs", nargs="+", help="run directories holding .cache/, one per replicate")
        p.add_argument("--system", required=True, help="complex system name")
        p.add_argument("--ligand-mask", default=":LIG")
        p.add_argument("-o", "--out", help="CSV output prefix")

    common(sub.add_parser("cancel", help="per-band and per-GB-step cancellation between legs"))

    p = sub.add_parser("saltbridge", help="salt-bridge states of a freed selection, per GB window and leg")
    common(p)
    p.add_argument("--free", required=True, help="AMBER mask of the freed selection, e.g. ':22-32'")
    p.add_argument("--cutoff", type=float, default=4.0, help="salt-bridge distance, A")
    p.add_argument("--stride", type=int, default=4)

    p = sub.add_parser("pose", help="ligand pose held per run vs the input structure")
    common(p)
    p.add_argument("--input-parm", required=True)
    p.add_argument("--input-rst7", required=True)
    p.add_argument("--eps", type=float, default=2.0, help="complex GB window to read")
    p.add_argument("--pocket-radius", type=float, default=8.0)
    p.add_argument("--stride", type=int, default=20)

    p = sub.add_parser("seam", help="receptor seam gap by region, plus endstate x reference cross check")
    common(p)
    p.add_argument("--input-parm", required=True)
    p.add_argument("--input-rst7", required=True)
    p.add_argument("--free", default=None, help="freed selection, reported as its own region")
    p.add_argument("--region", action="append", default=[], help="NAME=MASK, repeatable")
    p.add_argument("--receptor-endstate", nargs="+", help="one trajectory per run, if not auto-found")
    p.add_argument("--endstate-stride", type=int, default=10)
    p.add_argument("--window-stride", type=int, default=2)
    p.add_argument("--cross-stride", type=int, default=50)

    args = parser.parse_args()
    {"cancel": cmd_cancel, "saltbridge": cmd_saltbridge, "pose": cmd_pose, "seam": cmd_seam}[args.command](args)


if __name__ == "__main__":
    main()
