#!/usr/bin/env python
"""Extend the weak restraint windows of finished runs and re-solve the weak band with the extra frames."""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.io import netcdf_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cycle_audit import run_paths  # noqa: E402

KB = 0.0019872041
DIVERGED = re.compile(r"NaN|Infinity|\*{5,}")
LEGS = ("complex", "receptor")
ARMS = ("production", "extended", "extension_only")
RST_KEYS = re.compile(r"\b(r1|r2|r3|r4|rk2|rk3)\s*=\s*([-\d.eE+]+)")
IAT = re.compile(r"\biat\s*=\s*((?:-?\d+\s*,\s*)*-?\d+)")


# ---------------------------------------------------------------- discovery

def window_dirs(tree, name, leg, max_exponent):
    """Restraint window directories of one leg at or below a conformational exponent.

    Parameters
    ----------
    tree : str
        Parent of the per-system window trees.
    name : str
        Complex system name for the complex leg, receptor system name for the receptor leg.
    leg : str
        ``"complex"`` or ``"receptor"``.
    max_exponent : float
        Keep windows whose conformational exponent is at or below this.

    Returns
    -------
    dict
        ``{exponent: directory}``, strongest restraint first.
    """
    bases = glob.glob(os.path.join(tree, name, "lambda_window", "*", "*"))
    if len(bases) != 1:
        sys.exit(f"expected one lambda_window/<charge>/<extdiel> under {os.path.join(tree, name)}, found {len(bases)}")
    out = {}
    for d in glob.glob(os.path.join(bases[0], "*")):
        try:
            exponent = float(os.path.basename(d))
        except ValueError:
            continue
        if exponent > max_exponent:
            continue
        if leg == "complex":
            sub = [s for s in glob.glob(os.path.join(d, "*")) if os.path.isdir(s)]
            if len(sub) != 1:
                sys.exit(f"expected one orientational directory in {d}, found {len(sub)}")
            d = sub[0]
        out[exponent] = d
    return dict(sorted(out.items(), reverse=True))


def window_files(d):
    """Input and output files of one finished window.

    Parameters
    ----------
    d : str
        Window directory.

    Returns
    -------
    dict
        Paths keyed ``traj``, ``restrt``, ``parm``, ``rst``, ``mdin``, ``mdout``.
    """
    def one(pattern):
        hits = glob.glob(os.path.join(d, pattern))
        if len(hits) != 1:
            sys.exit(f"expected one {pattern} in {d}, found {len(hits)}")
        return hits[0]

    mdins = [f for f in glob.glob(os.path.join(d, "tmp*.tmp")) if re.search(r"\bnstlim\s*=", open(f).read())]
    if not mdins:
        sys.exit(f"no MD input (tmp*.tmp with nstlim) in {d}")
    return dict(traj=one("*_prod_traj.nc"), restrt=one("*_prod_restrt.rst7"), parm=one("*.parm7"),
                rst=one("restraint.RST"), mdin=mdins[0], mdout=one("mdout"))


def md_settings(mdout):
    """MD length and output stride as the finished run echoed them.

    Parameters
    ----------
    mdout : str
        AMBER output file.

    Returns
    -------
    dict
        ``nstlim``, ``ntwx``, ``ntpr`` (int) and ``dt`` (ps).
    """
    text = open(mdout).read()
    out = {}
    for key, cast in (("nstlim", int), ("ntwx", int), ("ntpr", int), ("dt", float)):
        m = re.search(rf"\b{key}\s*=\s*([\d.]+)", text)
        if not m:
            sys.exit(f"{key} not found in {mdout}")
        out[key] = cast(m.group(1))
    return out


def set_param(text, key, value):
    """Replace one ``key = value`` entry of an mdin namelist.

    Parameters
    ----------
    text : str
        mdin contents.
    key : str
        Namelist variable.
    value : object
        New value.

    Returns
    -------
    str
        Updated mdin.
    """
    new, n = re.subn(rf"(\b{key}\s*=\s*)[^,\s!]+", rf"\g<1>{value}", text, count=1)
    if n != 1:
        sys.exit(f"{key} not found in mdin")
    return new


def extension_state(job):
    """Whether an extension job directory holds a finished, sane trajectory.

    Parameters
    ----------
    job : str
        Extension job directory.

    Returns
    -------
    str
        ``"absent"``, ``"done"``, ``"diverged"`` or ``"incomplete"``.
    """
    if not os.path.isdir(job):
        return "absent"
    mdout = os.path.join(job, "extend.mdout")
    if not os.path.exists(mdout) or not os.path.exists(os.path.join(job, "extend_traj.nc")):
        return "incomplete"
    text = open(mdout).read()
    if DIVERGED.search(text):
        return "diverged"
    if "Final Performance Info" not in text and "Total wall time" not in text:
        return "incomplete"
    return "done"


# ---------------------------------------------------------------- setup

def cmd_setup(args):
    """Write one restart job per weak window, a command list and a SLURM array script."""
    out_root = os.path.abspath(args.out)
    jobs = []
    for run in args.runs:
        for system in args.systems:
            p = run_paths(run, system)
            for leg in LEGS:
                name = system if leg == "complex" else p["receptor"]
                for exponent, d in window_dirs(p["tree"], name, leg, args.max_exponent).items():
                    f = window_files(d)
                    s = md_settings(f["mdout"])
                    target = int(round(args.target_ns * 1000.0 / s["dt"]))
                    steps = target - s["nstlim"]
                    if steps <= 0:
                        print(f"skip {p['label']} {name} {exponent:g}: already {s['nstlim'] * s['dt'] / 1000:g} ns")
                        continue
                    if steps % s["ntwx"]:
                        sys.exit(f"{d}: {steps} extension steps is not a multiple of ntwx={s['ntwx']}")
                    job = os.path.join(out_root, p["label"], system, leg, f"{exponent:g}")
                    os.makedirs(job, exist_ok=True)
                    for src, dst in ((f["parm"], "window.parm7"), (f["restrt"], "window_restrt.rst7"),
                                     (f["rst"], "restraint.RST")):
                        link = os.path.join(job, dst)
                        if os.path.lexists(link):
                            os.remove(link)
                        os.symlink(os.path.abspath(src), link)
                    mdin = open(f["mdin"]).read()
                    if not re.search(r"DISANG\s*=\s*restraint\.RST", mdin):
                        sys.exit(f"{f['mdin']}: expected DISANG = restraint.RST")
                    for key, value in (("nstlim", steps), ("irest", 1), ("ntx", 5),
                                       ("ntwx", s["ntwx"]), ("ntpr", s["ntpr"])):
                        mdin = set_param(mdin, key, value)
                    with open(os.path.join(job, "extend.mdin"), "w") as fh:
                        fh.write(mdin)
                    cmd = (f"cd {job} && {args.pmemd} -O -i extend.mdin -p window.parm7 -c window_restrt.rst7 "
                           f"-o extend.mdout -r extend_restrt.rst7 -x extend_traj.nc -inf extend.mdinfo "
                           f"&& ! grep -Eq 'NaN|Infinity|[*]{{5}}' extend.mdout")
                    jobs.append(dict(run=os.path.abspath(run), label=p["label"], system=system, leg=leg,
                                     exponent=exponent, window=os.path.abspath(d), job=job,
                                     from_ns=s["nstlim"] * s["dt"] / 1000, to_ns=args.target_ns, command=cmd))
    if not jobs:
        sys.exit("nothing to extend")
    os.makedirs(os.path.join(out_root, "logs"), exist_ok=True)
    with open(os.path.join(out_root, "commands.txt"), "w") as fh:
        fh.write("\n".join(j["command"] for j in jobs) + "\n")
    with open(os.path.join(out_root, "manifest.json"), "w") as fh:
        json.dump(jobs, fh, indent=1)
    throttle = f"%{args.max_parallel}" if args.max_parallel else ""
    preamble = "\n".join(args.preamble)
    with open(os.path.join(out_root, "submit_extend.sbatch"), "w") as fh:
        fh.write(f"""#!/bin/bash
#SBATCH --job-name=extwin
#SBATCH --partition={args.partition}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time={args.time}
#SBATCH --array=1-{len(jobs)}{throttle}
#SBATCH --output={out_root}/logs/%A_%a.out
{preamble}
line=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {out_root}/commands.txt)
echo "$line"
bash -c "$line"
""")
    print(f"{len(jobs)} extension jobs -> {out_root}")
    print(f"submit: sbatch {os.path.join(out_root, 'submit_extend.sbatch')}")


# ---------------------------------------------------------------- restraint energies

def parse_disang(path):
    """Two-, three- and four-atom restraints of an AMBER DISANG file.

    Parameters
    ----------
    path : str
        Restraint file.

    Returns
    -------
    dict
        ``{n_atoms: dict(idx, r1, r2, r3, r4, rk)}`` with 0-based atom indices.
    """
    groups = {}
    for chunk in open(path).read().split("&rst")[1:]:
        body = chunk.split("/")[0]
        if "igr" in body:
            sys.exit(f"{path}: group restraints (igr) are not supported")
        m = IAT.search(body)
        iat = [int(v) for v in m.group(1).replace(" ", "").split(",") if v]
        if min(iat) < 1:
            sys.exit(f"{path}: negative iat (group restraint) is not supported")
        kv = {k: float(v) for k, v in RST_KEYS.findall(body)}
        if kv["rk2"] != kv["rk3"]:
            sys.exit(f"{path}: rk2 != rk3 in a restraint on atoms {iat}")
        g = groups.setdefault(len(iat), {k: [] for k in ("idx", "r1", "r2", "r3", "r4", "rk")})
        g["idx"].append([a - 1 for a in iat])
        for k in ("r1", "r2", "r3", "r4"):
            g[k].append(kv[k])
        g["rk"].append(kv["rk2"])
    return {n: {k: np.asarray(v) for k, v in g.items()} for n, g in groups.items()}


def unit_energy(x, r1, r2, r3, r4):
    """AMBER flat-bottom restraint energy at unit force constant, linear beyond r1 and r4."""
    return np.select(
        [x < r1, x < r2, x <= r3, x <= r4],
        [(r1 - r2) ** 2 + 2 * (r1 - r2) * (x - r1), (x - r2) ** 2, 0.0, (x - r3) ** 2],
        (r4 - r3) ** 2 + 2 * (r4 - r3) * (x - r4),
    )


def geometry(xyz, idx, r2, r3):
    """Restraint coordinate per frame: distance in A, angle or dihedral in degrees."""
    n = idx.shape[1]
    a = xyz[:, idx[:, 0]]
    b = xyz[:, idx[:, 1]]
    if n == 2:
        return np.linalg.norm(a - b, axis=2)
    c = xyz[:, idx[:, 2]]
    if n == 3:
        u, v = a - b, c - b
        cos = (u * v).sum(2) / (np.linalg.norm(u, axis=2) * np.linalg.norm(v, axis=2))
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    d = xyz[:, idx[:, 3]]
    b1, b2, b3 = b - a, c - b, d - c
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    y = np.linalg.norm(b2, axis=2) * (b1 * n2).sum(2)
    phi = np.degrees(np.arctan2(y, (n1 * n2).sum(2)))
    mid = 0.5 * (r2 + r3)
    return phi + 360.0 * np.round((mid - phi) / 360.0)


def band_restraints(rst_files):
    """Shared restraint geometry of the band and each state's force constants.

    Parameters
    ----------
    rst_files : list of str
        One restraint file per state, in state order.

    Returns
    -------
    dict
        ``{n_atoms: dict(idx, r1, r2, r3, r4, rk)}`` with ``rk`` of shape (terms, states).
    """
    parsed = [parse_disang(p) for p in rst_files]
    ref = parsed[0]
    for p, path in zip(parsed[1:], rst_files[1:]):
        if set(p) != set(ref):
            sys.exit(f"{path}: restraint types differ from {rst_files[0]}")
        for n in ref:
            for k in ("idx", "r1", "r2", "r3", "r4"):
                if not np.array_equal(p[n][k], ref[n][k]):
                    sys.exit(f"{path}: restraint {k} differs from {rst_files[0]}; only force constants may change")
    return {n: dict(ref[n], rk=np.stack([p[n]["rk"] for p in parsed], axis=1)) for n in ref}


def state_energies(xyz, terms, chunk=100):
    """Restraint energy of every frame under every band state.

    Parameters
    ----------
    xyz : numpy.ndarray
        Coordinates, shape (frames, atoms, 3).
    terms : dict
        As returned by ``band_restraints``.
    chunk : int
        Frames per block.

    Returns
    -------
    numpy.ndarray
        kcal/mol, shape (frames, states).
    """
    n_states = next(iter(terms.values()))["rk"].shape[1]
    out = np.zeros((len(xyz), n_states))
    for start in range(0, len(xyz), chunk):
        block = xyz[start:start + chunk]
        for n, t in terms.items():
            x = geometry(block, t["idx"], t["r2"], t["r3"])
            r = [t[k] for k in ("r1", "r2", "r3", "r4")]
            if n > 2:
                x, r = np.radians(x), [np.radians(v) for v in r]
            out[start:start + chunk] += unit_energy(x, *r) @ t["rk"]
    return out


def read_traj(path):
    """Coordinates and times of an AMBER NetCDF trajectory."""
    nc = netcdf_file(path, "r", mmap=False)
    xyz = np.asarray(nc.variables["coordinates"].data, dtype=np.float64)
    time = np.asarray(nc.variables["time"].data, dtype=np.float64)
    nc.close()
    if not np.isfinite(xyz).all() or np.abs(xyz).max() > 1e4:
        sys.exit(f"{path}: non-finite or exploded coordinates")
    return xyz, time


# ---------------------------------------------------------------- MBAR

def subsample(u_own):
    """Indices kept after equilibration detection and decorrelation of one state's series."""
    from pymbar import timeseries

    t0, g, _ = timeseries.detect_equilibration(u_own)
    return t0 + np.asarray(timeseries.subsample_correlated_data(u_own[t0:], g=g), dtype=int)


def solve_band(energies, kt):
    """MBAR over the band from per-window restraint energies.

    Parameters
    ----------
    energies : list of numpy.ndarray
        Per window, kcal/mol of shape (frames, states); window k was sampled at state k.
    kt : float
        kcal/mol.

    Returns
    -------
    dict
        ``dG`` and ``err`` (weakest state minus strongest, kcal/mol), ``min_overlap``,
        ``n_samples`` (list).
    """
    from pymbar import MBAR

    kept = [e[subsample(e[:, k] / kt)] for k, e in enumerate(energies)]
    u_kn = np.concatenate(kept, axis=0).T / kt
    n_k = np.array([len(k) for k in kept])
    mbar = MBAR(u_kn, n_k, solver_protocol="robust")
    r = mbar.compute_free_energy_differences()
    o = mbar.compute_overlap()["matrix"]
    adjacent = [min(o[i, i + 1], o[i + 1, i]) for i in range(len(n_k) - 1)]
    return dict(dG=r["Delta_f"][0, -1] * kt, err=r["dDelta_f"][0, -1] * kt,
                min_overlap=float(min(adjacent)), n_samples=n_k.tolist())


def production_columns(columns):
    """Map conformational exponent to the ``lambda_window`` state column of a cached leg."""
    return {float(c[3].split("_")[0]): c for c in columns if c[0] == "lambda_window"}


def check_against_cache(formatted, own_col, cols, e_own, n, tol, label):
    """Compare coordinate-based state energy differences with the cached post-analysis matrix."""
    df = pd.read_hdf(formatted)
    rows = df.loc[[own_col]]
    worst = 0.0
    for k, col in enumerate(cols):
        if col == own_col or rows[col].isna().all():
            continue
        cached = (rows[col] - rows[own_col]).values[:n]
        mine = e_own[:n, k] - e_own[:n, cols.index(own_col)]
        worst = max(worst, float(np.abs(cached - mine).max()))
    print(f"  check {label}: max |dU(coords) - dU(cache)| = {worst:.3f} kcal/mol")
    if worst > tol:
        sys.exit(f"{label}: coordinate restraint energies disagree with the cache by {worst:.3f} > {tol}")


# ---------------------------------------------------------------- analyze

def cmd_analyze(args):
    """Re-solve the weak band per arm and substitute it into each replicate's dG_bind."""
    kt = KB * args.temperature
    out_root = os.path.abspath(args.out)
    leg_rows, total_rows = [], []
    for run in args.runs:
        for system in args.systems:
            p = run_paths(run, system)
            dg_prod = float(pd.read_hdf(os.path.join(p["cache"], f"deltaG_{system}.h5"))["deltaG"].iloc[0])
            contrib = {arm: 0.0 for arm in ARMS}
            prod_contrib = 0.0
            complete = True
            for leg in LEGS:
                name = system if leg == "complex" else p["receptor"]
                stem = system if leg == "complex" else f"receptor_{name}"
                wins = window_dirs(p["tree"], name, leg, args.anchor)
                if max(wins) != args.anchor:
                    sys.exit(f"{name}: no window at the anchor exponent {args.anchor}")
                files = [window_files(d) for d in wins.values()]
                terms = band_restraints([f["rst"] for f in files])
                fe = pd.read_hdf(os.path.join(p["cache"], f"{stem}_fe.h5"))
                cols_by_exp = production_columns(fe.columns)
                cols = [cols_by_exp[e] for e in wins]
                band_prod = float(fe[cols[-1]].iloc[0] - fe[cols[0]].iloc[0])
                print(f"{p['label']} {name}: {len(wins)} states {max(wins):g} .. {min(wins):g}", flush=True)

                arm_energies = {arm: [] for arm in ARMS}
                extended = 0
                for k, (exponent, f) in enumerate(zip(wins, files)):
                    xyz, _ = read_traj(f["traj"])
                    e_orig = state_energies(xyz, terms)
                    if args.check and k in (0, len(wins) - 1):
                        check_against_cache(os.path.join(p["cache"], f"{stem}_formatted.h5"), cols[k], cols,
                                            e_orig, args.check_frames, args.check_tol, f"{name} {exponent:g}")
                    job = os.path.join(out_root, p["label"], system, leg, f"{exponent:g}")
                    state = extension_state(job)
                    if state in ("diverged", "incomplete"):
                        print(f"  {exponent:g}: extension {state} -- extended arms skipped for this leg")
                        complete = False
                    e_ext = None
                    if state == "done":
                        xyz, _ = read_traj(os.path.join(job, "extend_traj.nc"))
                        e_ext = state_energies(xyz, terms)
                        extended += 1
                    arm_energies["production"].append(e_orig)
                    arm_energies["extended"].append(e_orig if e_ext is None else np.vstack([e_orig, e_ext]))
                    arm_energies["extension_only"].append(e_orig if e_ext is None else e_ext)
                    print(f"  {exponent:g}: {len(e_orig)} frames" + (f" + {len(e_ext)} extension" if e_ext is not None else ""),
                          flush=True)

                sign = 1.0 if leg == "complex" else -1.0
                prod_contrib += sign * band_prod
                for arm in ARMS:
                    if arm != "production" and (not complete or not extended):
                        contrib[arm] = np.nan
                        continue
                    r = solve_band(arm_energies[arm], kt)
                    contrib[arm] += sign * r["dG"]
                    leg_rows.append(dict(run=p["label"], system=system, leg=leg, arm=arm, band_dG=r["dG"],
                                         err=r["err"], min_overlap=r["min_overlap"],
                                         n_samples=sum(r["n_samples"]), extended_windows=extended,
                                         production_band_dG=band_prod))
            for arm in ARMS:
                total_rows.append(dict(run=p["label"], system=system, arm=arm, band_contribution=contrib[arm],
                                       production_band_contribution=prod_contrib, dG_production=dg_prod,
                                       dG=dg_prod - contrib["production"] + contrib[arm]))

    legs, totals = pd.DataFrame(leg_rows), pd.DataFrame(total_rows)
    legs.to_csv(f"{args.o}_legs.csv", index=False)
    totals.to_csv(f"{args.o}_totals.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n== weak band per leg (dG = weakest state - anchor, kcal/mol)")
    print(legs.round(3).to_string(index=False))
    print("\n== dG_bind with the band re-solved per arm, across replicates")
    summary = totals.groupby(["system", "arm"]).agg(
        n=("dG", "count"), dG_mean=("dG", "mean"), dG_sd=("dG", "std"), band_sd=("band_contribution", "std"))
    print(summary.round(3).to_string())
    print(f"\nwrote {args.o}_legs.csv, {args.o}_totals.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("runs", nargs="+", help="run directories holding .cache/, one per replicate")
        p.add_argument("--systems", nargs="+", required=True, help="complex system names")
        p.add_argument("--out", required=True, help="root for the extension jobs")

    p = sub.add_parser("setup", help="write restart jobs and a SLURM array script")
    common(p)
    p.add_argument("--max-exponent", type=float, default=-8.5, help="extend windows at or below this exponent")
    p.add_argument("--target-ns", type=float, default=20.0)
    p.add_argument("--pmemd", default="pmemd.cuda")
    p.add_argument("--partition", default="gpu")
    p.add_argument("--time", default="02:00:00")
    p.add_argument("--max-parallel", type=int, default=None, help="SLURM array throttle")
    p.add_argument("--preamble", action="append", default=[],
                   help="shell line run before pmemd in each task, e.g. 'module load amber' (repeatable)")

    p = sub.add_parser("analyze", help="re-solve the weak band with and without the extension")
    common(p)
    p.add_argument("--anchor", type=float, default=-8.0, help="strongest band state; not extended")
    p.add_argument("--temperature", type=float, default=298.0)
    p.add_argument("--no-check", dest="check", action="store_false",
                   help="skip the comparison against the cached post-analysis energies")
    p.add_argument("--check-frames", type=int, default=20)
    p.add_argument("--check-tol", type=float, default=0.25, help="kcal/mol")
    p.add_argument("-o", default="extend_weak_windows", help="CSV output prefix")

    args = ap.parse_args()
    {"setup": cmd_setup, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
