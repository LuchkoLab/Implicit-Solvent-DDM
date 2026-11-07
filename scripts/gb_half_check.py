#!/usr/bin/env python
"""Rerun the GB-half-on window on CPU pmemd and compare BAR across gas, GB-half and water with the GPU run."""
import argparse
import concurrent.futures
import glob
import os
import re
import shutil
import subprocess
import sys

import numpy as np

KT = 0.0019872041 * 298.0
DIVERGED = re.compile(r"NaN|Infinity|\*{5,}")


# ---------------------------------------------------------------- discovery

def one(pattern):
    """The single path matching a glob pattern."""
    hits = glob.glob(pattern)
    if len(hits) != 1:
        sys.exit(f"expected one match for {pattern}, found {len(hits)}")
    return hits[0]


def gb_window(tree, eps):
    """GB-dielectric window directory of the complex leg at one external dielectric."""
    for d in glob.glob(os.path.join(tree, "gb_dielectric", "*", "*", "*", "*")):
        if abs(float(d.split(os.sep)[-3]) - eps) < 1e-6:
            return d
    sys.exit(f"no gb_dielectric window at eps={eps} under {tree}")


def band_windows(leg_dir):
    """GB-dielectric windows of one leg, keyed by external dielectric."""
    base = os.path.join(leg_dir, "gb_dielectric")
    out = {}
    for root, _, files in os.walk(base):
        if "mdout" in files:
            out[float(os.path.relpath(root, base).split(os.sep)[1])] = root
    if not out:
        sys.exit(f"no GB-dielectric windows under {base}")
    return dict(sorted(out.items()))


def start_coordinate(window):
    """The window's MD starting structure, as recorded in its simulations.log."""
    m = re.search(r"'-c', '([^']+)'", open(os.path.join(window, "simulations.log")).read())
    if not m:
        sys.exit(f"no -c argument in {window}/simulations.log")
    path = os.path.join(window, os.path.basename(m.group(1)))
    if not os.path.exists(path):
        sys.exit(f"start structure {path} not found")
    return path


def md_mdin(window):
    """The window's MD input (the tmp*.tmp carrying nstlim)."""
    for f in sorted(glob.glob(os.path.join(window, "tmp*.tmp"))):
        if re.search(r"\bnstlim\s*=", open(f).read()):
            return f
    sys.exit(f"no MD mdin in {window}")


def windows(tree, eps):
    """Gas, GB-half (GPU) and water windows of the complex leg."""
    return {
        "gas": one(os.path.join(tree, "interactions", "*", "*", "*", "*")),
        "half": gb_window(tree, eps),
        "water": one(os.path.join(tree, "electrostatics", "0.0", "78.5", "*", "*")),
    }


# ---------------------------------------------------------------- cv.in

def parse_colvars(mdout):
    """RMSD collective variables echoed by NFE in an mdout.

    Parameters
    ----------
    mdout : str
        AMBER output of a run with ``infe = 1``.

    Returns
    -------
    list of dict
        Per CV: ``atoms`` (1-based), ``ref`` (echoed reference, shape (n, 3)), ``position`` and
        ``strength``.
    """
    text = open(mdout).read()
    cvs = []
    for block in re.split(r"NFE : CV #\d+", text)[1:]:
        pos = [float(v) for v in re.search(r"position\s*=\s*([-\d.,\s]+)", block).group(1).split(",")[:4]]
        k = [float(v) for v in re.search(r"strength\s*=\s*([-\d.,\s]+)", block).group(1).split(",")[:2]]
        atoms_text = re.search(r"atoms = \((.*?)\)", block, re.S).group(1)
        atoms = [int(a) for a in re.findall(r"(\d+)\s*\[", atoms_text)]
        ref_text = block.split("reference coordinates :")[1].split("NFE : \n")[0]
        ref = np.array([[float(x) for x in m] for m in re.findall(
            r"NFE :\s+\d+ :\s+([-\d.]+),\s+([-\d.]+),\s+([-\d.]+)", ref_text)])
        if len(ref) != len(atoms):
            sys.exit(f"{mdout}: {len(atoms)} atoms but {len(ref)} reference rows in a CV")
        cvs.append(dict(atoms=atoms, ref=ref, position=pos, strength=k))
    if not cvs:
        sys.exit(f"{mdout}: no NFE collective variables echoed")
    return cvs


def write_cv(mdout, parm, coords, out):
    """Rebuild cv.in from the echo, with reference positions taken from the exact start structure."""
    import parmed as pmd
    from implicit_solvent_ddm.restraints import render_colvar

    xyz = pmd.load_file(parm, coords).coordinates
    blocks = ""
    for cv in parse_colvars(mdout):
        idx = np.array(cv["atoms"]) - 1
        ref = xyz[idx]
        worst = float(np.abs(ref - cv["ref"]).max())
        if worst > 2e-3:
            sys.exit(f"start structure differs from the echoed reference by {worst:.4f} A")
        r1, r2, r3, r4 = cv["position"]
        blocks += render_colvar(idx, ref, r3, cv["strength"][0], r1=r1, r2=r2, r4=r4)
        print(f"  CV: {len(idx)} atoms, r3={r3}, k={cv['strength'][0]}, reference matches echo to {worst:.4f} A")
    with open(out, "w") as fh:
        fh.write(blocks)


# ---------------------------------------------------------------- mdin

def set_param(text, key, value):
    """Replace the first ``key = value`` of a namelist."""
    new, n = re.subn(rf"(\b{key}\s*=\s*)[^,\s!]+", rf"\g<1>{value}", text, count=1)
    if n != 1:
        sys.exit(f"{key} not found in mdin")
    return new


def scoring_mdin(mdin):
    """Single-point (imin=5) scoring input with the state's own solvent model and no restraints."""
    text = open(mdin).read()
    for key, value in (("imin", 5), ("ntx", 1), ("irest", 0), ("nmropt", 0)):
        text = set_param(text, key, value)
    text = re.sub(r",?\s*infe\s*=\s*1", "", text)
    return re.sub(r"&pmd.*?/\s*\n", "", text, flags=re.S)


# ---------------------------------------------------------------- setup

def build_job(window, out, eps, ns):
    """Copy one GB window's inputs into ``out`` with a rebuilt cv.in and an ``ns``-long CPU mdin."""
    os.makedirs(out, exist_ok=True)
    parm = one(os.path.join(window, "*.parm7"))
    start = start_coordinate(window)
    for src, dst in ((parm, "system.parm7"), (start, "start.rst7"),
                     (os.path.join(window, "restraint.RST"), "restraint.RST")):
        shutil.copy(src, os.path.join(out, dst))
    write_cv(os.path.join(window, "mdout"), parm, start, os.path.join(out, "cv.in"))
    text = open(md_mdin(window)).read()
    dt = float(re.search(r"\bdt\s*=\s*([\d.]+)", text).group(1))
    text = set_param(text, "nstlim", int(round(ns * 1000 / dt)))
    text = set_param(text, "extdiel", eps)
    if not re.search(r"\binfe\s*=\s*1", text) or "cv.in" not in text:
        sys.exit(f"{window}: mdin lacks the NFE block")
    with open(os.path.join(out, "md.in"), "w") as fh:
        fh.write(text)


MD_LINE = ("{launcher} pmemd.MPI -O -i md.in -p system.parm7 -c start.rst7 "
           "-r restrt.rst7 -x traj.nc -o mdout -inf mdinfo\n"
           "grep -Eq 'NaN|Infinity|[*]{{5}}' mdout && {{ echo 'MD diverged'; exit 1; }}\n")


def cmd_setup(args):
    """Write the CPU GB-half job."""
    tree = os.path.abspath(args.tree)
    out = os.path.abspath(args.out)
    build_job(windows(tree, args.eps)["half"], out, args.eps, args.ns)
    with open(os.path.join(out, "run_md.sh"), "w") as fh:
        fh.write(f"#!/bin/bash\ncd {out}\n" + MD_LINE.format(launcher=f"mpirun -np {args.np}")
                 + "grep -m1 'EGB' mdout\n")
    print(f"CPU job ready: bash {os.path.join(out, 'run_md.sh')}  ({args.ns} ns, eps={args.eps}, {args.np} ranks)")


def cmd_setup_band(args):
    """Write one CPU batch job per GB-dielectric window of each leg, and a submit_all.sh."""
    run = os.path.abspath(args.run)
    out_root = os.path.abspath(args.out)
    legs = {"complex": os.path.join(run, args.system)}
    receptors = glob.glob(os.path.join(run, "MCL-1_receptor-*")) if args.receptor is None else [os.path.join(run, args.receptor)]
    if "receptor" in args.legs:
        if len(receptors) != 1:
            sys.exit(f"expected one receptor tree under {run}; pass --receptor")
        legs["receptor"] = receptors[0]
    submits = []
    for leg in args.legs:
        for eps, window in band_windows(legs[leg]).items():
            job = os.path.join(out_root, leg, f"eps_{eps:.4f}")
            print(f"{leg} eps={eps:.4f}")
            build_job(window, job, eps, args.ns)
            script = os.path.join(job, "submit.sh")
            with open(script, "w") as fh:
                fh.write(f"""#!/bin/bash
#SBATCH --job-name=gb{leg[0]}{eps:.3f}
#SBATCH --partition={args.partition}
#SBATCH --ntasks={args.np}
#SBATCH --cpus-per-task=1
#SBATCH --mem={args.mem}
#SBATCH --time={args.time}
#SBATCH --output={job}/slurm_%j.out
module purge
module load {args.amber_module}
cd {job}
""" + MD_LINE.format(launcher="mpirun -np ${SLURM_NTASKS}") + "grep 'ns/day' mdout | tail -1\n")
            submits.append(script)
    with open(os.path.join(out_root, "submit_all.sh"), "w") as fh:
        fh.write("#!/bin/bash\n" + "".join(f"sbatch {s}\n" for s in submits))
    print(f"{len(submits)} jobs -> bash {os.path.join(out_root, 'submit_all.sh')}")


# ---------------------------------------------------------------- analyze

def score(job):
    """Run one sander imin=5 pass and return its per-frame energies (kcal/mol)."""
    from implicit_solvent_ddm.mdout import min_to_dataframe

    label, mdin, parm, start, traj, workdir = job
    mdout = os.path.join(workdir, f"{label}.mdout")
    if not os.path.exists(mdout) or "Total wall time" not in open(mdout).read():
        subprocess.run(["sander", "-O", "-i", mdin, "-p", parm, "-c", start, "-y", traj,
                        "-o", mdout, "-r", os.path.join(workdir, f"{label}.rst"),
                        "-inf", os.path.join(workdir, f"{label}.info")], check=True)
    return label, min_to_dataframe(mdout)["ENERGY"].to_numpy(dtype=float)


def bar(u_a_own, u_a_other, u_b_own, u_b_other):
    """Two-state MBAR (BAR) from both trajectories' energies under both states.

    Returns
    -------
    tuple of float
        dG(a->b) kcal/mol, its error, overlap a->b and b->a.
    """
    from pymbar import MBAR, timeseries

    def keep(u):
        t0, g, _ = timeseries.detect_equilibration(u)
        return t0 + np.asarray(timeseries.subsample_correlated_data(u[t0:], g=g), dtype=int)

    ia, ib = keep(u_a_own / KT), keep(u_b_own / KT)
    u_kn = np.array([np.r_[u_a_own[ia], u_b_other[ib]], np.r_[u_a_other[ia], u_b_own[ib]]]) / KT
    mbar = MBAR(u_kn, np.array([len(ia), len(ib)]))
    r = mbar.compute_free_energy_differences()
    o = mbar.compute_overlap()["matrix"]
    return r["Delta_f"][0, 1] * KT, r["dDelta_f"][0, 1] * KT, o[0, 1], o[1, 0]


def mean_egb(mdout):
    """Mean EGB printed during an MD run."""
    vals = [float(v) for v in re.findall(r"EGB\s*=\s*([-\d.]+)", open(mdout).read())]
    return float(np.mean(vals)) if vals else float("nan")


def cmd_analyze(args):
    """Score all trajectories and print BAR for gas <-> half <-> water, CPU vs GPU half."""
    tree = os.path.abspath(args.tree)
    win = windows(tree, args.eps)
    out = os.path.abspath(args.out)
    cpu_mdout = os.path.join(out, "mdout")
    if not os.path.exists(cpu_mdout) or DIVERGED.search(open(cpu_mdout).read()):
        sys.exit(f"CPU run missing or diverged: {cpu_mdout}")
    work = os.path.join(out, "scoring")
    os.makedirs(work, exist_ok=True)
    parm, start = os.path.join(out, "system.parm7"), os.path.join(out, "start.rst7")
    mdins = {}
    for state, d in win.items():
        mdins[state] = os.path.join(work, f"score_{state}.in")
        with open(mdins[state], "w") as fh:
            fh.write(scoring_mdin(md_mdin(d)))
    trajs = {
        "gas": one(os.path.join(win["gas"], "*_prod_traj.nc")),
        "half_gpu": one(os.path.join(win["half"], "*_prod_traj.nc")),
        "half_cpu": os.path.join(out, "traj.nc"),
        "water": one(os.path.join(win["water"], "*_prod_traj.nc")),
    }
    needed = {"gas": ("gas", "half"), "half_gpu": ("gas", "half", "water"),
              "half_cpu": ("gas", "half", "water"), "water": ("half", "water")}
    jobs = [(f"{t}__{h}", mdins[h], parm, start, trajs[t], work) for t, hs in needed.items() for h in hs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.nproc) as pool:
        e = dict(pool.map(score, jobs))

    print("\nMD ensembles (mean EGB during MD, kcal/mol):")
    for name, mdout in (("gas", os.path.join(win["gas"], "mdout")), ("half GPU", os.path.join(win["half"], "mdout")),
                        ("half CPU", cpu_mdout), ("water", os.path.join(win["water"], "mdout"))):
        print(f"  {name:9s} {mean_egb(mdout):10.1f}")
    print(f"\nBAR (dG in kcal/mol; overlap forward / reverse), GB half on = eps {args.eps}:")
    print(f"  {'half from':9s} {'link':16s} {'dG':>10s} {'err':>7s} {'overlap':>17s}")
    for tag in ("cpu", "gpu"):
        h = f"half_{tag}"
        total = 0.0
        for link, (a, b) in (("gas -> half", ("gas", h)), ("half -> water", (h, "water"))):
            sa, sb = a.split("_")[0], b.split("_")[0]
            dg, err, f, r = bar(e[f"{a}__{sa}"], e[f"{a}__{sb}"], e[f"{b}__{sb}"], e[f"{b}__{sa}"])
            total += dg
            print(f"  {tag.upper():9s} {link:16s} {dg:10.3f} {err:7.3f}   {f:.4f} / {r:.4f}")
        print(f"  {tag.upper():9s} {'gas -> water':16s} {total:10.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("setup", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--tree", required=True, help="<run>/<label>/<complex system> of the GPU run")
        p.add_argument("--out", required=True, help="directory for the CPU job and its analysis")
        p.add_argument("--eps", type=float, default=2.0, help="external dielectric of the GB-half window")
        if name == "setup":
            p.add_argument("--ns", type=float, default=2.0)
            p.add_argument("--np", type=int, default=30, help="pmemd.MPI ranks")
        else:
            p.add_argument("--nproc", type=int, default=10, help="parallel sander jobs")
    p = sub.add_parser("setup-band", help="one CPU batch job per GB-dielectric window")
    p.add_argument("--run", required=True, help="<run>/<label> directory holding the system trees")
    p.add_argument("--system", required=True, help="complex system name")
    p.add_argument("--receptor", default=None, help="receptor system name (auto-detected if omitted)")
    p.add_argument("--legs", nargs="+", default=["complex", "receptor"], choices=["complex", "receptor"])
    p.add_argument("--out", required=True)
    p.add_argument("--ns", type=float, default=10.0)
    p.add_argument("--np", type=int, default=30, help="pmemd.MPI ranks per window")
    p.add_argument("--partition", default="cpu")
    p.add_argument("--time", default="12:00:00")
    p.add_argument("--mem", default="16G")
    p.add_argument("--amber-module", default="amber/25/gnu")
    args = ap.parse_args()
    {"setup": cmd_setup, "analyze": cmd_analyze, "setup-band": cmd_setup_band}[args.cmd](args)


if __name__ == "__main__":
    main()
