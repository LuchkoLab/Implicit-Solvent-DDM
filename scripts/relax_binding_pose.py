#!/usr/bin/env python
"""Relax a binding pose once under the production GB settings with the pose restrained."""
import argparse
import os
import re
import shutil
import subprocess as sp
import sys

import numpy as np
import pytraj as pt

from implicit_solvent_ddm.pose_clusters import build_pocket_atoms

# sander/pmemd declare restraintmask as character(len=256)
MAX_MASK_LENGTH = 256
DIVERGED = re.compile(r"NaN|Infinity|\*{5,}")
# A of RMSD rise, 2nd quarter -> last quarter of 03_equil, that counts as still drifting
PLATEAU_TOL = 0.25

MIN_TEMPLATE = """{title}
&cntrl
  imin = 1, maxcyc = {maxcyc}, ncyc = {ncyc}, drms = 0.1,
  ntb = 0, cut = 999.0,
  igb = {igb}, saltcon = {saltcon}, extdiel = {extdiel}, gbsa = 0,{rgbmax}
  ntr = 1, restraint_wt = {weight},
  restraintmask = '{mask}',
  ntpr = 500, ntxo = 1,
/
"""

MD_TEMPLATE = """{title}
&cntrl
  imin = 0, irest = {irest}, ntx = {ntx},
  nstlim = {nstlim}, dt = {dt},
  nscm = 0, ntb = 0, cut = 999.0,
  igb = {igb}, saltcon = {saltcon}, extdiel = {extdiel}, gbsa = 0,{rgbmax}
  ntt = 3, gamma_ln = 1.0, ig = {ig}, tempi = {tempi}, temp0 = {temp0},
  ntc = 2, ntf = 2,
  ntr = 1, restraint_wt = {weight},
  restraintmask = '{mask}',
  nmropt = {nmropt},
  ntpr = {ntpr}, ntwx = {ntwx}, ntxo = 1, ioutfm = 1,
/
{wt}"""

HEAT_WT = """&wt type = 'TEMP0', istep1 = 0, istep2 = {ramp}, value1 = {t0}, value2 = {t1} /
&wt type = 'TEMP0', istep1 = {ramp1}, istep2 = {nstlim}, value1 = {t1}, value2 = {t1} /
&wt type = 'END' /
"""


def compress(numbers):
    """Collapse sorted integers into an AMBER range list.

    Parameters
    ----------
    numbers : iterable of int
        1-based residue or atom numbers.

    Returns
    -------
    str
        e.g. ``"3-5,9,12-14"``.
    """
    numbers = sorted(set(int(n) for n in numbers))
    runs, start, prev = [], numbers[0], numbers[0]
    for n in numbers[1:]:
        if n != prev + 1:
            runs.append((start, prev))
            start = n
        prev = n
    runs.append((start, prev))
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def pose_restraint_mask(top, pocket_atoms, ligand_mask):
    """AMBER mask holding the ligand heavy atoms and the pocket atoms.

    Parameters
    ----------
    top : pytraj.Topology
        Complex topology.
    pocket_atoms : numpy.ndarray
        0-based pocket atom indices from ``build_pocket_atoms``.
    ligand_mask : str
        AMBER mask selecting the ligand.

    Returns
    -------
    str
        Mask for ``restraintmask``.

    Raises
    ------
    ValueError
        If the mask exceeds the 256-character ``restraintmask`` field.
    """
    names = {top.atom(int(i)).name for i in pocket_atoms}
    if names == {"CA"}:
        residues = compress(top.atom(int(i)).resid + 1 for i in pocket_atoms)
        pocket = f":{residues}@CA"
    else:
        pocket = "@" + compress(int(i) + 1 for i in pocket_atoms)
    mask = f"({ligand_mask} & !@H=) | ({pocket})"
    if len(mask) > MAX_MASK_LENGTH:
        raise ValueError(
            f"restraint mask is {len(mask)} characters, over the {MAX_MASK_LENGTH} "
            f"restraintmask limit: {mask}"
        )
    return mask


def gb_line(rgbmax):
    """Optional rgbmax entry. Left out, AMBER uses its 25 A default, as production does."""
    return "" if rgbmax is None else f" rgbmax = {rgbmax},"


def write_inputs(args, mask):
    """Write the four stage mdin files.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line.
    mask : str
        Positional-restraint mask.

    Returns
    -------
    list of (str, str, bool)
        ``(stage, mdin path, is_md)`` in run order.
    """
    common = dict(
        igb=args.igb, saltcon=args.saltcon, extdiel=args.extdiel,
        rgbmax=gb_line(args.rgbmax), weight=args.pose_weight, mask=mask,
    )
    heat_steps = int(round(args.heat_ps / 0.002))
    ramp = int(heat_steps * 0.8)
    equil_steps = int(round(args.equil_ns * 1000.0 / args.dt))
    # ~1 frame / 10 ps, so the RMSD plateau check has a few hundred points per ns
    equil_ntwx = max(1, int(round(10.0 / args.dt)))

    texts = {
        "01_min": MIN_TEMPLATE.format(
            title="Restrained minimisation, production GB", maxcyc=5000, ncyc=1000, **common
        ),
        "02_heat": MD_TEMPLATE.format(
            title="Restrained heating", irest=0, ntx=1, nstlim=heat_steps, dt=0.002,
            ig=args.ig, tempi=10.0, temp0=args.temp, nmropt=1,
            ntpr=max(1, heat_steps // 20), ntwx=max(1, heat_steps // 20),
            wt=HEAT_WT.format(ramp=ramp, ramp1=ramp + 1, nstlim=heat_steps, t0=10.0, t1=args.temp),
            **common,
        ),
        "03_equil": MD_TEMPLATE.format(
            title="Restrained equilibration, pose held", irest=1, ntx=5, nstlim=equil_steps,
            dt=args.dt, ig=args.ig, tempi=args.temp, temp0=args.temp, nmropt=0,
            ntpr=equil_ntwx, ntwx=equil_ntwx, wt="", **common,
        ),
        "04_min": MIN_TEMPLATE.format(
            title="Final restrained minimisation", maxcyc=2000, ncyc=500, **common
        ),
    }
    stages = []
    for stage, text in texts.items():
        path = os.path.join(args.output, f"{stage}.mdin")
        with open(path, "w") as handle:
            handle.write(text)
        stages.append((stage, path, stage.endswith(("heat", "equil"))))
    return stages


def check_mdout(mdout):
    """Fail on a diverged or unfinished AMBER run, which can still exit 0.

    Parameters
    ----------
    mdout : str
        Path to the stage mdout.

    Raises
    ------
    RuntimeError
        If the mdout holds NaN / Infinity / overflow asterisks or never reached the timings.
    """
    with open(mdout, errors="replace") as handle:
        text = handle.read()
    # restraintmask echo and the &cntrl echo can hold asterisks; only scan the results
    results = text.split("4.  RESULTS", 1)[-1]
    bad = DIVERGED.search(results)
    if bad:
        raise RuntimeError(f"{mdout}: diverged ({bad.group(0)!r} in the energies)")
    # sander prints "Total time", pmemd "Total wall time"
    if not re.search(r"Total (wall )?time", text):
        raise RuntimeError(f"{mdout}: run did not finish")


def run_stages(args, stages, parm, inpcrd, coords=None):
    """Run each stage, feeding its restart to the next, all referenced to the input pose.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line.
    stages : list of (str, str, bool)
        From ``write_inputs``.
    parm, inpcrd : str
        Complex topology and the input pose (also the ``-ref`` for every stage).
    coords : str, optional
        Starting coordinates of the first stage; the input pose when omitted.

    Returns
    -------
    str
        Restart written by the last stage.
    """
    coords = coords or inpcrd
    for stage, mdin, is_md in stages:
        prefix = os.path.join(args.output, stage)
        cmd = [
            args.engine, "-O", "-i", mdin, "-o", f"{prefix}.mdout", "-p", parm,
            "-c", coords, "-ref", inpcrd, "-r", f"{prefix}.rst7", "-inf", f"{prefix}.mdinfo",
        ]
        if is_md:
            cmd += ["-x", f"{prefix}.nc"]
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            proc = sp.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{stage}: {args.engine} exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
                )
            check_mdout(f"{prefix}.mdout")
        coords = f"{prefix}.rst7"
    return coords


def pose_rmsd(parm, inpcrd, traj, pocket_mask, ligand_heavy_mask):
    """Ligand heavy-atom RMSD to the input pose after superposing on the pocket.

    Parameters
    ----------
    parm, inpcrd : str
        Complex topology and input pose.
    traj : str
        Trajectory, or a single restart.
    pocket_mask, ligand_heavy_mask : str
        pytraj masks.

    Returns
    -------
    numpy.ndarray
        Per-frame RMSD in Angstrom.
    """
    ref = pt.load(inpcrd, parm)
    frames = pt.load(traj, parm)
    frames = pt.superpose(frames, ref=ref, mask=pocket_mask)
    return np.asarray(pt.rmsd(frames, ref=ref, mask=ligand_heavy_mask, nofit=True))


def protein_rmsd(parm, inpcrd, traj, fit_mask, regions):
    """Receptor RMSD to the input after superposing on ``fit_mask``, plus each region.

    Parameters
    ----------
    parm, inpcrd : str
        Complex topology and input pose.
    traj : str
        Trajectory, or a single restart.
    fit_mask : str
        Receptor atoms to superpose on and report (C-alphas, or heavy atoms for a host).
    regions : list of (str, str)
        ``(name, mask)`` pairs reported without refitting.

    Returns
    -------
    dict
        ``{"protein": per-frame RMSD, name: per-frame RMSD, ...}`` in Angstrom.
    """
    ref = pt.load(inpcrd, parm)
    frames = pt.superpose(pt.load(traj, parm), ref=ref, mask=fit_mask)
    out = {"protein": np.asarray(pt.rmsd(frames, ref=ref, mask=fit_mask, nofit=True))}
    for name, mask in regions:
        out[name] = np.asarray(pt.rmsd(frames, ref=ref, mask=mask, nofit=True))
    return out


def eptot_series(mdout):
    """Per-print EPtot from an MD mdout, excluding the averages block.

    Parameters
    ----------
    mdout : str
        Path to the mdout.

    Returns
    -------
    numpy.ndarray
        EPtot values in kcal/mol.
    """
    with open(mdout, errors="replace") as handle:
        text = handle.read().split("A V E R A G E S")[0]
    return np.array([float(x) for x in re.findall(r"EPtot\s*=\s*(-?[\d.]+)", text)])


def quarters(series):
    """Mean of the 2nd quarter and of the last quarter; the 1st is left out as heating tail.

    Parameters
    ----------
    series : numpy.ndarray
        Per-frame values.

    Returns
    -------
    tuple of (float, float)
        (2nd-quarter mean, last-quarter mean).
    """
    q = max(1, series.size // 4)
    return float(series[q:2 * q].mean()), float(series[-q:].mean())


def representative_frame(parm, traj, mask):
    """Medoid of the second half of ``traj``: the frame with the smallest mean RMSD to the rest.

    Parameters
    ----------
    parm, traj : str
        Complex topology and equilibration trajectory.
    mask : str
        Atoms the pairwise RMSD is fitted and measured on.

    Returns
    -------
    int
        0-based frame index into the full trajectory.
    """
    frames = pt.load(traj, parm)
    start = frames.n_frames // 2
    rmsd = np.asarray(pt.pairwise_rmsd(frames[start:], mask=mask))
    return start + int(np.argmin(rmsd.mean(axis=1)))


def write_frame(parm, traj, index, out):
    """Write one trajectory frame as a restart.

    Parameters
    ----------
    parm, traj : str
        Topology and trajectory.
    index : int
        0-based frame.
    out : str
        Output restart path.
    """
    # a one-frame trajin range, so cpptraj does not number the restart
    script = f"parm {parm}\ntrajin {traj} {index + 1} {index + 1}\ntrajout {out} restrt\ngo\n"
    proc = sp.run(["cpptraj"], input=script, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"cpptraj failed writing {out}\n{proc.stdout}\n{proc.stderr}")


def write_split(parm, rst7, mask, out):
    """Write the complex restart with ``mask`` stripped, in the complex's atom order.

    Parameters
    ----------
    parm, rst7 : str
        Complex topology and relaxed restart.
    mask : str
        Atoms to strip.
    out : str
        Output restart path.
    """
    # cpptraj, not pytraj.write_traj: pytraj numbers a single restart as out.1
    script = f"parm {parm}\ntrajin {rst7}\nstrip {mask}\ntrajout {out} restrt\ngo\n"
    proc = sp.run(["cpptraj"], input=script, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"cpptraj failed writing {out}\n{proc.stdout}\n{proc.stderr}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("parm", help="complex .parm7")
    parser.add_argument("inpcrd", help="complex .rst7 holding the one binding pose")
    parser.add_argument("-o", "--output", required=True, help="output directory")
    parser.add_argument("--ligand-mask", default=":LIG")
    parser.add_argument("--receptor-mask", default="!:LIG")
    parser.add_argument("--igb", type=int, default=2)
    parser.add_argument("--saltcon", type=float, default=0.3,
                        help="match the windows (default 0.3), not min.mdin's 0.0")
    parser.add_argument("--extdiel", type=float, default=78.5)
    parser.add_argument("--rgbmax", type=float, default=None,
                        help="omit to keep AMBER's default, as the production mdin does")
    parser.add_argument("--temp", type=float, default=298.0)
    parser.add_argument("--dt", type=float, default=0.004,
                        help="equilibration timestep in ps; 0.004 needs an HMR topology")
    parser.add_argument("--heat-ps", type=float, default=100.0)
    parser.add_argument("--equil-ns", type=float, default=2.0)
    parser.add_argument("--pose-weight", type=float, default=5.0,
                        help="kcal/mol/A^2 on ligand heavy atoms + pocket C-alphas, every stage")
    parser.add_argument("--pocket-radius", type=float, default=8.0,
                        help="A; same default as the flat-bottom pocket and pose clustering")
    parser.add_argument("--ig", type=int, default=71277,
                        help="fixed seed, so a re-run on the same build and GPU reproduces")
    parser.add_argument("--region", action="append", default=[],
                        help="NAME=MASK to track separately, e.g. loop=:22-32@CA; repeatable")
    parser.add_argument("--final-frame", choices=("representative", "last"), default="representative",
                        help="equil frame 04_min starts from: medoid of the 2nd half, or the last")
    parser.add_argument("--engine", default="pmemd.cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    parm, inpcrd = os.path.abspath(args.parm), os.path.abspath(args.inpcrd)
    os.makedirs(args.output, exist_ok=True)
    if not args.dry_run and shutil.which(args.engine) is None:
        sys.exit(f"{args.engine} not on PATH (module load amber)")

    start = pt.load(inpcrd, parm)
    pocket_atoms, ligand_atoms, pocket_label = build_pocket_atoms(
        start.top, start[0].xyz, args.ligand_mask, args.receptor_mask, radius=args.pocket_radius
    )
    mask = pose_restraint_mask(start.top, pocket_atoms, args.ligand_mask)
    print(f"pocket: {pocket_label}; {len(ligand_atoms)} ligand heavy atoms")
    print(f"restraintmask ({args.pose_weight} kcal/mol/A^2): {mask}")

    # a host has no backbone: fit and report on its heavy atoms instead
    fit_mask = f"({args.receptor_mask})&@CA"
    if pt.select_atoms(fit_mask, start.top).size == 0:
        fit_mask = f"({args.receptor_mask})&!@H="
    equil_nc = os.path.join(args.output, "03_equil.nc")

    stages = write_inputs(args, mask)
    min_start = run_stages(args, stages[:3], parm, inpcrd)
    chosen = None
    if args.final_frame == "representative" and not args.dry_run:
        chosen = representative_frame(parm, equil_nc, fit_mask)
        min_start = os.path.join(args.output, "03_equil_representative.rst7")
        write_frame(parm, equil_nc, chosen, min_start)
    final = run_stages(args, stages[3:], parm, inpcrd, coords=min_start)
    if args.dry_run:
        return

    stem = os.path.splitext(os.path.basename(inpcrd))[0]
    relaxed = os.path.join(args.output, f"{stem}_relaxed.rst7")
    shutil.copyfile(final, relaxed)
    write_split(parm, relaxed, args.ligand_mask, os.path.join(args.output, f"{stem}_receptor_relaxed.rst7"))
    write_split(parm, relaxed, args.receptor_mask, os.path.join(args.output, f"{stem}_ligand_relaxed.rst7"))

    pocket_mask = "@" + compress(int(i) + 1 for i in pocket_atoms)
    ligand_mask = "@" + compress(int(i) + 1 for i in ligand_atoms)
    regions = [tuple(r.split("=", 1)) for r in args.region]

    ligand = pose_rmsd(parm, inpcrd, equil_nc, pocket_mask, ligand_mask)
    protein = protein_rmsd(parm, inpcrd, equil_nc, fit_mask, regions)
    final_ligand = float(pose_rmsd(parm, inpcrd, relaxed, pocket_mask, ligand_mask)[0])
    final_protein = {k: float(v[0]) for k, v in protein_rmsd(parm, inpcrd, relaxed, fit_mask, regions).items()}
    eptot = eptot_series(os.path.join(args.output, "03_equil.mdout"))

    time_ns = np.arange(1, ligand.size + 1) * args.equil_ns / ligand.size
    columns = [time_ns, ligand] + list(protein.values())
    header = "time_ns,ligand_rmsd_A," + ",".join(f"{k}_rmsd_A" for k in protein)
    np.savetxt(os.path.join(args.output, "pose_rmsd.csv"), np.column_stack(columns),
               delimiter=",", header=header, comments="", fmt="%.4f")

    trends = {"ligand": quarters(ligand), **{k: quarters(v) for k, v in protein.items()}}
    e_early, e_late = quarters(eptot)
    lines = [
        f"input            {inpcrd}",
        f"relaxed          {relaxed}",
        f"GB               igb={args.igb} saltcon={args.saltcon} extdiel={args.extdiel}",
        f"pose restraint   {args.pose_weight} kcal/mol/A^2 on {mask}",
        f"equilibration    {args.equil_ns} ns at {args.temp} K, dt={args.dt}, ig={args.ig}",
        f"ligand RMSD to input (pocket-superposed): final {final_ligand:.2f} A, "
        f"equil 2nd quarter {trends['ligand'][0]:.2f} A -> last quarter {trends['ligand'][1]:.2f} A",
    ]
    for name in protein:
        label = f"{name} RMSD to input ({fit_mask} fit)" if name == "protein" else f"{name} RMSD to input"
        lines.append(f"{label}: final {final_protein[name]:.2f} A, "
                     f"equil 2nd quarter {trends[name][0]:.2f} A -> last quarter {trends[name][1]:.2f} A")
    lines.append(f"EPtot drift      2nd quarter {e_early:.1f} -> last quarter {e_late:.1f} kcal/mol "
                 f"({e_late - e_early:+.1f})")
    if chosen is None:
        lines.append("final frame      last equil frame")
    else:
        at = ", ".join(f"{k} {v[chosen]:.2f} A (last frame {v[-1]:.2f})" for k, v in protein.items())
        lines.append(f"final frame      equil frame {chosen + 1} of {ligand.size} "
                     f"(t={time_ns[chosen]:.2f} ns), medoid of the 2nd half by {fit_mask} RMSD: {at}")
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(args.output, "relax_summary.txt"), "w") as handle:
        handle.write(summary)
    print(summary, end="")
    # the ligand is held, so the protein and regions are what show an unfinished relaxation
    for name, (early, late) in trends.items():
        if late - early > PLATEAU_TOL:
            print(f"WARNING: {name} RMSD still rising ({early:.2f} -> {late:.2f} A); "
                  f"consider a longer --equil-ns")


if __name__ == "__main__":
    main()
