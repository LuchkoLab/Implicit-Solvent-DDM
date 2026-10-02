#!/usr/bin/env python
"""Unbiased apo receptor endstate: the workflow's minimization, then the basic-MD endstate mdin with no restraints."""

import argparse
import os
import re
import shutil
import subprocess
import sys

PKG_MIN_MDIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "implicit_solvent_ddm", "templates", "min.mdin",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prmtop", required=True, help="receptor topology")
    p.add_argument("--inpcrd", required=True, help="receptor start coordinates (e.g. *_receptor_relaxed.rst7)")
    p.add_argument("--md-mdin", required=True, help="basic-MD endstate mdin used for the complex")
    p.add_argument("--min-mdin", default=PKG_MIN_MDIN, help="minimization mdin (default: the workflow's)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--name", default="MCL-1_receptor", help="prefix for output files")
    p.add_argument("--ig", type=int, required=True, help="random seed, one per replicate")
    p.add_argument("--temp", type=float, default=298.0, help="only used in the output file name")
    p.add_argument("--exe", default="pmemd.cuda")
    p.add_argument("--dry-run", action="store_true", help="write inputs and print the commands only")
    return p.parse_args()


def prepare_mdin(text, ig=None):
    """Point DISANG at an empty restraint file and optionally fix the seed.

    Parameters
    ----------
    text : str
        mdin template.
    ig : int, optional
        Seed replacing the template's ``ig``.

    Returns
    -------
    str
        mdin text.
    """
    if not re.search(r"DISANG\s*=", text):
        sys.exit("template has no DISANG line")
    text = re.sub(r"DISANG\s*=\s*\S+", "DISANG=empty.restraint", text)
    if ig is not None:
        pattern = re.compile(r"\big\s*=\s*-?\d+", re.IGNORECASE)
        if not pattern.search(text):
            sys.exit("template has no ig to replace")
        text = pattern.sub(f"ig = {ig}", text, count=1)
    return text


def run(cmd, cwd, dry_run):
    print("[run]", " ".join(cmd), flush=True)
    if dry_run:
        return
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"{cmd[0]} exited {r.returncode}; see {cwd}")


def check_mdout(path):
    """Exit if the mdout is missing, never finished, or holds a NaN/overflow energy."""
    with open(path, errors="replace") as fh:
        text = fh.read()
    if "Final Performance Info" not in text and "FINAL RESULTS" not in text:
        sys.exit(f"{path} did not finish")
    if re.search(r"EPtot\s*=\s*(\*+|NaN)", text, re.IGNORECASE):
        sys.exit(f"{path} has a NaN or overflowed energy")


def main():
    args = parse_args()
    exe = shutil.which(args.exe)
    if exe is None and not args.dry_run:
        sys.exit(f"{args.exe} not found on PATH (module load amber/25/gnu?)")
    exe = exe or args.exe
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    prmtop = os.path.join(out, os.path.basename(args.prmtop))
    inpcrd = os.path.join(out, "start.rst7")
    shutil.copy(args.prmtop, prmtop)
    shutil.copy(args.inpcrd, inpcrd)
    open(os.path.join(out, "empty.restraint"), "w").close()
    with open(args.min_mdin) as fh:
        min_text = prepare_mdin(fh.read())
    with open(args.md_mdin) as fh:
        md_text = prepare_mdin(fh.read(), args.ig)
    for name, text in (("min.mdin", min_text), ("md.mdin", md_text)):
        with open(os.path.join(out, name), "w") as fh:
            fh.write(text)

    traj = f"{args.name}_{args.temp}K.nc"
    run([exe, "-O", "-i", "min.mdin", "-p", prmtop, "-c", inpcrd,
         "-o", "min.mdout", "-r", "min.rst7", "-inf", "min.mdinfo"], out, args.dry_run)
    if not args.dry_run:
        check_mdout(os.path.join(out, "min.mdout"))
    run([exe, "-O", "-i", "md.mdin", "-p", prmtop, "-c", "min.rst7",
         "-o", "md.mdout", "-r", f"{args.name}_{args.temp}K_lastframe.rst7", "-x", traj,
         "-inf", "md.mdinfo"], out, args.dry_run)
    if not args.dry_run:
        check_mdout(os.path.join(out, "md.mdout"))
    print(f"[done] receptor_coordinate_filename: {os.path.join(out, traj)}", flush=True)


if __name__ == "__main__":
    main()
