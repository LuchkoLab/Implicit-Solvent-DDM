#!/usr/bin/env python
"""Check that a ligand decoupled in GB water (charges, LJ to receptor and GB screening off) leaves the receptor unchanged."""
import argparse
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import parmed as pmd

# use the checkout this script lives in, not an older installed copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TERMS = ("EGB", "EELEC", "VDWAALS", "1-4 VDW", "1-4 EEL", "BOND", "ANGLE", "DIHED")
MDIN = """single point
&cntrl
  imin=0, nstlim=1, ntpr=1, ntb=0, ntx=1, irest=0,
  igb={igb}, saltcon={saltcon}, cut=999.0, gbsa=0,
/
"""


def build(args, out):
    """Write the topologies and coordinates for every test system.

    Parameters
    ----------
    args : argparse.Namespace
        Command line.
    out : str
        Output directory.

    Returns
    -------
    dict
        ``{system: (parm, coords)}``.
    """
    from implicit_solvent_ddm.alchemical import scale_ligand_receptor_lj

    complex_ = pmd.load_file(args.complex_parm, args.coords)
    ligand_atoms = np.array(list(pmd.amber.AmberMask(complex_, args.ligand_mask).Selected()))
    receptor_atoms = np.array(list(pmd.amber.AmberMask(complex_, args.receptor_mask).Selected()))
    if receptor_atoms.max() > ligand_atoms.min():
        sys.exit("receptor atoms must precede the ligand so the apo receptor shares their coordinates")

    def variant(name, lj_off, q0, screen0):
        parm = pmd.load_file(args.complex_parm, args.coords)
        if lj_off:  # before atom edits: addLJType reloads atoms from parm_data
            scale_ligand_receptor_lj(parm, ligand_atoms, receptor_atoms, 0.0)
        if q0:
            pmd.tools.actions.change(parm, "CHARGE", args.ligand_mask, 0.0).execute()
        if screen0:
            pmd.tools.actions.change(parm, "SCREEN", args.ligand_mask, 0.0).execute()
        path = os.path.join(out, f"{name}.parm7")
        parm.save(path, overwrite=True)
        return path

    systems = {
        "complex": (variant("complex", False, False, False), args.coords),
        "q0": (variant("q0", False, True, False), args.coords),
        "q0_screen0": (variant("q0_screen0", False, True, True), args.coords),
        "decoupled": (variant("decoupled", True, True, True), args.coords),
    }
    xyz = complex_.coordinates
    for name, parm_path, atoms in (("receptor", args.receptor_parm, receptor_atoms),
                                   ("ligand", args.ligand_parm, ligand_atoms)):
        parm = pmd.load_file(parm_path)
        if len(parm.atoms) != len(atoms):
            sys.exit(f"{parm_path} has {len(parm.atoms)} atoms, expected {len(atoms)}")
        parm.coordinates = xyz[atoms]
        crd = os.path.join(out, f"{name}.rst7")
        parm.save(crd, format="rst7", overwrite=True)
        systems[name] = (parm_path, crd)
    return systems


def energies(mdout):
    """Step-0 energy terms of an mdout."""
    text = open(mdout).read()
    block = text[text.index("NSTEP ="):]
    out = {}
    for term in TERMS:
        m = re.search(rf"{re.escape(term)}\s*=\s*([-\d.]+)", block)
        out[term] = float(m.group(1)) if m else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--complex-parm", required=True)
    ap.add_argument("--receptor-parm", required=True)
    ap.add_argument("--ligand-parm", required=True)
    ap.add_argument("--coords", required=True, help="complex coordinates")
    ap.add_argument("--ligand-mask", default=":LIG")
    ap.add_argument("--receptor-mask", default="!:LIG")
    ap.add_argument("--igb", type=int, default=2)
    ap.add_argument("--saltcon", type=float, default=0.3)
    ap.add_argument("--engines", nargs="+", default=["pmemd.cuda", "pmemd", "sander"])
    ap.add_argument("--tol", type=float, default=0.02, help="kcal/mol")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    systems = build(args, out)
    mdin = os.path.join(out, "single_point.in")
    with open(mdin, "w") as fh:
        fh.write(MDIN.format(igb=args.igb, saltcon=args.saltcon))

    engines = [e for e in args.engines if shutil.which(e)]
    for missing in set(args.engines) - set(engines):
        print(f"skipping {missing}: not on PATH")
    result = {}
    for engine in engines:
        for name, (parm, crd) in systems.items():
            stem = os.path.join(out, f"{name}.{engine}")
            subprocess.run([engine, "-O", "-i", mdin, "-p", parm, "-c", crd, "-o", stem + ".out",
                            "-r", stem + ".rst", "-inf", stem + ".info"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            result[(engine, name)] = energies(stem + ".out")

    print(f"\nterms (kcal/mol), igb={args.igb}, saltcon={args.saltcon}")
    for engine in engines:
        print(f"\n== {engine}")
        print(f"  {'system':11s}" + "".join(f"{t:>12s}" for t in ("EGB", "EELEC", "VDWAALS")))
        for name in systems:
            e = result[(engine, name)]
            print(f"  {name:11s}" + "".join(f"{e[t]:12.4f}" for t in ("EGB", "EELEC", "VDWAALS")))
        rec, dec, lig, q0 = (result[(engine, n)] for n in ("receptor", "decoupled", "ligand", "q0"))
        checks = [
            ("decoupled EGB == receptor EGB", dec["EGB"] - rec["EGB"]),
            ("decoupled EELEC == receptor EELEC", dec["EELEC"] - rec["EELEC"]),
            ("decoupled VDW == receptor + ligand intra VDW", dec["VDWAALS"] - rec["VDWAALS"] - lig["VDWAALS"]),
        ]
        for label, diff in checks:
            print(f"  {'PASS' if abs(diff) <= args.tol else 'FAIL'}  {label:46s} diff {diff:+.4f}")
        print(f"  info  descreening by an uncharged ligand (q0 EGB - receptor EGB) {q0['EGB'] - rec['EGB']:+.4f}")
    if len(engines) > 1:
        print("\n== engine agreement (EGB, relative to the first engine)")
        for name in systems:
            ref = result[(engines[0], name)]["EGB"]
            print(f"  {name:11s}" + "".join(f"  {e}: {result[(e, name)]['EGB'] - ref:+.4f}" for e in engines[1:]))


if __name__ == "__main__":
    main()
