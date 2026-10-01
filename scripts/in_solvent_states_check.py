#!/usr/bin/env python
"""Single-point check of every in-solvent cycle state built by the workflow's own alter_topology."""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import parmed as pmd

HERE = os.path.dirname(os.path.abspath(__file__))
# use the checkout this script lives in, not an older installed copy
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from ingb_decouple_check import MDIN, energies  # noqa: E402
from implicit_solvent_ddm.alchemical import alter_topology  # noqa: E402


class _FileStore:
    """Toil file store stand-in whose global files are plain paths."""

    def __init__(self, root):
        self.root = root

    def getLocalTempDir(self):
        return tempfile.mkdtemp(dir=self.root)

    def readGlobalFile(self, file_id, userPath=None):
        shutil.copyfile(file_id, userPath)
        return userPath

    def writeGlobalFile(self, path):
        return os.path.abspath(path)


def build(args, out):
    """Write the topology and coordinates of every state.

    Parameters
    ----------
    args : argparse.Namespace
        Command line.
    out : str
        Output directory.

    Returns
    -------
    dict
        ``{state: (parm, coords)}`` in cycle order, then ``receptor`` and ``ligand_q0``.
    """
    job = SimpleNamespace(fileStore=_FileStore(out))
    complex_ = pmd.load_file(args.complex_parm, args.coords)
    ligand_atoms = np.array(list(pmd.amber.AmberMask(complex_, args.ligand_mask).Selected()))
    receptor_atoms = np.array(list(pmd.amber.AmberMask(complex_, args.receptor_mask).Selected()))
    if receptor_atoms.max() > ligand_atoms.min():
        sys.exit("receptor atoms must precede the ligand so the apo receptor shares their coordinates")

    def state(name, parm, coords, **kwargs):
        os.chdir(tempfile.mkdtemp(dir=out))
        built = alter_topology(job, parm, coords, args.ligand_mask, args.receptor_mask, **kwargs)
        path = os.path.join(out, f"{name}.parm7")
        shutil.move(built, path)
        return path, coords

    systems = {"no_interactions": state("no_interactions", args.complex_parm, args.coords,
                                        set_charge=0.0, lj_scale=0.0, screen_scale=0.0)}
    for lj in args.lj_windows:
        systems[f"lj_{lj}"] = state(f"lj_{lj}", args.complex_parm, args.coords,
                                    set_charge=0.0, lj_scale=lj, screen_scale=0.0)
    systems["interactions"] = state("interactions", args.complex_parm, args.coords,
                                    set_charge=0.0, screen_scale=0.0)
    for screen in args.screening_windows:
        systems[f"screening_{screen}"] = state(f"screening_{screen}", args.complex_parm, args.coords,
                                               set_charge=0.0, screen_scale=screen)
    for charge in (0.0, 1.0):
        systems[f"electrostatics_{charge}"] = state(f"electrostatics_{charge}", args.complex_parm,
                                                    args.coords, set_charge=charge)

    xyz = complex_.coordinates
    for name, parm_path, atoms in (("receptor", args.receptor_parm, receptor_atoms),
                                   ("ligand", args.ligand_parm, ligand_atoms)):
        parm = pmd.load_file(parm_path)
        if len(parm.atoms) != len(atoms):
            sys.exit(f"{parm_path} has {len(parm.atoms)} atoms, expected {len(atoms)}")
        parm.coordinates = xyz[atoms]
        crd = os.path.join(out, f"{name}.rst7")
        parm.save(crd, format="rst7", overwrite=True)
        if name == "receptor":
            systems["receptor"] = (parm_path, crd)
        else:
            systems["ligand_q0"] = state("ligand_q0", parm_path, crd, set_charge=0.0)
    return systems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--complex-parm", required=True)
    ap.add_argument("--receptor-parm", required=True)
    ap.add_argument("--ligand-parm", required=True)
    ap.add_argument("--coords", required=True, help="complex coordinates")
    ap.add_argument("--ligand-mask", default=":LIG")
    ap.add_argument("--receptor-mask", default="!:LIG")
    ap.add_argument("--lj-windows", type=float, nargs="*", default=[0.2])
    ap.add_argument("--screening-windows", type=float, nargs="*", default=[0.5])
    ap.add_argument("--igb", type=int, default=2)
    ap.add_argument("--saltcon", type=float, default=0.3)
    ap.add_argument("--engines", nargs="+", default=["pmemd.cuda", "pmemd", "sander"])
    ap.add_argument("--tol", type=float, default=0.02, help="kcal/mol")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for path in ("complex_parm", "receptor_parm", "ligand_parm", "coords"):
        setattr(args, path, os.path.abspath(getattr(args, path)))

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

    failed = False
    print(f"\nterms (kcal/mol), igb={args.igb}, saltcon={args.saltcon}")
    for engine in engines:
        print(f"\n== {engine}")
        print(f"  {'state':20s}" + "".join(f"{t:>12s}" for t in ("EGB", "EELEC", "VDWAALS")))
        for name in systems:
            e = result[(engine, name)]
            print(f"  {name:20s}" + "".join(f"{e[t]:12.4f}" for t in ("EGB", "EELEC", "VDWAALS")))
        get = lambda name: result[(engine, name)]  # noqa: E731
        rec, dec, lig, inter = get("receptor"), get("no_interactions"), get("ligand_q0"), get("interactions")
        checks = [
            ("no_interactions EGB == receptor EGB", dec["EGB"] - rec["EGB"]),
            ("no_interactions EELEC == receptor EELEC", dec["EELEC"] - rec["EELEC"]),
            ("no_interactions VDW == receptor + ligand intra VDW",
             dec["VDWAALS"] - rec["VDWAALS"] - lig["VDWAALS"]),
            ("ligand q0 EGB == 0", lig["EGB"]),
            ("ligand q0 EELEC == 0", lig["EELEC"]),
            ("interactions EGB == receptor EGB", inter["EGB"] - rec["EGB"]),
            ("electrostatics q0 VDW == interactions VDW",
             get("electrostatics_0.0")["VDWAALS"] - inter["VDWAALS"]),
        ]
        for label, diff in checks:
            ok = abs(diff) <= args.tol
            failed |= not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {label:52s} diff {diff:+.4f}")
    if len(engines) > 1:
        print("\n== engine agreement (EGB, relative to the first engine)")
        for name in systems:
            ref = result[(engines[0], name)]["EGB"]
            print(f"  {name:20s}" + "".join(f"  {e}: {result[(e, name)]['EGB'] - ref:+.4f}" for e in engines[1:]))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
