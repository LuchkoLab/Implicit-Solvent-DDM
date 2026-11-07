#!/usr/bin/env python
"""Locate where a window trajectory blows up: first broken frame, the atoms that moved, and the contacts just before."""
import argparse
import os
import re

import numpy as np
import pytraj as pt


def label(top, i):
    """Return ``RESnum@ATOM`` for a 0-based atom index."""
    atom = top.atom(int(i))
    return f"{top.residue(atom.resid).name}{atom.resid + 1}@{atom.name}"


def first_broken_frame(traj, pairs, tol):
    """Index of the first frame whose bonded distances leave frame 0's by more than ``tol``.

    Parameters
    ----------
    traj : pytraj.TrajectoryIterator
        Window trajectory.
    pairs : numpy.ndarray
        ``(n, 2)`` bonded atom pairs.
    tol : float
        Allowed deviation, Angstrom.

    Returns
    -------
    tuple
        ``(frame, deviation)`` with ``frame`` None when no frame breaks.
    """
    d = pt.distance(traj, pairs)
    dev = np.abs(d - d[:, :1]).max(axis=0)
    bad = np.flatnonzero(dev > tol)
    return (int(bad[0]) if bad.size else None), dev


def close_contacts(xyz, top, cutoff, ligand):
    """Non-bonded pairs (beyond 1-3) closer than ``cutoff`` in one frame, closest first."""
    n = len(xyz)
    bonded = [set() for _ in range(n)]
    for a, b in top.bond_indices:
        bonded[a].add(b)
        bonded[b].add(a)
    near = [set(s) for s in bonded]
    for a in range(n):
        for b in bonded[a]:
            near[a] |= bonded[b]
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
    i, j = np.where(np.triu(d < cutoff, 1))
    out = [(d[a, b], a, b) for a, b in zip(i, j) if b not in near[a]]
    lig = set(ligand)
    return sorted((r, a, b, (a in lig) or (b in lig)) for r, a, b in out)


def boresch_terms(rst):
    """Angle and dihedral restraints in an AMBER DISANG file as ``(kind, 1-based atoms, target)``."""
    out = []
    for block in open(rst).read().split("&rst")[1:]:
        iat = [int(x) for x in re.search(r"iat\s*=\s*([\d,\s]+?),\s*r1", block).group(1).replace(",", " ").split()]
        if len(iat) in (3, 4):
            r2 = float(re.search(r"r2\s*=\s*([-\d.]+)", block).group(1))
            out.append(("angle" if len(iat) == 3 else "dihedral", iat, r2))
    return out


def report_boresch(traj, top, rst, bad):
    """Print each Boresch angle and dihedral over the good frames and just before the break."""
    good = traj[:bad]
    print(f"\nBoresch terms from {rst} (last 5 good frames; angles near 0 or 180 make the dihedrals singular):")
    for kind, iat, target in boresch_terms(rst):
        mask = " ".join(f"@{i}" for i in iat)
        v = (pt.angle if kind == "angle" else pt.dihedral)(good, mask)
        names = " ".join(label(top, i - 1) for i in iat)
        tail = " ".join(f"{x:7.1f}" for x in v[-5:])
        extra = f" min {v.min():6.1f} max {v.max():6.1f}" if kind == "angle" else ""
        print(f"  {kind:8s} {names:55s} target {target:7.1f}  last: {tail}{extra}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top", required=True)
    p.add_argument("--traj", required=True)
    p.add_argument("--ligand", default=":LIG")
    p.add_argument("--tol", type=float, default=1.0, help="bond deviation that marks a broken frame (A)")
    p.add_argument("--contact", type=float, default=1.6, help="non-bonded contact cutoff (A)")
    p.add_argument("--top-atoms", type=int, default=15)
    p.add_argument("--rst", help="DISANG file with the Boresch terms (default: restraint.RST next to --traj)")
    a = p.parse_args()

    traj = pt.iterload(a.traj, a.top)
    top = traj.top
    pairs = np.asarray(top.bond_indices)
    ligand = top.select(a.ligand)
    bad, dev = first_broken_frame(traj, pairs, a.tol)
    print(f"frames {traj.n_frames}, bonds {len(pairs)}, ligand atoms {len(ligand)}")
    rst = a.rst or os.path.join(os.path.dirname(a.traj), "restraint.RST")
    if bad is None:
        print(f"no frame breaks a bond by more than {a.tol} A (max {dev.max():.3f})")
        if os.path.exists(rst):
            report_boresch(traj, top, rst, traj.n_frames)
        return
    print(f"first broken frame: {bad + 1} (1-based), max bond deviation {dev[bad]:.2f} A")
    print("max bond deviation in the frames before: " + "  ".join(f"{k + 1}:{dev[k]:.2f}" for k in range(max(0, bad - 5), bad)))

    # get_coordinates returns an owned array; Frame.xyz is a view freed with the temporary Frame
    before, after = pt.get_coordinates(traj, frame_indices=[bad - 1, bad])
    lig_xyz = before[ligand]
    moved = np.linalg.norm(after - before, axis=1)
    print(f"\natoms that moved most between frame {bad} and {bad + 1} (distance to nearest ligand atom in frame {bad}):")
    for i in np.argsort(moved)[::-1][: a.top_atoms]:
        to_lig = np.linalg.norm(lig_xyz - before[i], axis=1).min()
        print(f"  {label(top, i):18s} moved {moved[i]:9.2f} A   to ligand {to_lig:6.2f} A")

    devs = np.abs(pt.distance(traj[bad - 1:bad + 1], pairs) - pt.distance(traj[:1], pairs))[:, 1]
    print(f"\nmost stretched bonds in frame {bad + 1}:")
    for k in np.argsort(devs)[::-1][:10]:
        i, j = pairs[k]
        print(f"  {label(top, i):18s} - {label(top, j):18s} deviation {devs[k]:9.2f} A")

    print(f"\nnon-bonded contacts < {a.contact} A in the last good frame {bad} (* = involves the ligand):")
    for r, i, j, lig in close_contacts(before, top, a.contact, ligand)[:20]:
        print(f"  {'*' if lig else ' '} {label(top, i):18s} {label(top, j):18s} {r:5.2f} A")

    if os.path.exists(rst):
        report_boresch(traj, top, rst, bad)


if __name__ == "__main__":
    main()
