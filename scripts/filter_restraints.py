#!/usr/bin/env python
"""filter_restraints.py -- drop restraints touching a selection from an AMBER DISANG file.

WHY
---
The receptor leg's conformational restraint is a whole-receptor contact network (82,522 pairwise
distance restraints for MCL-1). Measured against the real file, 76% of the endstate -> lambda_window
dU gap comes from residues 22-32 and 86% from residues more than 15 A from the ligand -- floppy
regions with no bearing on binding. Freeing them takes the gap from 31.3 kcal/mol (9.0 SD, no
distribution overlap at all) to 5.0 kcal/mol (2.4 SD).

This writes the filtered restraint file so that can be tested directly: rerun one window against it
and measure the overlap, rather than arguing from the static estimate.

WHAT THIS DOES
--------------
Parses ``&rst ... /`` blocks, drops every block whose ``iat`` pair has EITHER atom inside
``--free``, and writes the rest through byte-for-byte. Nothing is rescaled or renumbered; the kept
blocks are the original text.

USAGE
-----
  python scripts/filter_restraints.py \\
      --restraint <window>/restraint.RST \\
      --parm      <window>/<receptor>.parm7 \\
      --free      "resid 22:32" \\
      --out       restraint_freeloop.RST

  # several regions at once
  ... --free "resid 22:32 or resid 146:152"

NOTES
-----
* ``--free`` is an MDAnalysis selection against the RECEPTOR topology, which is what ``iat`` indexes
  (1-based). Check the reported atom count before trusting a selection string.
* A restraint is dropped if EITHER end touches the selection: a contact between the freed region and
  the rest of the protein is exactly the kind carrying the dU gap, so keeping it would defeat this.
* THERMODYNAMIC CYCLE: for a production run the same selection must go into the complex leg as well,
  or the conformational restraint free energies no longer cancel between the legs and dG_bind is
  wrong. Filtering the receptor leg alone is valid only as an overlap diagnostic.
* The workflow regenerates restraint.RST per window, so a filtered file placed by hand will be
  overwritten on the next workflow run. For a one-off test, run sander directly with an mdin whose
  DISANG points at the filtered file.
"""
import argparse
import re
import sys


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--restraint", required=True, help="input DISANG file (restraint.RST)")
    ap.add_argument("--parm", required=True, help="receptor topology that iat indexes")
    ap.add_argument("--free", required=True, help="MDAnalysis selection to leave UNrestrained")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import MDAnalysis as mda

    universe = mda.Universe(args.parm)
    freed = universe.select_atoms(args.free)
    if freed.n_atoms == 0:
        sys.exit(f"selection {args.free!r} matched no atoms")
    # iat is 1-based into this topology.
    freed_ix = set((freed.indices + 1).tolist())
    print(f"topology     : {universe.atoms.n_atoms} atoms")
    print(f"freed        : {freed.n_atoms} atoms, resids {freed.residues.resids.min()}-{freed.residues.resids.max()}")

    text = open(args.restraint).read()
    matches = list(re.finditer(r"&rst.*?\n\s*/\s*\n", text, re.S))
    declared = text.count("&rst")
    if len(matches) != declared:
        sys.exit(f"parsed {len(matches)} blocks but the file declares {declared} &rst records")
    # Anything outside the blocks is carried through verbatim -- the generator terminates the file
    # with a bare "&end" (restraints.py:1313), and dropping it would leave sander an unterminated
    # DISANG file.
    preamble = text[: matches[0].start()]
    trailer = text[matches[-1].end() :]
    blocks = [m.group(0) for m in matches]

    iat = re.compile(r"iat\s*=\s*(\d+)\s*,\s*(\d+)")
    kept, dropped = [], 0
    for block in blocks:
        match = iat.search(block)
        if not match:
            sys.exit("a &rst block has no parsable two-atom iat; refusing to guess")
        i, j = int(match.group(1)), int(match.group(2))
        if i in freed_ix or j in freed_ix:
            dropped += 1
        else:
            kept.append(block)

    with open(args.out, "w") as handle:
        handle.write(preamble + "".join(kept) + trailer)
    print(f"restraints   : {len(blocks)} in, {dropped} dropped, {len(kept)} kept "
          f"({dropped / len(blocks):.1%} removed)")
    print(f"trailer      : {trailer.strip()!r} carried through")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
