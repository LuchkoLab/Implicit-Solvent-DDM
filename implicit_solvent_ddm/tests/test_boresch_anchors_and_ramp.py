"""Pure unit tests for pocket-backbone Boresch anchors and the receptor restraint ramp.

MCL-1/ligand-17: the legacy screen anchored Boresch to ARG31:NH1, inside the freed loop 22-32 and
37 A from the ligand; the loop swung ~20 A and dragged the decoupled ligand into PHE58/PHE100, so
the no-LJ -> LJ step lost all overlap. Separately, a loop freed in every window sampled differently
in the complex and receptor legs, so the GB band stopped cancelling. These tests pin the two fixes.
They drive the real job code through a stand-in Toil file store -- no job store, no AMBER::

    pytest implicit_solvent_ddm/tests/test_boresch_anchors_and_ramp.py -v
"""
import os
import re
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import parmed as pmd
import pytest

from implicit_solvent_ddm.restraint_helper import count_heavy_neighbors, ramped_force
from implicit_solvent_ddm.restraints import (
    BoreschRestraints,
    get_conformational_restraints,
    write_restraint_forces,
)

HERE = os.path.dirname(os.path.realpath(__file__))
CB7_PARM = os.path.join(HERE, "structs", "cb7-mol01.parm7")
CB7_RST = os.path.join(HERE, "structs", "cb7-mol01.ncrst")
MCL1 = os.path.join(HERE, "..", "..", "fragment-opt-abfe-benchmark")
MCL1_PARM = os.path.join(MCL1, "topologies_and_structure/MCL-1/ligand-17/complex/amber/MCL-1_ligand-17.parm7")
MCL1_REF = os.path.join(MCL1, "relaxed/MCL-1_ligand-17_10ns/MCL-1_ligand-17_relaxed.rst7")


class _FileStore:
    """Stand-in for a Toil file store: 'global' files are plain paths."""

    def __init__(self, root):
        self.root, self.log = root, []

    def getLocalTempDir(self):
        return tempfile.mkdtemp(dir=self.root)

    def getLocalTempFile(self):
        return tempfile.mkstemp(dir=self.root)[1]

    def readGlobalFile(self, file_id, userPath=None):
        # the job code names coordinates basename(file_id[0]): 'f' for a real FileID, '' for a path
        if os.path.isdir(userPath):
            userPath = os.path.join(userPath, "f")
        shutil.copyfile(file_id, userPath)
        return userPath

    def writeGlobalFile(self, path):
        return os.path.abspath(path)

    def logToMaster(self, message):
        self.log.append(message)


def _boresch(parm, coords, ligand_mask, receptor_mask, **kwargs):
    return BoreschRestraints(parm, coords, restraint_type=1, ligand_mask=ligand_mask,
                             receptor_mask=receptor_mask, K_r=16.0, K_thetaA=256.0, K_thetaB=256.0,
                             K_phiA=256.0, K_phiB=256.0, K_phiC=256.0, **kwargs)


def test_heavy_neighbor_count_uses_the_bond_partner():
    # ARG NH1-like terminal N: one heavy partner (CZ) plus two H, bonds stored in both atom orders.
    # The legacy count read bond.atom2 and counted NH1 itself, so terminal tips passed the filter.
    s = pmd.Structure()
    for name in ("CZ", "NH1", "HH11", "HH12", "NE"):
        s.add_atom(pmd.Atom(name=name), "ARG", 1)
    cz, nh1, h1, h2, ne = s.atoms
    for a, b in ((cz, nh1), (h1, nh1), (nh1, h2), (ne, cz)):
        s.bonds.append(pmd.Bond(a, b))
    assert count_heavy_neighbors(nh1) == 1
    assert count_heavy_neighbors(cz) == 2


def test_ramp_is_the_network_at_the_top_and_gone_at_the_floor():
    assert ramped_force(16.0, 16.0, 2.0) == pytest.approx(16.0)
    assert ramped_force(2.0 ** -14, 16.0, 2.0) == pytest.approx(2.0 ** -32)
    ladder = [2.0 ** e for e in np.arange(-14.0, 4.25, 0.25)]
    ramp = [ramped_force(f, 16.0, 2.0) for f in ladder]
    assert all(r <= f + 1e-12 for r, f in zip(ramp, ladder))
    assert np.all(np.diff(ramp) > 0)


def test_ramped_pairs_get_their_own_force_in_both_legs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job = SimpleNamespace(fileStore=_FileStore(str(tmp_path)))
    complex_t, _, receptor_t = get_conformational_restraints(
        job, CB7_PARM, CB7_RST, ":CB7", ":M01", receptor_ramp_mask="@1-10")
    receptor_text, complex_text = open(receptor_t).read(), open(complex_t).read()
    assert "$frest_ramp" in receptor_text
    # the complex file carries the identical receptor network, so both legs ramp the same restraints
    assert receptor_text.replace("&end", "") in complex_text.replace("&end", "")
    for block in re.findall(r"&rst(.*?)/", receptor_text, re.S):
        i, j = (int(v) for v in re.search(r"iat\s*=\s*(\d+)\s*,\s*(\d+)", block).groups())
        assert ("$frest_ramp" in block) == (i <= 10 or j <= 10)

    written = write_restraint_forces(job, receptor_t, conformational_force=2.0 ** -14,
                                     ramp_slope=2.0, ramp_max_force=16.0)
    forces = sorted({float(v) for v in re.findall(r"rk2\s*=\s*([\d.eE+-]+)", open(written).read())})
    assert forces == pytest.approx([2.0 ** -32, 2.0 ** -14])
    with pytest.raises(ValueError, match="no ramp schedule"):
        write_restraint_forces(job, receptor_t, conformational_force=1.0)


def test_host_without_backbone_keeps_the_legacy_screen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = _FileStore(str(tmp_path))
    boresch = _boresch(CB7_PARM, CB7_RST, ":M01", ":CB7", anchor_pool="pocket_backbone")
    boresch.run(store)
    assert boresch._pool is None
    assert any(m.startswith("Boresch anchors (legacy)") for m in store.log)


@pytest.mark.skipif(not (os.path.exists(MCL1_PARM) and os.path.exists(MCL1_REF)), reason="MCL-1 benchmark files not present")
def test_protein_anchors_are_pocket_backbone_outside_the_freed_loop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = _FileStore(str(tmp_path))
    boresch = _boresch(MCL1_PARM, MCL1_REF, ":LIG", "!:LIG", anchor_pool="pocket_backbone",
                       pocket_radius=10.0, excluded_receptor_mask=":22-32")
    boresch.run(store)
    top = boresch.complex_traj.top
    ligand = boresch.complex_traj.xyz[0][top.select(":LIG")]
    for index in (boresch.rec_dist_atom, boresch.receptor_atom2, boresch.receptor_atom3):
        atom = top.atom(int(index) - 1)
        assert atom.name in ("CA", "C", "N")
        assert not 22 <= atom.resid + 1 <= 32
        # its residue lines the pocket
        residue = top.select(f":{atom.resid + 1}")
        assert np.linalg.norm(boresch.complex_traj.xyz[0][residue][:, None] - ligand[None], axis=2).min() <= 10.0
    assert any(m.startswith("Boresch anchors (pocket_backbone)") for m in store.log)
