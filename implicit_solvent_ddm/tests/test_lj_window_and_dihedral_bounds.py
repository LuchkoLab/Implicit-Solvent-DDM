"""Tests for the intermediate ligand-receptor LJ windows and the centred Boresch dihedral bounds."""
import os
import re
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import parmed as pmd
import pytest

from implicit_solvent_ddm import block_mbar
from implicit_solvent_ddm.alchemical import alter_topology
from implicit_solvent_ddm.matrix_order import CycleSteps
from implicit_solvent_ddm.restraints import BoreschRestraints

HERE = os.path.dirname(os.path.realpath(__file__))
PARM = os.path.join(HERE, "structs", "cb7-mol01.parm7")
COORD = os.path.join(HERE, "structs", "cb7-mol01.ncrst")


class _FileStore:
    """Stand-in Toil file store: global files are plain paths."""

    def __init__(self, root):
        self.root = root

    def getLocalTempDir(self):
        return tempfile.mkdtemp(dir=self.root)

    def readGlobalFile(self, file_id, userPath=None):
        shutil.copyfile(file_id, userPath)
        return userPath

    def writeGlobalFile(self, path):
        return os.path.abspath(path)


def _pair_coefficients(parm, i, j):
    """LJ A and B coefficients acting between two 0-based atoms.

    Parameters
    ----------
    parm : parmed.amber.AmberParm
        Topology.
    i, j : int
        Atom indices.

    Returns
    -------
    tuple of float
        (A, B).
    """
    ti, tj = parm.parm_data["ATOM_TYPE_INDEX"][i], parm.parm_data["ATOM_TYPE_INDEX"][j]
    slot = parm.parm_data["NONBONDED_PARM_INDEX"][parm.ptr("ntypes") * (ti - 1) + (tj - 1)] - 1
    return parm.parm_data["LENNARD_JONES_ACOEF"][slot], parm.parm_data["LENNARD_JONES_BCOEF"][slot]


def _cb7_topology(tmp_path, monkeypatch, **kwargs):
    monkeypatch.chdir(tmp_path)
    job = SimpleNamespace(fileStore=_FileStore(str(tmp_path)))
    return pmd.load_file(alter_topology(job, PARM, COORD, ":M01", ":CB7", set_charge=0.0, **kwargs))


def test_lj_scale_changes_only_ligand_receptor_pairs(tmp_path, monkeypatch):
    original = pmd.load_file(PARM)
    scaled = _cb7_topology(tmp_path, monkeypatch, lj_scale=0.2)
    ligand = [a.idx for a in original.atoms if a.residue.name == "M01"]
    receptor = [a.idx for a in original.atoms if a.residue.name == "CB7"]
    types = scaled.parm_data["ATOM_TYPE_INDEX"]
    assert not {types[i] for i in ligand} & {types[j] for j in receptor}
    for i in ligand:
        for j in receptor[::5]:
            assert np.allclose(_pair_coefficients(scaled, i, j), 0.2 * np.array(_pair_coefficients(original, i, j)))
        for k in ligand:
            assert np.allclose(_pair_coefficients(scaled, i, k), _pair_coefficients(original, i, k))
    for j in receptor[::7]:
        for k in receptor[::7]:
            assert np.allclose(_pair_coefficients(scaled, j, k), _pair_coefficients(original, j, k))
    assert all(scaled.atoms[i].charge == 0.0 for i in ligand)


def test_lj_scale_with_exclusions_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="lj_scale and exculsions"):
        _cb7_topology(tmp_path, monkeypatch, lj_scale=0.2, exculsions=True)


def _cycle(lj_windows):
    return CycleSteps(conformation_forces=[-8.0, 4.0], orientational_forces=[-4.0, 8.0],
                      charges_windows=[0.0, 0.5, 1.0], external_dielectic=[2.0, 8.0], lj_windows=lj_windows)


def test_lj_window_sits_between_the_lj_end_states_and_indices_follow():
    base, lj = _cycle([]), _cycle([0.2])
    labels = [state[0] for state in lj.complex_order]
    assert labels[:3] == ["no_interactions", "lj_0.2", "interactions"]
    assert lj.complex_order[lj.start_gb_extdiel_matrix][0] == "interactions"
    for index in ("start_gb_extdiel_matrix", "start_complex_charge_matrix", "halo_restraint_matrix"):
        assert lj.complex_order[getattr(lj, index)] == base.complex_order[getattr(base, index)]


def test_block_chaining_covers_both_lj_links():
    order = _cycle([0.2]).complex_order
    blocks = block_mbar.partition_blocks(len(order), 4, block_mbar.band_boundaries(order))
    assert [0, 1] in blocks and [1, 2] in blocks


def test_dihedral_bounds_are_centred_on_each_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    boresch = BoreschRestraints(PARM, COORD, 1, ":M01", ":CB7", 16.0, 256.0, 256.0, 256.0, 256.0, 256.0)
    for name, value in dict(rec_dist_atom=1, receptor_atom2=2, receptor_atom3=3, lig_dist_atom=127,
                            ligand_atom2=128, ligand_atom3=129, r_aA0=5.0, theta_A0=90.0, theta_B0=90.0,
                            phi_A0=0.16, phi_B0=356.6, phi_C0=143.4).items():
        setattr(boresch, name, value)
    text = open(boresch._write_orientational_template).read()
    dihedrals = [b for b in re.findall(r"&rst(.*?)/", text, re.S)
                 if len(re.findall(r"\d+", re.search(r"iat\s*=\s*([\d,\s]+)", b).group(1))) == 4]
    assert len(dihedrals) == 3
    for block in dihedrals:
        r1, r2, r3, r4 = (float(re.search(fr"{k}\s*=\s*([-\d.eE+]+)", block).group(1)) for k in ("r1", "r2", "r3", "r4"))
        assert r2 == r3
        assert r1 == pytest.approx(r2 - 180.0) and r4 == pytest.approx(r2 + 180.0)
