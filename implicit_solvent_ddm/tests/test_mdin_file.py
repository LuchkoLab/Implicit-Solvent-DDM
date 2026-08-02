"""Pure unit tests for mdin.make_mdin_file -- the mdin rewrite rules.

Covers the four production mdins plus the GB-band scoring mdin (saltcon=0). The band's MD is written
salt-free by generate_extdiel_mdin so the GB polar term stays exactly linear in lambda = 1 - 1/eps;
scoring it with the user's saltcon draws samples from one Hamiltonian and evaluates them under
another, and leaves the band unable to reach vacuum.

These tests do NOT touch Toil/AMBER. Run with::

    pytest implicit_solvent_ddm/tests/test_mdin_file.py -v
"""
import os
import re

import pytest

from implicit_solvent_ddm.mdin import get_mdins, make_mdin_file

# tests/input_files/intermediate.mdin -- `igb = 2, saltcon=0.3` on one line, `extdiel = $extdiel`
TEMPLATE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "input_files", "intermediate.mdin"
)


def build(tmp_path, monkeypatch, name, **kwargs):
    """make_mdin_file writes to CWD by bare name, so chdir first."""
    monkeypatch.chdir(tmp_path)
    return open(make_mdin_file(TEMPLATE, name, **kwargs)).read()


class _FakeFileStore:
    """Enough of a Toil file store for get_mdins: paths in, paths out."""

    def readGlobalFile(self, file_id):
        return file_id

    def writeGlobalFile(self, path):
        return path


class _FakeJob:
    fileStore = _FakeFileStore()


# --------------------------------------------------------------------------------------------------
# get_mdins emits the band scoring mdin
# --------------------------------------------------------------------------------------------------
def test_get_mdins_emits_saltfree_band_mdin(tmp_path, monkeypatch):
    """The fix: a 5th mdin, salt-free, alongside the four that keep the user's saltcon."""
    monkeypatch.chdir(tmp_path)
    default, no_solvent, post, post_nosolv, post_saltfree = get_mdins(_FakeJob(), TEMPLATE)

    assert re.search(r"saltcon\s*=\s*0\.0\b", open(post_saltfree).read())
    assert re.search(r"saltcon\s*=\s*0\.3", open(post).read())  # non-band columns unchanged
    assert re.search(r"saltcon\s*=\s*0\.3", open(default).read())
    # distinct filenames -- make_mdin_file writes to CWD, so a shared name would clobber
    assert len({default, no_solvent, post, post_nosolv, post_saltfree}) == 5


# --------------------------------------------------------------------------------------------------
# GB-dielectric band scoring mdin (the fix)
# --------------------------------------------------------------------------------------------------
def test_gb_band_scoring_mdin_is_saltfree(tmp_path, monkeypatch):
    out = build(tmp_path, monkeypatch, "post_saltfree_mdin", post_process=True, saltcon=0.0)
    assert re.search(r"saltcon\s*=\s*0\.0\b", out)
    assert not re.search(r"saltcon\s*=\s*0\.3", out)
    assert re.search(r"imin\s*=\s*5", out)  # single-point over an input trajectory
    assert re.search(r"igb\s*=\s*2", out)  # same GB model as the rest of the ladder
    assert "$extdiel" in out  # stays live; simulations.py fills it per window


def test_post_mdin_keeps_user_saltcon(tmp_path, monkeypatch):
    """The bug: plain post_mdin scores at the user's saltcon while the band's MD ran salt-free."""
    out = build(tmp_path, monkeypatch, "post_mdin", post_process=True)
    assert re.search(r"saltcon\s*=\s*0\.3", out)
    assert re.search(r"imin\s*=\s*5", out)


def test_gb_band_and_post_mdin_differ_only_in_saltcon(tmp_path, monkeypatch):
    """Same scoring Hamiltonian except the salt -- same igb, same Born radii, same cut."""
    post = build(tmp_path, monkeypatch, "post_mdin", post_process=True).splitlines()
    band = build(tmp_path, monkeypatch, "post_saltfree_mdin", post_process=True, saltcon=0.0).splitlines()
    differing = [(a, b) for a, b in zip(post, band) if a != b]
    assert len(post) == len(band)
    assert len(differing) == 1
    assert "saltcon" in differing[0][0]


# --------------------------------------------------------------------------------------------------
# The other production mdins stay as they were
# --------------------------------------------------------------------------------------------------
def test_vacuum_branch_unchanged(tmp_path, monkeypatch):
    out = build(tmp_path, monkeypatch, "no_solv_mdin", turn_off_solvent=True)
    assert re.search(r"igb\s*=\s*6", out)
    assert re.search(r"extdiel\s*=\s*0\.0", out)
    assert re.search(r"saltcon\s*=\s*0\.0", out)
    assert "$extdiel" not in out  # baked, not left for runtime substitution


def test_default_mdin_bakes_extdiel_and_keeps_saltcon(tmp_path, monkeypatch):
    out = build(tmp_path, monkeypatch, "_mdin", gb_extdiel=78.5)
    assert re.search(r"extdiel\s*=\s*78\.5", out)
    assert re.search(r"igb\s*=\s*2", out)
    assert re.search(r"saltcon\s*=\s*0\.3", out)  # MD path unaffected by the fix
    assert re.search(r"imin\s*=\s*0", out)


def test_post_nosolv_is_gas_and_single_point(tmp_path, monkeypatch):
    out = build(tmp_path, monkeypatch, "post_nosolv_mdin", turn_off_solvent=True, post_process=True)
    assert re.search(r"igb\s*=\s*6", out)
    assert re.search(r"imin\s*=\s*5", out)


# --------------------------------------------------------------------------------------------------
# Rewrite-rule regressions
# --------------------------------------------------------------------------------------------------
def test_saltcon_override_accepts_single_digit(tmp_path, monkeypatch):
    # The gas branch uses r"saltcon\s*=\s*\d+\.?\d+" (needs >=2 digits); the saltcon kwarg uses
    # r"saltcon\s*=\s*[0-9.]+", which also matches a bare `saltcon=0`. Pin the looser one.
    out = build(tmp_path, monkeypatch, "int_salt_mdin", post_process=True, saltcon=0)
    assert re.search(r"saltcon\s*=\s*0\b", out)
    assert not re.search(r"saltcon\s*=\s*0\.3", out)


def test_post_process_leaves_extdiel_live(tmp_path, monkeypatch):
    # simulations.py substitutes $extdiel per window at scoring time; if make_mdin_file baked it,
    # every gb_dielectric column would score at the same dielectric.
    out = build(tmp_path, monkeypatch, "post_mdin", post_process=True, gb_extdiel=78.5)
    assert "$extdiel" in out


def test_ntxo_survives_the_ntx_rewrite(tmp_path, monkeypatch):
    # `ntxo=2` contains the substring `ntx`, so the `if "ntx" in line` guard fires on it.
    out = build(tmp_path, monkeypatch, "_mdin")
    assert re.search(r"ntxo\s*=\s*2", out)
    assert re.search(r"ntx\s*=\s*1", out)


@pytest.mark.parametrize("name", ["_mdin", "post_mdin"])
def test_overrides_none_leaves_template_values(tmp_path, monkeypatch, name):
    """Flag-off regression: no nstlim/ntwx/saltcon/score_igb override touches those lines."""
    post = name == "post_mdin"
    out = build(tmp_path, monkeypatch, name, post_process=post)
    assert re.search(r"nstlim\s*=\s*100\b", out)
    assert re.search(r"ntwx\s*=\s*10\b", out)
    assert re.search(r"saltcon\s*=\s*0\.3", out)
    assert re.search(r"igb\s*=\s*2", out)
