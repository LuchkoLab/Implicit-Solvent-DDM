"""Pure unit tests for ligand pose clustering (pose_clusters).

Covers the branches a real trajectory does not reach: the host-guest pocket fallback
(no ``@CA`` to align on), the ambiguous-ensemble path that must warn rather than raise,
and the population-ordering invariant ``dominant_medoid`` silently depends on. The
algorithm itself is validated against a real trajectory, not here. No Toil, no AMBER.
Run with::

    pytest implicit_solvent_ddm/tests/test_pose_clusters.py -v
"""
import os

import numpy as np
import pytest

from implicit_solvent_ddm import pose_clusters as pc
from implicit_solvent_ddm.config import IntermediateStateArgs
from implicit_solvent_ddm.run_endstate import _pose_medoid_args
from implicit_solvent_ddm.simulations import ExtractTrajectories

RNG = np.random.default_rng(20260913)


@pytest.fixture(autouse=True)
def _deterministic_draws():
    """Reseed per test: otherwise the data depends on which tests ran first."""
    global RNG
    RNG = np.random.default_rng(20260913)


def _two_basin(n_a, n_b, n_atoms=6, spread=0.15, separation=6.0):
    """Frames drawn from two well-separated ligand poses, basin A first.

    Returns ``(X, n_atoms)`` with A occupying rows ``[:n_a]``.
    """
    pose_a = RNG.normal(scale=2.0, size=(n_atoms, 3))
    pose_b = pose_a + separation
    a = pose_a + RNG.normal(scale=spread, size=(n_a, n_atoms, 3))
    b = pose_b + RNG.normal(scale=spread, size=(n_b, n_atoms, 3))
    X = np.concatenate([a, b]).reshape(n_a + n_b, -1)
    return X, n_atoms


# --- stride ----------------------------------------------------------------

@pytest.mark.parametrize("n_frames,expected", [
    (1, 1), (2000, 1), (2001, 2), (10000, 5), (10001, 6),
])
def test_auto_stride_caps_the_frame_count(n_frames, expected):
    assert pc.resolve_stride(n_frames, 0) == expected


def test_auto_stride_actually_stays_under_the_cap():
    for n in (2001, 5000, 10000, 123456):
        stride = pc.resolve_stride(n, 0)
        assert len(range(0, n, stride)) <= pc.DEFAULT_FRAME_CAP


def test_explicit_stride_is_honoured():
    assert pc.resolve_stride(10000, 7) == 7


# --- clustering ------------------------------------------------------------

def test_two_basins_are_separated_with_the_right_populations():
    X, n_atoms = _two_basin(70, 30)
    labels, _, _ = pc.cluster_poses(X, cutoff=2.0, n_lig=n_atoms)
    assert set(labels) == {1, 2}
    assert np.sum(labels == 1) == 70
    assert np.sum(labels == 2) == 30


def test_cluster_one_is_always_the_most_populated():
    # basin A is the MINORITY here, so a correct relabelling must reorder
    X, n_atoms = _two_basin(20, 80)
    labels, _, _ = pc.cluster_poses(X, cutoff=2.0, n_lig=n_atoms)
    assert np.sum(labels == 1) == 80
    assert labels[0] == 2  # first frame belongs to the smaller basin


def test_medoid_is_the_most_central_frame_of_its_cluster():
    X, n_atoms = _two_basin(60, 40)
    labels, condensed, _ = pc.cluster_poses(X, cutoff=2.0, n_lig=n_atoms)
    frames = np.arange(len(X))
    summary = pc.cluster_summary(labels, condensed, frames)

    from scipy.spatial.distance import squareform
    rmsd = squareform(condensed)
    for row in summary:
        members = np.where(labels == row["cluster"])[0]
        totals = rmsd[np.ix_(members, members)].sum(axis=1)
        assert row["medoid_row"] == members[np.argmin(totals)]


def test_medoid_frame_is_the_original_frame_number_under_a_stride():
    X, n_atoms = _two_basin(60, 40)
    labels, condensed, _ = pc.cluster_poses(X, cutoff=2.0, n_lig=n_atoms)
    frames = np.arange(0, 500, 5)  # what pose_matrix would hand back at stride 5
    summary = pc.cluster_summary(labels, condensed, frames)
    for row in summary:
        assert row["medoid_frame"] == frames[row["medoid_row"]]


# --- dominant basin --------------------------------------------------------

def test_dominant_medoid_picks_the_larger_basin():
    X, n_atoms = _two_basin(25, 75)
    labels, condensed, _ = pc.cluster_poses(X, cutoff=2.0, n_lig=n_atoms)
    summary = pc.cluster_summary(labels, condensed, np.arange(len(X)))
    row, frame, fraction = pc.dominant_medoid(summary)

    assert fraction == pytest.approx(0.75)
    assert row >= 25  # the medoid came from the second, larger basin
    assert frame == row


def test_ambiguous_split_proceeds_instead_of_raising():
    # 45/40 with a 15-frame third basin: no majority, but the workflow must not die
    X, n_atoms = _two_basin(45, 40)
    third = X[:15] + 30.0
    X = np.concatenate([X, third])
    labels, condensed, _ = pc.cluster_poses(X, cutoff=2.0, n_lig=n_atoms)
    summary = pc.cluster_summary(labels, condensed, np.arange(len(X)))

    _, _, fraction = pc.dominant_medoid(summary)
    assert fraction < 0.5  # the caller's warn-and-proceed branch
    assert len(summary) == 3


# --- pocket selection ------------------------------------------------------

class _FakeAtom:
    def __init__(self, resid):
        self.resid = resid


class _FakeTopology:
    """Minimal stand-in for a pytraj Topology: mask -> atom indices."""

    def __init__(self, selections, resids=None):
        self._selections = selections
        self._resids = resids or {}

    def select(self, mask):
        return self._selections.get(mask)

    def atom(self, index):
        return _FakeAtom(self._resids.get(index, 0))


def _pocket_geometry():
    """Ligand at the origin, 12 receptor atoms at 5 A, 12 more at 20 A."""
    xyz = np.zeros((28, 3))
    xyz[:4] = RNG.normal(scale=0.3, size=(4, 3))       # ligand (2 heavy, 2 H)
    xyz[4:16] = RNG.normal(scale=0.3, size=(12, 3)) + 5.0
    xyz[16:28] = RNG.normal(scale=0.3, size=(12, 3)) + 20.0
    return xyz


def test_protein_pocket_uses_ca_atoms_within_the_radius():
    xyz = _pocket_geometry()
    top = _FakeTopology(
        {
            ":LIG & !@H*": np.array([0, 1]),
            ":LIG": np.array([0, 1, 2, 3]),
            ":1-10 & @CA": np.arange(4, 28),
            ":1-10 & !@H*": np.arange(4, 28),
        },
        resids={i: i for i in range(28)},
    )
    pocket, ligand, label = pc.build_pocket_atoms(top, xyz, ":LIG", ":1-10", radius=10.0)

    assert sorted(pocket) == list(range(4, 16))  # the far shell is excluded
    assert sorted(ligand) == [0, 1]              # descriptor is heavy atoms only
    assert "pocket CA residues" in label


def test_side_chain_contact_pulls_in_a_distant_ca():
    # the fragment case: every CA sits beyond the radius, only side chains touch. A
    # CA-distance rule finds nothing here and the run dies.
    xyz = np.zeros((26, 3))
    xyz[0] = [0.0, 0.0, 0.0]                                    # ligand heavy atom
    xyz[1] = [0.5, 0.0, 0.0]                                    # ligand hydrogen
    side_chains = np.arange(2, 14)
    cas = np.arange(14, 26)
    xyz[side_chains] = np.array([[5.0, 0.0, 0.0]]) + RNG.normal(scale=0.2, size=(12, 3))
    xyz[cas] = np.array([[12.0, 0.0, 0.0]]) + RNG.normal(scale=0.2, size=(12, 3))
    resids = {int(a): i for i, a in enumerate(side_chains)}
    resids.update({int(a): i for i, a in enumerate(cas)})       # CA i shares residue i
    top = _FakeTopology(
        {
            ":LIG & !@H*": np.array([0]),
            ":LIG": np.array([0, 1]),
            ":REC & @CA": cas,
            ":REC & !@H*": np.concatenate([side_chains, cas]),
        },
        resids=resids,
    )
    pocket, _, label = pc.build_pocket_atoms(top, xyz, ":LIG", ":REC", radius=8.0)

    assert sorted(pocket) == list(cas)
    assert "radius grown" not in label


def test_an_underfilled_pocket_grows_the_radius():
    # 8 CAs at 4.5 A, 12 more at 10.5 A: the requested radius is one atom short of a
    # trustworthy alignment, so the search widens instead of killing the workflow.
    xyz = np.zeros((22, 3))
    xyz[1] = [0.5, 0.0, 0.0]                                   # ligand hydrogen
    for k, atom in enumerate(range(2, 10)):
        xyz[atom] = [5.0, 0.1 * k, 0.0]
    for k, atom in enumerate(range(10, 22)):
        xyz[atom] = [11.0, 0.1 * k, 0.0]
    top = _FakeTopology(
        {
            ":LIG & !@H*": np.array([0]),
            ":LIG": np.array([0, 1]),
            ":REC & @CA": np.arange(2, 22),
            ":REC & !@H*": np.arange(2, 22),
        },
        resids={i: i for i in range(22)},
    )
    pocket, _, label = pc.build_pocket_atoms(top, xyz, ":LIG", ":REC", radius=6.0)

    assert len(pocket) == 20                 # the far shell came in at 12 A
    assert "radius grown 6 -> 12 A" in label


def test_host_guest_pocket_falls_back_to_heavy_atoms():
    # no @CA anywhere: the branch a protein trajectory can never exercise
    xyz = _pocket_geometry()
    top = _FakeTopology({
        ":M01 & !@H*": np.array([0, 1]),
        ":M01": np.array([0, 1, 2, 3]),
        ":CB7 & @CA": None,
        ":CB7 & !@H*": np.arange(4, 28),
    })
    pocket, ligand, label = pc.build_pocket_atoms(top, xyz, ":M01", ":CB7", radius=10.0)

    assert sorted(pocket) == list(range(4, 16))
    assert sorted(ligand) == [0, 1]
    assert "host-guest" in label


def test_pocket_search_uses_all_ligand_atoms_not_just_heavy():
    # the lone hydrogen reaches a shell the heavy atoms do not
    xyz = np.zeros((10, 3))
    xyz[0] = [0.0, 0.0, 0.0]   # heavy
    xyz[1] = [4.0, 0.0, 0.0]   # hydrogen, displaced toward the receptor
    xyz[2:] = np.array([[9.0, 0.0, 0.0]]) + RNG.normal(scale=0.05, size=(8, 3))
    top = _FakeTopology({
        ":LIG & !@H*": np.array([0]),
        ":LIG": np.array([0, 1]),
        ":REC & @CA": None,
        ":REC & !@H*": np.arange(2, 10),
    })
    # 6 A reaches the shell from the hydrogen at x=4 but not from the heavy atom at x=0
    pocket, _, _ = pc.build_pocket_atoms(top, xyz, ":LIG", ":REC", radius=6.0, min_atoms=8)
    assert len(pocket) == 8


def test_too_few_pocket_atoms_is_an_error():
    xyz = _pocket_geometry()
    top = _FakeTopology({
        ":LIG & !@H*": np.array([0, 1]),
        ":LIG": np.array([0, 1, 2, 3]),
        ":1-10 & @CA": np.arange(16, 28),     # only the far shell
        ":1-10 & !@H*": np.arange(16, 28),
    })
    with pytest.raises(ValueError, match="pocket"):
        pc.build_pocket_atoms(top, xyz, ":LIG", ":1-10", radius=10.0)


def test_empty_ligand_mask_is_an_error():
    top = _FakeTopology({":NOPE & !@H*": None, ":NOPE": None})
    with pytest.raises(ValueError, match="ligand mask"):
        pc.build_pocket_atoms(top, np.zeros((4, 3)), ":NOPE", ":1-10", radius=8.0)


# --- superposition ---------------------------------------------------------

def test_kabsch_recovers_a_known_rigid_motion():
    reference = RNG.normal(size=(20, 3))
    theta = 0.7
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    mobile = reference @ rotation.T + np.array([5.0, -2.0, 3.0])

    R, mc, rc = kabsch_result = pc.kabsch(mobile, reference)
    assert np.allclose((mobile - mc) @ R + rc, reference)
    assert np.linalg.det(kabsch_result[0]) == pytest.approx(1.0)


def test_kabsch_never_returns_a_reflection():
    reference = RNG.normal(size=(20, 3))
    mirrored = reference * np.array([1.0, 1.0, -1.0])
    R, _, _ = pc.kabsch(mirrored, reference)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_pose_matrix_is_invariant_to_rigid_motion_of_the_whole_system():
    """A tumbling complex must give identical poses -- that is what the pocket fit removes."""
    base = RNG.normal(scale=3.0, size=(12, 3))
    theta = 1.1
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    traj = np.stack([base, base @ rotation.T + np.array([10.0, 4.0, -7.0])])

    frames, X = pc.pose_matrix(traj, ligand_indices=[0, 1, 2], pocket_indices=range(3, 12))

    assert list(frames) == [0, 1]
    assert np.allclose(X[0], X[1], atol=1e-8)


def test_pose_matrix_respects_stride_and_skip():
    traj = RNG.normal(size=(20, 8, 3))
    frames, X = pc.pose_matrix(traj, [0, 1], range(2, 8), stride=5, skip=3)
    assert list(frames) == [3, 8, 13, 18]
    assert X.shape == (4, 6)


# --- PCA -------------------------------------------------------------------

def test_pca_explains_all_the_variance_across_full_rank():
    X = RNG.normal(size=(40, 6))
    _, explained, _, _ = pc.pca(X, n_components=6)
    assert np.sum(explained) == pytest.approx(1.0)
    assert np.all(np.diff(explained) <= 1e-12)  # descending


def test_pca_projection_round_trips_through_project():
    X = RNG.normal(size=(30, 9))
    proj, _, mean, components = pc.pca(X, n_components=9)
    assert np.allclose(pc.project(X[7], mean, components), proj[7])


# --- diagnostics -----------------------------------------------------------

def test_cosine_content_flags_a_random_walk():
    # Hess: PC1 of a multidimensional diffusion looks like a clean collective motion
    walk = np.cumsum(RNG.normal(size=(2000, 12)), axis=0)
    proj, _, _, _ = pc.pca(walk, n_components=3)
    assert pc.cosine_content(proj[:, 0], 1) > 0.7


def test_cosine_content_is_low_for_a_two_state_jump():
    jump = np.concatenate([np.zeros(500), np.ones(500)])
    assert pc.cosine_content(jump, 1) < 0.7


@pytest.mark.parametrize("labels,expected", [
    ([1, 1, 1, 1], 1.0),
    ([1, 2, 1, 2], 0.0),
    ([1, 1, 2, 2], 2 / 3),
])
def test_cluster_metastability(labels, expected):
    assert pc.cluster_metastability(np.array(labels)) == pytest.approx(expected)


# --- wiring ----------------------------------------------------------------

MDIN = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "input_files", "intermediate.mdin"
)


def _intermediate_args(**overrides):
    kwargs = dict(
        exponent_conformational_forces=[-14.0, 0.0],
        exponent_orientational_forces=[-10.0, 4.0],
        restraint_type=1,
        igb_solvent=2,
        mdin_intermediate_file=MDIN,
        temperature=298,
    )
    kwargs.update(overrides)
    return IntermediateStateArgs(**kwargs)


def test_restraint_reference_defaults_to_the_previous_behaviour():
    assert _intermediate_args().restraint_reference == "lastframe"


@pytest.mark.parametrize("value", ["lastframe", "pose_medoid"])
def test_restraint_reference_accepts_both_modes(value):
    assert _intermediate_args(restraint_reference=value).restraint_reference == value


def test_restraint_reference_rejects_anything_else():
    # a typo must fail at config load, not silently fall back to lastframe
    with pytest.raises(ValueError, match="restraint_reference"):
        _intermediate_args(restraint_reference="medoid")


def _config(**overrides):
    return type("Cfg", (), {"intermediate_args": _intermediate_args(**overrides)})()


def test_flag_off_passes_no_extra_arguments():
    """The regression gate: default config must build ExtractTrajectories as it always did."""
    assert _pose_medoid_args(_config(), {"runtype": "remd"}) == {}


def test_flag_off_leaves_job_resources_untouched():
    before = ExtractTrajectories("t.parm7", ["a.nc"], 298.0)
    after = ExtractTrajectories(
        "t.parm7", ["a.nc"], 298.0, **_pose_medoid_args(_config(), {"runtype": "remd"})
    )
    assert after.memory == before.memory
    assert after.pose_medoid is False


def test_clustering_job_asks_for_more_memory():
    lastframe = ExtractTrajectories("t.parm7", ["a.nc"], 298.0)
    medoid = ExtractTrajectories(
        "t.parm7", ["a.nc"], 298.0, pose_medoid=True, ligand_mask=":LIG"
    )
    assert medoid.memory > lastframe.memory


def test_pose_medoid_without_a_ligand_mask_is_an_error():
    with pytest.raises(ValueError, match="ligand_mask"):
        ExtractTrajectories("t.parm7", ["a.nc"], 298.0, pose_medoid=True)
