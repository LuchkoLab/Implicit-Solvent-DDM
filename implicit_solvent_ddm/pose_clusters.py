"""Ligand pose clustering on a pocket-aligned trajectory.

Identifies the dominant (most populated) ligand binding pose by:

  1. superposing every frame on the BINDING-SITE atoms (not on the ligand),
  2. treating the ligand's Cartesian coordinates in that pocket frame as the
     pose descriptor,
  3. clustering with average-linkage hierarchical clustering on the
     no-fit ligand RMSD matrix (the cpptraj-standard approach),
  4. projecting onto PCs for visualization only.

Why superpose on the pocket and not the ligand:
    Best-fit RMSD on the ligand removes exactly the rigid-body translation and
    rotation that *define* the binding pose. After pocket alignment, the no-fit
    ligand RMSD between frames i and j is

        RMSD_ij = ||x_i - x_j|| / sqrt(N_atoms)

    i.e. a plain Euclidean distance in the flattened 3N coordinate space,
    rescaled. That is what makes Cartesian PCA metrically consistent with the
    clustering distance here -- the PCA is a linear embedding of the same
    metric, not a separate and incompatible one.

The medoid of the dominant cluster is the restraint reference the workflow wants:
mean applied restraint energy goes as sum_k <(d(x) - d(r))^2>, which is minimised
at the medoid. The endstate last frame is a random draw from this distribution and
lands outside the dominant basin roughly a third of the time.

Everything here is numpy + scipy so it can be unit tested without a trajectory.
``load_trajectory`` is the only pytraj entry point, and matplotlib is imported
lazily inside the plot functions.
"""

import os

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform

DEFAULT_FRAME_CAP = 2000  # O(n^2): 2000 frames is 16 MB condensed, 10000 is 400 MB
MIN_POCKET_ATOMS = 10
MIN_CLUSTER_FRAMES = 10
MAX_POCKET_RADIUS = 16.0  # auto-expansion ceiling; past this it is no longer "the pocket"


# ----------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------

def cosine_content(pc, k=1):
    """Hess cosine content of a principal component projection.

    Random diffusion in a short/unconverged trajectory produces PCs that look
    like clean collective motion. Values near 1 mean PC_k is indistinguishable
    from a random walk and should NOT be interpreted as a real transition.

    Hess, Phys. Rev. E 62:8438 (2000); 65:031910 (2002).
    """
    pc = np.asarray(pc, dtype=float)
    n = len(pc)
    t = np.arange(n)
    num = np.sum(np.cos(k * np.pi * t / n) * pc) ** 2
    den = np.sum(pc ** 2)
    return (2.0 / n) * num / den


def cluster_metastability(labels):
    """Fraction of consecutive frame pairs that stay in the same cluster.

    Low values mean the 'clusters' are being crossed constantly and are
    probably not separated by any real barrier.
    """
    labels = np.asarray(labels)
    return float(np.mean(labels[:-1] == labels[1:]))


def resolve_stride(n_frames, stride=0, cap=DEFAULT_FRAME_CAP):
    """Frame stride, auto-sized to keep the O(n^2) matrix bounded when stride is 0."""
    if stride and stride > 0:
        return int(stride)
    if n_frames <= cap:
        return 1
    return int(-(-n_frames // cap))  # ceil


# ----------------------------------------------------------------------
# PCA (visualization / descriptive only -- see module docstring)
# ----------------------------------------------------------------------

def pca(X, n_components=10):
    """Principal components of X via SVD, as ``(projection, explained_variance_ratio,
    mean, components)``. Matches ``sklearn.decomposition.PCA`` without the dependency;
    the analysis only uses it to draw a scatter plot."""
    X = np.asarray(X, dtype=float)
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    mean = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    total = np.sum(S ** 2)
    components = Vt[:n_components]
    projection = (U[:, :n_components] * S[:n_components])
    explained = (S[:n_components] ** 2) / total
    return projection, explained, mean, components


def project(x, mean, components):
    """Project one flattened pose onto an existing PCA basis."""
    return (np.asarray(x, dtype=float) - mean) @ components.T


# ----------------------------------------------------------------------
# selection and superposition
# ----------------------------------------------------------------------

def _select(top, mask):
    """Atom indices for an AMBER mask, as a plain int array (never None)."""
    sel = top.select(mask)
    if sel is None:
        return np.empty(0, dtype=int)
    return np.asarray(sel, dtype=int)


def _radius_ladder(radius, max_radius, step=2.0):
    """The requested radius, then ``step`` increments up to ``max_radius`` inclusive."""
    ladder = [float(radius)]
    while ladder[-1] + step <= max_radius:
        ladder.append(ladder[-1] + step)
    return ladder


def build_pocket_atoms(top, xyz0, ligand_mask, receptor_mask, radius,
                       min_atoms=MIN_POCKET_ATOMS, max_radius=MAX_POCKET_RADIUS):
    """Freeze a static binding-site atom selection based on frame 0.

    A residue lines the site when any of its heavy atoms sits within ``radius`` of the
    ligand; the superposition uses the C-alphas of those residues, or the contacting heavy
    atoms when the receptor has no backbone (host-guest). The radius grows in 2 A steps up
    to ``max_radius`` when the site comes up short -- contact on the C-alpha alone
    undercounts a fragment's site. Distances are evaluated numerically because
    ``Topology.select`` cannot evaluate a cpptraj distance mask.

    Parameters
    ----------
    top : pytraj.Topology
        Topology the masks are evaluated against.
    xyz0 : numpy.ndarray
        ``(n_atoms, 3)`` coordinates of the frame that defines the site.
    ligand_mask, receptor_mask : str
        AMBER masks selecting the ligand and the receptor.
    radius : float
        Contact distance, in Angstrom.
    min_atoms : int, optional
        Fewest pocket atoms an alignment is trusted with.
    max_radius : float, optional
        Ceiling for the automatic radius growth.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, str)
        Pocket atom indices, ligand heavy-atom indices, and a label for the report.

    Raises
    ------
    ValueError
        If either mask selects nothing, or the site is still underfilled at
        ``max_radius``.
    """
    xyz0 = np.asarray(xyz0, dtype=float)

    # heavy atoms are the pose descriptor; all atoms define the pocket, since a hydrogen
    # in contact means the residue is in the site
    ligand_indices = _select(top, f"{ligand_mask} & !@H*")
    ligand_all = _select(top, ligand_mask)
    if ligand_indices.size == 0:
        raise ValueError(
            f"ligand mask {ligand_mask!r} selected no heavy atoms"
        )

    receptor_heavy = _select(top, f"{receptor_mask} & !@H*")
    if receptor_heavy.size == 0:
        raise ValueError(
            f"receptor mask {receptor_mask!r} selected no atoms"
        )

    ca = _select(top, f"{receptor_mask} & @CA")
    kind = "CA" if ca.size > 0 else "heavy"

    contact = np.linalg.norm(
        xyz0[receptor_heavy][:, None, :] - xyz0[ligand_all][None, :, :], axis=-1
    ).min(axis=1)

    for used in _radius_ladder(radius, max_radius):
        near = receptor_heavy[contact <= used]
        if kind == "CA":
            resids = {int(top.atom(int(i)).resid) for i in near}
            pocket_indices = np.array(
                [int(i) for i in ca if int(top.atom(int(i)).resid) in resids], dtype=int
            )
        else:
            pocket_indices = near
        if pocket_indices.size >= min_atoms:
            break

    if pocket_indices.size < min_atoms:
        raise ValueError(
            f"only {pocket_indices.size} pocket {kind} atoms within {used:g} A of the "
            f"ligand; increase the pocket radius or check the masks"
        )

    grown = "" if used == float(radius) else f" [radius grown {float(radius):g} -> {used:g} A]"
    if kind == "CA":
        resids = sorted({int(top.atom(int(i)).resid) + 1 for i in pocket_indices})
        label = f"pocket CA residues   : {resids}{grown}"
    else:
        label = (
            f"pocket heavy atoms   : {pocket_indices.size} (no backbone; host-guest)"
            f"{grown}"
        )

    return pocket_indices, ligand_indices, label


def _frame_xyz(frame):
    """Coordinates of one frame, from a pytraj Frame or a plain (n_atoms, 3) array."""
    xyz = getattr(frame, "xyz", frame)
    return np.asarray(xyz, dtype=float)


def kabsch(mobile, reference):
    """Optimal rigid rotation superposing ``mobile`` onto ``reference``.

    Both are ``(N, 3)``. Returns ``(rotation, mobile_centroid, reference_centroid)``
    such that ``(x - mobile_centroid) @ rotation + reference_centroid`` superposes any
    row-vector coordinate ``x``. No mass weighting, matching MDAnalysis ``AlignTraj``.
    """
    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)
    mc = mobile.mean(axis=0)
    rc = reference.mean(axis=0)
    cov = (mobile - mc).T @ (reference - rc)
    V, _, Wt = np.linalg.svd(cov)
    # reject the reflection SVD will otherwise hand back
    if np.linalg.det(V @ Wt) < 0:
        V[:, -1] *= -1.0
    return V @ Wt, mc, rc


def pose_matrix(traj, ligand_indices, pocket_indices, stride=1, skip=0):
    """Pocket-superposed ligand coordinates for every retained frame.

    Returns ``(frames, X)`` where ``frames`` holds the ORIGINAL frame numbers and
    ``X`` is ``(n_frames, 3 * n_ligand_atoms)``. Frame 0 is the superposition
    reference, so ``X`` is directly comparable to ``reference_pose``.
    """
    ligand_indices = np.asarray(ligand_indices, dtype=int)
    pocket_indices = np.asarray(pocket_indices, dtype=int)

    ref_pocket = _frame_xyz(traj[0])[pocket_indices]

    frames, coords = [], []
    for i in range(int(skip), len(traj), int(stride)):
        xyz = _frame_xyz(traj[i])
        rotation, mc, rc = kabsch(xyz[pocket_indices], ref_pocket)
        coords.append((xyz[ligand_indices] - mc) @ rotation + rc)
        frames.append(i)

    X = np.asarray(coords).reshape(len(frames), -1)
    return np.asarray(frames, dtype=int), X


def reference_pose(traj, ligand_indices):
    """Frame 0's ligand coordinates, flattened. It is the superposition reference, so
    it needs no rotation of its own."""
    ligand_indices = np.asarray(ligand_indices, dtype=int)
    return _frame_xyz(traj[0])[ligand_indices].ravel()


# ----------------------------------------------------------------------
# clustering
# ----------------------------------------------------------------------

def cluster_poses(X, cutoff, n_lig):
    """Average-linkage clustering on the no-fit ligand RMSD matrix.

    Returns ``(labels, condensed, linkage_matrix)``. Labels are renumbered by
    descending population, so cluster 1 is always the dominant basin.
    """
    X = np.asarray(X, dtype=float)
    condensed = pdist(X) / np.sqrt(n_lig)  # RMSD, Angstrom
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=cutoff, criterion="distance")

    uniq, counts = np.unique(labels, return_counts=True)
    order = uniq[np.argsort(-counts, kind="stable")]
    remap = {old: i + 1 for i, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])
    return labels, condensed, Z


def cluster_summary(labels, condensed, frames):
    """Per-cluster medoid, population and spread, ordered by descending population.

    ``medoid_row`` indexes into ``labels`` / ``X``; ``medoid_frame`` is the original
    trajectory frame number.
    """
    labels = np.asarray(labels)
    frames = np.asarray(frames)
    rmsd = squareform(condensed)
    n_frames = len(labels)

    summary = []
    for c in range(1, labels.max() + 1):
        idx = np.where(labels == c)[0]
        sub = rmsd[np.ix_(idx, idx)]
        local = int(np.argmin(sub.sum(axis=1)))
        summary.append(dict(
            cluster=c,
            medoid_row=int(idx[local]),
            population=len(idx),
            fraction=len(idx) / n_frames,
            medoid_frame=int(frames[idx[local]]),
            avg_rmsd_to_medoid=float(sub[local].mean()),
            first_frame=int(frames[idx[0]]),
            last_frame=int(frames[idx[-1]]),
        ))
    return summary


def dominant_medoid(summary):
    """The most populated cluster's medoid: ``(row, frame, fraction)``. Callers should
    warn when the fraction is not a clear majority -- there is no representative pose in
    that case, only a least-bad one."""
    top = summary[0]
    return top["medoid_row"], top["medoid_frame"], top["fraction"]


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------

def build_report(summary, labels, frames, X, x_init, explained, pocket_label,
                 n_lig, cutoff, stride, skip, init_label="initial (frame 0)"):
    """The ``summary.txt`` body, as a single string (no trailing newline)."""
    labels = np.asarray(labels)
    frames = np.asarray(frames)
    n_frames = len(frames)

    proj, _, mean, components = pca(X)
    cc1 = cosine_content(proj[:, 0], 1)
    cc2 = cosine_content(proj[:, 1], 1)
    meta = cluster_metastability(labels)

    d_init = np.linalg.norm(X - x_init, axis=1) / np.sqrt(n_lig)
    nearest_cluster = int(labels[np.argmin(d_init)])
    for r in summary:
        r["rmsd_init_to_medoid"] = float(
            np.linalg.norm(X[r["medoid_row"]] - x_init) / np.sqrt(n_lig)
        )

    best = min(summary, key=lambda r: r["rmsd_init_to_medoid"])
    dominant = summary[0]

    lines = []
    lines.append(f"frames analyzed      : {n_frames} (skip={skip}, stride={stride})")
    lines.append(f"ligand atoms         : {n_lig}")
    lines.append(pocket_label)
    lines.append(f"linkage cutoff       : {cutoff} A")
    lines.append(f"PC1/PC2 variance     : {explained[0]:.1%} / {explained[1]:.1%}")
    lines.append(f"PC1 cosine content   : {cc1:.2f}   (>0.7 => random-walk-like, do not over-read)")
    lines.append(f"PC2 cosine content   : {cc2:.2f}")
    lines.append(f"cluster metastability: {meta:.2f}   (frames staying in the same cluster)")
    lines.append("")
    lines.append(f"reference pose       : {init_label}")
    lines.append(f"  RMSD over traj     : mean {d_init.mean():.2f} A, "
                 f"min {d_init.min():.2f} A, max {d_init.max():.2f} A")
    lines.append(f"  RMSD to LAST frame : {d_init[-1]:.2f} A   (frame {int(frames[-1])})")
    lines.append(f"  closest single frame: {int(frames[np.argmin(d_init)])} "
                 f"({d_init.min():.2f} A, cluster {nearest_cluster})")
    lines.append("")
    lines.append(f"  nearest medoid     : cluster {best['cluster']} at "
                 f"{best['rmsd_init_to_medoid']:.2f} A "
                 f"(pop {best['fraction']:.1%}, medoid frame {best['medoid_frame']})")
    lines.append(f"  dominant medoid    : cluster {dominant['cluster']} at "
                 f"{dominant['rmsd_init_to_medoid']:.2f} A "
                 f"(pop {dominant['fraction']:.1%}, medoid frame {dominant['medoid_frame']})")
    if best["cluster"] != dominant["cluster"]:
        lines.append("  NOTE: the pose closest to the reference is NOT the most "
                     "populated one. The dominant basin has moved away from the "
                     "starting structure.")
    elif best["rmsd_init_to_medoid"] > 2.0:
        lines.append("  NOTE: the dominant pose is also the closest to the reference, "
                     "but has drifted >2 A from it.")
    lines.append("")
    lines.append(f"{'clu':>4} {'pop':>7} {'frac':>7} {'medoid':>8} {'spread':>8} "
                 f"{'d_init':>8} {'first':>8} {'last':>8}")
    for r in summary:
        lines.append(f"{r['cluster']:>4} {r['population']:>7} {r['fraction']:>7.1%} "
                     f"{r['medoid_frame']:>8} {r['avg_rmsd_to_medoid']:>8.2f} "
                     f"{r['rmsd_init_to_medoid']:>8.2f} "
                     f"{r['first_frame']:>8} {r['last_frame']:>8}")

    return "\n".join(lines), proj, d_init, project(x_init, mean, components)


def write_report(path, report):
    with open(path, "w") as fh:
        fh.write(report + "\n")


def write_assignments(path, frames, labels, proj, d_init):
    np.savetxt(path,
               np.column_stack([frames, labels, proj[:, 0], proj[:, 1], proj[:, 2], d_init]),
               delimiter=",", header="frame,cluster,PC1,PC2,PC3,rmsd_to_initial",
               comments="", fmt="%.4f")


# ----------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------

def _pyplot():
    """matplotlib with a headless backend, imported only when a plot is actually drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_pose_analysis(path, summary, frames, labels, proj, explained, d_init,
                       proj_init, init_label="initial (frame 0)"):
    plt = _pyplot()
    labels = np.asarray(labels)
    frames = np.asarray(frames)
    meta = cluster_metastability(labels)
    best = min(summary, key=lambda r: r["rmsd_init_to_medoid"])

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    ax[0].scatter(proj[:, 0], proj[:, 1], c=labels, s=6, cmap="tab10", alpha=0.7)
    for r in summary[:5]:
        k = np.where(frames == r["medoid_frame"])[0][0]
        ax[0].annotate(f"C{r['cluster']}", (proj[k, 0], proj[k, 1]),
                       fontsize=11, fontweight="bold")
    # initial / docked pose: big open red circle + crosshairs
    ax[0].scatter(proj_init[0], proj_init[1], s=320, facecolors="none",
                  edgecolors="red", linewidths=2.5, zorder=10, label=init_label)
    ax[0].scatter(proj_init[0], proj_init[1], s=25, c="red", marker="+",
                  linewidths=1.5, zorder=11)
    ax[0].axvline(proj_init[0], color="red", ls=":", lw=0.8, alpha=0.5, zorder=1)
    ax[0].axhline(proj_init[1], color="red", ls=":", lw=0.8, alpha=0.5, zorder=1)

    # link the reference to the medoid closest to it, labelled with the RMSD
    k_best = np.where(frames == best["medoid_frame"])[0][0]
    ax[0].plot([proj_init[0], proj[k_best, 0]], [proj_init[1], proj[k_best, 1]],
               color="red", ls="--", lw=1.2, alpha=0.8, zorder=9)
    ax[0].scatter(proj[k_best, 0], proj[k_best, 1], s=140, facecolors="none",
                  edgecolors="red", linewidths=1.8, marker="s", zorder=10,
                  label=f"nearest medoid C{best['cluster']} "
                        f"({best['rmsd_init_to_medoid']:.2f} A)")
    ax[0].scatter(proj[-1, 0], proj[-1, 1], s=140, c="red", marker="X",
                  edgecolors="black", linewidths=0.6, zorder=10,
                  label=f"last frame ({d_init[-1]:.2f} A)")
    ax[0].legend(loc="best", fontsize=8, frameon=False)
    ax[0].set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax[0].set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax[0].set_title("ligand pose space (pocket-aligned)")

    ax[1].scatter(frames, labels, c=labels, s=6, cmap="tab10", zorder=3)
    ax[1].set_xlabel("frame")
    ax[1].set_ylabel("cluster")
    ax[1].set_title(f"occupancy vs time (metastability {meta:.2f})")
    ax1b = ax[1].twinx()
    ax1b.plot(frames, d_init, color="red", lw=0.8, alpha=0.6, zorder=1)
    ax1b.set_ylabel("RMSD to initial pose (A)", color="red")
    ax1b.tick_params(axis="y", colors="red")

    ax[2].bar([r["cluster"] for r in summary], [r["fraction"] for r in summary],
              color="steelblue")
    ax[2].set_xlabel("cluster")
    ax[2].set_ylabel("population fraction")
    ax[2].set_title("cluster populations")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dendrogram(path, Z, cutoff):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(Z, no_labels=True, color_threshold=cutoff, ax=ax)
    ax.axhline(cutoff, ls="--", c="k", lw=1)
    ax.set_ylabel("no-fit ligand RMSD (A)")
    ax.set_title("average-linkage dendrogram (check the cutoff is not arbitrary)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------

def load_trajectory(topology, trajectories):
    """pytraj trajectory for one topology and one or more NetCDF files."""
    import pytraj as pt

    if isinstance(trajectories, (str, bytes, os.PathLike)):
        trajectories = [trajectories]
    return pt.iterload(list(trajectories), top=topology)


def analyze(traj, ligand_mask, receptor_mask, cutoff=2.0, radius=8.0, stride=0,
            skip=0, frame_cap=DEFAULT_FRAME_CAP):
    """Run the whole analysis and return everything the caller might want to write.

    Returns a dict with ``summary``, ``labels``, ``frames``, ``linkage``, ``report``,
    ``proj``, ``d_init``, ``proj_init``, ``stride`` and ``pocket_label``.
    """
    stride = resolve_stride(len(traj) - int(skip), stride, cap=frame_cap)
    pocket_indices, ligand_indices, pocket_label = build_pocket_atoms(
        traj.top, _frame_xyz(traj[0]), ligand_mask, receptor_mask, radius
    )
    frames, X = pose_matrix(traj, ligand_indices, pocket_indices, stride=stride, skip=skip)
    n_lig = len(ligand_indices)

    labels, condensed, Z = cluster_poses(X, cutoff, n_lig)
    summary = cluster_summary(labels, condensed, frames)

    x_init = reference_pose(traj, ligand_indices)
    _, explained, _, _ = pca(X)
    report, proj, d_init, proj_init = build_report(
        summary, labels, frames, X, x_init, explained, pocket_label,
        n_lig, cutoff, stride, skip
    )

    return dict(summary=summary, labels=labels, frames=frames, linkage=Z,
                report=report, proj=proj, d_init=d_init, proj_init=proj_init,
                explained=explained, stride=stride, pocket_label=pocket_label,
                n_ligand_atoms=n_lig)


def write_artifacts(result, out_dir, prefix="", cutoff=2.0):
    """Write the four artifacts and return their paths, in a stable order."""
    os.makedirs(out_dir, exist_ok=True)
    paths = dict(
        summary=os.path.join(out_dir, f"{prefix}summary.txt"),
        assignments=os.path.join(out_dir, f"{prefix}assignments.csv"),
        pose_analysis=os.path.join(out_dir, f"{prefix}pose_analysis.png"),
        dendrogram=os.path.join(out_dir, f"{prefix}dendrogram.png"),
    )
    write_report(paths["summary"], result["report"])
    write_assignments(paths["assignments"], result["frames"], result["labels"],
                      result["proj"], result["d_init"])
    plot_pose_analysis(paths["pose_analysis"], result["summary"], result["frames"],
                       result["labels"], result["proj"], result["explained"],
                       result["d_init"], result["proj_init"])
    plot_dendrogram(paths["dendrogram"], result["linkage"], cutoff)
    return paths
