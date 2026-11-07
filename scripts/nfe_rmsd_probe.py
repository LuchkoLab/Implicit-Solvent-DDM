#!/usr/bin/env python
"""nfe_rmsd_probe.py -- can AMBER give us a GPU RMSD restraint, and is it best-fit?

WHY
---
We want to replace the pairwise conformational restraint network (82,077 ``&rst`` terms on MCL-1)
with a single best-fit RMSD collective variable under a flat-bottom umbrella, so the restraint ladder
walks a bounded coordinate in Angstrom that terminates exactly at the unrestrained endstate.

The obvious mechanism, Targeted MD (``itgtmd=1``), is sander-only -- the AMBER 25 manual lists it
under functionality "missing entirely" from PMEMD (Sec. 23.3), and there is no ``sander.cuda``. The
route that survives is AMBER's NFE module: ``infe=1`` + ``&pmd`` + ``cv_type='MULTI_RMSD'``, which
Sec. 26.4.4 says "can be fully applied in GPU accelerated PMEMD".

The manual leaves several load-bearing questions unanswered, and one of them can kill the design:

* Sec. 26.4.4/26.4.7 promise the *modules* are GPU-capable; they never enumerate which CV types have
  CUDA kernels.
* ``MULTI_RMSD`` cites Coutsias/Seok/Dill (quaternion RMSD), which IS an optimal-superposition
  method -- but the manual never says the NFE driver superimposes before calling it. A lab-frame CV
  would fight the Boresch orientational restraints and invalidate the standard-state correction, and
  the approach is invalid as specified.
* No documented limit on ``cv_ni``/``cv_nr``. This workflow has already been bitten once by an
  undocumented Fortran array cap -- ``igr1``/``igr2`` top out at 200 atoms, which is what forced the
  ``@CA`` workaround in the flat-bottom COM restraint (release v1.1.3). HSP90 has 207 CA atoms.

This script answers those questions in minutes, before any workflow code is touched.

WHAT THIS DOES
--------------
Each probe builds its own self-contained mdin + ``cv.in`` + ``restraint.RST`` in a scratch directory
and runs AMBER directly. Nothing in ``implicit_solvent_ddm/`` is modified; probe 4 imports the
package's own mdout parser so it measures exactly what MBAR would see.

  1 runs      does ``infe=1`` + ``&pmd`` + MULTI_RMSD start at all on pmemd.cuda with igb != 0, with
              a real DISANG present so nmropt=1 and NFE are active in the same run?
  2 bestfit   THE GATE. Is the CV rotation/translation invariant, and does it equal cpptraj
              ``rms ... mass``? Runs short MD to move the structure off the reference, then
              evaluates the CV at step 0 twice -- once against the reference as given, once against
              a randomly rotated and translated copy of it -- and cross-checks both against cpptraj.
  3 bias      does the restraint apply force, or only monitor? Two short runs at r3=0, k=0 vs
              k=``--k``, compared by their RMSD distributions.
  4 rescore   does ``imin=5`` evaluate NFE biases, and does the bias reach the ENERGY column MBAR
              reads? Informational -- the analytic post-analysis route ships either way.
  5 scale     probe 2 on the real system, to catch a silent ``cv_r`` truncation. Compares the
              reported CV against cpptraj rather than merely checking for an error message.

``--probe gate`` (the default) runs 1, 2 and 5 -- everything that can stop the design.

HOW TO READ IT
--------------
Every probe prints GO or NO-GO with the number it decided on. Probe 2 is the one to read first: if
``rotated`` differs from ``as-given`` by more than ``--tol``, MULTI_RMSD is lab-frame and the design
needs rethinking, not patching.

EXAMPLE
-------
    python scripts/nfe_rmsd_probe.py --probe gate \\
        --prmtop .../DRY_complex.parm7 --inpcrd .../complex.rst7 --workdir nfe_probe_work

    python scripts/nfe_rmsd_probe.py --probe 3 --nstlim 25000 \\
        --prmtop ... --inpcrd ... --k 50
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

CPPTRAJ = os.environ.get("CPPTRAJ", "cpptraj")


# --------------------------------------------------------------------------------------------------
# Shell
# --------------------------------------------------------------------------------------------------
def run_cpptraj(script, workdir, tag, dry=False):
    """Write a cpptraj input file and execute it.

    Parameters
    ----------
    script : str
        Complete cpptraj input. Every input path in it must be absolute, since cpptraj runs with
        ``cwd=workdir`` so that bare output names land there.
    workdir : str
        Directory the job runs in and writes its outputs to.
    tag : str
        Basename for the generated ``<tag>.cpptraj`` input file.
    dry : bool, optional
        Print the script instead of running it, by default False.

    Raises
    ------
    SystemExit
        If cpptraj exits non-zero.
    """
    path = os.path.abspath(os.path.join(workdir, f"{tag}.cpptraj"))
    with open(path, "w") as handle:
        handle.write(script)
    if dry:
        print(f"--- {tag}.cpptraj ---\n{script}")
        return
    proc = subprocess.run([CPPTRAJ, "-i", path], capture_output=True, text=True, cwd=workdir)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"cpptraj failed on {tag} (exit {proc.returncode})")


def run_amber(executable, workdir, tag, extra, dry=False):
    """Run one AMBER job and capture its output.

    A non-zero exit is itself a probe result, so this returns the code rather than raising.

    Parameters
    ----------
    executable : str
        AMBER binary to invoke, e.g. ``pmemd.cuda`` or ``sander``.
    workdir : str
        Directory the job runs in.
    tag : str
        Basename of ``<tag>.mdin`` / ``<tag>.mdout`` / ``<tag>.stdout``.
    extra : list of str
        Additional command-line arguments (``-p``, ``-c``, ``-x``, ``-y``, ``-r``).
    dry : bool, optional
        Print the command instead of running it, by default False.

    Returns
    -------
    int
        The process exit code, or 0 when ``dry`` is set.
    """
    cmd = [executable, "-O", "-i", f"{tag}.mdin", "-o", f"{tag}.mdout"] + extra
    if dry:
        print(f"--- {tag} ---\n{' '.join(cmd)}")
        return 0
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir)
    with open(os.path.join(workdir, f"{tag}.stdout"), "w") as handle:
        handle.write(proc.stdout + proc.stderr)
    return proc.returncode


def read_dat(path):
    """Read a two-column cpptraj ``.dat`` (frame, value).

    Parameters
    ----------
    path : str
        Path to the cpptraj data file.

    Returns
    -------
    numpy.ndarray
        The value column.
    """
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                rows.append(float(parts[1]))
    return np.asarray(rows)


def read_pmd(path, column=1):
    """Read the collective-variable series out of an NFE ``&pmd`` output file.

    The pmd.txt layout is not documented, so this parses defensively: comments are stripped, only
    all-numeric rows are kept, and ``column`` is taken (NFE writes time or step first). Use
    ``--show-pmd`` to print the raw head if a layout surprise is suspected.

    Parameters
    ----------
    path : str
        Path to the ``&pmd`` ``output_file``.
    column : int, optional
        Zero-based column holding the CV, by default 1. Falls back to the last available column if
        the file is narrower.

    Returns
    -------
    numpy.ndarray
        The CV values, one per reported step.

    Raises
    ------
    SystemExit
        If no numeric rows are found.
    """
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(("#", "!")):
                continue
            try:
                rows.append([float(p) for p in line.split()])
            except ValueError:
                continue
    if not rows:
        raise SystemExit(f"no numeric rows in {path} -- inspect it by hand (--show-pmd)")
    width = min(len(r) for r in rows)
    idx = column if width > column else width - 1
    return np.asarray([r[idx] for r in rows])


def describe(name, values):
    """Summarise a distribution.

    Parameters
    ----------
    name : str
        Label for the row.
    values : numpy.ndarray
        Sample values.

    Returns
    -------
    dict
        Keys ``set``, ``n``, ``mean``, ``sd``, ``min``, ``p50``, ``max``.
    """
    return {
        "set": name,
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "p50": float(np.median(values)),
        "max": float(values.max()),
    }


# --------------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------------
def select_atoms(topology, mask=None):
    """Resolve the restrained atom set.

    Degrades the way ``restraints.FlatBottom._determine_restraint_atoms`` already does: backbone CA
    for a protein, all heavy atoms for a host-guest system with no backbone.

    Parameters
    ----------
    topology : pytraj.Topology
        Topology to select against.
    mask : str, optional
        Explicit AMBER mask. When omitted, ``@CA`` is tried first and ``!@H*`` is the fallback.

    Returns
    -------
    indices : numpy.ndarray
        Zero-based atom indices.
    mask : str
        The mask actually used.

    Raises
    ------
    SystemExit
        If the resolved mask selects no atoms.
    """
    if mask:
        sel = topology.select(mask)
        if sel is None or sel.size == 0:
            raise SystemExit(f"mask {mask!r} selected no atoms")
        return np.asarray(sel), mask
    sel = topology.select("@CA")
    if sel is not None and sel.size > 0:
        return np.asarray(sel), "@CA"
    sel = topology.select("!@H*")
    if sel is None or sel.size == 0:
        raise SystemExit("neither @CA nor !@H* selected any atoms")
    return np.asarray(sel), "!@H*"


def random_rigid_transform(coords, seed=20260830):
    """Rotate and translate a coordinate set as a rigid body.

    A best-fit RMSD must not notice this; a lab-frame one will.

    Parameters
    ----------
    coords : numpy.ndarray, shape (n, 3)
        Coordinates to transform.
    seed : int, optional
        RNG seed, so the transform is reproducible across runs.

    Returns
    -------
    numpy.ndarray, shape (n, 3)
        The transformed coordinates.
    """
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return coords @ rot.T + rng.uniform(-25.0, 25.0, size=3)


# --------------------------------------------------------------------------------------------------
# AMBER input generation
# --------------------------------------------------------------------------------------------------
def render_colvar(indices, reference, r1=-10.0, r2=0.0, r3=0.0, r4=1000.0, k=0.0):
    """Render one single-group ``MULTI_RMSD`` ``&colvar`` block.

    Single-group rather than the multi-group form on purpose: multi-group combines as
    ``sqrt(sum_g M_g/M R_g^2)``, whereas separate blocks give ``U = sum_g U_g``, which is additive
    the way the existing complex restraint file already is and maps 1:1 onto one cpptraj
    ``rms ... mass`` series.

    Parameters
    ----------
    indices : numpy.ndarray
        Zero-based atom indices; emitted one-based as ``cv_i``.
    reference : numpy.ndarray, shape (n, 3)
        Reference positions, emitted flattened as ``cv_r``.
    r1, r2, r3, r4 : float, optional
        Flat-bottom anchor positions. With ``r1 = -10`` and ``r2 = 0`` only the upper wall is live,
        so ``r3`` is a leash length in Angstrom and ``r3 = 0`` degenerates to a pure harmonic.
    k : float, optional
        Force constant for both walls, by default 0.0 (monitor only).

    Returns
    -------
    str
        The ``&colvar`` block, newline-terminated.
    """
    one_based = [int(i) + 1 for i in indices]
    flat = np.asarray(reference, dtype=float).reshape(-1)
    idx_lines = ",\n         ".join(
        ", ".join(str(v) for v in one_based[i : i + 10]) for i in range(0, len(one_based), 10)
    )
    ref_lines = ",\n         ".join(
        ", ".join(f"{v:.6f}" for v in flat[i : i + 6]) for i in range(0, len(flat), 6)
    )
    return (
        "&colvar\n"
        "  cv_type = 'MULTI_RMSD'\n"
        f"  cv_ni = {len(one_based)},\n"
        f"  cv_i = {idx_lines},\n"
        f"  cv_nr = {flat.size},\n"
        f"  cv_r = {ref_lines},\n"
        f"  anchor_position = {r1}, {r2}, {r3}, {r4}\n"
        f"  anchor_strength = {k}, {k}\n"
        "/\n"
    )


def write_inert_disang(path):
    """Write a parseable, zero-force ``&rst`` file.

    Lets nmropt=1 and NFE be exercised in the same run without perturbing the energy, which is the
    combination the complex leg will actually use (Boresch ``&rst`` plus the RMSD CV).

    Parameters
    ----------
    path : str
        Destination path for the DISANG file.
    """
    with open(path, "w") as handle:
        handle.write(
            "&rst iat = 1, 2,\n"
            "r1=0, r2 = 0.0, r3 = 1000.0, r4 = 1000.0\n"
            "rk2=0.0, rk3=0.0,\n"
            "/\n"
            "&end\n"
        )


def write_mdin(path, nstlim, igb=2, thermostat=False, ntwx=0, freq=1):
    """Write a self-contained implicit-solvent mdin with NFE enabled.

    ``&pmd`` is placed after ``&cntrl`` and BEFORE ``&wt``/DISANG -- the position the workflow's
    planned mdin injector will use, so this probe validates that choice rather than a different one.

    Parameters
    ----------
    path : str
        Destination path for the mdin.
    nstlim : int
        Number of MD steps.
    igb : int, optional
        Generalized Born model, by default 2.
    thermostat : bool, optional
        Use Langevin dynamics. False gives ``ntt=0``, which makes the step-0 energy and CV
        deterministic -- required by the best-fit probe.
    ntwx : int, optional
        Trajectory write interval; 0 writes none.
    freq : int, optional
        ``ntpr`` and ``&pmd`` ``output_freq``, by default 1.
    """
    temp = (
        "  ntt = 3, gamma_ln = 2.0, temp0 = 300.0, tempi = 300.0, ig = 20260830,\n"
        if thermostat
        else "  ntt = 0,\n"
    )
    with open(path, "w") as handle:
        handle.write(
            "NFE MULTI_RMSD probe\n"
            " &cntrl\n"
            "  imin = 0, ntx = 1, irest = 0,\n"
            f"  nstlim = {nstlim}, dt = 0.002,\n"
            "  ntc = 2, ntf = 2,\n"
            "  ntb = 0, cut = 999.0,\n"
            f"  igb = {igb}, saltcon = 0.0, rgbmax = 999.0, gbsa = 0,\n"
            f"{temp}"
            f"  ntpr = {freq}, ntwx = {ntwx}, ntwr = {max(nstlim, 1)},\n"
            "  nmropt = 1, infe = 1,\n"
            " /\n"
            " &pmd\n"
            "  output_file = 'pmd.txt'\n"
            f"  output_freq = {freq}\n"
            "  cv_file = 'cv.in'\n"
            " /\n"
            " &wt TYPE='END' /\n"
            "DISANG=restraint.RST\n"
        )


def stage_case(workdir, name, cv_text, nstlim, thermostat, ntwx, freq, igb):
    """Create a per-case subdirectory holding a complete AMBER input set.

    Parameters
    ----------
    workdir : str
        Parent scratch directory.
    name : str
        Case name; becomes the subdirectory name.
    cv_text : str
        Contents of ``cv.in``.
    nstlim, thermostat, ntwx, freq, igb
        Passed through to :func:`write_mdin`.

    Returns
    -------
    str
        Absolute path to the case directory.
    """
    case = os.path.abspath(os.path.join(workdir, name))
    os.makedirs(case, exist_ok=True)
    with open(os.path.join(case, "cv.in"), "w") as handle:
        handle.write(cv_text)
    write_inert_disang(os.path.join(case, "restraint.RST"))
    write_mdin(os.path.join(case, f"{name}.mdin"), nstlim, igb, thermostat, ntwx, freq)
    return case


def cpptraj_rmsd(prmtop, trajectory, reference, mask, workdir, tag, dry=False):
    """Mass-weighted best-fit RMSD of a trajectory to a reference structure.

    This is the value ``MULTI_RMSD`` must reproduce if it superimposes.

    Parameters
    ----------
    prmtop : str
        Topology path.
    trajectory : str
        Trajectory or coordinate file to measure.
    reference : str
        Reference coordinate file.
    mask : str
        AMBER mask defining the fitted and measured atoms.
    workdir : str
        Scratch directory for the cpptraj input and its ``<tag>.dat`` output.
    tag : str
        Basename for the generated files.
    dry : bool, optional
        Print the cpptraj script instead of running it.

    Returns
    -------
    numpy.ndarray
        Per-frame RMSD in Angstrom, or ``[nan]`` under ``dry``.
    """
    run_cpptraj(
        f"parm {os.path.abspath(prmtop)}\n"
        f"trajin {os.path.abspath(trajectory)}\n"
        f"reference {os.path.abspath(reference)} [r]\n"
        f"rms probe ref [r] {mask} mass out {tag}.dat\n"
        "run\n",
        workdir,
        tag,
        dry,
    )
    if dry:
        return np.asarray([float("nan")])
    return read_dat(os.path.join(workdir, f"{tag}.dat"))


def count_frames(prmtop, trajectory):
    """Count the frames in a trajectory without loading their coordinates.

    Parameters
    ----------
    prmtop : str
        Topology path.
    trajectory : str
        Trajectory to inspect.

    Returns
    -------
    int
        Frame count.
    """
    import pytraj as pt

    return int(pt.iterload(trajectory, prmtop).n_frames)


def extract_frames(prmtop, trajectory, workdir, n_frames=3, dry=False):
    """Write evenly spaced frames of a trajectory out as AMBER restarts.

    Sampling several points rather than one lets the best-fit probe check the CV across the RMSD
    range the ladder will actually traverse, instead of at a single value.

    Parameters
    ----------
    prmtop : str
        Topology path.
    trajectory : str
        Source trajectory.
    workdir : str
        Directory the cpptraj job runs in and writes the restarts to.
    n_frames : int, optional
        How many frames to extract, by default 3. The last frame is always included.
    dry : bool, optional
        Print the cpptraj script instead of running it.

    Returns
    -------
    list of str
        Absolute paths to the written restarts, in trajectory order.
    """
    if dry:
        return [os.path.abspath(os.path.join(workdir, "check.rst7"))]
    total = count_frames(prmtop, trajectory)
    picks = sorted({int(round(x)) for x in np.linspace(1, total, min(n_frames, total))})
    # One trajin per pick so cpptraj emits exactly those frames, in order, as separate restarts.
    script = [f"parm {os.path.abspath(prmtop)}"]
    script += [f"trajin {os.path.abspath(trajectory)} {f} {f}" for f in picks]
    script += ["trajout check.rst7 restart", "run"]
    run_cpptraj("\n".join(script) + "\n", workdir, "extract", dry)
    print(f"  extracted frames {picks} of {total}")
    # cpptraj suffixes multi-frame restart output with the frame ordinal.
    written = []
    for i in range(1, len(picks) + 1):
        candidate = os.path.join(workdir, f"check.rst7.{i}")
        if not os.path.exists(candidate) and len(picks) == 1:
            candidate = os.path.join(workdir, "check.rst7")
        if os.path.exists(candidate):
            written.append(os.path.abspath(candidate))
    if not written:
        raise SystemExit(f"cpptraj wrote no restarts into {workdir} (expected check.rst7[.N])")
    return written


def write_rescore_mdin(path, r3, k, igb=2, saltcon=0.0, nfe=True):
    """Write an ``imin=5`` single-point re-scoring mdin, optionally with NFE active.

    Mirrors what ``runner.only_post_analysis`` does for every cell of the MBAR matrix, so the ENERGY
    column this produces is the one MBAR would consume.

    Parameters
    ----------
    path : str
        Destination path for the mdin.
    r3 : float
        Flat-bottom upper edge in Angstrom, written into the accompanying ``cv.in`` by the caller.
        Recorded here only in the title, for traceability.
    k : float
        Force constant, likewise recorded only in the title.
    igb : int, optional
        Generalized Born model, by default 2.
    saltcon : float, optional
        Salt concentration, by default 0.0.
    nfe : bool, optional
        Emit ``infe=1`` and the ``&pmd`` block. False gives the unbiased reference scoring.
    """
    nfe_cntrl = "  nmropt = 1, infe = 1,\n" if nfe else "  nmropt = 1,\n"
    nfe_block = (
        " &pmd\n"
        "  output_file = 'pmd.txt'\n"
        "  output_freq = 1\n"
        "  cv_file = 'cv.in'\n"
        " /\n"
        if nfe
        else ""
    )
    with open(path, "w") as handle:
        handle.write(
            f"imin=5 rescore r3={r3} k={k} nfe={nfe}\n"
            " &cntrl\n"
            "  imin = 5, ntx = 1, irest = 0,\n"
            "  maxcyc = 1, ncyc = 0,\n"
            "  ntb = 0, cut = 999.0,\n"
            f"  igb = {igb}, saltcon = {saltcon}, rgbmax = 999.0, gbsa = 0,\n"
            "  ntpr = 1, ntwx = 0,\n"
            f"{nfe_cntrl}"
            " /\n"
            f"{nfe_block}"
            " &wt TYPE='END' /\n"
            "DISANG=restraint.RST\n"
        )


# --------------------------------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------------------------------
def probe_runs(args, indices, mask, reference, nstlim=None, write_traj=True):
    """Probe 1 -- does ``infe=1`` + ``&pmd`` + MULTI_RMSD start at all?

    Runs MD with the CV in monitor-only mode (``anchor_strength = 0``) and a live DISANG, so
    nmropt=1 and NFE are exercised together.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line.
    indices : numpy.ndarray
        Zero-based restrained atom indices.
    mask : str
        Mask the indices came from, for reporting.
    reference : numpy.ndarray, shape (n, 3)
        Reference positions for the CV.
    nstlim : int, optional
        Step count. Defaults to ``--short-nstlim``; callers that already have a trajectory pass 1,
        since the only question left is whether the run starts.
    write_traj : bool, optional
        Write ``probe1.nc``, which probe 2 harvests frames from when no ``--traj`` was supplied.

    Returns
    -------
    ok : bool
        Whether the run started and produced a CV series.
    case : str
        Path to the case directory.
    """
    nstlim = args.short_nstlim if nstlim is None else nstlim
    print(f"\n=== probe 1: does it run? ({len(indices)} atoms, mask {mask}, {nstlim} steps) ===")
    cv = render_colvar(indices, reference, r3=0.0, k=0.0)
    interval = max(nstlim // 10, 1)
    case = stage_case(
        args.workdir, "probe1", cv, nstlim, nstlim > 1,
        interval if write_traj else 0, interval, args.igb,
    )
    extra = ["-p", os.path.abspath(args.prmtop), "-c", os.path.abspath(args.inpcrd),
             "-r", "probe1.rst7"]
    if write_traj:
        extra += ["-x", "probe1.nc"]
    rc = run_amber(args.executable, case, "probe1", extra, args.dry_run)
    if args.dry_run:
        return True, case
    pmd = os.path.join(case, "pmd.txt")
    if rc != 0:
        print(f"NO-GO: {args.executable} exited {rc}. See {case}/probe1.mdout and probe1.stdout")
        return False, case
    if not os.path.exists(pmd):
        print(f"NO-GO: run succeeded but no pmd.txt in {case} -- &pmd was not honoured")
        return False, case
    values = read_pmd(pmd, args.pmd_column)
    if args.show_pmd:
        with open(pmd) as handle:
            print("--- pmd.txt head ---")
            print("".join(handle.readlines()[:8]))
    print(f"GO: exit 0, pmd.txt has {values.size} CV rows, first={values[0]:.4f} last={values[-1]:.4f}")
    return True, case


def probe_bestfit(args, indices, mask, reference, frames, label="probe 2"):
    """Probe 2/5 -- is the CV best-fit, and does it match cpptraj?

    For each supplied frame, evaluates the CV at step 0 twice: once against the reference as given,
    once against a rigidly rotated and translated copy. A best-fit RMSD is invariant to that; a
    lab-frame one is not. Both are cross-checked against cpptraj ``rms ... mass``.

    Several frames rather than one, when a trajectory is available, so the CV is checked across the
    RMSD range the ladder will traverse instead of at a single value -- a CV that is right near zero
    and wrong at 3 A would pass a single-point test.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line.
    indices : numpy.ndarray
        Zero-based restrained atom indices.
    mask : str
        Mask the indices came from; also used for the cpptraj cross-check.
    reference : numpy.ndarray, shape (n, 3)
        Reference positions.
    frames : list of str
        Coordinate files to evaluate from. Each must differ from the reference, or the test is
        degenerate at that point.
    label : str, optional
        Heading text, so probe 5 can reuse this with its own name.

    Returns
    -------
    bool
        True when the CV is invariant within ``--tol`` and matches cpptraj within ``--tol-cpptraj``
        at every frame.
    """
    print(f"\n=== {label}: best-fit gate ({len(indices)} atoms, mask {mask}, "
          f"{len(frames)} frame(s)) ===")
    if len(indices) > 200:
        print(f"note: cv_ni = {len(indices)} exceeds the igr1/igr2 cap of 200 that bit the COM "
              "restraint; a silent truncation would show up as a cpptraj mismatch below")

    rotated_reference = random_rigid_transform(reference)
    tag = label.replace(" ", "_").replace("(", "").replace(")", "")
    d_rot = d_cpp = 0.0
    covered = []

    for f_i, frame_path in enumerate(frames, start=1):
        cases = {}
        for name, ref in (("asgiven", reference), ("rotated", rotated_reference)):
            cv = render_colvar(indices, ref, r3=0.0, k=0.0)
            case_name = f"bestfit_{tag}_f{f_i}_{name}"
            case = stage_case(args.workdir, case_name, cv, 1, False, 0, 1, args.igb)
            rc = run_amber(
                args.executable, case, case_name,
                ["-p", os.path.abspath(args.prmtop), "-c", frame_path, "-r", f"{name}.rst7"],
                args.dry_run,
            )
            if args.dry_run:
                cases[name] = float("nan")
                continue
            if rc != 0:
                print(f"NO-GO: {case_name} exited {rc}. See {case}/{case_name}.mdout")
                return False
            cases[name] = float(read_pmd(os.path.join(case, "pmd.txt"), args.pmd_column)[0])

        series = cpptraj_rmsd(
            args.prmtop, frame_path, args.ref or args.inpcrd, mask, args.workdir,
            f"{tag}_f{f_i}_cpptraj", args.dry_run,
        )
        if args.dry_run:
            continue

        cpptraj_value = float(series[0])
        this_rot = abs(cases["asgiven"] - cases["rotated"])
        this_cpp = abs(cases["asgiven"] - cpptraj_value)
        d_rot = max(d_rot, this_rot)
        d_cpp = max(d_cpp, this_cpp)
        covered.append(cpptraj_value)
        print(f"  frame {f_i}: CV {cases['asgiven']:.6f} | rotated {cases['rotated']:.6f} "
              f"(d {this_rot:.2e}) | cpptraj {cpptraj_value:.6f} (d {this_cpp:.2e})")

    if args.dry_run:
        return True

    print(f"  RMSD range covered: {min(covered):.3f} - {max(covered):.3f} A")
    print(f"  worst rotation delta {d_rot:.2e} A, worst cpptraj delta {d_cpp:.2e} A")

    if d_rot > args.tol:
        print(f"NO-GO: CV moved by {d_rot:.4f} A under a rigid transform of the reference. "
              "MULTI_RMSD is LAB-FRAME -- it would fight the Boresch restraints and invalidate the "
              "standard-state correction. Stop and re-plan.")
        return False
    # Looser than --tol on purpose: NFE may report the CV after the single step rather than before
    # it, which cpptraj (measuring the input coordinates) cannot see.
    if d_cpp > args.tol_cpptraj:
        print(f"NO-GO: CV differs from cpptraj by {d_cpp:.4f} A. Either the CV is not the mass-"
              "weighted best-fit RMSD we assumed, or cv_r was silently truncated. If the gap is "
              "small and the invariance test passed, re-check with --tol-cpptraj before concluding.")
        return False
    print("GO: rotation-invariant and equal to cpptraj -- best-fit confirmed, and the analytic "
          "post-analysis route is exact.")
    return True


def probe_bias(args, indices, mask, reference):
    """Probe 3 -- does the restraint apply force, or only monitor?

    Two runs from the same start at ``r3 = 0``, one with ``k = 0`` and one with ``k = --k``. The
    biased trajectory's RMSD distribution must be compressed toward zero.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line.
    indices : numpy.ndarray
        Zero-based restrained atom indices.
    mask : str
        Mask for the cpptraj cross-check.
    reference : numpy.ndarray, shape (n, 3)
        Reference positions.

    Returns
    -------
    bool
        True when the biased run samples a smaller mean RMSD than the unbiased one.
    """
    print(f"\n=== probe 3: does the bias bite? (k = 0 vs {args.k}) ===")
    stats = {}
    for name, k in (("free", 0.0), ("biased", args.k)):
        cv = render_colvar(indices, reference, r3=args.r3, k=k)
        case = stage_case(
            args.workdir, f"bias_{name}", cv, args.nstlim, True,
            max(args.nstlim // 100, 1), max(args.nstlim // 100, 1), args.igb
        )
        rc = run_amber(
            args.executable, case, f"bias_{name}",
            ["-p", os.path.abspath(args.prmtop), "-c", os.path.abspath(args.inpcrd),
             "-r", f"{name}.rst7", "-x", f"{name}.nc"],
            args.dry_run,
        )
        if args.dry_run:
            continue
        if rc != 0:
            print(f"NO-GO: {name} run exited {rc}. See {case}/bias_{name}.mdout")
            return False
        values = cpptraj_rmsd(
            args.prmtop, os.path.join(case, f"{name}.nc"), args.ref or args.inpcrd,
            mask, args.workdir, f"bias_{name}_rmsd", args.dry_run,
        )
        stats[name] = describe(name, values)
    if args.dry_run:
        return True

    for row in stats.values():
        print(f"  {row['set']:<7} n={row['n']:<6} mean={row['mean']:.3f}  sd={row['sd']:.3f}  "
              f"max={row['max']:.3f} A")
    kt = 0.001987 * 300.0
    print(f"  anchor energy at the free mean: {0.5 * args.k * stats['free']['mean'] ** 2 / kt:.1f} kT")
    if stats["biased"]["mean"] < stats["free"]["mean"]:
        print("GO: the biased ensemble is compressed toward the reference -- forces are applied.")
        return True
    print("NO-GO: the biased run is not compressed. &pmd may be monitoring without applying force.")
    return False


def probe_rescore(args, indices, mask, reference, trajectory):
    """Probe 4 -- does ``imin=5`` evaluate NFE, and does the bias reach the ENERGY column?

    Informational: the plan scores the restraint band analytically, so a negative result here costs
    only the cross-check, not the design. A positive result gives a numerical oracle for the
    analytic energy function.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line.
    indices : numpy.ndarray
        Zero-based restrained atom indices.
    mask : str
        Unused here; kept for signature symmetry with the other probes.
    reference : numpy.ndarray, shape (n, 3)
        Reference positions.
    trajectory : str
        Trajectory to re-score.

    Returns
    -------
    bool
        True when the two tolerances give different ENERGY columns.
    """
    print(f"\n=== probe 4: does imin=5 see NFE? (sander = {args.sander}) ===")
    try:
        from implicit_solvent_ddm.mdout import min_to_dataframe
    except ImportError as exc:
        print(f"skipped: cannot import the package mdout parser ({exc})")
        return False

    energies = {}
    for name, r3 in (("tight", 0.0), ("loose", args.r3 if args.r3 > 0 else 5.0)):
        cv = render_colvar(indices, reference, r3=r3, k=args.k)
        case = os.path.abspath(os.path.join(args.workdir, f"rescore_{name}"))
        os.makedirs(case, exist_ok=True)
        with open(os.path.join(case, "cv.in"), "w") as handle:
            handle.write(cv)
        write_inert_disang(os.path.join(case, "restraint.RST"))
        write_rescore_mdin(os.path.join(case, f"rescore_{name}.mdin"), r3, args.k, args.igb)
        rc = run_amber(
            args.sander, case, f"rescore_{name}",
            ["-p", os.path.abspath(args.prmtop), "-c", os.path.abspath(args.inpcrd),
             "-y", os.path.abspath(trajectory), "-r", f"{name}.rst7"],
            args.dry_run,
        )
        if args.dry_run:
            continue
        if rc != 0:
            print(f"result: sander exited {rc} with infe=1 -- imin=5 does NOT accept NFE. "
                  "The analytic route is mandatory (it was already the plan).")
            return False
        frame = min_to_dataframe(os.path.join(case, f"rescore_{name}.mdout"))
        energies[name] = frame[["ENERGY", "RESTRAINT"]].to_numpy()
    if args.dry_run:
        return True

    d_energy = float(np.abs(energies["tight"][:, 0] - energies["loose"][:, 0]).max())
    d_restraint = float(np.abs(energies["tight"][:, 1] - energies["loose"][:, 1]).max())
    print(f"  max |dENERGY|   = {d_energy:.6f} kcal/mol")
    print(f"  max |dRESTRAINT| = {d_restraint:.6f} kcal/mol")
    if d_energy > 1e-6:
        print("result: the bias DOES reach ENERGY -- usable as a numerical oracle for the analytic "
              "energy function.")
        return True
    if d_restraint > 1e-6:
        print("result: the bias lands in RESTRAINT but NOT in ENERGY, which is the column MBAR "
              "reads (adaptive_restraints.py:583). Analytic route mandatory.")
        return False
    print("result: imin=5 ran but ignored the bias entirely. Analytic route mandatory.")
    return False


# --------------------------------------------------------------------------------------------------
def main():
    """Parse arguments, resolve inputs, and run the selected probes.

    Returns
    -------
    int
        0 when every selected probe passed, 1 otherwise -- so the script can gate a shell pipeline.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prmtop", required=True, help="topology (.parm7)")
    ap.add_argument("--inpcrd", required=True, help="starting coordinates (.rst7)")
    ap.add_argument("--ref", help="reference structure for the CV; defaults to --inpcrd")
    ap.add_argument("--mask", help="restrained atoms; default @CA, falling back to !@H*")
    ap.add_argument("--workdir", default="nfe_probe_work", help="scratch directory")
    ap.add_argument("--executable", default="pmemd.cuda", help="MD engine to probe")
    ap.add_argument("--sander", default="sander", help="binary for the imin=5 probe")
    ap.add_argument(
        "--probe", default="gate",
        help="gate (=1,2,5), all, or a comma-separated subset of 1,2,3,4,5",
    )
    ap.add_argument("--traj", help="existing trajectory to take check frames from and to re-score "
                                   "in probe 4; without it probe 1 generates a short one")
    ap.add_argument("--frames", type=int, default=3,
                    help="how many frames probe 2/5 check, spread through the trajectory")
    ap.add_argument("--nstlim", type=int, default=25000, help="steps for probe 3 (50 ps at 2 fs)")
    ap.add_argument("--short-nstlim", type=int, default=500,
                    help="steps for probe 1 when no --traj is given")
    ap.add_argument("--k", type=float, default=10.0, help="anchor_strength for probes 3 and 4")
    ap.add_argument("--r3", type=float, default=0.0, help="flat-bottom upper edge, Angstrom")
    ap.add_argument("--igb", type=int, default=2, help="GB model")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="rotation-invariance tolerance, Angstrom (should be near-exact)")
    ap.add_argument("--tol-cpptraj", type=float, default=1e-2,
                    help="CV-vs-cpptraj tolerance, Angstrom; looser because NFE may report the CV "
                         "after the single step rather than before it")
    ap.add_argument("--pmd-column", type=int, default=1, help="zero-based CV column in pmd.txt")
    ap.add_argument("--show-pmd", action="store_true", help="print the raw pmd.txt head")
    ap.add_argument("--dry-run", action="store_true", help="write inputs, run nothing")
    args = ap.parse_args()

    if args.probe == "gate":
        wanted = {"1", "2", "5"}
    elif args.probe == "all":
        wanted = {"1", "2", "3", "4", "5"}
    else:
        wanted = {p.strip() for p in args.probe.split(",")}

    os.makedirs(args.workdir, exist_ok=True)
    if not args.dry_run and shutil.which(args.executable) is None:
        raise SystemExit(f"{args.executable} not on PATH")

    import pytraj as pt

    complex_traj = pt.load(args.inpcrd, args.prmtop)
    indices, mask = select_atoms(complex_traj.top, args.mask)
    reference = pt.load(args.ref or args.inpcrd, args.prmtop).xyz[0][indices]
    print(f"prmtop {args.prmtop}\nmask   {mask} -> {len(indices)} atoms "
          f"(cv_ni={len(indices)}, cv_nr={3 * len(indices)})")

    results = {}
    case1 = None
    if wanted & {"1", "2", "5"}:
        # With a trajectory already in hand there is nothing to harvest, so probe 1 shrinks to the
        # one question it still answers: does the run start?
        ok, case1 = probe_runs(
            args, indices, mask, reference,
            nstlim=1 if args.traj else args.short_nstlim,
            write_traj=args.traj is None,
        )
        results["1"] = ok
        if not ok and not args.dry_run:
            print("\nstopping: nothing downstream is meaningful if the run does not start.")
            return 1

    frames = [os.path.abspath(args.inpcrd)]
    if wanted & {"2", "5"} and not args.dry_run:
        # Evaluating at the reference itself is degenerate, so take frames that have moved off it.
        source = args.traj or os.path.join(case1, "probe1.nc")
        frames = extract_frames(args.prmtop, source, args.workdir, args.frames, args.dry_run)
        print(f"\nchecking {len(frames)} frame(s) from {source}")
    if "2" in wanted:
        results["2"] = probe_bestfit(args, indices, mask, reference, frames)
    if "5" in wanted:
        results["5"] = probe_bestfit(args, indices, mask, reference, frames,
                                     label="probe 5 (scale)")
    if "3" in wanted:
        results["3"] = probe_bias(args, indices, mask, reference)
    if "4" in wanted:
        traj = args.traj or (os.path.join(case1, "probe1.nc") if case1 else args.inpcrd)
        results["4"] = probe_rescore(args, indices, mask, reference, traj)

    print("\n=== summary ===")
    for key in sorted(results):
        print(f"  probe {key}: {'GO' if results[key] else 'NO-GO'}")
    blocking = [k for k in ("1", "2", "5") if k in results and not results[k]]
    if blocking:
        print(f"\nBLOCKED on probe(s) {', '.join(blocking)} -- do not start the refactor.")
        return 1
    print("\nGate clear.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
