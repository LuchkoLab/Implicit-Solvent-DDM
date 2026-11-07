#!/usr/bin/env python
"""stride_extended_window.py -- glue an extended lambda_window run back into 10,000 frames.

WHY
---
The -14.0 receptor window ran 2 ns (nstlim=500000, dt=0.004) and wrote 10,000 frames at ntwx=50,
i.e. one every 0.2 ps. The receptor endstate it has to overlap with (``remd/*_298.0K.nc``) is 20 ns
per replica sampled once per 2 ps exchange -- also 10,000 frames. Same sample count, 10x less
simulated time on the window side. Extending the window to 5 ns and 10 ns tests whether that time
gap, not a basin mismatch, is what the endstate seam's overlap is measuring.

The frame count has to come back to 10,000. Post-analysis stores one rectangular frames x states
matrix per leg (``*_formatted.h5``, ``postTreatment.py:296``), so every state contributes the same
number of rows; and against the endstate an unequal sample count would change the BAR comparison as
well as the sampling. So the finished 2 ns is strided down and the extension is written at a coarser
ntwx, and the two are concatenated:

     5 ns   every 2nd old frame -> 5,000 @ 0.4 ps  +  3 ns at ntwx=150 -> 5,000 @ 0.6 ps
            The first 2 ns ends up sampled 1.5x more densely than the last 3 ns. A single uniform
            grid would need 0.5 ps spacing = every 2.5th old frame, which cpptraj cannot stride to.
            Both segments are equilibrium samples of the same ensemble, so the density step does not
            bias the ensemble average; it only makes the concatenated series non-uniform in time,
            which affects pymbar's statistical-inefficiency estimate and not the free energy.

    10 ns   every 5th old frame -> 2,000 @ 1.0 ps  +  8 ns at ntwx=250 -> 8,000 @ 1.0 ps
            = 10,000 frames on one uniform 1.0 ps grid.

Nothing here is hardcoded: the stride is derived from what ``extend.mdin`` actually says, so editing
the extension's nstlim/ntwx keeps the arithmetic correct.

WHAT THIS DOES
--------------
One cpptraj pass per target: strided original + whole extension -> ``<base>_prod_traj.nc`` in the
``5ns``/``10ns`` directory, then verifies the frame count. The output directory is left looking like
a window directory, so ``endstate_window_rmsd.py``, ``gb_junction_overlap.py`` and the scoring
machinery find the files where they expect them.

USAGE
-----
  module load amber                     # needs cpptraj

  W=<run>/MCL-1_receptor-...-IRN_IRN/lambda_window/1.0/78.5/-14.0

  # run the extensions first, from inside each directory (DISANG is ../restraint.RST):
  #   cd $W/5ns  && pmemd.cuda -O -i extend.mdin -p ../*.parm7 -c ../*_prod_restrt.rst7 \
  #                            -r ext_restrt.rst7 -x ext_traj.nc -o mdout
  # then:
  python scripts/stride_extended_window.py "$W" --target-ns 5 10

  python scripts/stride_extended_window.py "$W" --target-ns 5 --dry-run   # print the plan only

NOTES
-----
* The extension trajectory is whichever ``.nc`` is in the target directory that is not the merged
  output; pass ``--ext-traj`` if there is more than one.
* A short extension is caught here rather than downstream: if the merged trajectory does not come
  out at ``--total-frames`` the script fails and points at the extension's mdout.
"""
import argparse
import glob
import os
import re
import subprocess as sp
import sys


def parse_cntrl(path, keys=("dt", "nstlim", "ntwx")):
    """Read ``&cntrl`` values out of an mdin.

    Parameters
    ----------
    path : str
        Path to an AMBER mdin file.
    keys : tuple[str]
        Namelist names to pull out.

    Returns
    -------
    dict
        Name -> float, for the keys present. Trailing ``!`` comments are stripped first so a
        commented-out value is never picked up.
    """
    with open(path) as handle:
        text = "\n".join(line.split("!")[0] for line in handle)
    found = {}
    for key in keys:
        match = re.search(rf"\b{key}\s*=\s*([-+0-9.eE]+)", text)
        if match:
            found[key] = float(match.group(1))
    return found


def only(pattern, what, exclude=()):
    """Resolve a glob that must match exactly one file.

    Parameters
    ----------
    pattern : str
        Glob pattern.
    what : str
        Human name of the file, used in the error message.
    exclude : tuple[str]
        Paths to drop from the matches.

    Returns
    -------
    str
        The single matching path.

    Raises
    ------
    SystemExit
        If the glob does not match exactly one file.
    """
    hits = sorted(p for p in glob.glob(pattern) if p not in exclude)
    if len(hits) != 1:
        raise SystemExit(
            f"expected exactly one {what} ({pattern}), found {len(hits)}"
            + (": " + ", ".join(os.path.basename(h) for h in hits) if hits else "")
        )
    return hits[0]


def window_mdin_values(window_dir):
    """Read dt and ntwx off the mdin the finished window was run with.

    The workflow leaves one ``tmp*.tmp`` per AMBER invocation, so there is usually more than one
    copy; they only conflict if the window was re-run with different settings.

    Parameters
    ----------
    window_dir : str
        The finished ``lambda_window/.../<exponent>`` directory.

    Returns
    -------
    dict
        ``dt``, ``nstlim`` and ``ntwx``.

    Raises
    ------
    SystemExit
        If no MD mdin is found, or the copies disagree.
    """
    seen = {}
    for path in sorted(glob.glob(os.path.join(window_dir, "tmp*.tmp"))
                       + glob.glob(os.path.join(window_dir, "*.mdin"))):
        values = parse_cntrl(path, ("dt", "nstlim", "ntwx", "imin"))
        if values.get("imin") == 0 and {"dt", "nstlim", "ntwx"} <= values.keys():
            seen.setdefault((values["dt"], values["nstlim"], values["ntwx"]), path)
    if not seen:
        raise SystemExit(f"no MD mdin (imin=0 with dt, nstlim and ntwx) in {window_dir}")
    if len(seen) > 1:
        listed = ", ".join(os.path.basename(p) for p in seen.values())
        raise SystemExit(f"{window_dir} holds mdins with conflicting settings ({listed})")
    dt, nstlim, ntwx = next(iter(seen))
    return {"dt": dt, "nstlim": nstlim, "ntwx": ntwx}


def frame_count(parm, traj):
    """Frames in a trajectory according to cpptraj.

    Parameters
    ----------
    parm : str
        Topology path.
    traj : str
        Trajectory path.

    Returns
    -------
    int

    Raises
    ------
    SystemExit
        If cpptraj is missing or does not report a frame count.
    """
    try:
        result = sp.run(["cpptraj", "-p", parm, "-y", traj, "-tl"],
                        capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("cpptraj not on PATH (module load amber)")
    found = re.search(r"Frames:\s*(\d+)", result.stdout)
    if not found:
        sys.stderr.write(result.stdout[-2000:])
        raise SystemExit(f"cpptraj could not read {traj}")
    return int(found.group(1))


def plan(old_frames, ext_frames, total_frames):
    """Work out how many old frames to keep and at what stride.

    Parameters
    ----------
    old_frames : int
        Frames in the finished trajectory.
    ext_frames : int
        Frames the extension writes, from its own mdin.
    total_frames : int
        Frames the merged trajectory must have.

    Returns
    -------
    tuple[int, int]
        ``(stride, kept)``.

    Raises
    ------
    SystemExit
        If the extension already meets or exceeds the budget, or if the frames it leaves for the
        original are not reachable by an integer stride.
    """
    kept = total_frames - ext_frames
    if kept <= 0:
        raise SystemExit(
            f"the extension alone writes {ext_frames} frames, leaving no room under a "
            f"{total_frames}-frame budget; raise ntwx in extend.mdin"
        )
    if old_frames % kept:
        raise SystemExit(
            f"cannot take {kept} frames out of {old_frames} by an integer stride "
            f"({old_frames}/{kept} = {old_frames / kept:.3f}). Pick an extension ntwx whose frame "
            f"count leaves a divisor of {old_frames}."
        )
    return old_frames // kept, kept


def merge(window_dir, target_ns, total_frames, ext_traj, dry_run, force):
    """Concatenate the strided original and the extension for one target.

    Parameters
    ----------
    window_dir : str
        The finished ``lambda_window/.../<exponent>`` directory.
    target_ns : float
        Which extension directory to build, in ns (``<window_dir>/<N>ns``).
    total_frames : int
        Frames the merged trajectory must have.
    ext_traj : str or None
        Override for the extension trajectory.
    dry_run : bool
        Print the plan and the cpptraj input, run nothing.
    force : bool
        Rebuild the merged trajectory if it already exists.

    Returns
    -------
    str or None
        Path to the merged trajectory, or None under ``--dry-run``.
    """
    outdir = os.path.join(window_dir, f"{target_ns:g}ns")
    if not os.path.isdir(outdir):
        raise SystemExit(f"no such directory: {outdir}")

    parm = only(os.path.join(window_dir, "*.parm7"), "topology")
    old_traj = only(os.path.join(window_dir, "*_prod_traj.nc"), "production trajectory")
    base = os.path.basename(old_traj)[: -len("_prod_traj.nc")]
    merged = os.path.join(outdir, f"{base}_prod_traj.nc")

    ext_mdin = os.path.join(outdir, "extend.mdin")
    if not os.path.exists(ext_mdin):
        ext_mdin = only(os.path.join(outdir, "*.mdin"), "extension mdin")
    values = parse_cntrl(ext_mdin)
    ext_frames = int(values["nstlim"]) // int(values["ntwx"])
    ext_spacing = values["ntwx"] * values["dt"]

    ext_traj = ext_traj or only(os.path.join(outdir, "*.nc"), "extension trajectory",
                                exclude=(merged,))

    old_frames = frame_count(parm, old_traj)
    old_values = window_mdin_values(window_dir)
    old_spacing = old_values["ntwx"] * old_values["dt"]
    stride, kept = plan(old_frames, ext_frames, total_frames)

    print(f"\n=== {target_ns:g} ns -> {outdir}")
    print(f"    existing   {old_frames * old_spacing:8.1f} ps  {old_frames:6d} frames @ "
          f"{old_spacing:.3f} ps")
    print(f"    keep every {stride} -> {' ' * 14}{kept:6d} frames @ {stride * old_spacing:.3f} ps")
    print(f"    extension  {ext_frames * ext_spacing:8.1f} ps  {ext_frames:6d} frames @ "
          f"{ext_spacing:.3f} ps   ({os.path.basename(ext_traj)})")
    print(f"    merged     {target_ns * 1000:8.1f} ps  {total_frames:6d} frames"
          + ("  (uniform grid)" if abs(ext_spacing - stride * old_spacing) < 1e-9
             else f"  (density step {ext_spacing / (stride * old_spacing):.2f}x at the seam)"))

    script = (
        f"parm {parm}\n"
        # stride, 2*stride, ... old_frames -- the last kept frame is the last old frame, no seam gap
        f"trajin {old_traj} {stride} {old_frames} {stride}\n"
        f"trajin {ext_traj}\n"
        f"trajout {merged} netcdf\n"
        "go\n"
    )
    if dry_run:
        print("\n" + script)
        return None
    if os.path.exists(merged) and not force:
        raise SystemExit(f"{merged} already exists; pass --force to rebuild it")

    script_path = os.path.join(outdir, "merge.cpptraj")
    with open(script_path, "w") as handle:
        handle.write(script)
    result = sp.run(["cpptraj", "-i", script_path], capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(merged):
        sys.stderr.write(result.stdout[-2000:])
        raise SystemExit(f"cpptraj could not write {merged}")

    frames = frame_count(parm, merged)
    if frames != total_frames:
        raise SystemExit(
            f"{merged} has {frames} frames, expected {total_frames}. The extension probably did "
            f"not finish -- check {os.path.join(outdir, 'mdout')}."
        )
    print(f"    -> {merged}  ({frames} frames)")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("window_dir",
                        help="the finished lambda_window/<...>/<exponent> directory")
    parser.add_argument("--target-ns", type=float, nargs="+", required=True,
                        help="which extension directories to build, in ns (e.g. --target-ns 5 10)")
    parser.add_argument("--total-frames", type=int, default=10000,
                        help="frames the merged trajectory must have (default: 10000, what the rest "
                             "of the cycle and the endstate carry)")
    parser.add_argument("--ext-traj", default=None,
                        help="extension trajectory; only valid with a single --target-ns "
                             "(default: the one .nc in <N>ns/ that is not the merged output)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the cpptraj input, run nothing")
    parser.add_argument("--force", action="store_true",
                        help="rebuild the merged trajectory if it already exists")
    args = parser.parse_args()

    window_dir = os.path.abspath(args.window_dir)
    if not os.path.isdir(window_dir):
        raise SystemExit(f"not a directory: {window_dir}")
    if args.ext_traj and len(args.target_ns) > 1:
        raise SystemExit("--ext-traj applies to a single --target-ns")

    for target_ns in args.target_ns:
        merge(window_dir, target_ns, args.total_frames, args.ext_traj, args.dry_run, args.force)


if __name__ == "__main__":
    main()
