#!/usr/bin/env python
"""Extract the target-temperature trajectory from a finished REMD leg.

An AMBER T-REMD run writes one trajectory per replica (``remd.nc.001`` ... ``remd.nc.0NN``).
Each of those is a single *replica* walking up and down the ladder, so none of them is a
constant-temperature ensemble. cpptraj's ``remdtraj`` reads the whole set at once and pulls
out the frames that sat at one temperature, producing the single trajectory that the
free-energy analysis actually wants.

This is the standalone version of ``simulations.ExtractTrajectories``, for re-extracting
from run output on disk without driving Toil. It uses the same cpptraj input as the
workflow (``implicit_solvent_ddm/templates/cpptraj_remd.sh``).

The target temperature must match a ladder rung EXACTLY -- cpptraj matches the value, not
the nearest rung, and silently yields an empty trajectory if nothing matches. This script
reads the rungs out of the mdout files and refuses to run on a mismatch.

Usage
-----
    # every solute under a remd/ directory
    python scripts/extract_remd_temperature.py path/to/mcl1_rep1/mcl1_rep1/remd -T 298.0

    # one solute, explicit output
    python scripts/extract_remd_temperature.py path/to/remd/MCL-1_ligand-1 -T 298.0 -o out.nc

    # see the cpptraj input without running it
    python scripts/extract_remd_temperature.py path/to/remd -T 298.0 --dry-run

Needs ``cpptraj`` on PATH (``module load amber``).
"""
import argparse
import glob
import os
import re
import subprocess as sp
import sys

REPLICA_GLOB = "*.nc.[0-9][0-9][0-9]"


def replica_trajectories(solute_dir):
    """Replica trajectories in ladder order, or [] if this is not a finished REMD dir."""
    trajs = glob.glob(os.path.join(solute_dir, REPLICA_GLOB))
    # Sort on the numeric suffix, not the string: .nc.010 must follow .nc.009, and a
    # >99-replica ladder must not put .nc.100 before .nc.011.
    return sorted(trajs, key=lambda p: int(p.rsplit(".", 1)[1]))


def topology(solute_dir):
    parms = sorted(glob.glob(os.path.join(solute_dir, "*.parm7")))
    if not parms:
        raise SystemExit(f"no .parm7 in {solute_dir}")
    if len(parms) > 1:
        raise SystemExit(f"ambiguous topology in {solute_dir}: {[os.path.basename(p) for p in parms]}")
    return parms[0]


def exchanges(solute_dir):
    """False when the groupfile says ``-rem 0``.

    The equilibration leg runs every replica at a FIXED rung with no exchange, so its
    trajectories carry no temperature record and remdtraj has nothing to sort: the
    target-temperature ensemble is simply one replica. Only the REMD leg (``-rem 1``)
    needs the extraction.
    """
    groupfile = os.path.join(solute_dir, "group.groupfile")
    if not os.path.exists(groupfile):
        return True
    with open(groupfile, errors="replace") as handle:
        return "-rem 0" not in handle.read()


def replica_instant_temp(solute_dir, index):
    """Last instantaneous TEMP(K) for a replica, as a rough sanity check on the rung."""
    matches = sorted(glob.glob(os.path.join(solute_dir, f"*mdinfo.{index:03d}")))
    if not matches:
        return None
    with open(matches[0], errors="replace") as handle:
        found = re.findall(r"TEMP\(K\)\s*=\s*([0-9.]+)", handle.read())
    return float(found[-1]) if found else None


def ladder_temperatures(trajs):
    """The ladder rungs, read from the trajectories' per-frame ``temp0``.

    The mdout files are NOT available here: Calculation.export_files drops remd.mdout.* and
    rem.log unless the run was in debug mode, and mdinfo only holds the last instantaneous
    temperature of a replica mid-ladder. The trajectories themselves are the only exported
    record of the rungs. Best-effort: returns [] if neither reader is installed, in which
    case the empty-output check after cpptraj runs is what catches a bad temperature.
    """
    for reader in (_temps_via_scipy, _temps_via_pytraj):
        try:
            temps = reader(trajs)
        except Exception:
            continue
        if temps:
            return temps
    return []


def _temps_via_scipy(trajs):
    from scipy.io import netcdf_file

    temps = set()
    for traj in trajs:
        handle = netcdf_file(traj, "r", mmap=False)
        try:
            if "temp0" not in handle.variables:
                return []
            temps.update(round(float(t), 2) for t in handle.variables["temp0"][:])
        finally:
            handle.close()
    return sorted(temps)


def _temps_via_pytraj(trajs):
    import pytraj as pt

    temps = set()
    for traj in trajs:
        temps.update(round(float(t), 2) for t in pt.iterload(traj).temperatures)
    return sorted(temps)


def cpptraj_input(parm, trajs, target_temp, out_traj):
    """Same form as implicit_solvent_ddm/templates/cpptraj_remd.sh."""
    return (
        f"parm {parm}\n"
        f"trajin {trajs[0]} remdtraj remdtrajtemp {target_temp} "
        f"trajnames {','.join(trajs[1:])}\n"
        f"trajout {out_traj} nobox\n"
        "go\n"
    )


def write_endpoint_frames(parm, traj, prefix):
    """Write the first and last frame of ``traj`` as Amber restarts.

    Uses a cpptraj input file rather than `-y/-ya`: `-ya` receives its arguments as a
    single argv token, so a frame RANGE like "1 1" is not parsed as trajin start/stop --
    cpptraj silently reads every frame and, with a restart output format, writes one
    numbered restart PER FRAME (10,000 of them for a 10 k-frame trajectory).
    """
    written = []
    for label, selector in (("firstframe", "1 1"), ("lastframe", "lastframe")):
        out = f"{prefix}_{label}.rst7"
        script_path = f"{prefix}_{label}.cpptraj"
        with open(script_path, "w") as handle:
            handle.write(
                f"parm {parm}\ntrajin {traj} {selector}\ntrajout {out} restart\ngo\n"
            )
        result = sp.run(["cpptraj", "-i", script_path], capture_output=True, text=True)

        # A frame selector that failed to apply shows up as numbered siblings; remove them
        # rather than leaving thousands of files behind.
        strays = glob.glob(f"{out}.[0-9]*")
        if strays:
            for stray in strays:
                os.remove(stray)
            raise SystemExit(
                f"cpptraj wrote {len(strays)} numbered restarts instead of one for "
                f"{label} (selector {selector!r}); removed them. Its trajin syntax differs "
                "from what this script assumes."
            )
        if result.returncode != 0 or not os.path.exists(out):
            print(result.stdout[-2000:], file=sys.stderr)
            raise SystemExit(f"cpptraj could not write {out}")
        written.append(out)
    return written


def frame_count(parm, traj):
    """Frames in the extracted trajectory, or None if cpptraj cannot report it."""
    result = sp.run(
        ["cpptraj", "-p", parm, "-y", traj, "-tl"], capture_output=True, text=True
    )
    found = re.search(r"Frames:\s*(\d+)", result.stdout)
    return int(found.group(1)) if found else None


def extract(solute_dir, target_temp, out_traj=None, dry_run=False, force=False,
            endpoints=True, replica=1):
    trajs = replica_trajectories(solute_dir)
    if not trajs:
        return None

    name = os.path.basename(os.path.normpath(solute_dir))
    parm = topology(solute_dir)
    out_traj = out_traj or os.path.join(solute_dir, f"{name}_{target_temp}K.nc")

    if not exchanges(solute_dir):
        source = trajs[replica - 1]
        instant = replica_instant_temp(solute_dir, replica)
        print(f"\n=== {name}  ({len(trajs)} replicas, -rem 0: no exchange)")
        print(f"    replica {replica:03d} runs at a fixed rung and IS the {target_temp} K "
              "trajectory; nothing to extract.")
        if instant is not None:
            print(f"    {os.path.basename(source)}  (last instantaneous T = {instant:g} K)")
        if dry_run:
            return source
        if endpoints:
            prefix = os.path.join(solute_dir, f"{name}_{target_temp}K")
            for path in write_endpoint_frames(parm, source, prefix):
                print(f"    -> {path}")
        return source

    rungs = ladder_temperatures(trajs)
    if rungs and not any(abs(rung - target_temp) < 1e-6 for rung in rungs):
        message = (
            f"{name}: {target_temp} K is not a rung of this ladder "
            f"({', '.join(f'{rung:g}' for rung in rungs)}). cpptraj matches the value "
            f"exactly and would write an empty trajectory."
        )
        if not force:
            raise SystemExit(message + " Pass --force to run anyway.")
        print(f"  WARNING {message}", file=sys.stderr)

    script = cpptraj_input(parm, trajs, target_temp, out_traj)
    print(f"\n=== {name}  ({len(trajs)} replicas)")
    if rungs:
        print(f"    ladder: {', '.join(f'{rung:g}' for rung in rungs)} K")
    else:
        print("    ladder: unreadable (no scipy/pytraj); relying on the frame count below")
    if dry_run:
        print(script)
        return out_traj

    script_path = os.path.join(solute_dir, f"{name}_extract_{target_temp}K.cpptraj")
    with open(script_path, "w") as handle:
        handle.write(script)

    result = sp.run(["cpptraj", "-i", script_path], capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(out_traj):
        print(result.stdout[-3000:], file=sys.stderr)
        print(result.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"{name}: cpptraj failed (exit {result.returncode})")

    frames = frame_count(parm, out_traj)
    print(f"    -> {out_traj}" + (f"  ({frames} frames)" if frames is not None else ""))
    if frames == 0:
        raise SystemExit(
            f"{name}: extracted 0 frames. The ladder has no replica at {target_temp} K, "
            "or the replica trajectories carry no temperature information."
        )

    if endpoints:
        for path in write_endpoint_frames(parm, out_traj, os.path.splitext(out_traj)[0]):
            print(f"    -> {path}")
    return out_traj


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("remd_dir", help="a remd/ directory, or a single solute directory inside one")
    parser.add_argument("-T", "--target-temp", type=float, default=298.0,
                        help="target temperature in K; must match a ladder rung exactly (default: 298.0)")
    parser.add_argument("-o", "--output", default=None,
                        help="output trajectory (single-solute mode only; default <solute>_<T>K.nc beside the replicas)")
    parser.add_argument("--dry-run", action="store_true", help="print the cpptraj input and stop")
    parser.add_argument("--force", action="store_true", help="run even if the temperature is not a ladder rung")
    parser.add_argument("--no-endpoints", dest="endpoints", action="store_false",
                        help="skip writing the first/last frame restarts")
    parser.add_argument("--replica", type=int, default=1,
                        help="no-exchange (-rem 0) legs only: which replica is the target rung "
                             "(default: 1, the lowest, which this workflow pins to the target temperature)")
    args = parser.parse_args()

    root = os.path.abspath(args.remd_dir)
    if not os.path.isdir(root):
        raise SystemExit(f"not a directory: {root}")

    # A solute directory holds the replicas directly; a remd/ directory holds solute dirs.
    if replica_trajectories(root):
        solute_dirs = [root]
    else:
        solute_dirs = sorted(
            entry.path for entry in os.scandir(root)
            if entry.is_dir() and replica_trajectories(entry.path)
        )
        if args.output:
            raise SystemExit("-o applies to a single solute directory; drop it to process a whole remd/ tree")
    if not solute_dirs:
        raise SystemExit(f"no replica trajectories ({REPLICA_GLOB}) under {root}")

    written = [extract(d, args.target_temp, args.output, args.dry_run, args.force,
                       args.endpoints, args.replica) for d in solute_dirs]
    print(f"\n{len([w for w in written if w])} trajector"
          f"{'y' if len(written) == 1 else 'ies'} at {args.target_temp} K.")


if __name__ == "__main__":
    main()
