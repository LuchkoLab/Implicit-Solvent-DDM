#!/usr/bin/env python
"""endstate_seam_overlap.py -- 2-state MBAR across the endstate <-> lambda_window seam.

WHY
---
``gb_junction_overlap.py`` answers the same question for the GB-band <-> gas junction, but it
writes ``nmropt=0`` on the stated grounds that both states there carry a byte-identical
``restraint.RST`` whose energy cancels in every difference. At the endstate seam that cancellation
is exactly what does NOT happen: the endstate and the ``lambda_window`` each have their own
``restraint.RST``, and their difference is the whole Hamiltonian difference. Scoring with
``nmropt=0`` drops it and returns overlap ~ 1 -- a pass that means nothing.

So both sides here run ``nmropt=1``, each against its own DISANG, which is what production does:
the two post-processing mdouts straddling this seam on MCL-1/ligand-1 rep2 agree on ``igb=2``,
``saltcon=0.3``, ``extdiel=78.5``, ``nmropt=1`` and differ only in ``RESTRAINT`` (28.56 vs 0.0000
on the same frame). A 0.0000 there is not proof the endstate file is zero-force -- a flat-bottom
well wide enough to contain the endstate ensemble also reads 0.0000 on those frames while being
non-zero on the window's. ``--endstate-restraint`` is therefore required, not inferred, and
:func:`check_restraints` reports what each file actually contains.

WHAT THIS DOES
--------------
Four ``sander imin=5`` jobs -- {endstate, window} trajectories x {endstate, window} Hamiltonians --
then a 2-state MBAR giving dG, its error, and the min-direction overlap that ``min_degree_overlap``
(0.04) is tested against. Each job is cached on a (trajectory, Hamiltonian) fingerprint, so a
re-run after a failure resumes.

There are exactly four jobs, so ``--nproc 4 --ranks 10`` fills a 40-core allocation with one
10-rank ``sander.MPI`` step per job and no idle cores.

USAGE
-----
  #SBATCH --nodes=1 --ntasks=40 --cpus-per-task=1

  module load amber
  export SANDER_MPI=$AMBERHOME/bin/sander.MPI
  export CPPTRAJ=$AMBERHOME/bin/cpptraj

  R=<run>/mcl1_remd_2ns_rep2
  S=MCL-1_receptor-ligand-1_with_mcl1_uxU_uxU-ligand-1_with_mcl1_uxU_IRN_IRN

  W=$R/$S/lambda_window/1.0/78.5/-14.0
  E=$R/$S/<endstate state dir>          # the endstate's OWN restraint.RST lives here

  # the 2 ns baseline -- 4 jobs x 10 ranks, every frame
  python scripts/endstate_seam_overlap.py \
      --window-dir  $W --endstate-restraint $E/restraint.RST \
      --endstate-traj $R/remd/$S/${S}_298.0K.nc \
      --nproc 4 --ranks 10 --stride 1 --workdir ./seam_2ns

  # the extended window: same seam, --window-traj points at the merged 10 ns
  python scripts/endstate_seam_overlap.py \
      --window-dir  $W --endstate-restraint $E/restraint.RST \
      --window-traj $W/10ns/*_prod_traj.nc \
      --endstate-traj $R/remd/$S/${S}_298.0K.nc \
      --nproc 4 --ranks 10 --stride 1 --workdir ./seam_10ns

  # print the mdins and launcher commands without running them
  python scripts/endstate_seam_overlap.py --window-dir ... --endstate-traj ... --dry-run

NOTES
-----
* ``--ntasks=40`` matters. ``--ntasks=1 --cpus-per-task=40`` grants ONE launcher slot, and every
  ``srun``/``mpirun`` step then refuses to place 10 ranks -- the same fault that killed the REMD
  ladder (an ``-n`` taken from the core reservation rather than the rank count).
* ``--launcher mpirun`` (default) does not touch Slurm's PMI. ``--launcher srun`` needs the
  allocation to hold ranks x jobs TASKS; ``srun-pmix`` and raw ``{ranks}`` templates are there for
  builds where PMI2 is not what Slurm speaks.
* ``--ranks 1`` (default) is 4 concurrent single-core jobs -- parallel, but only 4 cores. Getting
  all 40 busy needs ``--ranks 10`` and therefore a working MPI launcher.
* GB here is O(N^2) over 2,444 atoms with ``cut=999`` and no pairlist, so 10 ranks scale well but
  not linearly -- expect ~6-7x, not 10x.
* ``--stride 100`` (default) keeps every 100th frame: ~100 frames per job, a couple of minutes
  serial. With 40 cores use ``--stride 1`` and score all 10,000 frames -- that is the production
  frame set, and it is the only way the error bar means anything.
* GB settings are read from the window's own production ``mdout``, not assumed -- ``rgbmax`` in
  particular, which production omits so AMBER defaults it to 25 A. Hardcoding 999 changes the Born
  radii and silently invalidates the comparison.
* The restraints are distance-based (``iat = i, j``), so no ``-ref`` structure is needed; the
  reference geometry lives in each ``&rst`` record's ``r2``/``r3``.
* Before any sander runs, :func:`check_restraints` verifies the window's ``rk2``/``rk3`` equals
  ``2 ** exponent`` for the exponent in its own directory name, that both files restrain the same
  atom-pair network, and reports the endstate file's force constants. A stale or swapped DISANG
  changes du and therefore the overlap with nothing else going wrong.
* Reports the per-ensemble RESTRAINT distribution alongside the overlap. Disjoint supports (window
  max below endstate min) mean the seam is a structural mismatch, not a sampling deficit, and no
  run length closes it -- see ``endstate_window_rmsd.py``.
"""
import argparse
import os
import re
import subprocess
import sys
import time
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Everything that is not specific to the restraint comes from the GB-junction script: the same
# mdout-derived GB settings, the same cpptraj striding and -c extraction, the same worker pool.
from gb_junction_overlap import (  # noqa: E402
    SANDER, first_frame, kcals_per_kt, one, read_gb_settings, stride, _run_pool,
)

SANDER_MPI = os.environ.get("SANDER_MPI", "sander.MPI")

# One launcher step per sander job.
#
# `srun` needs the allocation to actually hold ranks x jobs TASKS (--ntasks=40, not
# --ntasks=1 --cpus-per-task=40); with one task it dies in PMI2 with "invalid kvs seq ...
# failed to send temp kvs to srun". It also needs the OpenMPI build to speak Slurm's PMI --
# check `srun --mpi=list` and use the raw-template form below if pmix is what is available.
#
# `mpirun` is the safer default under an interactive salloc: OpenMPI launches its own daemons and
# does not depend on Slurm's PMI at all. --oversubscribe is required because a --ntasks=1
# allocation advertises one slot, and --bind-to none stops all four concurrent jobs pinning
# themselves to core 0 upward.
#
# --launcher also accepts a raw template containing {ranks}, for anything not covered here.
LAUNCHERS = {
    "mpirun": "mpirun --oversubscribe --bind-to none -np {ranks}",
    "srun": "srun --exclusive --ntasks={ranks} --cpus-per-task=1",
    "srun-pmix": "srun --mpi=pmix --exclusive --ntasks={ranks} --cpus-per-task=1",
    "none": "",
}


def gb_settings(mdout):
    """GB settings for the seam: the shared reader plus the two keys it does not carry.

    ``gb_junction_overlap.read_gb_settings`` omits ``igb`` and ``extdiel`` because that script
    sets them per state (GB window vs gas). Across this seam both states share whatever the
    window's own MD ran with, so they are read from the same mdout as everything else.

    Parameters
    ----------
    mdout : str
        Path to the window's production MD mdout.

    Returns
    -------
    dict
        The shared GB settings plus ``igb`` (int) and ``extdiel`` (float).
    """
    params = read_gb_settings(mdout)
    head = open(mdout).read().split("3.  ATOMIC")[0]
    for key, default, cast in (("igb", 2, int), ("extdiel", 78.5, float)):
        m = re.search(rf"\b{key}\s*=\s*([0-9.]+)", head)
        params[key] = cast(float(m.group(1))) if m else cast(default)
    return params


RST_RE = re.compile(r"iat\s*=\s*(\d+)\s*,\s*(\d+)")
RK_RE = re.compile(r"rk2\s*=\s*([0-9.eE+-]+)\s*,\s*rk3\s*=\s*([0-9.eE+-]+)")


def describe_restraints(path):
    """Summarise a ``restraint.RST``: how many records, which force constants, which atom pairs.

    Parameters
    ----------
    path : str
        Path to an AMBER NMR restraint (DISANG) file.

    Returns
    -------
    dict
        ``n`` record count, ``rk`` the set of distinct ``(rk2, rk3)`` pairs, and ``pairs`` the
        set of restrained atom-index pairs.
    """
    txt = open(path).read()
    return {
        "n": txt.count("&rst"),
        "rk": {(float(a), float(b)) for a, b in RK_RE.findall(txt)},
        "pairs": {tuple(sorted((int(i), int(j)))) for i, j in RST_RE.findall(txt)},
    }


def check_restraints(window_rst, endstate_rst, window_dir):
    """Verify the two DISANG files are the ones this seam is supposed to compare.

    The measured quantity is a difference of restraint energies, so a stale or mismatched file
    changes the answer without any error surfacing. Three things are checked: the window's force
    constant equals ``2 ** exponent`` for the exponent in its own directory name, the two files
    restrain the same atom-pair network, and the endstate's force constants are reported rather
    than assumed to be zero.

    Parameters
    ----------
    window_rst, endstate_rst : str
        Paths to the two restraint files.
    window_dir : str
        The ``lambda_window`` state directory, whose trailing component is the exponent.

    Returns
    -------
    list of str
        Human-readable findings; any entry starting with ``WARNING`` means stop and look.
    """
    win, end = describe_restraints(window_rst), describe_restraints(endstate_rst)
    out = [f"window   {win['n']:>7,} records  rk2/rk3={sorted(win['rk'])}",
           f"endstate {end['n']:>7,} records  rk2/rk3={sorted(end['rk'])}"]

    try:
        exponent = float(os.path.basename(os.path.normpath(window_dir)))
    except ValueError:
        out.append(f"WARNING: cannot read an exponent from {window_dir!r}; force constant unchecked")
        exponent = None
    if exponent is not None:
        expect = 2.0 ** exponent
        got = {k for pair in win["rk"] for k in pair}
        if len(got) != 1 or abs(next(iter(got)) - expect) > 1e-12 * max(expect, 1.0):
            out.append(f"WARNING: window rk should be 2**{exponent} = {expect:.6e}, file has {got}")
        else:
            out.append(f"window force constant matches 2**{exponent} = {expect:.6e}")

    if end["n"] == 0:
        # An empty DISANG is the endstate being genuinely unrestrained, which is the expected
        # shape of this seam -- not a mismatch. du is then the window's restraint term alone.
        out.append("endstate DISANG has no &rst records: the endstate is unrestrained, so du is "
                   "the window's restraint term alone. Expected for this seam.")
    elif win["pairs"] != end["pairs"]:
        only_w, only_e = len(win["pairs"] - end["pairs"]), len(end["pairs"] - win["pairs"])
        out.append(f"WARNING: restraint networks differ -- {only_w} pairs only in the window file, "
                   f"{only_e} only in the endstate file. du is then not a pure force-constant "
                   f"difference.")
    else:
        out.append(f"both files restrain the same {len(win['pairs']):,} atom pairs")

    if end["rk"] and end["rk"] != {(0.0, 0.0)}:
        out.append("NOTE: the endstate file is NOT zero-force; its restraint term is carried "
                   "explicitly (nmropt=1 on both sides), not dropped.")
    return out


def link_into(path, workdir, name):
    """Symlink ``path`` into ``workdir`` under a short ``name``, and return the new path.

    sander truncates any filename past 256 characters, and these run trees are deep enough to
    exceed that on the topology and DISANG alone. Every sander argument is therefore a basename
    resolved from ``workdir``.

    Parameters
    ----------
    path : str
        The real file, anywhere on disk.
    workdir : str
        Directory the sander jobs run in.
    name : str
        Short basename to expose it under.

    Returns
    -------
    str
        Absolute path to the link inside ``workdir``.
    """
    dest = os.path.join(workdir, name)
    if os.path.islink(dest) or os.path.exists(dest):
        os.remove(dest)
    os.symlink(os.path.abspath(path), dest)
    return dest


def write_mdin(path, params, disang=None):
    """Write a single-point ``imin=5`` mdin, with the conformational restraint on or off.

    Parameters
    ----------
    path : str
        Where to write the mdin.
    params : dict
        GB settings from :func:`read_gb_settings`.
    disang : str, optional
        Absolute path to a ``restraint.RST``. Both seam states normally have one -- the endstate
        its own and the window its own -- so this is ``None`` only if a state genuinely ran
        unrestrained.

    Returns
    -------
    str
        ``path``, for chaining.
    """
    nmropt = 0 if disang is None else 1
    with open(path, "w") as fh:
        fh.write(
            "single-point re-score across the endstate seam (imin=5)\n&cntrl\n"
            f"  imin=5, ntx=1, irest=0, ntb=0, nmropt={nmropt},\n"
            f"  igb={params['igb']}, intdiel={params['intdiel']}, extdiel={params['extdiel']},\n"
            f"  saltcon={params['saltcon']}, gbsa={params['gbsa']},\n"
            f"  cut={params['cut']}, rgbmax={params['rgbmax']},\n"
            "  ntpr=1, ntwr=0, ntwx=0,\n/\n"
        )
        if disang is not None:
            fh.write(f"&wt type='END'\n/\nDISANG={disang}\n")
    return path


def score(traj, parm, coord, params, disang, workdir, tag, dry=False, launcher="", sander_bin=None):
    """Re-score ``traj`` under one Hamiltonian; return per-frame total and restraint energies.

    Parameters
    ----------
    traj : str
        Trajectory to score.
    parm : str
        Topology matching ``traj``.
    coord : str
        Single-frame restart for sander's ``-c``.
    params : dict
        GB settings from :func:`read_gb_settings`.
    disang : str or None
        Restraint file, or ``None`` for the unrestrained endstate Hamiltonian.
    workdir : str
        Directory for the mdin/mdout/cache files.
    tag : str
        Unique name for this (trajectory, Hamiltonian) pair.
    dry : bool, optional
        Print the sander command and mdin instead of running.
    launcher : str, optional
        MPI launcher prefix, already formatted (e.g. ``"srun --exclusive --ntasks=10"``). Empty
        for a serial run.
    sander_bin : str, optional
        sander executable; defaults to ``SANDER`` serial or ``SANDER_MPI`` when ``launcher`` is set.

    Returns
    -------
    energy : numpy.ndarray
        Per-frame total potential energy in kcal/mol.
    restraint : numpy.ndarray
        Per-frame restraint energy in kcal/mol (zeros when ``disang`` is ``None``).
    cached : bool
        True when the result was reused from a previous run.
    """
    from implicit_solvent_ddm.mdout import min_to_dataframe

    mdout = os.path.join(workdir, f"{tag}.mdout")
    done = os.path.join(workdir, f"{tag}.done")
    key = f"{os.path.abspath(traj)}|disang={disang}|" + str(sorted(params.items()))
    if os.path.exists(done) and os.path.exists(mdout) and open(done).read().strip() == key:
        df = min_to_dataframe(mdout)
        return (df["ENERGY"].to_numpy(float),
                df.get("RESTRAINT", 0.0 * df["ENERGY"]).to_numpy(float), True)

    # sander truncates filenames at 256 characters (mdread1.F90) and then dies with a bare
    # "Fortran runtime error: End of file". These run trees are deep enough to blow that on their
    # own, so every argument is passed as a basename relative to workdir; parm and DISANG are
    # reached through short symlinks that main() drops there.
    write_mdin(os.path.join(workdir, f"{tag}.mdin"), params,
               os.path.basename(disang) if disang else None)
    exe = sander_bin or (SANDER_MPI if launcher else SANDER)
    cmd = launcher.split() + [exe, "-O", "-i", f"{tag}.mdin", "-p", os.path.basename(parm),
                              "-c", os.path.basename(coord), "-y", os.path.basename(traj),
                              "-o", f"{tag}.mdout", "-r", f"{tag}.restrt", "-x", f"{tag}.mdcrd"]
    if dry:
        print(f"  [dry] cd {workdir} && {' '.join(cmd)}")
        print("        " + open(os.path.join(workdir, f"{tag}.mdin"))
              .read().replace("\n", "\n        ").rstrip())
        return np.zeros(1), np.zeros(1), False

    r = subprocess.run(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not os.path.exists(mdout):
        tail = "".join(open(mdout).readlines()[-25:]) if os.path.exists(mdout) else ""
        raise SystemExit(f"ERROR: sander failed\n  {' '.join(cmd)}\n"
                         f"{r.stdout.decode(errors='replace')[-1500:]}\n--- mdout tail ---\n{tail}")
    df = min_to_dataframe(mdout)
    if len(df) == 0:
        raise SystemExit(f"ERROR: no ENERGY frames parsed from {mdout}")
    open(done, "w").write(key)
    return (df["ENERGY"].to_numpy(float),
            df.get("RESTRAINT", 0.0 * df["ENERGY"]).to_numpy(float), False)


def _score_worker(job):
    """Run one re-score; module-level so ProcessPoolExecutor can pickle it."""
    tag, traj, parm, coord, params, disang, workdir, dry, launcher, sander_bin = job
    t0 = time.perf_counter()
    try:
        e, rst, cached = score(traj, parm, coord, params, disang, workdir, tag, dry,
                               launcher, sander_bin)
        return tag, e, rst, cached, None, time.perf_counter() - t0
    except BaseException:
        return tag, None, None, False, traceback.format_exc(), time.perf_counter() - t0


def seam_mbar(du_end, du_win, kT, subsample=True):
    """Solve the 2-state seam through ``pandasmbar``, exactly as production MBAR does.

    ``pandasmbar`` wants one row per frame, the index naming the state it was sampled at and the
    columns the states it was evaluated at, in reduced units. Only differences enter MBAR, so
    state ``endstate`` is taken as zero and ``window`` as ``du``: passing ``du`` rather than the
    two absolute energies avoids the mdout ENERGY header's ~5 significant figures leaving
    ~0.1 kcal/mol of quantisation noise on a du of ~3.

    Parameters
    ----------
    du_end, du_win : numpy.ndarray
        The window-minus-endstate potential difference (kcal/mol), on the endstate and window
        ensembles respectively.
    kT : float
        kT in kcal/mol.
    subsample : bool, optional
        Run ``detect_equilibration`` + ``subsample_correlated_data`` first, as production does.

    Returns
    -------
    dG : float
        Free energy of window minus endstate, kcal/mol.
    err : float
        Its statistical uncertainty, kcal/mol.
    overlap : float
        ``min(O[0,1], O[1,0])`` -- the quantity ``min_degree_overlap`` is tested against.
    counts : dict
        Frames surviving per state.
    """
    import pandas as pd
    from implicit_solvent_ddm import pandasmbar as pdmbar

    labels = ["endstate"] * len(du_end) + ["window"] * len(du_win)
    du = np.concatenate([du_end, du_win]) / kT
    df = pd.DataFrame({"endstate": np.zeros_like(du), "window": du},
                      index=pd.Index(labels, name="state"))

    if subsample:
        df = pdmbar.subsample_correlated_data(df, pdmbar.detect_equilibration(df, nskip=1))
    free_energies, errors, mb = pdmbar.mbar(df)

    O = np.asarray(mb.compute_overlap()["matrix"])
    dG = float(free_energies.loc["endstate", "window"]) * kT
    err = float(errors.loc["endstate", "window"]) * kT
    counts = df.groupby(level="state").size().to_dict()
    return dG, err, float(min(O[0, 1], O[1, 0])), counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-dir", required=True,
                    help="the lambda_window state dir (holds restraint.RST, *_prod_traj.nc, mdout)")
    ap.add_argument("--endstate-traj", required=True,
                    help="the endstate ensemble, e.g. remd/<solute>/<solute>_298.0K.nc")
    ap.add_argument("--window-traj", default=None,
                    help="override the window trajectory (e.g. the merged 10 ns)")
    ap.add_argument("--parm", default=None, help="topology (default: the .parm7 in --window-dir)")
    ap.add_argument("--window-restraint", default=None,
                    help="the window's DISANG (default: restraint.RST in --window-dir)")
    ap.add_argument("--endstate-restraint", required=True,
                    help="the ENDSTATE's own DISANG -- it has one, and production scores the "
                         "endstate with nmropt=1 against it. Pass the restraint.RST from the "
                         "endstate's own state directory, not the window's")
    ap.add_argument("--stride", type=int, default=100, help="keep every Nth frame (default 100)")
    ap.add_argument("--nproc", type=int, default=4,
                    help="sander jobs run concurrently (default 4 -- there are only 4)")
    ap.add_argument("--ranks", type=int, default=1,
                    help="MPI ranks per sander job (default 1 = serial). Total cores used is "
                         "--nproc x --ranks, so 4 x 10 fills a 40-core allocation")
    ap.add_argument("--launcher", default="mpirun",
                    help=f"how to launch a multi-rank job: one of {sorted(LAUNCHERS)}, or a raw "
                         f"template containing {{ranks}}. Default mpirun, which does not depend on "
                         f"Slurm's PMI. Ignored when --ranks 1")
    ap.add_argument("--sander", default=None,
                    help="sander executable (default: $SANDER serial, $SANDER_MPI when --ranks > 1)")
    ap.add_argument("--workdir", default="./endstate_seam_work")
    ap.add_argument("--temperature", type=float, default=298.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    template = LAUNCHERS.get(args.launcher, args.launcher)
    if args.ranks > 1 and "{ranks}" not in template and template:
        raise SystemExit(f"ERROR: --launcher template {template!r} has no {{ranks}} placeholder")
    launcher = "" if args.ranks <= 1 else template.format(ranks=args.ranks)

    wdir = os.path.abspath(args.window_dir)
    parm = os.path.abspath(args.parm or one(f"{wdir}/*.parm7", "topology"))
    rst = {"window": os.path.abspath(args.window_restraint
                                     or one(f"{wdir}/restraint.RST", "window restraint file")),
           "endstate": os.path.abspath(args.endstate_restraint)}
    for side, path in rst.items():
        if not os.path.isfile(path):
            raise SystemExit(f"ERROR: {side} restraint file not found: {path}")
    win_traj = os.path.abspath(args.window_traj or one(f"{wdir}/*_prod_traj.nc", "window trajectory"))
    end_traj = os.path.abspath(args.endstate_traj)
    params = gb_settings(one(f"{wdir}/mdout", "window production mdout"))
    kT = kcals_per_kt(args.temperature)

    os.makedirs(args.workdir, exist_ok=True)
    work = os.path.abspath(args.workdir)

    print(f"parm       {parm}")
    print(f"rst window {rst['window']}")
    print(f"rst endstt {rst['endstate']}")
    print(f"endstate   {end_traj}")
    print(f"window     {win_traj}")
    print(f"GB         igb={params['igb']} extdiel={params['extdiel']} saltcon={params['saltcon']} "
          f"rgbmax={params['rgbmax']} cut={params['cut']}")
    # There are only ever 4 jobs, so --nproc above that is capped and buys nothing; the only way
    # to use more cores is --ranks (MPI within each job).
    workers = min(args.nproc, 4)
    capped = "" if workers == args.nproc else f" (--nproc {args.nproc} capped: only 4 jobs exist)"
    print(f"layout     {workers} concurrent jobs x {args.ranks} ranks = {workers * args.ranks} "
          f"cores{capped}" + (f"   [{launcher}]" if launcher else "   [serial]"))

    print("\n--- restraint check ---")
    findings = check_restraints(rst["window"], rst["endstate"], wdir)
    for line in findings:
        print(f"  {line}")
    if any(f.startswith("WARNING") for f in findings):
        print("  ^ du is a difference of these two restraint terms, so a mismatch here changes "
              "the answer silently.")
    print()

    # sander's 256-char filename limit means nothing outside workdir can be named directly; give
    # the topology and both DISANG files short names inside it.
    parm = link_into(parm, work, "top.parm7")
    rst = {side: link_into(path, work, f"rst_{side}.RST") for side, path in rst.items()}

    trajs = {}
    for name, raw in (("endstate", end_traj), ("window", win_traj)):
        t = stride(raw, parm, args.stride, work, name, args.dry_run)
        trajs[name] = (t, first_frame(t, parm, work, name, args.dry_run))

    # 4 jobs: both ensembles under both Hamiltonians. Each Hamiltonian carries its OWN restraint
    # file, matching production -- the endstate is not scored unrestrained.
    jobs = [(f"{ens}__at__{ham}", trajs[ens][0], parm, trajs[ens][1], params,
             rst[ham], work, args.dry_run, launcher, args.sander)
            for ens in ("endstate", "window") for ham in ("endstate", "window")]
    print(f"launching {len(jobs)} sander jobs, {min(args.nproc, len(jobs))} at a time"
          + (f" x {args.ranks} ranks" if args.ranks > 1 else " (serial, 1 core each)")
          + " -- no output until the first one finishes; check with `pgrep -c sander`")
    _t0 = time.perf_counter()
    results = {r[0]: r for r in _run_pool(_score_worker, jobs, args.nproc, "score")}
    print(f"\n  wall {time.perf_counter() - _t0:.0f} s   "
          + "  ".join(f"{t.replace('__at__', '@')}={r[5]:.0f}s{' cached' if r[3] else ''}"
                      for t, r in sorted(results.items())))

    for tag, _, _, _, err, _ in results.values():
        if err:
            raise SystemExit(f"ERROR in {tag}:\n{err}")
    if args.dry_run:
        return

    e = {tag: r[1] for tag, r in results.items()}
    rterm = {tag: r[2] for tag, r in results.items()}

    # du IS the difference of the two restraint terms -- nothing else in the two Hamiltonians
    # differs. Take it from the RESTRAINT columns (4 decimals) rather than by differencing the
    # mdout ENERGY header, which carries only ~5 significant figures on a ~-2900 kcal/mol total
    # and so quantises a du of ~3 at the 0.1 kcal/mol level. The header difference is kept as the
    # consistency check, with a tolerance set by that precision.
    du = {}
    for ens in ("endstate", "window"):
        du[ens] = rterm[f"{ens}__at__window"] - rterm[f"{ens}__at__endstate"]
        from_header = e[f"{ens}__at__window"] - e[f"{ens}__at__endstate"]
        resid = np.abs(du[ens] - from_header).max()
        tol = 1e-4 * np.abs(e[f"{ens}__at__endstate"]).max()  # mdout ENERGY prints ~5 sig figs
        flag = "" if resid < tol else "   <-- du is NOT restraint-only, check the mdins"
        print(f"{ens:9s} ensemble: du mean={du[ens].mean():8.2f} sd={du[ens].std():6.2f} "
              f"range=[{du[ens].min():.2f}, {du[ens].max():.2f}]"
              f"   (R_win {rterm[f'{ens}__at__window'].mean():.2f}, "
              f"R_end {rterm[f'{ens}__at__endstate'].mean():.2f})"
              f"  |du - dE|max={resid:.2e} (tol {tol:.2e}){flag}")

    lo, hi = du["window"].max(), du["endstate"].min()
    if lo < hi:
        print(f"\nNOTE: du supports are DISJOINT -- window max {lo:.2f} < endstate min {hi:.2f} "
              f"({hi - lo:.2f} kcal/mol gap). A structural mismatch, not a sampling deficit.")

    print()
    for sub in (True, False):
        dG, err, ov, counts = seam_mbar(du["endstate"], du["window"], kT, subsample=sub)
        tag = "subsampled (production)" if sub else "all frames"
        detail = "  " + "  ".join(f"{k} n={n}" for k, n in sorted(counts.items()))
        thin = "  <-- too few samples to trust" if min(counts.values()) < 50 else ""
        print(f"{tag:24s} dG = {dG:8.3f} +- {err:8.3f} kcal/mol   "
              f"overlap = {ov:.4e}   {'PASS' if ov >= 0.04 else 'FAIL'}{detail}{thin}")
    print("\n(min_degree_overlap = 0.04. Subsampling leaves dG alone and shrinks the error bar's\n"
          " optimism; it is reported first because that is what production MBAR solves. Note this\n"
          " is a 2-state overlap -- the full-cycle matrix entry also depends on every other state.)")


if __name__ == "__main__":
    main()
