#!/usr/bin/env python
"""gb_junction_overlap.py -- measure the GB-band <-> gas junction overlap, with and without the
scoring-saltcon fix.

WHY
---
The GB-dielectric band's MD is written by `generate_extdiel_mdin` with saltcon=0.0 (mdin.py:16-42),
because with salt the GB polar term is no longer linear in lambda = 1 - 1/eps. But the MBAR matrix is
filled by `post_mdin`, built at mdin.py:67-69 with NO saltcon argument -- so the guard at mdin.py:216
never fires and the user's saltcon (e.g. 0.3) survives into scoring.

With saltcon > 0 the GB prefactor is (1/intdiel - exp(-kappa*f_GB)/extdiel), which does NOT vanish at
extdiel = 1. So the scored ladder never reaches vacuum: it retains ~350 kcal/mol of Debye-screened GB
energy at eps -> 1, while the adjacent gas state (igb=6) has EGB = 0 exactly. That cliff is what
destroys the overlap at the junction.

Measured on MCL-1 rep1, receptor leg (100 frames):
    scoring saltcon=0.3 (production) : dG = +370.65   overlap = 3.52e-08
    scoring saltcon=0.0 (fix)        : dG =  +40.95   overlap = 6.33e-02
(the production .h5 itself says dG = +369.66, overlap = 1.12e-08 -- the 0.3 run reproduces it)

WHAT THIS DOES
--------------
Re-scores the two production trajectories that straddle the junction under three Hamiltonians:
  GB_s03  igb=2, extdiel=<eps>, saltcon=<--saltcon-prod>   (what post_mdin actually used)
  GB_s00  igb=2, extdiel=<eps>, saltcon=0.0                (what the MD used -- the fix)
  GAS     igb=6                                            (the gas anchor)
then reports 2-state BAR dG + MBAR overlap for GB_s03<->GAS and GB_s00<->GAS.

No MD, no Toil. Only sander imin=5 single points over existing trajectories.

USAGE
-----
  export SANDER=$AMBERHOME/bin/sander        # or sander.MPI / pmemd -- any single-point-capable build
  export CPPTRAJ=$AMBERHOME/bin/cpptraj

  # receptor leg (the one already validated)
  python scripts/gb_junction_overlap.py --leg receptor \
      --root <run>/mcl1_rep1/MCL-1_receptor-ligand-1_with_mcl1_uxU_uxU

  # complex leg
  python scripts/gb_junction_overlap.py --leg complex \
      --root <run>/mcl1_rep1/MCL-1_ligand-1

  # fully explicit (any leg / any junction)
  python scripts/gb_junction_overlap.py --gb-dir <path> --gas-dir <path> --eps 1.0204081632653061

  --dry-run prints the mdins and sander commands without executing them.

NOTES
-----
* Production GB settings are read off the GB state's own mdout so the re-score matches it exactly
  (rgbmax in particular: the production mdin omits it, so AMBER defaults to 25 Ang -- hardcoding 999
  would change the Born radii and silently invalidate the comparison).
* nmropt=0. Both junction states sit at the same restraint window with byte-identical restraint.RST,
  so the restraint energy is a constant that cancels in every energy difference. Absolute energies
  will differ from the production .h5 by that constant; no dG or overlap number is affected.
* Every sander job is cached on a (trajectory, igb, params) fingerprint -- safe to re-run / resume.
"""
import argparse
import concurrent.futures as cf
import glob
import os
import re
import subprocess
import sys
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SANDER = os.environ.get("SANDER", "sander")
CPPTRAJ = os.environ.get("CPPTRAJ", "cpptraj")

# Cycle geometry (implicit_solvent_ddm/matrix_order.py):
#   receptor_order: ... -> gb_dielectric(eps DESC) -> no_gb          (gas is LAST)
#   complex_order : no_interactions -> interactions -> gb_dielectric(eps ASC) -> ...  (gas is FIRST)
# Either way the junction is the smallest-eps GB window against its gas neighbour.
LEGS = {
    "receptor": dict(gb="gb_dielectric/1.0/{eps}/4.0", gas="no_gb/1.0/0.0/4.0"),
    "complex": dict(gb="gb_dielectric/0.0/{eps}/4.0/8.0", gas="interactions/0.0/0.0/4.0/8.0"),
}
DEFAULT_EPS = "1.0204081632653061"
AVAGADRO, BOLTZMAN, JOULES_PER_KCAL = 6.0221367e23, 1.380658e-23, 4184


def kcals_per_kt(t):
    return (BOLTZMAN * AVAGADRO / JOULES_PER_KCAL) * t


def one(pattern, what):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"ERROR: no {what} matching {pattern}")
    return hits[0]


def read_gb_settings(mdout):
    """Pull the GB parameters AMBER actually used, straight out of a production mdout."""
    txt = open(mdout).read()
    head = txt.split("3.  ATOMIC")[0]
    out = {}
    for key, default in (("intdiel", 1.0), ("cut", 999.0), ("rgbmax", 25.0),
                         ("gbsa", 0), ("saltcon", 0.3)):
        m = re.search(rf"\b{key}\s*=\s*([0-9.]+)", head)
        out[key] = float(m.group(1)) if m else float(default)
    out["gbsa"] = int(out["gbsa"])
    return out


def write_mdin(path, igb, extdiel, saltcon, p):
    with open(path, "w") as fh:
        fh.write(
            "single-point GB re-score (imin=5)\n&cntrl\n"
            "  imin=5, ntx=1, irest=0, ntb=0, nmropt=0,\n"
            f"  igb={int(igb)}, intdiel={p['intdiel']}, extdiel={extdiel},\n"
            f"  saltcon={saltcon}, gbsa={p['gbsa']},\n"
            f"  cut={p['cut']}, rgbmax={p['rgbmax']},\n"
            "  ntpr=1, ntwr=0, ntwx=0,\n/\n"
        )
    return path


def stride(traj, parm, n, workdir, tag, dry=False):
    out = os.path.join(workdir, f"{tag}.stride{n}.nc")
    if os.path.exists(out):
        return out
    tmp = out + ".tmp.nc"
    script = f"parm {parm}\ntrajin {traj} 1 last {n}\ntrajout {tmp} netcdf\nrun\nquit\n"
    if dry:
        print(f"  [dry] cpptraj <<< {script!r}")
        return out
    r = subprocess.run([CPPTRAJ], input=script.encode(),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not os.path.exists(tmp):
        raise SystemExit(f"ERROR: cpptraj failed on {traj}\n{r.stdout.decode(errors='replace')[-1500:]}")
    os.replace(tmp, out)
    return out


def first_frame(traj, parm, workdir, tag, dry=False):
    out = os.path.join(workdir, f"{tag}.ref.rst7")
    if os.path.exists(out) or dry:
        return out
    tmp = out + ".tmp"
    script = f"parm {parm}\ntrajin {traj} 1 1\ntrajout {tmp} restart\nrun\nquit\n"
    subprocess.run([CPPTRAJ], input=script.encode(),
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    got = tmp if os.path.exists(tmp) else (tmp + ".1" if os.path.exists(tmp + ".1") else None)
    if got is None:
        raise SystemExit(f"ERROR: cpptraj could not extract frame 1 of {traj}")
    os.replace(got, out)
    return out


def score(traj, parm, coord, igb, extdiel, saltcon, p, workdir, tag, dry=False):
    """sander imin=5 over `traj`; returns per-frame ENERGY (kcal/mol)."""
    from implicit_solvent_ddm.mdout import min_to_dataframe

    mdout = os.path.join(workdir, f"{tag}.mdout")
    done = os.path.join(workdir, f"{tag}.done")
    key = f"{os.path.abspath(traj)}|igb={igb}|extdiel={extdiel}|saltcon={saltcon}|" + str(sorted(p.items()))
    if os.path.exists(done) and os.path.exists(mdout) and open(done).read().strip() == key:
        return min_to_dataframe(mdout)["ENERGY"].to_numpy(float), True

    mdin = write_mdin(os.path.join(workdir, f"{tag}.mdin"), igb, extdiel, saltcon, p)
    cmd = [SANDER, "-O", "-i", mdin, "-p", parm, "-c", coord, "-y", traj, "-o", mdout,
           "-r", os.path.join(workdir, f"{tag}.restrt"), "-x", os.path.join(workdir, f"{tag}.mdcrd")]
    if dry:
        print(f"  [dry] {' '.join(cmd)}")
        print("        " + open(mdin).read().replace("\n", "\n        ").rstrip())
        return np.zeros(1), False
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not os.path.exists(mdout):
        tail = "".join(open(mdout).readlines()[-25:]) if os.path.exists(mdout) else ""
        raise SystemExit(f"ERROR: sander failed\n  {' '.join(cmd)}\n"
                         f"{r.stdout.decode(errors='replace')[-1500:]}\n--- mdout tail ---\n{tail}")
    e = min_to_dataframe(mdout)["ENERGY"].to_numpy(float)
    if len(e) == 0:
        raise SystemExit(f"ERROR: no ENERGY frames parsed from {mdout}")
    open(done, "w").write(key)
    return e, False


# --------------------------------------------------------------------------------------------------
# Parallel workers. Module-level so ProcessPoolExecutor can pickle them. Every job writes only to
# files keyed by its own tag, so concurrent jobs never touch the same path.
# --------------------------------------------------------------------------------------------------
def _prep_worker(job):
    """Stride one state's trajectory and extract its -c reference frame."""
    sn, traj_raw, parm, n, workdir, dry = job
    try:
        traj = stride(traj_raw, parm, n, workdir, sn, dry)
        coord = first_frame(traj, parm, workdir, sn, dry)
        return sn, traj, coord, None
    except BaseException:
        return sn, None, None, traceback.format_exc()


def _score_worker(job):
    """Run one sander single-point re-score."""
    sn, hn, traj, parm, coord, igb, eps, salt, p, workdir, dry = job
    try:
        e, cached = score(traj, parm, coord, igb, eps, salt, p, workdir, f"{sn}__at__{hn}", dry)
        return sn, hn, e, cached, None
    except BaseException:
        return sn, hn, None, False, traceback.format_exc()


def _run_pool(fn, jobs, nproc, label):
    """Map `fn` over `jobs` with up to `nproc` workers; returns results in completion order."""
    if nproc <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    out = []
    with cf.ProcessPoolExecutor(max_workers=min(nproc, len(jobs))) as ex:
        futs = [ex.submit(fn, j) for j in jobs]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            out.append(fut.result())
            print(f"  [{label} {i}/{len(jobs)}]", end=" ", flush=True)
    return out


def bar(w_f, w_r, kT):
    """2-state BAR dG (kcal/mol), error, and MBAR overlap, from forward/reverse work in kT."""
    import pymbar
    try:
        from implicit_solvent_ddm.block_mbar import BLOCK_SOLVER_PROTOCOL as proto
    except Exception:  # keep working if block_mbar isn't on this branch
        proto = (dict(method="adaptive", options=dict(min_sc_iter=0, maxiter=1000)),
                 dict(method="L-BFGS-B", options=dict(maxiter=1000)))
    nf, nr = len(w_f), len(w_r)
    u = np.zeros((2, nf + nr))
    u[1, :nf], u[0, nf:] = w_f, w_r
    mb = pymbar.MBAR(u, np.array([nf, nr]), solver_protocol=proto)
    res = mb.compute_free_energy_differences()
    ov = np.asarray(mb.compute_overlap()["matrix"])[0, 1]
    return float(res["Delta_f"][0, 1]) * kT, float(res["dDelta_f"][0, 1]) * kT, float(ov)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", choices=sorted(LEGS), help="use the built-in directory layout for this leg")
    ap.add_argument("--root", help="leg root (the dir containing gb_dielectric/, no_gb/ or interactions/)")
    ap.add_argument("--gb-dir", help="explicit path to the smallest-eps GB window")
    ap.add_argument("--gas-dir", help="explicit path to the adjacent gas (igb=6) state")
    ap.add_argument("--eps", default=DEFAULT_EPS, help=f"external dielectric (default {DEFAULT_EPS})")
    ap.add_argument("--saltcon-prod", type=float, default=None,
                    help="scoring saltcon that production used (default: read from the GB mdout)")
    ap.add_argument("--stride", type=int, default=100, help="keep every Nth frame (default 100)")
    ap.add_argument("--nproc", type=int, default=6,
                    help="concurrent sander jobs (default 6 = all of them; 0 = os.cpu_count(); "
                         "1 = serial). Keep SANDER a SERIAL build -- pointing this at sander.MPI "
                         "and running 6 at once oversubscribes the node.")
    ap.add_argument("--workdir", default="./gb_junction_work")
    ap.add_argument("--temperature", type=float, default=298.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.gb_dir and a.gas_dir:
        gb_dir, gas_dir = a.gb_dir, a.gas_dir
    elif a.leg and a.root:
        gb_dir = os.path.join(a.root, LEGS[a.leg]["gb"].format(eps=a.eps))
        gas_dir = os.path.join(a.root, LEGS[a.leg]["gas"])
    else:
        raise SystemExit("need either (--leg and --root) or (--gb-dir and --gas-dir)")

    os.makedirs(a.workdir, exist_ok=True)
    kT = kcals_per_kt(a.temperature)

    states = {}
    for name, d in (("GB", gb_dir), ("GAS", gas_dir)):
        if not os.path.isdir(d):
            raise SystemExit(f"ERROR: not a directory: {d}")
        states[name] = dict(dir=d,
                            traj=one(os.path.join(d, "*_traj.nc"), "trajectory"),
                            parm=one(os.path.join(d, "*.parm7"), "prmtop"),
                            mdout=os.path.join(d, "mdout"))

    p = read_gb_settings(states["GB"]["mdout"])
    assumed = a.saltcon_prod is None
    salt_prod = 0.3 if assumed else a.saltcon_prod
    print(f"GB  state : {states['GB']['dir']}")
    print(f"GAS state : {states['GAS']['dir']}")
    print(f"GB params read from its own MD mdout: {p}")
    print(f"  (note: the MD saltcon above is {p['saltcon']}; the SCORING saltcon is the bug and is "
          f"NOT in this mdout --\n   grep it from post_processing/.../mdout and pass --saltcon-prod "
          f"if it is not 0.3)")
    print(f"scoring saltcon compared: production={salt_prod}"
          f"{' [assumed]' if assumed else ' [given]'}  vs  fix=0.0")
    print(f"eps={a.eps}   stride={a.stride}   kT={kT:.7f} kcal/mol\n")

    hams = [("GB_s03", 2, a.eps, salt_prod), ("GB_s00", 2, a.eps, 0.0), ("GAS", 6, 0.0, 0.0)]
    nproc = a.nproc if a.nproc > 0 else (os.cpu_count() or 1)
    print(f"nproc={nproc}  ({len(states)} prep jobs, then {len(states)*len(hams)} sander jobs)\n")

    # --- prep: stride + reference frame, one job per state (I/O bound -- overlaps well) ------------
    prep_jobs = [(sn, s["traj"], s["parm"], a.stride, a.workdir, a.dry_run)
                 for sn, s in states.items()]
    prepped = {}
    for sn, traj, coord, err in _run_pool(_prep_worker, prep_jobs, nproc, "prep"):
        if err:
            raise SystemExit(f"\nERROR preparing {sn}:\n{err}")
        prepped[sn] = (traj, coord)
    print(f"\n  prepped {len(prepped)} trajectories\n" if nproc > 1 else "")

    # --- score: every (state, Hamiltonian) pair is independent -------------------------------------
    score_jobs = [(sn, hn, prepped[sn][0], states[sn]["parm"], prepped[sn][1],
                   igb, eps, salt, p, a.workdir, a.dry_run)
                  for sn in states for hn, igb, eps, salt in hams]
    E, results = {}, _run_pool(_score_worker, score_jobs, nproc, "sander")
    print()
    errs = [(sn, hn, err) for sn, hn, _, _, err in results if err]
    if errs:
        raise SystemExit("\n".join(f"ERROR in {sn}__at__{hn}:\n{err}" for sn, hn, err in errs))
    for sn, hn, e, cached, _ in sorted(results, key=lambda r: (r[0], r[1])):
        E[(sn, hn)] = e
        if not a.dry_run:
            print(f"  [{'cache' if cached else 'sander':>6}] {sn}__at__{hn:<10} n={len(e):>4} "
                  f"<E>={e.mean():+13.4f} kcal/mol")
    if a.dry_run:
        return 0

    print(f"\n{'='*78}\nJUNCTION   GB(eps={a.eps})  <->  GAS(igb=6)\n{'='*78}")
    print(f"{'scoring':<26} {'dG kcal/mol':>13} {'err':>9} {'overlap':>12} {'<W_F> kT':>10} {'sd':>7}")
    res = {}
    for label, hn in ((f"saltcon={salt_prod} (PRODUCTION)", "GB_s03"), ("saltcon=0.0 (FIX)", "GB_s00")):
        w_f = (E[("GB", "GAS")] - E[("GB", hn)]) / kT     # sampled in the GB window
        w_r = (E[("GAS", hn)] - E[("GAS", "GAS")]) / kT   # sampled in the gas state
        dG, err, ov = bar(w_f, w_r, kT)
        res[hn] = ov
        print(f"{label:<26} {dG:>13.4f} {err:>9.4f} {ov:>12.4e} {w_f.mean():>10.2f} {w_f.std():>7.2f}")

    imp = res["GB_s00"] / max(res["GB_s03"], 1e-300)
    print(f"\noverlap improvement (fix / production): {imp:.4e}x")
    print(f"work dir: {os.path.abspath(a.workdir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
