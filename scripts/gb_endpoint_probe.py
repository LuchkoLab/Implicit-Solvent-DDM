#!/usr/bin/env python
"""gb_endpoint_probe.py -- is the `no_gb` endpoint (igb=6) the eps->1 limit of the GB ladder?

STANDALONE (no Toil, no MD). Re-scores the receptor leg's existing production trajectories under the
GB-dielectric ladder plus TWO competing vacuum endpoints, then builds two MBAR problems that differ in
exactly one column and compares their overlap matrices.

  VAC_old : igb=6, extdiel=0.0   -- what production runs today
  VAC_new : igb=2, extdiel=1.0   -- the proposed fix; the GB prefactor (1/intdiel - 1/extdiel) is
                                    analytically EXACTLY zero, on the same code path and the same
                                    Born radii as every other window in the ladder.

Background
----------
The ladder's free energy is exactly linear in x = 1 - 1/eps (slope 1799.36 +/- 0.21 kcal/mol over five
windows, 0.01% fit). Solving each frame for the effective x of the production `no_gb` column gives
x_eff = -0.1951 +/- 0.0015, where true vacuum requires x = 0 -- that back-solves to eps = 0.84, which
is impossible. The archived mdouts show why: igb=2 prints a full GB block (saltcon/offset/gbalpha/
rgbmax/extdiel) and igb=6 prints none of it, so `extdiel=0.0` is silently discarded and igb=6 is a
different Hamiltonian, not the eps->1 limit.

Usage
-----
  export SANDER=/opt/miniconda3/envs/isddm_env/bin/sander
  export CPPTRAJ=/opt/miniconda3/envs/isddm_env/bin/cpptraj
  python scripts/gb_endpoint_probe.py --stride 50 --workdir /tmp/gb_endpoint_probe

  # smoke test first (2 jobs, ~20 frames)
  python scripts/gb_endpoint_probe.py --stride 500 --smoke --workdir /tmp/gb_endpoint_probe

Write --workdir to LOCAL disk: the repo lives on an sshfs mount that can drop mid-run. Every sander
job is cached on a (trajectory, igb, params) fingerprint, so a dropped mount resumes.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import gb_ddG_bar as gbb  # noqa: E402  -- reuse rescore/first_frame_coord/env overrides

BASE = os.path.join(
    REPO,
    "fragment-opt-abfe-benchmark/isdmm_runs/mcl1_rep1/mcl1_rep1",
    "MCL-1_receptor-ligand-1_with_mcl1_uxU_uxU",
)
PARM = "MCL-1_receptor-ligand-1_with_mcl1_uxU_uxU.parm7"
TEMPERATURE = 298.0

# --------------------------------------------------------------------------------------------------
# Production GB settings, read off the archived mdouts. These MUST match or the comparison is void.
#   rgbmax=25 : the production mdin omits rgbmax, so AMBER defaults it to 25 Ang (confirmed in the
#               mdout's "rgbmax = 25.00000"). gb_ddG_bar._GB_DEFAULTS ships rgbmax=0 -- wrong here.
#   saltcon=0 : confirmed "saltcon = 0.00000" in the production mdout (_GB_DEFAULTS ships 0.3).
#   cut=999   : confirmed (_GB_DEFAULTS ships 9999).
# --------------------------------------------------------------------------------------------------
GB_PARAMS = dict(intdiel=1.0, cut=999.0, rgbmax=25, gbsa=0)

# The two scoring salt concentrations.
#   0.3 : what post_mdin actually used (the user's template value survives -- mdin.py:67-69 passes
#         saltcon=None). This reproduces the production matrix.
#   0.0 : what generate_extdiel_mdin forced for the MD (mdin.py:16-42). This is the fix.
# With saltcon > 0 the GB prefactor is (1/intdiel - exp(-kappa*f)/extdiel), which does NOT vanish at
# extdiel=1 -- so the ladder never reaches vacuum and EGB(x=0) = -349 kcal/mol instead of 0.
SALT_PROD = 0.3
SALT_FIX = 0.0

# The ladder, in cycle order (water -> gas). x = 1 - 1/eps.
LADDER = [
    ("L78.5", "lambda_window/1.0/78.5/4.0", 78.5),
    ("L2.0", "gb_dielectric/1.0/2.0/4.0", 2.0),
    ("L1.3158", "gb_dielectric/1.0/1.3157894736842106/4.0", 1.3157894736842106),
    ("L1.1364", "gb_dielectric/1.0/1.1363636363636365/4.0", 1.1363636363636365),
    ("L1.0638", "gb_dielectric/1.0/1.0638297872340425/4.0", 1.0638297872340425),
    ("L1.0204", "gb_dielectric/1.0/1.0204081632653061/4.0", 1.0204081632653061),
]
VACUUM_DIR = "no_gb/1.0/0.0/4.0"

# The two competing endpoints: (name, igb, extdiel).
VAC_OLD = ("VAC_old", 6, 0.0)   # reproduces production exactly
VAC_NEW = ("VAC_new", 2, 1.0)   # the proposed fix


def xcoord(eps):
    """GB coupling coordinate x = 1/intdiel - 1/extdiel (intdiel=1). Vacuum is x = 0."""
    return 1.0 - 1.0 / eps


def find_traj(state_dir):
    """Locate the production trajectory + prmtop for one state directory."""
    import glob

    full = os.path.join(BASE, state_dir)
    hits = glob.glob(os.path.join(full, "*_traj.nc"))
    if not hits:
        raise FileNotFoundError(f"no *_traj.nc under {full}")
    return hits[0], os.path.join(full, PARM)


def stride_traj(traj, parm, stride, workdir, tag):
    """Write every `stride`-th frame of `traj` to a small local .nc (cached)."""
    out = os.path.join(workdir, f"{tag}.stride{stride}.nc")
    if os.path.exists(out):
        return out
    tmp = out + ".tmp.nc"
    script = f"parm {parm}\ntrajin {traj} 1 last {stride}\ntrajout {tmp} netcdf\nrun\nquit\n"
    res = subprocess.run([gbb.CPPTRAJ], input=script.encode(),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not os.path.exists(tmp):
        raise RuntimeError(
            f"cpptraj could not stride {traj}:\n{res.stdout.decode(errors='replace')[-1500:]}"
        )
    os.replace(tmp, out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", default="/tmp/gb_endpoint_probe")
    p.add_argument("--stride", type=int, default=50,
                   help="keep every Nth frame (default 50 -> ~200 frames from ~10000)")
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--smoke", action="store_true",
                   help="only score L1.0204 and the two vacuum endpoints against T_1.0204 (2 jobs)")
    p.add_argument("--force", action="store_true", help="ignore the sander cache")
    a = p.parse_args()

    os.makedirs(a.workdir, exist_ok=True)
    kT = gbb.kcals_per_kt(a.temperature)

    # ---------------------------------------------------------------- states and Hamiltonians
    # Two 7-state MBAR problems sharing the vacuum column:
    #   PROD : ladder scored at saltcon=0.3  -> must reproduce the production matrix
    #   FIX  : ladder scored at saltcon=0.0  -> matches the MD that generated the samples
    # (igb, extdiel, saltcon). VAC is igb=6; the smoke test proved igb=6 == igb=2/extdiel=1.0
    # exactly at saltcon=0, so one vacuum column serves both.
    ham = (
        [(f"{n}_s03", 2, e, SALT_PROD) for (n, _, e) in LADDER]
        + [(f"{n}_s00", 2, e, SALT_FIX) for (n, _, e) in LADDER]
        + [("VAC", 6, 0.0, SALT_FIX)]
    )
    traj_specs = [(n, d) for (n, d, _) in LADDER] + [("T_vac", VACUUM_DIR)]

    if a.smoke:
        ham = [h for h in ham if h[0] in ("L1.0204_s03", "L1.0204_s00", "VAC")]
        traj_specs = [t for t in traj_specs if t[0] == "L1.0204"]

    print(f"[probe] workdir={a.workdir}  stride={a.stride}  kT={kT:.7f} kcal/mol")
    print(f"[probe] {len(traj_specs)} trajectories x {len(ham)} Hamiltonians "
          f"= {len(traj_specs)*len(ham)} sander jobs\n", flush=True)

    # ---------------------------------------------------------------- stride the trajectories
    trajs = {}
    for tname, tdir in traj_specs:
        raw, parm = find_traj(tdir)
        small = stride_traj(raw, parm, a.stride, a.workdir, tname)
        trajs[tname] = (small, parm)
        print(f"  [stride] {tname:<10} {os.path.basename(small)}", flush=True)

    # ---------------------------------------------------------------- re-score everything
    rows = []
    comps = {}
    for tname, (traj, parm) in trajs.items():
        for hname, igb, eps, salt in ham:
            params = dict(GB_PARAMS, extdiel=eps, saltcon=salt)
            tag = f"{tname}__at__{hname}"
            e, cached = gbb.rescore(traj, parm, None, igb, params, a.workdir, tag, force=a.force)
            print(f"  [{'cache' if cached else 'sander':>6}] {tag:<28} n={len(e):>5} "
                  f"<E>={e.mean():+14.4f} kcal/mol", flush=True)
            for i, val in enumerate(e):
                rows.append({"traj": tname, "ham": hname, "igb": igb, "extdiel": eps,
                             "saltcon": salt,
                             "x": xcoord(eps) if igb == 2 else 0.0,
                             "frame": i, "E": val})
            # keep the component table for the decomposition (P5)
            from implicit_solvent_ddm.mdout import min_to_dataframe
            comps[(tname, hname)] = min_to_dataframe(os.path.join(a.workdir, f"{tag}.mdout"))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(a.workdir, "results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[probe] wrote {out_csv} ({len(df)} rows)")

    np.save(os.path.join(a.workdir, "energies.npy"),
            {"df": df, "comps": comps, "kT": kT}, allow_pickle=True)
    print(f"[probe] wrote {os.path.join(a.workdir, 'energies.npy')}")
    print("\n[probe] re-scoring complete -- run gb_endpoint_analyze.py for P1-P7")
    return 0


if __name__ == "__main__":
    sys.exit(main())
