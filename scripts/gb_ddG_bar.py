#!/usr/bin/env python
"""gb_ddG_bar.py -- relative binding free-energy difference between two GB models via BAR.

For a host-guest pair whose END-STATE trajectories were sampled under two GB models (e.g. OBC igb=2
and OBC2 igb=5) for each leg (complex, host/receptor, guest/ligand), this:

  1. re-scores EVERY frame under BOTH GB models with a sander imin=5 single-point,
  2. runs a 2-state BAR (== 2-column MBAR) per leg, pooling ALL replicas, and
  3. combines the legs:

        ddG_bind(A->B) = dG_complex(A->B) - dG_host(A->B) - dG_guest(A->B)

It reports the per-leg 2-state overlap (the previously-unknown OBC<->OBC2 phase-space overlap) and
an across-replica bootstrap CI (fixes the single-replica fragility), and can compare against the
ABFE-derived difference dG_bind(B) - dG_bind(A).

This is a STANDALONE analysis (no Toil). It needs AMBER `sander` on PATH (`module load amber`) and
the `implicit_solvent_ddm` package importable. It reuses the package's mdout parser
(`mdout.min_to_dataframe`) and MBAR wrapper (`pandasmbar._mbar`); a 2-column MBAR is numerically BAR.

Usage
-----
  # verify the BAR math + reduced-potential convention locally (no sander needed):
  python gb_ddG_bar.py --selftest

  # run the analysis over a manifest of pairs (on the cluster, sander available):
  python gb_ddG_bar.py --manifest pairs.yaml --out ddG_gb.csv

Manifest (YAML) -- drives "any endstate, any GB model, entire dataset"
----------------------------------------------------------------------
  temperature: 298.0
  igb_A: 2                 # sampled model A (e.g. OBC)
  igb_B: 5                 # sampled model B (e.g. OBC2)
  n_bootstrap: 200         # across-replica bootstrap (0 to skip -> analytical BAR error only)
  reference_mdin: /path/to/production_endstate.mdin   # GB params (cut/rgbmax/gbsa/saltcon/extdiel)
  pairs:
    - name: cb7_adn
      legs:
        complex: {parm: DRY_complex.parm7, coord: complex_lastframe.rst7,
                  A: [complex_obc_rep1.nc, complex_obc_rep2.nc],
                  B: [complex_obc2_rep1.nc, complex_obc2_rep2.nc]}
        host:    {parm: receptor.parm7,    coord: host_lastframe.rst7,
                  A: [host_obc_rep1.nc],    B: [host_obc2_rep1.nc]}
        guest:   {parm: ligand.parm7,      coord: guest_lastframe.rst7,
                  A: [guest_obc_rep1.nc],   B: [guest_obc2_rep1.nc]}
      # optional comparison: one FINAL dG_bind per replica per model (lists); ddG_ABFE = <B> - <A>
      abfe: {dG_bind_A: [-12.1, -12.4, -11.9, -12.6, -12.2], dG_bind_B: [-10.0, -10.3, -9.8, -10.5, -10.1]}

Per-model radii (igb=8 / GBn2)
------------------------------
GB radii live in the prmtop, not the mdin, so `igb=8` on an mbondi2 prmtop is NOT real GBn2 -- GBn2
is parameterized for mbondi3. When igb_A and igb_B need DIFFERENT radii, give `parm_A` and `parm_B`
on a leg (per-replica dicts accept them too) INSTEAD of a single `parm`; each model's re-score then
uses its own topology. A plain `parm` still serves both models (mbondi2 igb=2<->5). Make the mbondi3
copy once with parmed (same atoms/order, radii only):  parmed -p sys.mbondi2.parm7  ->
`changeRadii mbondi3` -> `outparm sys.mbondi3.parm7`. Then e.g.:
        complex: {parm_A: sys.mbondi2.parm7, parm_B: sys.mbondi3.parm7, coord: last.rst7,
                  A: [...igb2 trajs...], B: [...igb2 trajs re-scored, or igb8-sampled trajs...]}
"""
import argparse
import concurrent.futures
import os
import re
import subprocess

import numpy as np
import pandas as pd

# ---- reduced-potential convention: identical to adaptive_restraints.py / postTreatment.py ----
AVAGADRO = 6.0221367e23
BOLTZMAN = 1.380658e-23
JOULES_PER_KCAL = 4184


def kcals_per_kt(temperature):
    """kcal/mol per kT at `temperature` (K) -- matches the package's `kcals_per_Kt`."""
    return (BOLTZMAN * AVAGADRO / JOULES_PER_KCAL) * temperature


SANDER = os.environ.get("SANDER", "sander")
CPPTRAJ = os.environ.get("CPPTRAJ", "cpptraj")
LEGS = ("complex", "host", "guest")  # ddG = complex - host - guest


def first_frame_coord(traj, parm, workdir, coord_tag):
    """Write frame 1 of `traj` (read under `parm`) as an rst7, to use as sander's -c reference.
    Guarantees the reference matches this trajectory's own topology (removes coord/atom mismatches).
    Cached + atomic (write to a temp, then rename) so parallel jobs sharing it don't race."""
    out = os.path.join(workdir, f"{coord_tag}.ref.rst7")
    if os.path.exists(out):
        return out
    tmp = os.path.join(workdir, f"{coord_tag}.ref.{os.getpid()}.tmp.rst7")
    script = f"parm {parm}\ntrajin {traj} 1 1\ntrajout {tmp} restart\nrun\nquit\n"
    res = subprocess.run([CPPTRAJ], input=script.encode(),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # cpptraj may append a frame suffix (e.g. .1) to restart names
    written = tmp if os.path.exists(tmp) else (tmp + ".1" if os.path.exists(tmp + ".1") else None)
    if written is None:
        raise RuntimeError(
            f"cpptraj could not extract a reference frame from {traj} under {parm} "
            f"(atom mismatch?):\n{res.stdout.decode(errors='replace')[-1500:]}"
        )
    if not os.path.exists(out):
        os.replace(written, out)
    return out


# --------------------------------------------------------------------------------------------------
# GB scoring mdin (parse production params, emit a clean imin=5 single-point mdin)
# --------------------------------------------------------------------------------------------------
_GB_DEFAULTS = dict(extdiel=78.5, intdiel=1.0, saltcon=0.3, cut=9999.0, rgbmax=0, gbsa=0)


def parse_gb_params(reference_mdin):
    """Pull the GB parameters (cut/rgbmax/gbsa/saltcon/intdiel/extdiel) from a production mdin so the
    re-score matches production settings except the GB model. `$extdiel` placeholders (and missing
    keys) fall back to water/implicit-solvent defaults."""
    params = dict(_GB_DEFAULTS)
    if reference_mdin and os.path.exists(reference_mdin):
        with open(reference_mdin) as fh:
            text = fh.read()
        for key in ("extdiel", "intdiel", "saltcon", "cut", "rgbmax", "gbsa"):
            m = re.search(rf"{key}\s*=\s*([0-9]+\.?[0-9]*)", text)
            if m:
                params[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    return params


def write_score_mdin(path, igb, params):
    """Write a minimal imin=5 (single-point over an input trajectory) GB-energy mdin at `igb`."""
    with open(path, "w") as fh:
        fh.write(
            "single-point GB energy re-score (imin=5)\n"
            "&cntrl\n"
            "  imin=5, ntx=1, irest=0, ntb=0, nmropt=0,\n"
            f"  igb={int(igb)}, intdiel={params['intdiel']}, extdiel={params['extdiel']},\n"
            f"  saltcon={params['saltcon']}, gbsa={int(params['gbsa'])},\n"
            f"  cut={params['cut']}, rgbmax={params['rgbmax']},\n"
            "  ntpr=1, ntwr=0, ntwx=0,\n"
            "/\n"
        )
    return path


# --------------------------------------------------------------------------------------------------
# Re-scoring: sander imin=5 over a trajectory -> per-frame ENERGY (kcal/mol)
# --------------------------------------------------------------------------------------------------
def _cache_key(traj, igb, params):
    """Fingerprint of everything that determines the re-scored energies -- so a cached mdout is only
    reused when the trajectory, GB model, and GB params are all unchanged."""
    return (f"{os.path.abspath(traj)}|igb={igb}|"
            + ",".join(f"{k}={params[k]}" for k in sorted(params)))


def rescore(traj, parm, coord, igb, params, workdir, tag, force=False):
    """Single-point re-score every frame of `traj` under GB model `igb`. Returns
    (energy_ndarray_kcalmol, was_cached). `tag` must be UNIQUE per (leg, ensemble, replica, igb) --
    several replicas share the same trajectory basename, so basename-derived names would collide.

    Caching: if a valid `{tag}.mdout` already exists in `workdir` AND its `{tag}.done` fingerprint
    matches this (traj, igb, params), the existing mdout is parsed and sander is NOT re-run. Pass
    `force=True` (or delete the scratch dir) to recompute."""
    from implicit_solvent_ddm.mdout import min_to_dataframe

    mdout = os.path.join(workdir, f"{tag}.mdout")
    done = os.path.join(workdir, f"{tag}.done")
    key = _cache_key(traj, igb, params)

    if not force and os.path.exists(mdout) and os.path.exists(done):
        try:
            if open(done).read().strip() == key:
                df = min_to_dataframe(mdout)
                if "ENERGY" in df and len(df) > 0:
                    return df["ENERGY"].to_numpy(dtype=float), True
        except Exception:
            pass  # stale/corrupt cache -> fall through and recompute

    if coord is None:  # derive the -c reference from frame 1 of this trajectory (shared across igb)
        coord = first_frame_coord(traj, parm, workdir, tag.rsplit("_igb", 1)[0])
    mdin = write_score_mdin(os.path.join(workdir, f"{tag}.mdin"), igb, params)
    cmd = [SANDER, "-O", "-i", mdin, "-p", parm, "-c", coord, "-y", traj,
           "-o", mdout, "-r", os.path.join(workdir, f"{tag}.restrt"),
           "-x", os.path.join(workdir, f"{tag}.mdcrd")]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if res.returncode != 0 or not os.path.exists(mdout):
        mdout_tail = ""
        if os.path.exists(mdout):  # AMBER writes fatal errors into the mdout, not stdout
            mdout_tail = "\n--- mdout tail ---\n" + "".join(open(mdout).readlines()[-25:])
        raise RuntimeError(
            f"sander re-score failed (igb={igb}, traj={traj})\n"
            f"  parm={parm}\n  coord={coord}\n  cmd: {' '.join(cmd)}\n"
            f"--- stdout/stderr ---\n{res.stdout.decode(errors='replace')[-1500:]}{mdout_tail}"
        )
    df = min_to_dataframe(mdout)
    if "ENERGY" not in df or len(df) == 0:
        raise RuntimeError(f"no ENERGY frames parsed from {mdout} (igb={igb}, traj={traj})")
    with open(done, "w") as fh:  # mark valid + record the fingerprint
        fh.write(key)
    return df["ENERGY"].to_numpy(dtype=float), False


def _subsample_native(e_native):
    """Discard equilibration + decorrelate on the NATIVE-Hamiltonian energy series; return the frame
    indices to keep (falls back to all frames if timeseries analysis is unavailable/degenerate)."""
    try:
        from pymbar import timeseries
        t0, g, _ = timeseries.detect_equilibration(e_native)
        idx = timeseries.subsample_correlated_data(e_native[t0:], g=g)
        keep = np.asarray(idx, dtype=int) + int(t0)
        return keep if keep.size >= 2 else np.arange(len(e_native))
    except Exception:
        return np.arange(len(e_native))


# --------------------------------------------------------------------------------------------------
# Parallel re-scoring: every (leg, ensemble, replica) endstate under BOTH GB models runs concurrently
# --------------------------------------------------------------------------------------------------
def _parm_for(igb, igb_A, parm_A, parm_B):
    """Pick the topology whose GB radii match this model: `parm_A` for the igb_A re-score, `parm_B`
    for igb_B. Lets a model with different radii (e.g. igb=8/GBn2 needs mbondi3, while igb=2/5 use
    mbondi2) be scored under its OWN prmtop -- GB (PB) radii live in the prmtop, not the mdin, so
    `igb=8` on an mbondi2 prmtop is NOT real GBn2. BAR between (igb_A, radii_A) and (igb_B, radii_B)
    is still exact: both reduced potentials are evaluated on the same frames (parm_A/parm_B must be
    the SAME system in the SAME atom order -- e.g. a `parmed changeRadii mbondi3` copy of parm_A)."""
    return parm_A if igb == igb_A else parm_B


def _resolve_entry(entry, spec):
    """Resolve one replica entry to (traj, parm_A, parm_B, coord). An entry is EITHER a trajectory
    path (uses the leg-level parm/coord) OR a dict {traj, parm?/parm_A?/parm_B?, coord?} overriding
    them -- needed when replicas were parameterized in separate runs (so each trajectory is scored
    with its OWN topology / matching atom order) AND/OR when the two GB models need different radii.
    Resolution per field is replica-override > leg value; and `parm_A`/`parm_B` > `parm`, so a plain
    `parm` still serves BOTH models (old behavior, byte-identical when parm_A/parm_B are absent)."""
    if isinstance(entry, dict):
        traj = entry["traj"]
        coord = entry.get("coord", spec.get("coord"))
        base = entry.get("parm", spec.get("parm"))
        parm_A = entry.get("parm_A", spec.get("parm_A", base))
        parm_B = entry.get("parm_B", spec.get("parm_B", base))
    else:
        traj, coord, base = entry, spec.get("coord"), spec.get("parm")
        parm_A = spec.get("parm_A", base)
        parm_B = spec.get("parm_B", base)
    return traj, parm_A, parm_B, coord


def rescore_all(pair, igb_A, igb_B, params, workdir, n_jobs, force=False):
    """Fan out ALL sander imin=5 re-scores for one pair across `n_jobs` workers. Each task is one
    (leg, ensemble, replica, igb) single-point over a trajectory -- fully independent, its own tagged
    mdin/mdout -- so with 3 legs x 2 ensembles x n_replicas x 2 GB models they all run at once (e.g.
    ~60 jobs for 5 replicas). Returns {(leg, ens, rep_idx, igb): energy_ndarray}. ThreadPoolExecutor
    is fine because each worker blocks in the sander subprocess (the GIL is released there)."""
    tasks = []
    for leg in LEGS:
        spec = pair["legs"][leg]
        for ens in ("A", "B"):
            for i, entry in enumerate(spec[ens]):
                traj, parm_A, parm_B, tcoord = _resolve_entry(entry, spec)  # per-replica/-model parm allowed
                for igb in (igb_A, igb_B):
                    tparm = _parm_for(igb, igb_A, parm_A, parm_B)  # this model's radii (e.g. mbondi2 vs mbondi3)
                    tag = f"{leg}_{ens}_rep{i}_igb{igb}"
                    tasks.append(((leg, ens, i, igb), traj, tparm, tcoord, igb, tag))

    def _run(task):
        key, traj, parm, coord, igb, tag = task
        energy, cached = rescore(traj, parm, coord, igb, params, workdir, tag, force=force)
        return key, energy, cached

    results, n_cached = {}, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, n_jobs)) as ex:
        for fut in concurrent.futures.as_completed([ex.submit(_run, t) for t in tasks]):
            key, energy, cached = fut.result()
            results[key] = energy
            n_cached += int(cached)
    return results, len(tasks), n_cached


def assemble_leg(leg, leg_spec, results, igb_A, igb_B, temperature):
    """Build one leg's per-replica (u_igbA, u_igbB) reduced potentials (E/kT) from the parallel
    re-score results, equilibration-trimmed + decorrelated on each replica's native Hamiltonian.
    Returns {"A": [ (u_igbA, u_igbB), ... ], "B": [ ... ]}."""
    kT = kcals_per_kt(temperature)
    out = {"A": [], "B": []}
    for ens, native_igb in (("A", igb_A), ("B", igb_B)):
        for i in range(len(leg_spec[ens])):
            e_igbA = results[(leg, ens, i, igb_A)]
            e_igbB = results[(leg, ens, i, igb_B)]
            native = e_igbA if native_igb == igb_A else e_igbB
            keep = _subsample_native(native)
            out[ens].append((e_igbA[keep] / kT, e_igbB[keep] / kT))
    return out


# --------------------------------------------------------------------------------------------------
# 2-state BAR (== 2-column MBAR) on pooled reduced potentials
# --------------------------------------------------------------------------------------------------
def bar_two_state(rows_A, rows_B, igb_A, igb_B):
    """rows_A / rows_B: lists of (u_igbA, u_igbB) arrays (reduced potentials) sampled at igb_A / igb_B.
    Builds the pooled 2-state df and runs `pandasmbar._mbar`. Returns (dG_kT, err_kT, overlap_2x2)
    where dG = f(igb_B) - f(igb_A) in kT units, and overlap[0,1] is the A<->B superdiagonal overlap."""
    from implicit_solvent_ddm.pandasmbar import _mbar

    uA = [np.column_stack(r) for r in rows_A]  # each (n_r, 2): cols [u@igbA, u@igbB]
    uB = [np.column_stack(r) for r in rows_B]
    data = np.vstack(uA + uB) if (uA or uB) else np.empty((0, 2))
    index = [igb_A] * sum(len(r[0]) for r in rows_A) + [igb_B] * sum(len(r[0]) for r in rows_B)
    df = pd.DataFrame(data, columns=[igb_A, igb_B], index=pd.Index(index, name="state"))

    free_energies, errors, mbar = _mbar(df)
    dG = float(free_energies.loc[igb_A, igb_B])   # f_B - f_A, in kT
    err = float(errors.loc[igb_A, igb_B])
    overlap = mbar.compute_overlap()["matrix"]
    return dG, err, np.asarray(overlap)


# --------------------------------------------------------------------------------------------------
# Per-pair driver: 3 legs -> ddG, with an across-replica bootstrap CI
# --------------------------------------------------------------------------------------------------
def ddG_pair(pair, igb_A, igb_B, params, temperature, n_bootstrap, workdir, rng, n_jobs, force=False):
    """Compute ddG_bind(A->B) for one host-guest pair.

    DEFAULT: run BAR SEPARATELY per replica -- pair OBC replica r with OBC2 replica r, BAR each leg,
    combine complex-host-guest -> one ddG per replica -> report **mean +/- SEM** across replicas.
    This is directly comparable to ddG_ABFE (also a mean +/- SEM over replicas) and doesn't lean on a
    small-N bootstrap. OPTIONAL (n_bootstrap>0): additionally report the all-replicas-POOLED BAR ddG
    and its across-replica bootstrap std. All sander re-scores fan out in parallel up front; the BAR
    solves only re-pool cached energies, so they are cheap and serial."""
    kT = kcals_per_kt(temperature)
    results, n_tasks, n_cached = rescore_all(pair, igb_A, igb_B, params, workdir, n_jobs, force=force)
    print(f"[gb_ddG] {pair['name']}: {n_tasks} sander imin=5 jobs on up to {n_jobs} cores "
          f"({n_cached} reused from cache, {n_tasks - n_cached} run)")
    cache = {leg: assemble_leg(leg, pair["legs"][leg], results, igb_A, igb_B, temperature)
             for leg in LEGS}

    # --- PER-REPLICA ddG (default): one BAR per leg per replica (OBC rep r vs OBC2 rep r) ---
    n_rep = min(min(len(cache[leg]["A"]), len(cache[leg]["B"])) for leg in LEGS)
    per_leg_dG = {leg: [] for leg in LEGS}
    per_leg_ov = {leg: [] for leg in LEGS}
    ddG_reps = []
    for r in range(n_rep):
        legs = {}
        for leg in LEGS:
            dG_kT, _, ov = bar_two_state([cache[leg]["A"][r]], [cache[leg]["B"][r]], igb_A, igb_B)
            legs[leg] = dG_kT * kT
            per_leg_dG[leg].append(dG_kT * kT)
            per_leg_ov[leg].append(float(ov[0, 1]))
        ddG_reps.append(legs["complex"] - legs["host"] - legs["guest"])

    mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
    sem = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
    row = {"pair": pair["name"], "n_rep": n_rep,
           "ddG_BAR": mean(ddG_reps), "ddG_sem": sem(ddG_reps)}
    for leg in LEGS:
        row[f"dG_{leg}"] = mean(per_leg_dG[leg])
        row[f"dG_{leg}_sem"] = sem(per_leg_dG[leg])
        row[f"overlap_{leg}"] = mean(per_leg_ov[leg])   # mean per-replica 2-state overlap

    # --- OPTIONAL: all-replicas-POOLED BAR + across-replica bootstrap (only when n_bootstrap>0) ---
    if n_bootstrap > 0:
        pooled = {}
        for leg in LEGS:
            dG_kT, _, _ = bar_two_state(cache[leg]["A"], cache[leg]["B"], igb_A, igb_B)
            pooled[leg] = dG_kT * kT
        row["ddG_BAR_pooled"] = pooled["complex"] - pooled["host"] - pooled["guest"]
        boot = []
        for _ in range(n_bootstrap):
            legs, ok = {}, True
            for leg in LEGS:
                rows = {}
                for ens in ("A", "B"):
                    reps = cache[leg][ens]
                    pick = rng.integers(0, len(reps), size=len(reps))
                    rows[ens] = [reps[i] for i in pick]
                try:
                    dG_kT, _, _ = bar_two_state(rows["A"], rows["B"], igb_A, igb_B)
                    legs[leg] = dG_kT * kT
                except Exception:
                    ok = False
                    break
            if ok:
                boot.append(legs["complex"] - legs["host"] - legs["guest"])
        row["ddG_boot_std"] = float(np.std(boot, ddof=1)) if len(boot) > 2 else float("nan")
    abfe = pair.get("abfe")
    if abfe and abfe.get("dG_bind_A") is not None and abfe.get("dG_bind_B") is not None:
        # One final dG_bind PER REPLICA in each model (scalar or list); the ABFE comparison is the
        # AVERAGE difference:  ddG_ABFE = <dG_bind(B)> - <dG_bind(A)>, with an across-replica SEM.
        A = np.atleast_1d(np.asarray(abfe["dG_bind_A"], dtype=float))
        B = np.atleast_1d(np.asarray(abfe["dG_bind_B"], dtype=float))
        sem = lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
        row["ddG_ABFE"] = float(B.mean() - A.mean())
        row["ddG_ABFE_err"] = float(np.hypot(sem(A), sem(B)))
        row["n_abfe"] = f"{len(A)}/{len(B)}"
        row["ddG_diff"] = row["ddG_BAR"] - row["ddG_ABFE"]
    return row


# --------------------------------------------------------------------------------------------------
# Manifest driver
# --------------------------------------------------------------------------------------------------
def run_manifest(manifest_path, out_csv, n_jobs=None, scratch=None, force=False):
    import yaml

    with open(manifest_path) as fh:
        man = yaml.safe_load(fh)
    igb_A, igb_B = int(man["igb_A"]), int(man["igb_B"])
    temperature = float(man.get("temperature", 298.0))
    n_bootstrap = int(man.get("n_bootstrap", 0))   # default per-replica mean+/-SEM; >0 adds pooled+bootstrap
    params = parse_gb_params(man.get("reference_mdin"))
    rng = np.random.default_rng(int(man.get("seed", 0)))
    # parallel sander workers: CLI --nproc > manifest n_jobs > all cores
    if n_jobs is None:
        n_jobs = int(man.get("n_jobs", os.cpu_count() or 1))
    # PERSISTENT scratch: per-pair mdouts are kept and reused on re-run (CLI --scratch > manifest
    # workdir > ./gb_ddG_scratch). Cheap analysis tweaks re-run without re-scoring; use --force or
    # delete the dir to recompute.
    scratch_root = scratch or man.get("workdir") or os.path.join(os.getcwd(), "gb_ddG_scratch")

    rows = []
    for pair in man["pairs"]:
        workdir = os.path.join(scratch_root, pair["name"])
        os.makedirs(workdir, exist_ok=True)
        print(f"[gb_ddG] {pair['name']}: re-scoring under igb={igb_A} & igb={igb_B}  (scratch: {workdir})")
        rows.append(ddG_pair(pair, igb_A, igb_B, params, temperature, n_bootstrap,
                             workdir, rng, n_jobs, force=force))
        r = rows[-1]
        extra = (f"  (ABFE {r['ddG_ABFE']:+.2f}+/-{r['ddG_ABFE_err']:.2f} n={r['n_abfe']}, "
                 f"diff {r['ddG_diff']:+.2f})") if "ddG_ABFE" in r else ""
        pooled = (f"  [pooled {r['ddG_BAR_pooled']:+.2f}, boot +/-{r['ddG_boot_std']:.2f}]"
                  if "ddG_BAR_pooled" in r else "")
        print(f"[gb_ddG] {pair['name']}: ddG(igb{igb_A}->igb{igb_B}) = "
              f"{r['ddG_BAR']:+.2f} +/- {r['ddG_sem']:.2f} kcal/mol  (mean+/-SEM over {r['n_rep']} reps)  "
              f"overlaps {r['overlap_complex']:.3f}/{r['overlap_host']:.3f}/{r['overlap_guest']:.3f}"
              f"{pooled}{extra}")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[gb_ddG] wrote {out_csv} ({len(df)} pairs)")
    return df


# --------------------------------------------------------------------------------------------------
# Self-test: verify BAR math + reduced-potential convention WITHOUT sander
# --------------------------------------------------------------------------------------------------
def selftest():
    """Synthetic check: two states whose reduced potentials differ by a constant shift c have
    dG = c (in kT), and re-scoring under the same model twice gives dG ~ 0 with overlap ~ 1."""
    rng = np.random.default_rng(0)
    igb_A, igb_B = 2, 5
    N, c = 4000, 1.7  # true dG(A->B) = c kT

    # sample u_A ~ N(0,1); u_B = u_A + c for BOTH ensembles (a rigid energy shift => exact dG = c)
    def make(n):
        uA = rng.normal(0, 1, n)
        return (uA, uA + c)
    rows_A = [make(N)]
    rows_B = [make(N)]
    dG, err, ov = bar_two_state(rows_A, rows_B, igb_A, igb_B)
    assert abs(dG - c) < 0.05, f"BAR dG {dG:.3f} != {c} (rigid shift)"
    assert ov[0, 1] > 0.9, f"overlap {ov[0,1]:.3f} should be ~1 for a rigid shift"
    print(f"[selftest] rigid-shift dG = {dG:.4f} kT (expect {c}); overlap = {ov[0,1]:.3f}  OK")

    # self-consistency: identical Hamiltonians -> dG ~ 0, overlap ~ 1
    same = [(rng.normal(0, 1, N),) * 2]
    dG0, _, ov0 = bar_two_state(same, [(rng.normal(0, 1, N),) * 2], igb_A, igb_B)
    assert abs(dG0) < 0.05, f"identical-model dG {dG0:.3f} should be ~0"
    print(f"[selftest] identical-model dG = {dG0:.4f} kT (expect 0); overlap = {ov0[0,1]:.3f}  OK")

    # kT convention sanity
    assert abs(kcals_per_kt(298.0) - 0.5921) < 1e-3, kcals_per_kt(298.0)
    print(f"[selftest] kcals_per_kt(298) = {kcals_per_kt(298.0):.4f} kcal/mol  OK")
    print("[selftest] all checks passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="YAML manifest of host-guest pairs")
    ap.add_argument("--out", default="ddG_gb.csv", help="output CSV path")
    ap.add_argument("--nproc", type=int, default=None,
                    help="parallel sander workers (default: manifest n_jobs, else all cores)")
    ap.add_argument("--scratch", default=None,
                    help="persistent scratch dir for the re-scored mdouts (default ./gb_ddG_scratch); "
                         "cached mdouts are reused on re-run")
    ap.add_argument("--force", action="store_true", help="recompute all re-scores, ignoring the cache")
    ap.add_argument("--selftest", action="store_true", help="verify BAR math locally (no sander)")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.manifest:
        ap.error("provide --manifest (or --selftest)")
    run_manifest(args.manifest, args.out, n_jobs=args.nproc, scratch=args.scratch, force=args.force)


if __name__ == "__main__":
    main()
