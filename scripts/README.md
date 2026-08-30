# `scripts/` — standalone analysis scripts and run drivers

Everything here is a **standalone** script: it is not imported by `implicit_solvent_ddm` and nothing
in the package depends on it. Each one is run directly, reads existing run output (or drives a run),
and writes CSV/HDF results. None of them modify package code.

All of them need `implicit_solvent_ddm` importable — either `pip install -e .` from the repo root, or
run them with the repo root on `PYTHONPATH`. Use the `isddm_env` conda environment (pandas, pymbar,
pytables); the AMBER-dependent ones additionally need `sander`/`cpptraj` on `PATH` (`module load amber`).

| script | what it does | needs AMBER? |
|---|---|---|
| `bar_vs_mbar.py` | Adjacent-window BAR vs full MBAR — how much ΔG accuracy the N²→tridiagonal post-analysis cut would cost | no |
| `plot_bar_vs_mbar.py` | Figures for the above: MBAR-vs-BAR correlation, per-system/leg error heatmap, error-vs-overlap | no |
| `frame_stride_mbar.py` | MBAR vs MBAR at fewer **frames** — how much ΔG moves if post-analysis scores 500/1000/5000 instead of all 10,000 frames | no |
| `block_mbar_sweep.py` | Block-chained MBAR accuracy vs **block size K** — how much ΔG moves when the cycle is solved as a chain of K-state blocks instead of one dense N² solve (K=2 is the BAR chain) | no |
| `gb_ddG_bar.py` | ΔΔG between two GB models (OBC / OBC2 / GBn2) by re-scoring end-state trajectories under both | yes |
| `gb_junction_overlap.py` | **GB-band ↔ gas junction overlap, with and without the scoring-`saltcon` fix** — re-scores the two trajectories straddling the junction and reports BAR ΔG + MBAR overlap both ways | yes |
| `gb_endpoint_probe.py` | Re-scores a whole GB ladder under both scoring salt concentrations plus the gas endpoint (the full N×N version of the above) | yes |
| `gb_endpoint_analyze.py` | Builds two MBAR problems from `gb_endpoint_probe.py` output that differ only in scoring `saltcon`, and compares their overlap matrices | no |
| `endstate_window_rmsd.py` | **Is the endstate in the same basin as the weakest restraint window?** — cross-RMSD between the endstate and a `lambda_window` trajectory, to tell a structural mismatch (which ALS cannot fix) from a stiffness gap (which it can) | yes |
| `extract_remd_temperature.py` | **Pull the target-temperature trajectory out of a finished REMD leg** — cpptraj `remdtraj` over the `remd.nc.00*` replica set, one output per solute under a `remd/` tree, plus first/last-frame restarts. Also handles the `-rem 0` equilibration leg, where each replica is already a fixed rung and only the endpoints are written (standalone form of `simulations.ExtractTrajectories`) | yes |
| `_run_cb7_overlap.py` | Scratch driver: run the cb7 workflow on local scratch and emit MBAR overlap matrices | yes |

---

## `bar_vs_mbar.py` — is adjacent-window BAR good enough to replace MBAR?

**Question.** `only_post_analysis` (`runner.py`) re-scores every trajectory under every window in a
leg — a full N² of `sander imin=5` evaluations — because MBAR needs every frame's energy at every
state. That CPU tail, not the GPU MD, dominates the DDM budget (see `PERFORMANCE_TIMING.md`). If a
chain of 2-state BAR calculations between *adjacent* windows is accurate enough, post-analysis only
needs the **tridiagonal**: `3N−2` blocks instead of `N²`. For the cb7 complex leg (N=39) that is
115 vs 1521 (~13×); for a 74-window protein–ligand leg, 220 vs 5476 (~25×).

**Method.** Uses only the already-computed `.cache/<system>/*_formatted.h5` matrices — no MD, no
`sander`. Per (system, replica, leg):

1. read `*_formatted.h5` (cycle-ordered energies, kcal/mol) and divide by `kcals_per_Kt`;
2. run the production front-end once — `pdmbar.detect_equilibration` → `subsample_correlated_data`
   (verbatim from `compute_mbar`, `adaptive_restraints.py:694`);
3. **reference:** full MBAR over all N states → ΔG = f(last) − f(first), plus the overlap matrix;
4. **test:** for each adjacent column pair, slice the 2-column / 2-state sub-block and run
   `pdmbar.mbar()` on it. A 2-state MBAR *is* Bennett acceptance ratio — same self-consistent
   equation, same analytic error — so there is no separate BAR implementation. Sum the chain.

Both estimators consume the **identical** subsampled frames, so the only difference is the estimator.

Two design notes worth knowing before editing it:

- Adjacency comes from `df.columns` as stored, **not** from a rebuilt `CycleSteps`. `_ordered()`
  (`adaptive_restraints.py:608`) already put the columns in canonical cycle order before the file was
  written, and today's `matrix_order.py` includes `receptor_GB_exl_windows` (commit `5e7d0ef`,
  2026-06-24) which post-dates the Dec-2025 data — a re-derived `receptor_order` would not match.
- The formatted file is the **un-subsampled** matrix (MBAR was solved on `df_subsampled`, but
  `df_mbar` is what got written), so step 2 must be redone here. That gives a free correctness check:
  the recomputed full MBAR must reproduce the stored `*_fe.h5` to within `--fe-tol`. If it does not,
  the temperature is wrong — use `--calibrate`.

```bash
python scripts/bar_vs_mbar.py --selftest                          # verify the math, no data needed
python scripts/bar_vs_mbar.py --root <run_dir> --calibrate        # find the temperature, then exit
python scripts/bar_vs_mbar.py --root <run_dir> --systems cb7_adn  # one system, all reps
python scripts/bar_vs_mbar.py --root <run_dir> --nproc 8          # everything
```

`<run_dir>` holds `rep1..repN`, each with `.cache/<system>/*_formatted.h5`, e.g.
`.../different_gbs/igb_2/cb7_re_parameterization`. `--systems` matches substrings, so `cb7_adn`
selects `cb7_adn_ligand_1_Hmass`. Temperature must match the production
`intermediate_args.temperature` (298 K for the igb_2 cb7 runs, and for the alibay MCL-1 runs); it
is not a free scale factor, since MBAR is solved on `u = E/kT`.

**Replicate layouts** (applies to all three discovery-based scripts — `find_tasks` is shared).
Replicate dirs are found with `--rep-glob`, default `rep*`. Two non-default cases:

```bash
# replicate dirs named something else, e.g. isdmm_runs/mcl1_rep{1,2,3}
python scripts/block_mbar_sweep.py --root .../isdmm_runs --rep-glob 'mcl1_rep*'

# --root IS a single replicate (it holds .cache/ directly) -- no glob needed
python scripts/block_mbar_sweep.py --root .../isdmm_runs/mcl1_rep1
```

The glob is tried first; the single-replicate fallback only fires when it matches nothing, so a
tree with both a root-level `.cache/` and replicate dirs still resolves to the replicates. A silent
`no tasks found` almost always means the layout missed one of these two shapes — check with
`python -c "import sys; sys.path.insert(0,'scripts'); import bar_vs_mbar as b; print(b.find_tasks(ROOT, None, None, GLOB))"`
before assuming the caches are missing.

**Outputs** (`--out-prefix`, default `bar_vs_mbar`):

- `*_pairs.csv` — one row per adjacent pair: `dG_bar`, `dG_mbar_pair`, `abs_diff`, and that pair's
  MBAR overlap. This is the diagnostic that shows *where* the chain degrades.
- `*_summary.csv` — one row per (system, rep, leg) plus a per-(system, rep) 3-leg total, with
  `n_evals_mbar` (N²) vs `n_evals_bar` (3N−2) and the resulting speedup.

Per-task results are cached under `--scratch`, so a dropped mount or a killed run resumes.

**Runtime.** The slow step is `pdmbar.detect_equilibration`, which production calls with `nskip=1` —
pymbar then scans every frame as a candidate equilibration start, per state, for 35–39 states per leg.
Kept at the production default for fidelity. Run the full sweep on titan where the data is local, not
across the sshfs mount.

**Caveats.** cb7 is a useful stress case (its complex leg contains the gas→water desolvation cliff,
adjacent overlap ~0.008) but a protein–ligand leg has more adjacent steps for error to accumulate
across. Protein–ligand `_formatted.h5` caches now exist for MCL-1/ligand-1 — all three legs, reps 1
and 2, under `fragment-opt-abfe-benchmark/isdmm_runs/mcl1_rep{1,2}/.cache/MCL-1_ligand-1/` (rep3 has
no cache yet) — so the protein–ligand question can be settled directly rather than bounded from cb7.
The BAR chain error `√Σσᵢ²` assumes independent pairs — mildly optimistic, since adjacent BARs share
trajectories.

---

## `block_mbar_sweep.py` — how much ΔG moves when you solve blocks instead of the full N²

**Question.** MBAR needs a *dense* `u_kn`, so post-analysis re-scores every trajectory under every
window. A band of the matrix cannot be handed to MBAR as one global solve — but the cycle *can* be
solved as a chain of small dense sub-solves that share endpoints. Block size K=2 is exactly the
adjacent-window BAR chain (`3N−2` evaluations); larger K costs more evaluations and pools more states
per solve, which is where a 2-state chain is weakest. For the MCL-1 complex leg (N=43): K=2 → 127
evaluations (14.6×), K=3 → 169 (10.9×), K=4 → 211 (8.8×), K=5 → 249 (7.4×).

This script measures the accuracy cost per K against the dense reference, using only the cached
`*_formatted.h5` — no MD, no `sander`. It is the measurement that decides which K to set in
`intermediate_args.post_analysis_block_size`.

The estimator lives in the package (`implicit_solvent_ddm/block_mbar.py`), not in this script, because
the same partitioning drives production: `runner.only_post_analysis` skips any (trajectory,
Hamiltonian) pair outside `block_mbar.required_pairs`, so a banded run schedules that many `sander`
child jobs instead of N².

Two things to know:

- **`align_bands` (default on)** isolates every sub-leg transition into its own 2-state block, so a
  block never straddles a force-field model switch. On MCL-1 the `igb=6` ↔ ε≈1 pair has measured
  overlap **0.0000**; a wider block containing it is internally disconnected and buys nothing. Note
  this makes blocks *smaller* around junctions, so it slightly *reduces* the evaluation count while
  giving those links less context — use `--no-align-bands` to see the difference.
- **K=2 must reproduce `bar_vs_mbar.py`.** Verified: it matches that script's independent 2-state BAR
  chain to 1e-5 kcal/mol on cb7_adn/rep1.

```bash
python scripts/block_mbar_sweep.py --root <run_dir> --systems cb7_adn --blocks 2 3 4
python scripts/block_mbar_sweep.py --root <run_dir> --blocks 2 3 4 5 7 --nproc 8
```

Outputs `*_legs.csv` (per system/rep/leg/K) and `*_totals.csv` (3-leg total + delta vs the dense
reference). Per-task results are cached under `--scratch`, so a dropped mount resumes.

---

## `frame_stride_mbar.py` — how much ΔG moves when you score fewer frames

**Question.** Orthogonal to the N² question above. Each `sander imin=5` job re-scores **every frame**
of its source trajectory — production writes `ntwx=250` over 2.5 M steps, so 10,000 frames per job —
and the full N² matrix pays that 10,000 times over. Measured on MCL-1/ligand-1: 4,344 s per job,
4,208 CPU-h per replicate, 23.4 h on 180 cores (`research/STATUS.md`). But `subsample_correlated_data`
runs *after* the expensive pass and discards everything inside the correlation time. If striding to
500 frames leaves ΔG unchanged, post-analysis gets ~20× cheaper — **multiplicatively** with the
tridiagonal cut that `bar_vs_mbar.py` measures.

**Method.** MBAR vs MBAR; the only difference between arms is how many frames enter. Uses only the
cached `*_formatted.h5` — no MD, no `sander`. Each file is read **once** (they are ~110 MB and
usually on a network mount, so re-reading per target would dominate), then per frame target:

1. **stride** the raw matrix to ~`target` frames per state — `iloc[::stride]` *within each sampled
   state*, which is what changing production `ntwx` (or a `cpptraj` pre-stride) actually produces;
2. re-run the **full production front-end** on the thinned data — `detect_equilibration` →
   `subsample_correlated_data`. This must be redone per arm: `g` is a property of the thinned series;
3. solve the **full MBAR** over all N states → ΔG = f(last) − f(first).

The largest frame count is the reference arm (it is the complete data set, so it reproduces
production); every other arm is reported as a delta against it.

Two things to know before editing:

- Striding **must** group by `df.index`. The rows are stacked per sampled state, so a global
  `[::stride]` would thin states unevenly and break pymbar's `sum(N_k) == n_samples` invariant.
- It uses `bar_vs_mbar.MBAR_PROTOCOL`, not pymbar's default — pymbar's `(hybr, adaptive)` default
  stalls on this data and returns non-converged free energies (`research/STATUS.md`). Every row
  carries `mbar_grad_norm`; check it before believing any ΔG. `--warm-start` initializes from the
  arm's own adjacent-pair BAR chain (identical treatment across arms; safe only because the gradient
  norm is reported — MBAR's objective is strictly convex, so a *converged* solve is the unique
  optimum regardless of where it started).

**Side benefit:** `detect_equilibration` also returns the statistical inefficiency, so the output
answers "what stride is actually justified?" directly — `mean_g` is the correlation length in frames
and `mean_Neff` is how many uncorrelated samples MBAR really had. Read those before choosing a stride
for a new system; `g` is system-dependent and does **not** transfer between host–guest and
protein–ligand.

```bash
# headline: 500 frames vs all 10,000, one system, every replica
python scripts/frame_stride_mbar.py --root <run_dir> --systems cb7_adn --frames 10000 500

# full sweep
python scripts/frame_stride_mbar.py --root <run_dir> --frames 10000 5000 1000 500 --nproc 8
```

Outputs `*_legs.csv` (per system/rep/leg/target), `*_totals.csv` (3-leg total + delta vs reference)
and `*_pivot.csv`. Per-task results are cached under `--scratch`, so a dropped mount resumes.
Boresch and flat-bottom terms are analytic and frame-count-independent, so they cancel in every
delta and are omitted — same convention as `bar_vs_mbar.consolidate`.

---

## `plot_bar_vs_mbar.py` — figures for the BAR-vs-MBAR comparison

Reads the two CSVs `bar_vs_mbar.py` writes and produces three PNGs:

```bash
python scripts/plot_bar_vs_mbar.py                        # reads ./bar_vs_mbar_{summary,pairs}.csv
python scripts/plot_bar_vs_mbar.py --prefix bar_vs_mbar --outdir figures
```

- `*_correlation.png` — MBAR vs BAR ΔG, one panel per leg plus the 3-leg total, points are
  (system, replica), dashed line is y = x. **Read RMSE/MAE, not R²** — ΔG spans >100 kcal/mol
  across systems, which inflates R² to ~0.9999 regardless of estimator agreement.
- `*_heatmap.png` — system × leg, mean signed (BAR − MBAR) over replicas, diverging about 0.
- `*_overlap.png` — per adjacent pair, |BAR − MBAR| vs that pair's MBAR overlap. The mechanistic
  plot: shows the error concentrating at the low-overlap steps.

It also prints a table view of the same summary statistics.

---

## `gb_ddG_bar.py` — ΔΔG between two GB models via BAR

For a host–guest pair whose end-state trajectories were sampled under two GB models (e.g. OBC `igb=2`
and OBC2 `igb=5`) for each leg, this re-scores **every frame under both models** with a `sander
imin=5` single point, runs a 2-state BAR per leg, and combines:

```
ddG_bind(A→B) = dG_complex(A→B) − dG_host(A→B) − dG_guest(A→B)
```

It reports the per-leg 2-state overlap (the OBC↔OBC2 phase-space overlap) and can compare against the
ABFE-derived difference `dG_bind(B) − dG_bind(A)`.

```bash
module load amber                                       # sander + cpptraj on PATH
python scripts/gb_ddG_bar.py --selftest                 # verify the BAR math, no sander needed
python scripts/gb_ddG_bar.py --manifest scripts/gb_ddG_ign2igb8.yaml --out ddG_gb2togb8.csv
```

**Manifests in this directory:**
- `gb_ddG_manifest.example.yaml` — igb=2 → igb=5 (OBC → OBC2), annotated template
- `gb_ddG_ign2igb8.yaml` — igb=2 → igb=8 (OBC → GBn2), with per-model `parm_A`/`parm_B`

**Replica handling (read before quoting results).** By default it runs BAR **separately per
replica** — pairing replica *r* of ensemble A with replica *r* of ensemble B, one BAR per leg,
combining the three legs into one ΔΔG per replica — and reports **mean ± SEM across replicas**.
Replicas are *not* pooled. Setting `n_bootstrap > 0` additionally reports the all-replicas-pooled BAR
and an across-replica bootstrap.

**Per-model radii.** GB radii live in the prmtop, not the mdin, so `igb=8` on an mbondi2 prmtop is
not real GBn2. When the two models need different radii, give `parm_A` and `parm_B` on a leg instead
of a single `parm`. Make the mbondi3 copy once with `parmed`: `changeRadii mbondi3` → `outparm`.

Re-scored mdouts are cached in a persistent scratch dir and reused on re-run; use `--force` to
recompute.

---

## `gb_junction_overlap.py` — the GB-band ↔ gas junction, with and without the saltcon fix

**The bug.** The GB-dielectric band's MD is written by `generate_extdiel_mdin` (`mdin.py:16-42`) with
`saltcon=0.0`, because with salt the GB polar term is no longer linear in λ = 1 − 1/ε. But the MBAR
matrix is filled by `post_mdin`, built at `mdin.py:67-69` with **no `saltcon` argument** — so the
guard at `mdin.py:216` never fires and the user's `saltcon` (0.3) survives into scoring. Every
`gb_dielectric` window is therefore **sampled salt-free and scored with salt**.

With `saltcon > 0` the GB prefactor is `(1/intdiel − exp(−κ·f_GB)/extdiel)`, which does **not** vanish
at `extdiel = 1`. The scored ladder never reaches vacuum — it retains ~350 kcal/mol of Debye-screened
GB energy as ε → 1 — while the adjacent gas state (`igb=6`) has `EGB = 0` exactly. That cliff is what
collapses the overlap at the junction.

**Measured (MCL-1 rep1, receptor leg, 100 frames):**

| scoring | ΔG kcal/mol | err | overlap |
|---|---|---|---|
| `saltcon=0.3` (production) | +370.65 | 313.15 | 3.52e−08 |
| `saltcon=0.0` (fix) | **+40.95** | **0.22** | **6.33e−02** |

The production `.h5` itself gives +369.66 / 1.12e−08, so the `0.3` run reproduces it — that is the
control that validates the harness. `igb=6` is **not** the problem: it returns byte-identical energies
to `igb=2, extdiel=1.0`.

**Usage.**

```bash
export SANDER=$AMBERHOME/bin/sander
export CPPTRAJ=$AMBERHOME/bin/cpptraj

# receptor leg: junction is gb_dielectric(eps=1.0204) -> no_gb
python scripts/gb_junction_overlap.py --leg receptor \
    --root <run>/mcl1_rep1/MCL-1_receptor-ligand-1_with_mcl1_uxU_uxU

# complex leg: cycle runs gas -> water, so the junction is interactions -> gb_dielectric(eps=1.0204)
python scripts/gb_junction_overlap.py --leg complex \
    --root <run>/mcl1_rep1/MCL-1_ligand-1

# any other junction
python scripts/gb_junction_overlap.py --gb-dir <path> --gas-dir <path> --eps 1.0204081632653061

# print the mdins and sander commands without running them
python scripts/gb_junction_overlap.py --leg complex --root <path> --dry-run
```

6 sander `imin=5` jobs. `--stride 100` (default) keeps every 100th frame; ~15 min for a 2444-atom
system on one core. Each job is cached on a (trajectory, igb, params) fingerprint, so it resumes.

**Gotchas this script handles for you.** GB settings are read from the GB state's own `mdout` rather
than assumed — in particular `rgbmax`, which production omits so AMBER defaults it to 25 Å; hardcoding
999 changes the Born radii and silently invalidates the comparison. (`gb_ddG_bar.py`'s `_GB_DEFAULTS`
ships `rgbmax=0, saltcon=0.3, cut=9999`, all three wrong for this system.) It writes `nmropt=0`: both
junction states share a byte-identical `restraint.RST`, so the restraint energy cancels in every
difference. The production *scoring* `saltcon` is **assumed 0.3** — it is not recorded in the MD
`mdout`; grep it from `post_processing/.../mdout` and pass `--saltcon-prod` if yours differs.

---

## `endstate_window_rmsd.py` — is the endstate in the same basin as the weakest restraint window?

**Question.** On MCL-1 rep1 the receptor `endstate → lambda_window(-14.0)` transition has an MBAR
overlap of 0.0039 — worst in the cycle, ~60× below the median — with ΔG = +5.96 kT and an error of
0.352 kT, 60× the next step's 0.006.

That shouldn't happen. Force constants are `2**exponent` (`workflow_phases.py:920`), so the −14.0
window runs at k = 6.1e-5 kcal/mol/Å² — the restraint is off. The ladder itself behaves (ΔG roughly
doubles as k doubles), and −14.0 is the *best*-connected state going forward (overlap 0.528 to
−13.0). But the endstate seam has the **smallest** change in k of any step and costs **more than
double** the next one. So the endstate is not the k→0 limit of the ladder — it's a different
ensemble. The receptor endstate is built by a separate phase (`user_defined_endstate` /
`run_endstate`, trajectory `*_basicMD_traj.nc`) and `apo_endstate_dirstruct` sets `runtype: "remd"`.

**Why it matters.** The two diagnoses have opposite fixes. A stiffness gap is solved by inserting
windows (ALS). A basin mismatch is not — subdividing 0 → 6.1e-5 gives sub-windows that all inherit
the same mismatch. This script tells them apart before you spend a schedule on it.

**Method.** No MD, no MBAR — four `cpptraj` passes over two existing trajectories:

1. average structure of each ensemble;
2. per-frame RMSD of both to the **restraint reference** (the coordinates the restraint pulls
   toward), showing whether either has drifted off it;
3. cross RMSD — each ensemble against the *other's* average;
4. the separation: average-to-average.

Verdict compares separation to within-basin spread: `≥2×` → different basins (ALS won't help);
`<1×` → same basin, so look at the restraint definition and at what the endstate Hamiltonian
actually is instead.

The mask defaults to the atoms the restraint file really restrains, parsed from the `iat=` records
of `restraint.RST` and collapsed to ranges — on the MCL-1 receptor that is all 2,444 atoms
(`@1-2444`), which is itself worth knowing: it explains why the low-k ladder still costs ~119
kcal/mol despite per-atom force constants near zero.

```bash
module load amber        # or export CPPTRAJ=$AMBERHOME/bin/cpptraj
python scripts/endstate_window_rmsd.py \
    --leg-root <run>/mcl1_rep1/MCL-1_receptor-ligand-1_with_mcl1_uxU_uxU \
    --endstate-traj <path>/MCL-1_receptor-ligand-1_with_mcl1_uxU_uxU_basicMD_traj.nc
```

`--window` picks a different exponent (compare against a healthy seam), `--stride` subsamples,
`--dry-run` prints the cpptraj input without running it.

---

## `_run_cb7_overlap.py` — scratch workflow driver with overlap output

Mirrors `conftest.run_workflow` but (1) turns on `plot_overlap_matrix`, (2) lets `NSTLIM` override
the intermediate MD length, and (3) routes **all** outputs and the Toil jobstore to a local scratch
dir (default `/tmp/cb7_als_run`) instead of the sshfs mount.

The local filesystem supports xattrs natively, so macOS does not create the `._` AppleDouble sidecars
that the workflow's `os.listdir`-based file discovery would otherwise pick up (the
`FileNotFoundError: ._charge_*.parm7` failure). Inputs are still read from the mount.

---

## Related scripts still at the repo root

These are **tracked in git** and were left in place to avoid churn on a branch with other pending
work. They belong here too — move them with `git mv` when convenient.

- `timing_report.py` — parses `[TIMING]` records from a run log into a per-phase wall-clock +
  CPU-h/GPU-h breakdown (`endstate` / `adaptive` / `intermediate_md` / `post_analysis` / `mbar`).
  Handles partial and hung logs.
- `_run_cb7_gpu.py` — driver: cb7 DDM on GPU+CPU, demonstrating the merged MD→post pipeline overlap.
- `_run_cb7_pilot.py` — driver: cb7 with the ALS pilot (Phase 4.5) enabled; Step 5a smoke test.
- `_run_cb7_pilot_real_production.py` — driver: real production run (4 fs / 10 ns windows) with the
  ALS pilot enabled.

`new_scheduler.py` (untracked, repo root) is a **rejected dead end** — its per-runner round-robin GPU
assignment double-books GPU 0. See "Ruled Out / Dead Ends" in `research/STATUS.md`. Build on
`runner.py` instead.
