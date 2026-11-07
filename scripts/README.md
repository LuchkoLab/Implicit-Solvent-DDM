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
| `endstate_seam_overlap.py` | **endstate ↔ `lambda_window` seam overlap, each side under its OWN `restraint.RST`** — the restraint-carrying sibling of the above: 4 `sander imin=5` jobs (both ensembles × both Hamiltonians, `nmropt=1` both sides), then a 2-state MBAR giving ΔG and the min-direction overlap. Use this and **not** `gb_junction_overlap.py` at this seam, which writes `nmropt=0` and would return overlap ≈ 1. Validates both DISANG files before running | yes |
| `gb_endpoint_probe.py` | Re-scores a whole GB ladder under both scoring salt concentrations plus the gas endpoint (the full N×N version of the above) | yes |
| `gb_endpoint_analyze.py` | Builds two MBAR problems from `gb_endpoint_probe.py` output that differ only in scoring `saltcon`, and compares their overlap matrices | no |
| `endstate_window_rmsd.py` | **Is the endstate in the same basin as the weakest restraint window?** — cross-RMSD between the endstate and a `lambda_window` trajectory, to tell a structural mismatch (which ALS cannot fix) from a stiffness gap (which it can) | yes |
| `stride_extended_window.py` | **Glue an extended `lambda_window` run back into 10,000 frames** — strides the finished trajectory and concatenates the extension, so a 2 ns window re-run to 5 ns or 10 ns still hands post-analysis the frame count the rest of the cycle carries | yes |
| `remd_ensemble_clusters.py` | **Do the complex and apo receptor REMD ensembles share basins?** — strips the ligand, clusters both ensembles together, and cross-tabulates cluster membership by origin. Answers whether a REMD floor window could cross the holo↔apo barrier, and emits apo seed frames for a multi-seeded window. On MCL-1: 14 clusters, **0 mixed** | yes |
| `filter_restraints.py` | **Drop restraints touching a selection from a DISANG file** — writes the kept `&rst` blocks through verbatim, terminator included. Used to free a floppy region whose restraint energy dominates the endstate seam without bearing on binding | no |
| `two_state_overlap.py` | **MBAR overlap between the endstate and one restraint window, with and without a freed selection** — no sander: the two states differ only by the conformational restraint, so the reduced potential difference *is* the restraint energy, computed straight from coordinates and the DISANG file | no |
| `extract_remd_temperature.py` | **Pull the target-temperature trajectory out of a finished REMD leg** — cpptraj `remdtraj` over the `remd.nc.00*` replica set, one output per solute under a `remd/` tree, plus first/last-frame restarts, in NetCDF or DCD (`--format`). Also handles the `-rem 0` equilibration leg, where each replica is already a fixed rung and only the endpoints are written (standalone form of `simulations.ExtractTrajectories`) | yes |
| `remd_mixing.py` | **Is the REMD ladder actually mixing?** — per-pair acceptance, round trips, per-rung residence and occupancy from a `rem.log`. Catches the failure acceptance rates hide: a replica trapped on the top rung, which stops every round trip while the other rungs keep swapping | no |
| `dg_vs_experiment.py` | **Computed ΔG vs experiment, and vs the replicates** — reads every `deltaG_*.h5` under one or more run directories, labels each run by the flat-bottom protocol its own `restraint.RST` carries (so mixed-protocol ligands are dropped rather than averaged), and plots replicate-vs-replicate, each run vs experiment, and the published ABFE reference on shared axes | no |
| `lj_step_estimators.py` | **Is the complex LJ step's ΔG trustworthy despite near-zero overlap?** From the cached energies only: BAR, forward (insertion) and reverse (deletion) exponential averaging with bootstrap errors, the cached MBAR value, BAR on each half of the run, and the effective number of frames carrying each one-sided average. Agreement within ~1 kcal/mol means the thin link is not biasing the step; disagreement means it needs intermediate LJ windows | no |
| `pocket_compression_check.py` | **Why does the complex LJ-off → LJ-on step have no overlap even when the ligand stays put?** Compares the `no_interactions` and `interactions` windows (same restraints, only ligand–protein LJ differs): where the conformational-network pairs around the pocket sit in their ±0.5 Å flat-bottom well, how far pocket atoms close on the decoupled ligand, and the restraint energy a narrower well would add. Prints a provisional verdict: A (pocket closes inside the free play → narrow the well), B (walls engaged but soft → force constant), C (network doesn't see it) | no (pytraj) |
| `cycle_audit.py` | **Why do replicates disagree? Where the complex and receptor legs fail to cancel** — four subcommands: `cancel` (per-band complex+receptor sums and per-GB-step mismatch, cache only), `saltbridge` (salt-bridge state of a freed selection's charged residues per GB window and leg), `pose` (the ligand pose each replicate held vs the input), `seam` (receptor endstate → weakest-window gap by region, plus an endstate × reference cross check that tells a bad endstate from a bad reference). MBAR overlap cannot see any of these — each leg is its own MBAR problem | yes (pytraj, not AMBER) |
| `extend_weak_windows.py` | **Does more sampling in the weak restraint windows lower the replicate spread?** `setup` writes one restart job per window at or below `--max-exponent` (default −8.5) continuing it to `--target-ns` (default 20), plus a SLURM array script; each task fails on NaN/Infinity/overflow in its mdout. `analyze` re-solves the band from the `--anchor` window (default −8.0) down to the weakest with MBAR on restraint energies computed from coordinates (states differ only by force constants), for the original frames, original + extension, and extension only, then substitutes each into the replicate's ΔG_bind and reports the SD across replicates. Aborts if its energies disagree with the cached post-analysis matrix | setup: no; runs: pmemd |
| `gb_half_check.py` | **Does CPU pmemd fix the GB band?** `pmemd.cuda` ignores `extdiel`, so a GPU GB-dielectric window samples water. `setup` rebuilds one GB window (default ε=2, "GB half on") as a CPU `pmemd.MPI` job with the same mdin, restraint file and a `cv.in` reconstructed from the mdout NFE echo; `analyze` scores gas, CPU half, GPU half and water under the gas/half/water Hamiltonians with sander and prints BAR for gas→half→water with the CPU and the GPU half window; `setup-band` writes one CPU batch job (`submit.sh`) per GB-dielectric window of each leg plus a `submit_all.sh` | yes |
| `ingb_decouple_check.py` | **Can the ligand be decoupled inside GB water instead of in vacuum?** Builds complex variants (ligand charges 0; + GB screening 0; + ligand-receptor LJ 0) from one structure, runs a single step with pmemd.cuda, pmemd and sander, and checks that the fully decoupled complex reproduces the apo receptor's EGB and EELEC and the receptor + ligand-intramolecular VDW, plus GPU/CPU agreement | yes |
| `in_solvent_states_check.py` | **Are the in-solvent cycle topologies right before any MD?** Builds every `decouple_in_solvent` state (no_interactions, LJ windows, interactions, screening windows, uncharged/charged electrostatics, ligand q=0) through the workflow's own `alter_topology`, runs a single step per engine, and checks the decoupled complex against the apo receptor + ligand, the uncharged ligand's zero GB/EELEC, and that interactions and screening-off leave receptor EGB unchanged | yes |
| `relax_binding_pose.py` | **Pre-step: relax the one input binding pose ONCE, before any replicate** — restrained minimise → heat → equilibrate → minimise under the production GB settings (`igb`/`saltcon`/`extdiel`, not `min.mdin`'s `saltcon=0`), with the ligand heavy atoms + pocket C-alphas held to the input pose in every stage. Writes the complex/receptor/ligand restarts every replicate shares, a pose-RMSD plateau check, and fails on NaN/Infinity mdouts | yes |
| `nfe_rmsd_probe.py` | **Can AMBER give us a GPU RMSD restraint, and is it best-fit?** — stands up `infe=1` + `&pmd` + `MULTI_RMSD` from scratch and checks that it runs, that the CV is rotation/translation invariant and equals cpptraj `rms … mass`, that it applies force rather than only monitoring, and that `cv_ni` does not silently truncate | yes |
| `_run_cb7_overlap.py` | Scratch driver: run the cb7 workflow on local scratch and emit MBAR overlap matrices | yes |
| `gpu_packing_benchmark.py` | **How many windows can share one GPU?** — runs N concurrent `pmemd.cuda` copies of one finished window on a single device (sweep of N, MPS off and on) and reports total and per-copy ns/day, GPU utilisation and memory, CPU time per copy, and whether every copy's EPtot series matches the first run (same seed) | yes (GPU) |
| `run_apo_receptor_md.py` | **Unbiased apo receptor endstate, shared across ligands** — the workflow's minimization (`templates/min.mdin`) then the complex's basic-MD endstate mdin with an empty DISANG and a fixed seed, so one 100 ns 298 K receptor trajectory per replicate can be passed as `receptor_coordinate_filename` (with `--ignore_receptor`) instead of a REMD ladder that unfolds the protein | yes (GPU) |

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

## `stride_extended_window.py` — extend a window's sampling without changing its frame count

> For a **fresh** run, `intermediate_states_arguments.long_restraint_window_ns` (default 10.0) does
> this natively, at the same frame count. It does **not** rescue the endstate seam on its own —
> under the full whole-receptor network a 10 ns floor window still had zero shared support with the
> endstate, because the window is seeded from a holo structure while the endstate is apo. It pays
> off alongside `unrestrained_receptor_mask`: with `':22-32'` freed, endstate overlap was 0.0404 at
> 2 ns (exactly the 0.04 threshold) and 0.0819 at 10 ns. This script stays for extending a run that
> has already finished.

**Question.** `endstate_window_rmsd.py` (above) asks whether the endstate seam's bad overlap is a
*structural* mismatch. This asks the other half: is it a **sampling-time** gap? The two sides of
that seam are sampled for wildly different lengths of time —

| | simulated time | frames | spacing |
|---|---|---|---|
| receptor endstate (`remd/*_298.0K.nc`) | 20 ns/replica | 10,000 | 2 ps (one per exchange) |
| window `lambda_window/.../-14.0` | 2 ns | 10,000 | 0.2 ps (`ntwx=50`, `dt=0.004`) |

Same sample count, 10× less simulated time on the window side — and at the −14.0 rung
*k* = 2⁻¹⁴ = 6.1e-5 kcal/mol/Å², so the restraint is effectively off and that window is a free
receptor given 2 ns to relax, compared against an endstate given 20 ns.

**Why the frame count is fixed at 10,000.** Post-analysis stores one rectangular frames × states
matrix per leg (`*_formatted.h5`, `postTreatment.py:296`), so every state has to contribute the same
number of rows; a window carrying 25,000 or 50,000 frames cannot be scored alongside the ladder. So
the finished 2 ns is strided down and the extension is written at a coarser `ntwx`:

| target | keep from the 2 ns | extension | merged |
|---|---|---|---|
| 5 ns | every 2nd → 5,000 @ 0.4 ps | 3 ns, `nstlim=750000 ntwx=150` → 5,000 @ 0.6 ps | 10,000, 1.5× density step at the seam |
| 10 ns | every 5th → 2,000 @ 1.0 ps | 8 ns, `nstlim=2000000 ntwx=250` → 8,000 @ 1.0 ps | 10,000 on one uniform 1.0 ps grid |

The 5 ns case cannot be uniform: that would need 0.5 ps spacing, i.e. every 2.5th old frame. Both
segments are equilibrium samples of the same ensemble, so the density step does not bias the
ensemble average — it only makes the concatenated series non-uniform in time, which affects pymbar's
statistical-inefficiency estimate and not the free energy.

The stride is **derived from what `extend.mdin` says**, not hardcoded, so editing the extension's
`nstlim`/`ntwx` keeps the arithmetic correct (and the script refuses splits that need a fractional
stride).

```bash
module load amber
W=<run>/MCL-1_receptor-...-IRN_IRN/lambda_window/1.0/78.5/-14.0

# 1. run each extension from inside its own directory — DISANG is ../restraint.RST
cd $W/5ns && pmemd.cuda -O -i extend.mdin -p ../*.parm7 -c ../*_prod_restrt.rst7 \
                        -r ext_restrt.rst7 -x ext_traj.nc -o mdout

# 2. build the 10,000-frame trajectories
python scripts/stride_extended_window.py "$W" --target-ns 5 10
```

Writes `<base>_prod_traj.nc` into each `<N>ns/` directory and verifies the frame count, failing with
a pointer at the extension's `mdout` if it came up short. The output directory is left looking like
a window directory so `endstate_window_rmsd.py`, `gb_junction_overlap.py` and the scoring machinery
find the files where they expect them. `--dry-run` prints the plan and the cpptraj input only.

The `extend.mdin` files are the window's own mdin with `ntx=5`/`irest=1` (continue from the saved
velocities) and the new `nstlim`/`ntwx`/`ntpr`; `igb`, `saltcon`, `extdiel`, `dt`, SHAKE and the
`nmropt`/DISANG restraint block are untouched, and `ig=-1` keeps a fresh Langevin seed.

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

---

## `remd_mixing.py` — is the ladder mixing, or just swapping?

**Question.** Acceptance rate is the number everyone quotes and it is the wrong one on its own. What
the endstate needs is **round trips**: a configuration that walks from the target rung to the top of
the ladder and back, which is what decorrelates the 298 K ensemble the restraints and the windows are
built from. A ladder can swap busily at every rung and deliver zero round trips if one replica gets
stuck at an end.

**What it reports.** Per adjacent pair: attempts, accepted, acceptance, and the energy gap in units
of the fluctuation (`dE/sigma`). Per replica: completed round trips and rungs visited, split into
first/second half so a ladder that degrades mid-run shows up. Per rung: mean and longest residence
in exchange attempts, against the `1/acceptance` a diffusive walk would give. Plus the occupancy
table — every replica should spend ~1/N of its time on every rung.

    python scripts/remd_mixing.py <run>/remd/<system>/rem.log --ps-per-exchange 2.0

`--ps-per-exchange` is `nstlim * dt` from the REMD mdin (500 * 0.004 = 2 ps here) and only converts
round-trip counts into ns.

**Read it like this.** Acceptance under ~15% on a pair is a bottleneck; over ~60% means two rungs are
redundant. A rung whose residence is an order of magnitude past `1/rate` is trapping whatever lands
on it — look at that replica's `Eptot`, and if it sits hundreds of kcal/mol above the rest, the
ladder has lost it for good (`dBeta * dE` of 7-8 is an acceptance of ~1e-4).

---

## `relax_binding_pose.py` — one shared, relaxed starting pose for every replicate

**Why.** The benchmark structures are raw tleap output that has never been through dynamics, and
`restraint_reference: pose_medoid` failed because each replicate derived its reference from its own
stochastic sampling (`research/STATUS.md`, Ruled Out). The invariant that matters is that runs with
different seeds see the *same* reference file. This script makes that file once, from the fixed input
pose. **Run it once per system and point every replicate at the output. Re-running it per replicate
brings the problem back.**

**What it does.** Four `ntr=1` stages, each referenced to the **input** pose (`-ref` never advances,
so the pose cannot drift stage by stage): `01_min` → `02_heat` (10 K → T, 2 fs, TEMP0 ramp) →
`03_equil` (`--equil-ns`, production `dt`) → `04_min`. The restraint (`--pose-weight`, default
5 kcal/mol/Å²) sits on the ligand heavy atoms plus the pocket C-alphas from `build_pocket_atoms`,
the same selection the flat-bottom restraint and pose clustering use. The rest of the receptor relaxes
freely. `ig` is fixed (`--ig`), so a re-run on the same build and GPU reproduces.

    python scripts/relax_binding_pose.py MCL-1_ligand-1.parm7 MCL-1_ligand-1.rst7 \
        -o relaxed/MCL-1_ligand-1 --igb 2 --saltcon 0.3 --equil-ns 2

    # MCL-1: track the floppy regions, which is where an unfinished relaxation shows
    python scripts/relax_binding_pose.py ... --region loop=:22-32@CA --region Cterm=:146-152@CA

    # igb5 benchmark configs
    python scripts/relax_binding_pose.py ... --igb 5

Match `--igb`/`--saltcon`/`--extdiel`/`--rgbmax` to the run's intermediate mdin. `--dt 0.004` (the
default) needs the HMR topology, as production does. `--dry-run` writes the mdins and prints the
commands.

**Outputs.** `<stem>_relaxed.rst7` (→ `intermediate_states_arguments.restraint_reference`),
`<stem>_receptor_relaxed.rst7`, `<stem>_ligand_relaxed.rst7` (atom order of the complex topology, so
check that it matches the separate receptor/ligand `.parm7`), `pose_rmsd.csv` and `relax_summary.txt`.
Per equilibration frame it tracks the ligand heavy-atom RMSD to the input (pocket-superposed), the
receptor C-alpha RMSD (heavy atoms for a host with no backbone), and every `--region NAME=MASK`; the
summary adds the EPtot drift. It warns when any of them rises by more than 0.25 Å between the 2nd
and last quarter of the equilibration.

**The ligand RMSD cannot tell you the relaxation is finished** — the ligand is held, so it stays flat
whatever the protein does. Watch the protein and, for MCL-1, **always pass
`--region loop=:22-32@CA --region Cterm=:146-152@CA`**: on ligand-17 at 2 ns the whole-protein RMSD
rose only 0.20 Å (under the tolerance) while the loop rose 0.64 Å and EPtot fell 44 kcal/mol. The
loop's final shape also matters downstream: it is the conformation the loop ramp will hold the loop
to at the top of the restraint ladder.

**Limits.** It only produces the files. The workflow's `restraint_reference` knows only `lastframe`
and `pose_medoid`, so feeding this structure in as the start point does not by itself make it the
restraint reference.

---

## `cycle_audit.py` — where do the complex and receptor legs stop cancelling?

**Why.** The two protein legs are each ~2,700 kcal/mol and ΔG_bind is their small difference, so the
cycle only works when both legs sample the *same* protein. MBAR overlap cannot see a failure of that
kind: each leg is a separate MBAR problem, and a leg can be well overlapped and precise (per-step errors
~0.05 kcal/mol) for an ensemble that differs from the other leg's. On MCL-1/ligand-17 (2 ns, igb2) the
three replicates gave −5.2 / −1.8 / +18.9 kcal/mol and each failed at a different step. This script
finds which.

    R=fragment-opt-abfe-benchmark/isdmm_runs
    A=fragment-opt-abfe-benchmark/topologies_and_structure/MCL-1/ligand-17/complex/amber
    python scripts/cycle_audit.py cancel     $R/mcl1_7remd_2ns_rep{1,2,3} --system MCL-1_ligand-17
    python scripts/cycle_audit.py saltbridge $R/mcl1_7remd_2ns_rep{1,2,3} --system MCL-1_ligand-17 --free ':22-32'
    python scripts/cycle_audit.py pose       $R/mcl1_7remd_2ns_rep{1,2,3} --system MCL-1_ligand-17 \
        --input-parm $A/MCL-1_ligand-17.parm7 --input-rst7 $A/MCL-1_ligand-17.rst7
    python scripts/cycle_audit.py seam       $R/mcl1_7remd_2ns_rep{1,2,3} --system MCL-1_ligand-17 \
        --input-parm $A/MCL-1_ligand-17.parm7 --input-rst7 $A/MCL-1_ligand-17.rst7 --free ':22-32' --region C-term=:146-152

Each run directory is one replicate (the directory holding `.cache/`). `-o PREFIX` writes CSVs.

**Read it in this order.**

1. **`cancel`** (seconds, cache only). The paired rows (`gas -> gb`, `gb`, `gb -> charge`, `restraint`,
   `restraint -> endstate`) should be small and equal across replicates; a replicate that is off in one
   row names the failing step. The GB per-step table should be ~0 everywhere. A mismatch many times the
   step's MBAR error is a real ensemble difference, not noise.
2. **`seam`** if `restraint -> endstate` is off. `disjoint=True` means no endstate frame looks like a window
   frame (overlap ~0). The region split says whether one floppy region or the whole protein carries the gap.
   In the cross-check, **a high column is a bad reference, a high row a bad endstate.** On ligand-17, rep2's
   column was ~3× the others for every endstate, so its `pose_medoid` reference was the problem, not its REMD.
3. **`saltbridge`** if `gb` is off. `mismatch` is the total-variation distance between the two legs'
   partner-state occupancies (0 same, 1 disjoint). On ligand-17 rep3, ASP25↔LYS27 reached 0.70 at ε=4 while
   rep1/rep2 had every loop salt bridge identical in both legs — the freed loop sampling differently per leg.
   Differing *between replicates* is harmless; differing *between legs* is what fails to cancel.
4. **`pose`** to see how far the held reference pose sits from the input (rep2/rep3: ~7 Å, rep1: 3.2 Å).

**Assumptions.** Receptor residue numbering is the complex's with the ligand removed (checked; the script
exits if not). The receptor endstate is auto-found under `<run>/*/remd/*receptor*/*K.nc`; pass
`--receptor-endstate` (one per run) otherwise. `seam` scores restraint energy straight from coordinates and the
weakest window's `restraint.RST` — exact, since that is the only term differing between the two states (see
`two_state_overlap.py`). The pymbar `****` banner during `seam` is its normal convergence printout.

---

## `pocket_compression_check.py` — is the pocket closing inside the restraints' free play?

**Why.** With the Boresch restraints holding (MCL-1 ligands 17 and 4), the complex `no_interactions → interactions`
step still has overlap 0.002–0.03: with LJ off, ligand–protein contacts close by ~0.35 Å, and the r⁻¹² wall turns that
into a +27 kcal/mol median penalty (0.1% of LJ-off frames resemble LJ-on ones). The conformational network is a
flat-bottom well — zero force within d₀ ± 0.5 Å — so a stronger force constant cannot stop motion inside it. This
script tells whether that free play is where the pocket closes, before any restraint change.

    python scripts/pocket_compression_check.py fragment-opt-abfe-benchmark/isdmm_runs/mcl1_rampbb_10ns_rep1 \
        --system MCL-1_ligand-17

The two windows share one `restraint.RST` and differ only in ligand–protein LJ, so the comparison is controlled (the
script warns if their restraints differ). Pocket-contact atoms are receptor atoms within `--contact` (4.5 Å) of the
ligand in the reference; far control pairs lie beyond `--far` (15 Å) and should not move between windows.

**Read it in this order:** section 1's median shift for `pocket heavy-heavy` (negative = compressed with LJ off) against
its `past_wall` fraction; section 2 confirms the atoms really close on the ligand; section 3 says where the pairs of the
clashing atoms sit (inside the free zone vs past the wall); section 4 compares the restraint energy a `--narrow` (±0.1 Å)
well would put on the LJ-off frames with the LJ penalty they carry now. The verdict line is a heuristic — the tables
are the result. Default `--stride 10` (200 frames per 10 ns window).

**Result on MCL-1 ligands 17 and 4 (verdict C):** the pocket does not deform — pocket pair distances are identical
with LJ on and off (shift ≤ 0.003 Å, 0% past the wall) and clashing atoms sit at d₀ — yet 18–25% of pocket atoms end up
> 0.3 Å closer to the LJ-off ligand. A ±0.1 Å well would add ~1,400 kcal/mol of restraint energy, so narrowing it is out.
Sections 5–7 test the follow-up explanation, the decoupled ligand wobbling as a rigid body inside the harmonic Boresch
restraints: 5 the ligand's pose spread relative to the pocket per window (RMSD, centroid shift, rotation); 6 each Boresch
coordinate's SD per window next to the thermal prediction √(kT/2k); 7 which ligand atoms come within `--lig-contact`
(2.2 Å) of the protein with LJ off, with their lever arm from the anchor L1 (far atoms loaded = rotation about the
anchor). A broader LJ-off pose with near-thermal Boresch SDs points to stiffer Boresch force constants as the fix.

## `nfe_rmsd_probe.py` — can AMBER give us a GPU RMSD restraint, and is it best-fit?

**Question.** We want to replace the pairwise conformational restraint network (82,077 `&rst` terms
on MCL-1) with one best-fit RMSD collective variable under a flat-bottom umbrella, so the restraint
ladder walks a bounded coordinate in Angstrom that terminates exactly at the unrestrained endstate.
Targeted MD (`itgtmd=1`) is the obvious mechanism and is unusable: the AMBER 25 manual lists it under
functionality "missing entirely" from PMEMD (Sec. 23.3), and there is no `sander.cuda`. The surviving
route is AMBER's NFE module, `infe=1` + `&pmd` + `cv_type='MULTI_RMSD'`, which Sec. 26.4.4 says "can
be fully applied in GPU accelerated PMEMD". The manual leaves the load-bearing details unstated.

**What it decides.**

| probe | question | blocking? |
|---|---|---|
| 1 `runs` | does `infe=1` + `&pmd` + MULTI_RMSD start on `pmemd.cuda` with `igb != 0`, alongside a live `nmropt=1` DISANG? | yes |
| 2 `bestfit` | is the CV rotation/translation invariant, and does it equal cpptraj `rms … mass`? | **yes — this is the gate** |
| 3 `bias` | does the restraint apply force, or only monitor? | no |
| 4 `rescore` | does `imin=5` evaluate NFE, and does the bias reach the `ENERGY` column MBAR reads? | no |
| 5 `scale` | does `cv_r` survive at protein size without silent truncation? | yes |

`--probe gate` (default) runs 1, 2 and 5.

**Why probe 2 is the gate.** `MULTI_RMSD` cites Coutsias/Seok/Dill (quaternion RMSD), which *is* an
optimal-superposition method, but the manual never says the NFE driver superimposes before calling
it. A lab-frame CV would fight the Boresch orientational restraints and invalidate the standard-state
correction — the design would be wrong, not merely slow. The probe settles it by evaluating the CV at
step 0 from the same coordinates twice, once against the reference as given and once against a
rigidly rotated and translated copy of it, and cross-checking both against cpptraj. **It does not
need a GPU**, so it can be answered with `--executable sander` before any GPU allocation.

**Why probe 5 exists.** This workflow has already been bitten by an undocumented Fortran array cap:
`igr1`/`igr2` top out at 200 atoms, which is what forced the `@CA` workaround in the flat-bottom COM
restraint (release v1.1.3). There is no documented limit on `cv_ni`/`cv_nr`. HSP90 has 207 CA atoms,
so the probe compares the reported CV against cpptraj rather than merely checking for an error.

**Method.** Each probe writes its own self-contained mdin + `cv.in` + `restraint.RST` into a case
subdirectory and runs AMBER directly — nothing in `implicit_solvent_ddm/` is touched, except that
probe 4 imports the package's own `mdout.min_to_dataframe` so it measures exactly what MBAR would
see. The DISANG is a real but zero-force `&rst`, so `nmropt=1` and NFE are exercised together without
perturbing the energy. `&pmd` is emitted after `&cntrl` and *before* `&wt`/`DISANG`, which is the
placement the workflow's planned mdin injector will use, so the probe validates that choice.

The restrained set degrades the way `restraints.FlatBottom` already does — `@CA` when present, all
heavy atoms otherwise — so cb7 exercises the host–guest fallback path.

**Reading it.** Every probe prints GO or NO-GO with the number it decided on, and the exit status is
non-zero if any blocking probe fails. The `pmd.txt` layout is undocumented, so the CV column is a
defensible guess (`--pmd-column`, default 1); pass `--show-pmd` on the first run to eyeball the raw
head before trusting the parse.

**Existing trajectories.** Pass `--traj` and the script stops generating its own: probe 1 shrinks to
a single step (the only question it still answers is whether the run starts), and probes 2/5 draw
`--frames` check points spread through the real trajectory instead of a 1 ps relaxation. That is the
better test — a CV that is right near zero and wrong at 3 Å would pass a single-point check, so
covering the RMSD range the ladder will actually traverse matters. Probe 4 re-scores the same
trajectory.

    # the design-critical question, CPU, no GPU allocation needed
    python scripts/nfe_rmsd_probe.py --probe 1,2 --executable sander --show-pmd \
        --prmtop implicit_solvent_ddm/tests/structs/cb7-mol01.parm7 \
        --inpcrd implicit_solvent_ddm/tests/structs/cb7-mol01.ncrst \
        --traj  implicit_solvent_ddm/tests/structs/cb7-mol01_300K.nc \
        --workdir nfe_probe_cb7

    # then the GPU question, and the force check
    python scripts/nfe_rmsd_probe.py --probe gate --executable pmemd.cuda \
        --prmtop ... --inpcrd ... --traj ... --workdir nfe_probe_cb7_gpu
    python scripts/nfe_rmsd_probe.py --probe 3 --nstlim 25000 --k 50 --prmtop ... --inpcrd ...

Probe 3 always runs new MD — you cannot show that a force is applied without integrating it.

## `gpu_packing_benchmark.py` — how many windows can share one GPU?

A ~2,500-atom GB window leaves most of a GPU idle, but the workflow gives each window a whole device.
This runs 1, 2, 4, ... copies of one finished window on device 0 at once, first time-sliced and then
under a private CUDA MPS daemon, and writes one CSV row per (MPS, N) cell (rewritten after every cell).
The mdin is rebuilt from the window's own `mdout` echo with `nstlim` (default 125000 = 500 ps) and a fixed
`ig` overridden, so the restraint network and output frequency match production. Every copy uses the same
seed, so `max_abs_dEPtot_vs_first` checks that packing and MPS leave the trajectory unchanged.
`sum_ns_day` (the copies' own mdout ns/day added up) is the steady-state throughput; `total_ns_day`
also charges each copy's ~4 s startup. Ask for one CPU core per copy: `cpu_frac_per_copy` near 1
means each copy holds a core, and with fewer cores than copies the result measures the CPU, not the GPU.

    salloc --partition=gpu --gres=gpu:1 --cpus-per-task=20 --mem=40G --time=00:30:00
    python scripts/gpu_packing_benchmark.py <window dir> --nstlim 50000 --copies 1 2 4 8 12 16 --mps on

MCL-1/ligand-17 (2,471 atoms, 85k restraints) on an RTX 5000 Ada: time-slicing gives no gain; under MPS the
GPU total is 1,735 / 2,726 / 3,327 / 3,480 / 3,489 ns/day at 1 / 2 / 4 / 8 / 16 copies.

## `run_apo_receptor_md.py` — unbiased apo receptor endstate

    python scripts/run_apo_receptor_md.py --prmtop <receptor.parm7> --inpcrd <receptor_relaxed.rst7> \
        --md-mdin input_files/four_fento_basic_md.mdin --out <dir> --ig 1001

Writes `<name>_298.0K.nc` (and a last-frame restart) under `--out`; `--dry-run` writes the inputs and prints
the two pmemd commands. Both steps are checked for completion and NaN energies.
