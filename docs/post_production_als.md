# Post-production adaptive lambda scheduling (ALS)

ALS adds windows where adjacent states overlap poorly. It runs **after** production: the run
first finishes on the windows in the config (round 0). Then each adjacent pair whose overlap is
too low gets a full-length window in its gap, and Phases 3–7 run again with warm resume, so only
the new windows and the new energy cells are computed. This repeats until no pair is weak or
`als_max_rounds` is reached. Turn it on with `workflow_jobs.adaptive_lambda: true`.

The earlier pre-production pilot (50 ps windows, scheduled before production) has been removed.
Its config keys are dropped with a warning (`config._REMOVED_PILOT_KEYS`).

## The trigger: two-state overlap

A pair is weak when `min(O_ab, O_ba)` from a **two-state** MBAR of just those two states is below
`min_degree_overlap`. The scale is 0 to 0.5: two identical states give 0.5.

The full-N overlap matrix that Phase 7 writes (`*_O_MBAR.h5`) cannot be used for this. In a
dense band of near-identical states, each row spreads over every state, so its superdiagonal is
about 1/N even when neighbours overlap perfectly. On the in-solvent host–guest runs, every row of
the 36-state ligand leg is about 1/36. A full-N threshold of 0.04 flags 448 of 551 runs, almost
all in over-dense ligand and receptor restraint bands, while the real gaps (complex charge
0 → 0.5 → 1 for sampl9 under igb 5) sit orders of magnitude lower.

Subsampling is done per sampled state (each state's own column), so slicing a pair out of the
subsampled full matrix gives the same samples a two-state analysis would use.

## What gets filled

| band | states | axis | legs sharing the list |
|---|---|---|---|
| restraint | complex q=1 anchor → `lambda_window`s; apo `lambda_window`s | log2 conformational exponent | complex, ligand, receptor |
| charge | `electrostatics` q 0 → 1 (complex); q=1 anchor → q 0 (ligand) | charge fraction | complex, ligand |
| lj | `no_interactions` (0) → `lj_x` → `interactions` (1) | LJ scale | complex |
| screening | `interactions` (0) → `screening_x` → `electrostatics` q=0 (1) | SCREEN scale | complex |
| dielectric (gas cycle, opt-in) | gas → `gb_dielectric` → water | 1 − 1/ε | complex, receptor |

- **One shared list per band.** A gap's decision overlap is the minimum over the legs that share
  its list, and one insertion adds the window to all of them.
- **Where the new value goes.** It is the 3-decimal midpoint of the gap. For restraints the
  midpoint is in log2 space, and the orientational partner keeps the constant offset. With an
  explicit `candidate_*_pool`, the candidate nearest the midpoint is used instead.
- **When a gap is not split.** If the split would land on an existing label, or make a half-gap
  narrower than `als_min_step`, the gap is reported as `weak_pool_exhausted` instead. LJ coupling
  is linear with no soft-core, so the LJ floor (0.005) is what stops bisection toward 0.
- **Never filled.** Pairs touching the unrestrained endstate are reported only (`endstate_report`).
  The band ends never move, because insertions land strictly inside a gap. That keeps the
  max-restraint anchor, the Boresch term and the weakest restraint window unchanged.

## Config (`intermediate_states_arguments`)

| key | default | meaning |
|---|---|---|
| `min_degree_overlap` | 0.04 | two-state overlap threshold |
| `als_max_rounds` | 3 | insertion rounds after production; 0 = report overlaps only |
| `als_bands` | restraint, charge, lj, screening | bands that may be filled |
| `als_max_new_windows_per_round` | none | MD-window budget per round (restraint 3, charge 2, lj/screening 1); weakest gaps first |
| `als_min_step` | see `als_refine.DEFAULT_MIN_STEP` | smallest half-gap per band |
| `candidate_{conformational,charge,lj,screening,dielectric}_pool` | empty | explicit candidates |
| `refine_expected_deltaG` | none | round 0 must reproduce this dG (refine mode) |

Validation (`Config._check_adaptive_lambda`):
- **Raises** without `export_intermediate_files`, because warm resume reads the window mdouts and
  scored parquets from disk.
- **Raises** with `md_only`, or with post-analysis or consolidation off.
- **Raises** with user-supplied restraint files or templates.
- **Drops a band with a warning** when the cycle has no such band.

## How a round runs (`workflow_phases.py`)

```
Phase 1 -> Phase 2 (endstate, once) -> als_round(0) -> refine_schedule(0)
als_round(k):       Phase 3 -> Phase 4 -> audit_existing_windows -> Phases 5/6 -> Phase 7 (+ leg_pair_overlaps)
refine_schedule(k): plan -> snapshot -> stop, or extend_schedule -> als_round(k+1) -> refine_schedule(k+1)
```

- **Rebuilding from lists.** Phase 3 and Phase 4 rebuild everything from the extended window
  lists. Directory names and the MBAR order come from the lists, so new values get new
  directories. Existing windows skip MD (`_check_mdout`) and existing cells load from their
  parquet.
- **`audit_existing_windows`.** It compares every existing window's on-disk `restraint.RST` and
  topology with the regenerated ones, and stops the run if any differ. Otherwise old trajectories
  would be scored under a different Hamiltonian.
- **Outputs.** The last round's results are the normal `.cache/<complex>/*.h5`. Round history is
  in `.cache/<complex>/als/`:
  - `round_XX/` holds deltaG, fe, error, O_MBAR, boresch, `pairs.csv` (every adjacent pair, its
    band, overlap and decision) and `plan.json`;
  - `history.csv` has one row per round; `reason` is `inserted`, or the stop reason on the last round;
  - `final.json` holds the stop reason, the final lists, unresolved gaps and endstate gaps. On
    `max-rounds`, the gaps the next round would have filled are listed as `weak_rounds_exhausted`.

## Block-chained post-analysis

With `post_analysis_block_size: K`, blocks were cut by position. An insertion would then shift
every later block and drop cells already scored. So the first insertion records each leg's block
ends (`block_anchor_states`). After that, a block is the span between two anchors, and a new state
joins the one block spanning its neighbours. That block grows from m to m+1 states (2m+1 new
cells) and nothing else changes. A span longer than `2K − 1` is split at its middle, at no cost:
both halves are subsets of cells already scored. Scoring (`runner._post_analysis_pairs`) and
solving (`compute_mbar`) both use `block_mbar.leg_blocks`.

## Refining a finished run (`scripts/prepare_refine.py`)

A finished run is refined from disk without rerunning REMD:

- **Rebuild the endstates.** The 298 K complex and guest trajectories are rebuilt from the REMD
  replicas with the workflow's own cpptraj input. They are checked against what the run scored:
  frame count, the binding-mode frame, and every frame's restraint energy under the strongest
  restraint window (to 0.05 kcal/mol or 1e-4 relative). `--max-mismatched-frames N` lets up to N
  frames differ; they are listed in the manifest.
- **Write the refine config.** It reuses them through `run_endstate_method: False` plus
  `complex_endstate_trajectory` / `ligand_endstate_trajectory`. The endstate keeps its
  `remd_endstate` label, so every scored cell is reused.
- **Protect the originals.** Results go to `cache_directory_output: <W>/refine`, so the original
  `.cache` is never written.
- **Round-0 check.** Round 0 must reproduce the original dG (`refine_expected_deltaG`) before
  anything is inserted.
- **Pinned code.** Each submit script pins `PYTHONPATH` to a frozen checkout and checks its commit.

`scripts/als_census.py` runs the same planner over stored `*_formatted.h5` matrices, read-only,
to pick the threshold and the runs worth refining.
