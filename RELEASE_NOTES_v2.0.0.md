# Release Notes - Implicit Solvent DDM v2.0.0

## Summary

2.0.0 is the first release since 1.1.3. It adds an in-solvent ligand decoupling cycle, adaptive
lambda scheduling (ALS) that runs after production, block-chained MBAR, GPU window packing and a
set of restraint options aimed at protein-ligand systems. It also fixes the receptor endstate
trajectory, which had been wired to the ligand trajectory since 1.1.1. Several defaults changed in
ways that alter the computed ΔG, and output trees written by earlier versions cannot be
re-analysed. Read [Breaking changes and migration](#breaking-changes-and-migration) before
upgrading.

The new methods are opt-in:

| feature | switch | default |
|---|---|---|
| in-solvent decoupling cycle | `decouple_in_solvent` | off (gas-phase cycle) |
| post-production ALS | `workflow_jobs.adaptive_lambda` | off (static window schedule) |
| block-MBAR | `post_analysis_block_size` | `None` (full N² MBAR) |
| RMSD collective-variable restraints (experimental) | `rmsd_tolerance_windows` | empty (pairwise network) |

The full list of changes, with commit references, is in `CHANGELOG.md`.

## Highlights

### In-solvent decoupling cycle
With `decouple_in_solvent: true` the ligand is decoupled inside GB water (charges 0, GB `SCREEN` 0,
ligand-receptor LJ 0) instead of in the gas phase. The cycle has no gas state and no GB-dielectric
band, so every complex and receptor window can run on `pmemd.cuda`, which ignores `extdiel` in GB
MD. The end states are the same as in the gas-phase cycle, so the result is still an absolute
binding free energy. `screening_windows` adds ligand `SCREEN` windows, and `lj_scale_windows` (also
available in the gas-phase cycle) adds ligand-receptor LJ windows on the complex leg. See
`docs/in_solvent_cycle.md`.

```yaml
workflow:
  intermediate_states_arguments:
    decouple_in_solvent: True
    lj_scale_windows: [0.2]
    screening_windows: [0.5]
    gb_extdiel_windows: []          # must be empty
    charges_lambda_window: [0.5]    # 0 and 1 are always added
```

The intermediate mdin template must have `gbsa=0` (or no `gbsa`), and `remove_GB_solvent_receptor`
is forced off.

### Post-production ALS
With `workflow_jobs.adaptive_lambda: true`, the run first finishes on the configured windows
(round 0). After each round, ALS computes the two-state MBAR overlap `min(O_ab, O_ba)` of every
adjacent pair in each leg. A gap is weak when that overlap is below `min_degree_overlap` (0.04);
when several legs share the gap's window list, one weak leg is enough. Each weak gap in a band
listed in `als_bands` gets one new window strictly inside it, at the midpoint of the gap (log2
exponent for restraints, 1 - 1/ε for the dielectric band) or at the `candidate_*_pool` value
nearest the midpoint. A gap is not split when half of it is narrower than `als_min_step`, when the
value would land on an existing window, or when the pool has no value inside it. The restraint,
charge, LJ and screening bands are filled by default; add `dielectric` to `als_bands` for the gas
cycle's GB-dielectric band. Pairs outside these bands are never filled: pairs that touch the
unrestrained endstate, and in the gas cycle the ligand's water-to-gas step. The new windows run at
full length from the same mdin template, and Phases 3-7 run again with warm resume, so only the new
windows and energy cells are computed.

ALS stops when a round inserts nothing (no weak gap is left in an enabled band, or none of the weak
gaps can be filled) or after `als_max_rounds` (3) insertion rounds. With `als_max_rounds: 0` it
reports the overlaps and inserts nothing. `als_max_new_windows_per_round` caps the MD windows added
per round, weakest gaps first. Round history is written to `.cache/<complex>/als/`:
`round_XX/pairs.csv` gives every pair's band, overlap and decision, and `final.json` gives the stop
reason, the gaps still weak and the weak endstate pairs. See `docs/post_production_als.md`.

### Reusing finished endstate trajectories
With `workflow_jobs.run_endstate_method: false` and `endstate_method` set to `remd` or `basic_md`,
a run skips endstate sampling and uses the endstate trajectories of a finished run. They go under
`endstate_parameter_files`:

```yaml
workflow_jobs:
  run_endstate_method: False
endstate_parameter_files:
  complex_parameter_filename: path/to/complex.parm7
  complex_coordinate_filename: path/to/complex.rst7         # the run's starting structure
  complex_endstate_trajectory: path/to/complex_298K.nc
  ligand_parameter_filename: path/to/ligand.parm7           # required here
  ligand_endstate_trajectory: path/to/ligand_298K.nc
  receptor_parameter_filename: path/to/receptor.parm7
  receptor_coordinate_filename: path/to/receptor_298K.nc    # the receptor endstate trajectory
  ignore_unique_naming: True                                # stable directory names
```

Config load requires both `*_endstate_trajectory` files, each with at least two frames, and
`ligand_parameter_filename`. The receptor has no `*_endstate_trajectory` key: its endstate is read
from `receptor_parameter_filename` and `receptor_coordinate_filename`. If those two are unset, the
workflow writes the receptor from the single-frame starting structure, and that one frame becomes
the receptor endstate. The endstate keeps its method's label (`remd_endstate` or
`vanillaMD_endstate`), so cells the finished run already scored are reused. See also
`docs/config.rst`, section Reusing finished endstate trajectories.

### Refining finished runs with ALS
`scripts/als_census.py` and `scripts/prepare_refine.py` apply ALS to finished runs without
rerunning REMD. They were written for the lab's in-solvent host-guest campaign and assume its
layout. Runs laid out differently need the scripts adapted.

`als_census.py` picks the threshold and the runs worth refining. It is read-only on the runs, and
it needs configs named `<prefix>-<guest>.yaml` that set `hardware_parameters.cache_directory_output`
explicitly.

- `als_census.py scan --root <runs> --out <dir>` reads the configs under
  `<runs>/igb_<gb>/<system>/in_solvent/rep<rep>/config_files/`. From each run's stored
  `*_formatted.h5` it solves the two-state overlap of every adjacent pair in every leg, and it caches
  one table per leg in `<dir>`. `--gbs`, `--systems` and `--reps` default to the campaign's values
  (1 and 5; `acd`, `bcd`, `cb7`, `oah`, `sampl9`; 1 to 3).
- `als_census.py plan --out <dir>` applies the ALS planner to the cached tables at several
  thresholds (`--thresholds`, default 0.01 to 0.05), with the default bands, midpoint splits and no
  per-round cap. It writes the would-be insertions with a core-hour estimate and, for each
  threshold, `targets_t<threshold>.txt`, the configs whose plan inserts at least one window.

`prepare_refine.py` takes such a target list, or config paths, and turns each run into a refine
job. Besides the config naming above, it assumes:

- REMD endstates with 8 replicas, `remd.nc.001` to `remd.nc.008`, and cpptraj on `PATH`;
- a run scored with the full N² MBAR (no `post_analysis_block_size`), with its window directories
  and scored cells on disk (`export_intermediate_files: true`);
- configs that also set `hardware_parameters.working_directory` and
  `endstate_parameter_files.ligand_parameter_filename` explicitly;
- a SLURM script `<working_directory>/ddm_submit/<guest>.sh` with an `export PYTHONNOUSERSITE=1`
  line and a `"$CONDA_PREFIX/bin/run_implicit_ddm.py"` line, which it copies and pins to a frozen
  checkout.

It rebuilds the complex and ligand endstate trajectories from the REMD replicas, plus the
receptor's when the run ran its own receptor REMD, and checks them against what the run scored. It
then writes a refine config that reuses them (see
[Reusing finished endstate trajectories](#reusing-finished-endstate-trajectories)) and sends
results to `<working_directory>/refine`. Round 0 of the refine run has to reproduce the original
ΔG before anything is inserted.

### Block-MBAR
`post_analysis_block_size: K` solves each leg as a chain of K-state MBAR blocks and runs only the
`sander imin=5` evaluations those blocks need. K = 2 is the adjacent-window BAR chain, 3N - 2 cells
instead of N². Under ALS, a new window joins exactly one block and no scored cell is lost.

### Throughput and GPU use
- Each window's post-analysis is a follow-on of its own MD job, so CPU post-analysis fills free
  cores while GPU MD is still running.
- `hardware_parameters.windows_per_gpu: k` packs k `pmemd.cuda` windows on each GPU under
  `single_machine`. It needs CUDA MPS (`nvidia-cuda-mps-control -d`). Four per GPU gave about 1.9x
  the window-MD throughput on an RTX 5000 Ada.
- `hardware_parameters.remd_accelerators` lets the complex and receptor REMD jobs reserve GPUs.
  `hardware_parameters.remd_executable` sets a separate binary for the REMD legs, for example a CPU
  build when AMBER has no `pmemd.cuda.MPI`.
- `hardware_parameters.mbar_accelerators` (default 0) sets the GPUs per MBAR job. In 1.1.3 each
  MBAR job held `num_accelerators` GPUs although pymbar runs on CPU; now MBAR runs on CPU and the
  GPUs are free during the analysis tail.
- `intermediate_states_arguments.gb_dielectric_cpu_cores: n` runs the gas cycle's GB-dielectric
  windows on CPU `pmemd.MPI` with n ranks while the other windows stay on the GPU. `pmemd.cuda`
  ignores `extdiel` in GB MD, so without it those windows sample full water (see migration item 5).
- `hardware_parameters.export_intermediate_files: false` keeps intermediate outputs in the Toil
  jobStore instead of the network output directory.
- `workflow_jobs.md_only: true` runs the intermediate MD and stops. Running again with it off
  scores the windows without re-running MD.
- `[TIMING]` records in the Toil leader log feed `scripts/timing_report.py`, which gives a
  per-phase wall-clock and CPU-h/GPU-h breakdown.

### Restraint options
- `restraint_reference`: `lastframe` (default), `pose_medoid` (medoid of the dominant ligand pose
  cluster, see `pose_cluster_*`), or a path to a single-frame complex restart shared by every
  replicate.
- `unrestrained_receptor_mask` drops receptor atoms from the conformational network, and
  `receptor_ramp_mask` ramps them instead. The two can act on different atoms, for example
  `receptor_ramp_mask: ':22-32'` with `unrestrained_receptor_mask: '@H*'` for an H-free receptor
  network.
- `boresch_anchors` (`pocket_backbone` by default, see below) and `boresch_pocket_radius`.
- `exponent_boresch_distance_forces`, `boresch_distance_force_constant` and
  `boresch_distance_offset` set the ladder that releases the Boresch distance term. Left unset, the
  distance term follows `exponent_conformational_forces`, as before.
- `workflow.restraints.pocket_radius` (8.0 Å) defines the binding site used as the flat-bottom
  receptor group (see below).

### Per-leg overlap matrices
Phase 7 writes the MBAR overlap matrix of every leg to `.cache/<complex>/` as `*_O_MBAR.h5`, with
states labelled in cycle order. `plot_overlap_matrix` adds a heatmap PDF per leg. 1.1.3 wrote only
the complex heatmap.

### Experimental: RMSD collective-variable restraints
A non-empty `rmsd_tolerance_windows` replaces the pairwise conformational network with an NFE
`MULTI_RMSD` restraint laddered in tolerance (Å) at a fixed force constant. The post-analysis
restraint term is computed analytically. This mode cannot be combined with `receptor_ramp_mask`
or `unrestrained_receptor_mask`, and ALS does not insert restraint windows in it. It is not
validated for production. See `script_examples/cb7_rmsd_restraints.md`.

## Breaking changes and migration

The sections below cover configs and analyses from 1.1.3 and from `main` after 1.1.3. A config
that does not set any of the keys named here still loads, but its result can differ.

### 1. The flat-bottom term is no longer in the total ΔG
The complex leg already ends at the flat-bottom-restrained endstate, so adding
F(FB) - F(noFB) counted the confinement cost twice. `flat_bottom_contribution` is still reported
in the deltaG output as a wall-contact diagnostic.

- **Affected:** every total ΔG computed by an earlier version.
- **Migration:** no config change. To compare an older total with a 2.0.0 one, subtract the
  `flat_bottom_contribution` stored in the same deltaG output from the older total.

### 2. The flat-bottom receptor group defaults to the binding site
When `workflow.restraints.restrained_receptor_atoms` is not set, the receptor side of the endstate
flat-bottom restraint is now the CA atoms of residues within `workflow.restraints.pocket_radius`
(new key, 8.0 Å) of the ligand. For a receptor with no CA atoms (host-guest) it is the host heavy
atoms within that distance of the ligand. The radius grows in 2 Å steps, up to 16 Å, until at
least 10 atoms are selected. The site is taken from the starting structure, or from
`restraint_reference` when that is a file. In 1.1.3 the group was every receptor CA atom, or every
receptor heavy atom for a host.

- **Affected:** configs that leave `restrained_receptor_atoms` unset, when the selected pocket is a
  strict subset of the 1.1.3 group. That is the usual case for a protein receptor, and there the
  restrained endstate ensemble changes, and with it the ΔG. A small host can be covered whole: on
  the CB7/M01 example the 8 Å pocket already holds all 84 CB7 heavy atoms, so nothing changes.
- **Migration:** to keep the 1.1.3 group, set it explicitly:

  ```yaml
  workflow:
    restraints:
      restrained_receptor_atoms: ":REC & @CA"     # protein: every receptor CA atom
      # restrained_receptor_atoms: ":CB7 & !@H*"  # host: every host heavy atom
  ```

  Replace `:REC` or `:CB7` with your `receptor_mask`.
  `script_examples/config_files/in_solvent_paper.yaml` sets `":CB7 & !@H*"` explicitly to match the
  paper runs; for CB7 it selects the same atoms as the default.

### 3. Boresch anchors and dihedral bounds
`boresch_anchors` defaults to `pocket_backbone`: receptor anchors R1-R3 come from backbone CA/C/N
atoms of residues within `boresch_pocket_radius` (10 Å) of the ligand, excluding freed or ramped
atoms. The old rule (any receptor heavy atom, longest lever arm wins) is `legacy`. Separately, the
three Boresch dihedral restraints now have bounds of target ± 180° instead of 0°-360°, so a target
near 0° or 360° is no longer at the edge of its bounds.

- **Affected:** protein receptors get different anchor atoms, and with them different restraint
  files and a different Boresch standard-state term. Host-guest receptors (fewer than three CA
  atoms) always use `legacy`, so only the dihedral bounds change for them.
- **New check:** for a protein receptor, `pocket_backbone` checks that the receptor atoms come
  first in the complex topology, and that at least three pocket backbone atoms are left after the
  freed or ramped atoms are excluded. If not, the Boresch restraint job raises `ValueError`, after
  the endstate sampling (REMD) has already run. Both modes, like 1.1.3, number the anchor atoms as
  if the receptor came first. `legacy` and 1.1.3 do not check the order, so with a ligand-first
  topology they pick the wrong anchor atoms.
- **Migration:** put the receptor first in the complex topology before starting a run. Set
  `boresch_anchors: legacy` in `intermediate_states_arguments` to keep the old anchor choice. The
  dihedral bounds have no switch.

### 4. A 10 ns floor restraint window by default
`long_restraint_window_ns` defaults to `10.0`: the receptor leg's lowest conformational-restraint
window runs for 10 ns. The complex and ligand legs are unchanged. The window's frame count is
unchanged (`ntwx`/`ntpr` are rescaled), and a window that is already longer is not shortened.

- **Affected:** any config without `long_restraint_window_ns`. On short test ladders this one
  receptor window dominates the run time.
- **Migration:** set `long_restraint_window_ns: null` to keep every window at the mdin template
  length, as `implicit_solvent_ddm/tests/input_files/config.yaml` does.

### 5. The GB-dielectric band uses `saltcon=0`
The `gb_dielectric` windows of the gas-phase cycle now run their MD and their single-point scoring
with `saltcon=0`, which keeps the GB polar term linear in λ = 1 - 1/ε so that the band reaches
vacuum at ε = 1. All other states keep the user's `saltcon`.

- **Affected:** gas-cycle runs with a non-empty `gb_extdiel_windows` and a non-zero `saltcon`.
  1.1.x sampled the band with salt, so those windows do not match the 2.0.0 Hamiltonian. Feature
  builds between c009e30 and 40e45bb sampled the band salt-free but scored it with salt. On MCL-1
  that gave the receptor's GB-band/gas junction ΔG = +370.65 ± 313.15 kcal/mol, against
  +40.95 ± 0.22 when scored salt-free.
- **Also affected, independent of salt:** gas-cycle band windows whose MD ran on `pmemd.cuda`, in
  1.1.x or in 2.0.0 without `gb_dielectric_cpu_cores`. `pmemd.cuda` ignores `extdiel` in GB MD, so
  those windows sampled full water, while `sander` scored them at the reduced dielectric.
- **Migration:** rerun the band in a fresh output tree, for 1.1.x results and for those feature
  builds alike. 2.0.0 cannot re-analyse their trees, because they predate the endstate label change
  (item 8). `scripts/gb_junction_overlap.py` re-scores only the band-gas junction of an existing
  tree, with and without salt, as a check; it does not give a corrected total. `saltcon` is not part
  of the MBAR state key, so a run resumed in an existing tree reuses its cached scores without
  checking `saltcon`. For GPU runs, rerun the band on CPU by setting
  `intermediate_states_arguments.gb_dielectric_cpu_cores` (MPI ranks for `pmemd.MPI`).

### 6. `gb_extdiel_windows` takes dielectric constants
The list now holds explicit external dielectric values ε, not fractions of 78.5. Values outside
[`gb_extdiel_floor` (1.0), 78.5] are clamped, with one warning per clamped value, and duplicates are
dropped, so an old fractional list still loads and collapses to a single ε = 1.0 window.

- **Affected:** every config that sets `gb_extdiel_windows`.
- **Migration:** multiply the old values by 78.5, for example `[0.25, 0.5, 0.75]` becomes
  `[19.625, 39.25, 58.875]`. Drop any old `0` or `1` entries: the gas and full-water end states are
  always present, and ε = 1 is now a vacuum-like window. For even overlap, space the values evenly in
  λ = 1 - 1/ε, that is ε = 1/(1 - λ): λ = 0.2, 0.4, 0.6 and 0.8 give `[1.25, 1.67, 2.5, 5.0]`. The
  in-solvent cycle needs the list empty.

### 7. `adaptive_lambda` means post-production ALS
On `main` after 1.1.3 (#100), `adaptive_lambda: true` ran a 50 ps pilot before production. That
pilot is removed. ALS now adds full-length windows after production.

- **Affected:** configs from `main` that set `adaptive_lambda: true` or any pilot key. 1.1.3 had no
  `adaptive_lambda`.
- **Migration:**
  - Remove the pilot keys. `main` had `pilot_ps`, `pilot_frames`, `pilot_nstlim`,
    `max_adaptive_iterations` (use `als_max_rounds`), `candidate_pool_step` (use
    `candidate_conformational_pool` for explicit values) and `redistribution_mode`. Five more,
    `batch_insertion`, `prune_after_convergence`, `prune_margin`, `charge_pool_step` and
    `dielectric_pool_step`, come from feature builds only. All eleven are dropped with a warning.
  - Re-check `min_degree_overlap`. The default is still 0.04, but it is now compared with a
    two-state overlap on a 0-0.5 scale, not with the superdiagonal of the full-N overlap matrix.
  - Keep `hardware_parameters.export_intermediate_files: true` (the default), `md_only` off,
    post-analysis and consolidation on, and workflow-generated restraints. Config load raises
    otherwise.

### 8. Endstate labels name their method
The endstate row of each MBAR matrix, its directory name and its MBAR index key are now
`remd_endstate`, `vanillaMD_endstate` or `user_provided_endstate` instead of `endstate`. Before,
a parquet left by a vanilla-MD run could be loaded as a REMD run's endstate energies.

- **Affected:** output trees written by any earlier version. Re-analysing one with 2.0.0 raises
  `KeyError`.
- **Migration:** analyse old trees with the version that wrote them. Start every 2.0.0 run in a
  fresh output directory.

### 9. Test tiers and CI
The CB7 Toil + sander tests carry the `workflow` marker. Two tests carry `hpc`: the MCL-1 Boresch
anchor test, which needs MCL-1 benchmark files that are not in the repository, and the REMD ladder
test, which needs `mpirun` and `sander.MPI` from an MPI build of AMBER. `setup.cfg` sets
`addopts = --strict-markers -m "not workflow"`, which deselects only the workflow tier: plain
`pytest` runs the unit tier plus the two `hpc` tests, which skip where their files or binaries are
missing. Markers are strict, so an unregistered marker is an error. `test_adaptive_insertion.py`
and `test_pilot_mdin.py` were removed with the pilot.

- **Migration:** run `pytest -m "not workflow and not hpc"` for the unit tier alone (the command
  the CI unit job runs) and `pytest -m workflow` for the CB7 run. Scripts that named test files
  directly should switch to markers (see [Testing](#testing)).

### 10. Python 3.10 or newer
`setup.py` sets `python_requires=">=3.10"`.

## Fixes

- **Receptor endstate trajectory:** since 1.1.1 the receptor endstate scoring was handed the ligand
  trajectory, so the receptor endstate had no valid frames and was tied to the cycle only through
  the weakest restraint window. This matters most for flexible protein receptors.
- **Receptor starting coordinate:** a receptor REMD run reused across ligands gives a multi-frame
  `receptor_coordinate_filename`. The receptor post-processing leg now has its own single-frame
  starting coordinate.
- **Endstate launch:** a failed AMBER run raises with the command and AMBER's stderr. Single-replica
  GPU jobs run the serial build without `mpirun`. The ligand REMD leg stays on CPU, and REMD takes
  `-n` from `nthreads` and keeps it a multiple of `-ng`.
- **`rem.log`** is kept outside debug mode, so REMD mixing can be checked after a run.
- **GPU placement** (#99, on `main` after 1.1.3): one alchemical window per allocated GPU.
- **Basic-MD endstate on the GPU** (#100, on `main` after 1.1.3): under Toil `single_machine` the
  endstate `pmemd.cuda` jobs got an empty `CUDA_VISIBLE_DEVICES` and failed. The `basic_md`
  endstate jobs now request a GPU and run with the variable unset when Toil leaves it empty.

## Installation

```bash
git clone https://github.com/LuchkoLab/Implicit-Solvent-DDM.git
cd Implicit-Solvent-DDM
git checkout 2.0.0
micromamba create -n isddm_env -f devtools/conda-envs/test_env.conda-lock.yml
micromamba activate isddm_env
pip install -e .
```

`devtools/conda-envs/test_env.conda-lock.yml` lists the exact conda-forge packages (linux-64) that
CI installs. It installs with no solve, in about 30 s. With conda, install it through conda-lock:
`conda-lock install -n isddm_env devtools/conda-envs/test_env.conda-lock.yml`.

`devtools/conda-envs/test_env.yaml` pins only what the code depends on: `python=3.10`,
`ambertools=24.8`, `pymbar-core=4.2.0`, `alchemlyb=2.4.1`, `numpy<2` and `pandas>=2.1,<3`. The
other packages are unpinned. `conda env create -f devtools/conda-envs/test_env.yaml` solves it from
scratch, which took about 13 min cold. The env uses `pymbar-core`, which has the pymbar 4 API that
block-MBAR needs; the `pymbar` conda package also pulls in jax, which nothing here uses.

`pip install -e .` adds `toil==8.2.0`. The conda-forge AmberTools build in the env has no MPI and
no `pmemd`. REMD endstates need an MPI build (`sander.MPI` or `pmemd.MPI`), and `pmemd`,
`pmemd.MPI` and `pmemd.cuda` come from a separate AMBER install; put them on `PATH`.

Usage is unchanged:

```bash
run_implicit_ddm.py file:/path/to/job-store --config_file config.yaml --workDir /path/to/working/directory
```

## Testing

```bash
pytest -m "not workflow and not hpc"   # unit tier, the CI unit job's command; about 3 s
pytest -m workflow                     # CB7 Toil + sander workflow tier; about 65 s
pytest -m hpc                          # needs MPI AMBER or the MCL-1 files; skips what is missing
```

The times were measured locally in the env built from the lock file. Plain `pytest` (`addopts` sets
`-m "not workflow"`) runs the unit tier plus the two `hpc` tests, which skip where their files or
binaries are missing. There is no GPU test.

GitHub CI runs the unit and workflow tiers as two parallel CPU jobs on pull requests to `main` and
on pushes to `main`, plus a weekly scheduled run and manual runs (`workflow_dispatch`). Pull
requests and pushes that change only `docs/`, `scripts/`, Markdown files or the docs workflow do
not trigger it. Both jobs install the env from the lock file and deselect `hpc`: the unit job runs
`pytest -m "not workflow and not hpc"` and the workflow job `pytest -m "workflow and not hpc"`. No
job or test sets a timeout. GitHub runners have no MPI AMBER and no MCL-1 files, so the `hpc` tests
never run there; run them on a cluster node before a release.

## Example configurations

The four configs in `script_examples/config_files/` load with 2.0.0 and use the CB7/M01 host-guest
system in `script_examples/structs/`. Their paths are relative to the repository root, so start
runs from there.

- `in_solvent_paper.yaml` runs the in-solvent cycle with REMD endstates. It uses the cycle and
  window schedule of the in-solvent host-guest runs behind the paper: 33 restraint windows, a
  charge window at 0.5, one LJ window at 0.2 and one screening window at 0.5. ALS is left off, as in
  those runs. The REMD template it points to is shorter than the one those runs used (`numexchg`
  250000 against 2500000), and a comment in the file says what to change. Start a new in-solvent
  run from this file:

  ```bash
  run_implicit_ddm.py file:jobstore_cb7 --config_file script_examples/config_files/in_solvent_paper.yaml
  ```

- `remd_config.yaml`, `basic_md_config.yaml` and `user_defined_endstate.yaml` run the gas-phase
  cycle with, in that order, REMD endstates (8 replicas, 298 to 500 K, 1 ns from
  `endstate_templates_required/remd.template`), a basic-MD endstate from
  `endstate_templates_required/basic_md.template` (4 ps), and user-supplied endstate trajectories
  (`endstate_method: 0`, the 4 ps demo trajectories in `script_examples/structs/`). Each has a
  15-window restraint schedule and no charge, LJ or GB-dielectric windows. They are the configs of
  the `docs/examples/` walkthroughs. Their endstate sampling is a short demonstration, not a
  production protocol.

## Documentation

- `docs/config.rst`: the configuration reference for the new and changed keys. New sections:
  Packing GPU windows, Other hardware keys, Reusing finished endstate trajectories, Workflow jobs,
  Flat-bottom
  restraint, GB-dielectric band (gas-phase cycle), LJ windows, In-solvent decoupling cycle,
  Receptor restraint selection, Boresch anchors and distance ladder, Block-chained
  post-analysis, Post-production adaptive lambda scheduling (ALS), RMSD collective-variable
  restraints (Experimental), and Intermediate mdin template.
- `docs/api.rst` adds `als_refine`, `block_mbar`, `pose_clusters` and `nfe_energy`.
- `docs/in_solvent_cycle.md`: in-solvent cycle states, ΔG assembly and validation.
- `docs/post_production_als.md`: ALS trigger, bands, config keys, round outputs, block-chained
  post-analysis and refining finished runs.
- `script_examples/cb7_rmsd_restraints.md`: RMSD restraint mode.
- `scripts/README.md`: analysis and diagnostic scripts. `scripts/paper_2026/README.md` covers the
  one-off paper scripts kept for provenance.

The two `docs/*.md` pages are read in the repository; the Sphinx build covers only `.rst` files.

Known issue: in the API reference, the `IntermediateStateArgs` docstring is out of date for two
fields. It describes `gb_extdiel_windows` as scaling factors normalized to 0-1, and
`min_degree_overlap` as a replica-exchange overlap with default 0.03. The field takes dielectric
constants ε, and `min_degree_overlap` is the ALS two-state overlap threshold with default 0.04.
`docs/config.rst` and these notes are correct. The docstrings will be fixed in the next release.

## Full changelog

- Work merged to `main` after 1.1.3: #99 (GPU placement) and #100. From #100, the ALS pilot is
  replaced in this release; the basic-MD endstate on GPU, `mbar_accelerators` and the per-leg
  overlap matrices are kept.
- Feature work: `1f3f5c9..7454d90`, 70 commits. It lands as `land/v2-engine` (`1f3f5c9..0aa0541`,
  plus a commit that restores `main`'s `CI.yaml`), `land/v2-paper-protocol` (`0aa0541..6ab9434`)
  and the start of `release/2.0.0` (`6ab9434..7454d90`). The package code in 2.0.0
  (`implicit_solvent_ddm/` outside `tests/`) is identical to 7454d90.
- Release cleanup on `release/2.0.0`: the CB7 drivers and `timing_report.py` moved from the
  repository root into `scripts/`, eight one-off paper scripts moved into `scripts/paper_2026/`
  (provenance only), lab paths in the kept scripts replaced with placeholders, `als_census.py`
  takes `--root`, the unused test input removed, `hpc` markers on two tests, version 2.0.0 and
  `python_requires` in `setup.py`, `export-subst` for `_version.py`, the 2.0 keys documented in
  `docs/config.rst`, the four new modules added to `docs/api.rst`, the `in_solvent_paper.yaml`
  example, and the example configs `remd_config.yaml` and `user_defined_endstate.yaml` and the
  `docs/examples/` walkthroughs updated to the 2.0 keys (in 1.1.3 those two configs did not load).
- CI (`ci/fast-tests`): two-job CPU CI (a5ece3f), docs build on the same env (ca5143b), pytest
  markers (61788d8, 54fcf47), a test env pinned to what the code depends on (bc6db7b), a conda-lock
  file (4332d7f) that the CI and docs workflows install from (5e3ac4b).

The branches merge into `main` in this order, each with a merge commit: `ci/fast-tests`,
`land/v2-engine`, `land/v2-paper-protocol`, `release/2.0.0`.

### New modules
- `implicit_solvent_ddm/als_refine.py`: post-production ALS planner.
- `implicit_solvent_ddm/block_mbar.py`: block-chained MBAR.
- `implicit_solvent_ddm/pose_clusters.py`: ligand pose clustering for `pose_medoid`.
- `implicit_solvent_ddm/nfe_energy.py`: analytic NFE restraint energy for the RMSD mode.
