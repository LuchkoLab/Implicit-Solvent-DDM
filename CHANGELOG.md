# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - unreleased

First release since 1.1.3. It contains the work merged to `main` after 1.1.3 (#99, #100), the
feature work in `1f3f5c9..7454d90`, the release cleanup on `release/2.0.0` and the test/CI work on
`ci/fast-tests`. The package code (`implicit_solvent_ddm/` outside `tests/`) is the code at
7454d90. Migration notes are in `RELEASE_NOTES_v2.0.0.md`.

The in-solvent cycle (`decouple_in_solvent`), post-production ALS (`adaptive_lambda`), block-MBAR
(`post_analysis_block_size`) and the RMSD restraints (`rmsd_tolerance_windows`) are opt-in. With
them off, a run uses the gas-phase cycle and a full MBAR over the configured windows. Some defaults
did change, and those changes alter results; they are listed under Breaking.

### Breaking
- The flat-bottom term is no longer added to the binding free energy (2f34da9). The complex leg
  already ends at the flat-bottom-restrained endstate, so adding F(FB) - F(noFB) counted the
  confinement cost twice. `flat_bottom_contribution` is still written to the deltaG output as a
  wall-contact diagnostic. Totals from earlier versions include it.
- The flat-bottom restraint's receptor group defaults to the binding site (d53b529). When
  `workflow.restraints.restrained_receptor_atoms` is not set, it is the CA atoms of residues within
  the new key `workflow.restraints.pocket_radius` (8.0 Å) of the ligand, or, for a receptor with no
  CA atoms (host-guest), the host heavy atoms within that distance of the ligand. The radius grows in
  2 Å steps, up to 16 Å, until at least 10 atoms are selected. The site is taken from the starting
  structure, or from `restraint_reference` when that is a file. In 1.1.3 the group was every
  receptor CA atom, or every receptor heavy atom for a host. For a config that leaves the key
  unset, the restrained endstate ensemble, and with it the ΔG, change whenever the selected pocket
  is a strict subset of the 1.1.3 group, which is the usual case for a protein. A small host can be
  covered whole: on the CB7/M01 example the 8 Å pocket already holds all 84 CB7 heavy atoms, so the
  group is the 1.1.3 one. To keep the 1.1.3 group, set `restrained_receptor_atoms` to
  `"<receptor_mask> & @CA"` (protein) or `"<receptor_mask> & !@H*"` (host).
- Boresch receptor anchors default to `boresch_anchors: pocket_backbone` (d53b529): R1-R3 are
  chosen from backbone CA/C/N atoms of residues within `boresch_pocket_radius` (10 Å) of the
  ligand, excluding any freed or ramped receptor mask. `legacy` keeps the old rule (any receptor
  heavy atom, longest lever arm wins), which on MCL-1 picked side-chain tips in a floppy loop. A
  receptor with fewer than three CA atoms (host-guest) always uses `legacy`. For a protein receptor,
  `pocket_backbone` checks that the receptor atoms come first in the complex topology and that at
  least three pocket backbone atoms are left after the exclusion. If not, the Boresch restraint job
  raises `ValueError`, after the endstate sampling has already run. Both modes, like 1.1.3, number
  the anchor atoms as if the receptor came first; `legacy` and 1.1.3 do not check the order, so
  with a ligand-first topology they pick the wrong anchor atoms. Put the receptor first.
- The three Boresch dihedral restraints have bounds centred on their targets (target ± 180°)
  instead of a fixed 0°-360° range, so a target near 0° or 360° no longer sits at the edge of its
  bounds (d53b529). There is no switch for this.
- The receptor leg's lowest conformational-restraint window runs for 10 ns by default
  (`long_restraint_window_ns: 10.0`, 927ef30); the complex and ligand legs are unchanged. Its frame
  count is kept (`ntwx`/`ntpr` are rescaled) and a window that is already longer is not shortened.
  `null` turns it off.
- The GB-dielectric band (`gb_dielectric` states) runs its MD and its single-point scoring with
  `saltcon=0` (c009e30, 40e45bb), so the GB polar term is linear in λ = 1 - 1/ε and the band
  reaches vacuum at ε = 1. Every other state keeps the user's `saltcon`. Builds between those two
  commits ran the band salt-free but scored it with salt. On MCL-1, re-scoring the receptor's
  GB-band/gas junction salt-free took its ΔG from +370.65 ± 313.15 to +40.95 ± 0.22 kcal/mol.
  Trees from those builds also predate bcf58ec (below), so 2.0.0 cannot re-analyse them; rerun the
  band in a fresh output tree. `saltcon` is not part of the MBAR state key, so a run resumed in an
  existing tree reuses its cached scores without checking `saltcon`.
- `gb_extdiel_windows` takes explicit dielectric constants ε, not fractions of 78.5 (c009e30).
  Values outside [`gb_extdiel_floor` (1.0), 78.5] are clamped, with one warning per clamped value,
  and duplicates are dropped. An old list such as `[0.25, 0.5]` therefore still loads and
  collapses, with only a warning, to a single ε = 1.0 window.
- `workflow_jobs.adaptive_lambda` now runs post-production ALS (7a509c9, see Added). The 50 ps
  pre-production pilot merged to `main` in #100 is removed, and its keys are dropped with a warning
  (see Removed). `min_degree_overlap` (0.04, as on `main`) is now compared with a two-state MBAR
  overlap, whose scale runs from 0 to 0.5, not with the superdiagonal of the full-N overlap matrix.
  Config load rejects ALS without `export_intermediate_files: true`, with `md_only`, with
  post-analysis or consolidation off, or with user-supplied restraint files or templates.
- The endstate row of every MBAR matrix is labelled after the method that produced it:
  `remd_endstate`, `vanillaMD_endstate` or `user_provided_endstate` instead of `endstate`
  (bcf58ec). The label is also the endstate's directory name and MBAR index key. Re-analysing an
  output tree written before this change raises `KeyError`. Before the change, a parquet left by a
  vanilla-MD run could be loaded as a REMD run's endstate energies.
- Tests are split into tiers, and the test env and CI changed (61788d8, 54fcf47, 663ee8f, a5ece3f,
  ca5143b, bc6db7b, 4332d7f, 5e3ac4b). The CB7 Toil + sander tests are marked `workflow`. Two tests
  are marked `hpc`: the MCL-1 Boresch anchor test, which needs MCL-1 benchmark files that are not in
  the repository, and the `mpirun`/`sander.MPI` REMD ladder test, which needs an MPI build of AMBER.
  `addopts = --strict-markers -m "not workflow"` deselects only the workflow tier, so plain `pytest`
  runs the unit tier plus the two `hpc` tests, which skip where their files or binaries are missing.
  The unit tier alone is `pytest -m "not workflow and not hpc"`, and the CB7 run is
  `pytest -m workflow`. `devtools/conda-envs/test_env.yaml` pins only `python=3.10`,
  `ambertools=24.8`, `pymbar-core=4.2.0`, `alchemlyb=2.4.1`, `numpy<2` and `pandas>=2.1,<3`, and
  `devtools/conda-envs/test_env.conda-lock.yml` lists the exact conda-forge packages (linux-64)
  of one solve of it. CI runs the unit tier (`-m "not workflow and not hpc"`) and the workflow tier
  (`-m "workflow and not hpc"`) as two parallel CPU jobs on pull requests to `main`, on pushes to
  `main`, weekly and on manual dispatch. Changes that touch only `docs/`, `scripts/`, Markdown files
  or the docs workflow do not trigger it. Both jobs install the lock file into a micromamba env named
  `isddm_env`, with no solve, then run `pip install -e .`; no job or test sets a timeout. The docs
  workflow installs from the same lock file and deploys only from `main`.
- `setup.py` requires Python >= 3.10 (d36356e).

### Added
- In-solvent decoupling cycle, opt-in with `decouple_in_solvent: true` (ef30028). The ligand is
  decoupled in GB water (charges 0, GB `SCREEN` 0, ligand-receptor LJ 0), so the cycle has no gas
  state and no GB-dielectric band, and every complex and receptor window can run on `pmemd.cuda`.
  `screening_windows` adds ligand `SCREEN` windows between `interactions` and the uncharged
  `electrostatics` state. Config load requires an empty `gb_extdiel_windows` and `gbsa=0` in the
  intermediate mdin template, and forces `remove_GB_solvent_receptor` off. The deltaG output labels
  the receptor leg endstate->restrained (f59e264). See `docs/in_solvent_cycle.md`.
- Post-production adaptive lambda scheduling (ALS), opt-in with
  `workflow_jobs.adaptive_lambda: true` (b9b70cb, 7a509c9). The run first finishes on the configured
  windows (round 0). After each round, ALS computes the two-state MBAR overlap `min(O_ab, O_ba)` of
  every adjacent pair in each leg. A gap is weak when that overlap is below `min_degree_overlap`
  (0.04) in any leg that shares its window list. Each weak gap in a band listed in
  `als_bands` (default restraint, charge, LJ and screening; add `dielectric` for the gas cycle's
  GB-dielectric band) gets one window strictly inside it, at the midpoint of the gap (log2 exponent
  for restraints, 1 - 1/ε for the dielectric band) or at the `candidate_*_pool` value nearest the
  midpoint. A gap is not split when half of it is narrower than `als_min_step`, when the value would
  land on an existing window, or when the pool has no value inside the gap. A restraint insertion
  gets an orientational exponent at the schedule's constant offset, or interpolated when the offset
  varies. `als_max_new_windows_per_round` caps the MD windows added per round (a restraint insertion
  adds 3, charge and dielectric 2, LJ and screening 1), weakest gaps first. Pairs outside these
  bands are never filled: pairs that touch the unrestrained endstate (weak ones are logged and
  listed in `final.json`) and, in the gas cycle, the ligand's water-to-gas step. The new windows run
  at full length from the same mdin template, and Phases 3-7 run again with warm resume, so only the
  new windows and energy cells are computed. ALS stops when a round's plan inserts nothing (no weak
  gap left in an enabled band, or none of the weak gaps can be split) or after `als_max_rounds` (3)
  insertion rounds; with 0 it reports the overlaps and inserts nothing. Other new key:
  `refine_expected_deltaG`. Round history is written to `.cache/<complex>/als/` (`round_XX/`,
  `history.csv`, `final.json`); `round_XX/pairs.csv` gives every pair's band, overlap and decision,
  and `final.json` gives the stop reason, the gaps still weak at the stop and the weak endstate
  pairs. See `docs/post_production_als.md`.
- `scripts/als_census.py` and `scripts/prepare_refine.py` apply ALS to finished in-solvent REMD
  runs laid out like the lab's host-guest campaign (a0368b8, 809fa39, 4c4812d, 7454d90). Both expect
  configs named `<prefix>-<guest>.yaml` that set `hardware_parameters.cache_directory_output`
  explicitly. `als_census.py scan --root <runs> --out <dir>` reads the configs under
  `<runs>/igb_<gb>/<system>/in_solvent/rep<rep>/config_files/`, solves the two-state overlap of
  every adjacent pair in every leg from each run's stored `*_formatted.h5`, and caches one table per
  leg in `<dir>`. `als_census.py plan --out <dir>` applies the ALS planner to those cached tables at
  several thresholds (`--thresholds`, default 0.01 to 0.05), with the default bands, midpoint splits
  and no per-round cap. It writes the would-be insertions, a core-hour estimate and, for each
  threshold, the list of configs worth refining. Both subcommands are read-only on the runs.
  `prepare_refine.py` takes such a list or config paths. It also assumes REMD endstates with 8
  replicas (`remd.nc.001` to `remd.nc.008`), a run scored with the full N² MBAR whose window
  directories and scored cells are on disk, configs that also set
  `hardware_parameters.working_directory` and `endstate_parameter_files.ligand_parameter_filename`,
  and a SLURM script `<working_directory>/ddm_submit/<guest>.sh`, which it copies and pins to a
  frozen checkout. It rebuilds the complex and ligand endstate trajectories from the REMD replicas
  with cpptraj, plus the receptor's when the run ran its own receptor REMD, and checks them against
  what the run scored. It then writes a refine config that reuses them (see Endstate trajectory
  reuse) and sends results to `<working_directory>/refine`. Round 0 has to reproduce the original
  ΔG before anything is inserted.
- Endstate trajectory reuse (1b5b264). With `workflow_jobs.run_endstate_method: false` and
  `endstate_method` set to `remd` or `basic_md`, the trajectories given under
  `endstate_parameter_files` as `complex_endstate_trajectory` and `ligand_endstate_trajectory` are
  used as the complex and ligand endstates. Config load requires both, each with at least two
  frames, and `ligand_parameter_filename`. The receptor endstate is read from
  `receptor_parameter_filename` and `receptor_coordinate_filename`, so give the receptor's endstate
  trajectory there, as `prepare_refine.py` does. If the two are unset, the workflow writes the
  receptor from the single-frame starting structure, and that one frame becomes the receptor
  endstate. The endstate keeps its method's label, so cells that are already scored are reused.
- Block-MBAR, opt-in with `post_analysis_block_size: K` (fe5b082). Post-analysis runs only the
  `sander imin=5` evaluations needed by a chain of K-state blocks, and `compute_mbar` solves that
  chain as dense per-block MBARs (`block_mbar.py`). K = 2 is the adjacent-window BAR chain: 3N - 2
  cells instead of N². Under ALS, block ends are fixed at anchor states after the first
  insertion, so a new window joins one block and no scored cell is dropped (b78af23). The default,
  `None`, keeps the full N² MBAR.
- Per-window MD->post pipeline (9f0a1d2). Each window's post-analysis is a follow-on of that
  window's own MD job, so CPU post-analysis fills free cores as trajectories finish instead of
  waiting for all MD. GPU MD windows request `cores=0`.
- `hardware_parameters.export_intermediate_files` (default `true`, 598f23d). With `false`,
  intermediate MD and post-analysis outputs stay in the Toil jobStore and pass between phases as
  FileIDs, and nothing is written to the network output directory. ALS needs it on.
- `hardware_parameters.windows_per_gpu` packs several `pmemd.cuda` windows onto each GPU under
  `single_machine` (default 1, 193d14d). It needs CUDA MPS. Four windows per GPU give about 1.9x
  the window-MD throughput of one on an RTX 5000 Ada. `maxCores` is lowered by one core per slot,
  REMD scales its slot request so it still claims whole devices, and the leader warns if the MPS
  daemon is missing.
- `hardware_parameters.remd_accelerators` (default 0) lets the complex and receptor REMD jobs
  reserve GPUs (a21b957). `hardware_parameters.remd_executable` overrides the AMBER binary for the
  REMD legs only (272573f), for example to run the ladder on CPU when AMBER has `pmemd.cuda` but no
  `pmemd.cuda.MPI`.
- `hardware_parameters.mbar_accelerators` (default 0) sets the GPUs requested by each MBAR job
  (#100, 8d599cb). In 1.1.3 the MBAR jobs requested `num_accelerators`, so on a GPU run each one
  held a GPU while pymbar ran on CPU. With the default, MBAR runs on CPU and the GPUs are free during
  the analysis tail. Values above 1 are clamped to 1.
- `intermediate_states_arguments.gb_dielectric_cpu_cores` (default `null`, ef30028) runs the MD of
  the gas cycle's GB-dielectric windows on CPU `pmemd.MPI` with that many MPI ranks, while the other
  windows keep the configured engine. `pmemd.cuda` ignores `extdiel` in GB MD, so a GB-dielectric
  window run on the GPU samples full water. The value must be >= 1.
- Boresch distance ladder keys (2a57f19). `exponent_boresch_distance_forces` sets the log2 ladder
  that releases the Boresch distance term; it must have as many entries as
  `exponent_orientational_forces`. `null` reuses `exponent_conformational_forces`, as earlier
  versions did. When neither list is given (RMSD mode), the distance ladder is the orientational one
  plus `boresch_distance_offset` (-4.0). `boresch_distance_force_constant` sets the distance force
  constant at the anchor and in the analytic standard-state correction; `null` takes the largest
  value of the ladder.
- Per-leg MBAR overlap matrices (#100, f5fd220). Phase 7 writes the overlap matrix of each leg to
  `.cache/<complex>/` as `<complex>_O_MBAR.h5`, `receptor_<receptor>_O_MBAR.h5` and
  `ligand_<ligand>_O_MBAR.h5`, with rows and columns labelled by state in cycle order, whether or
  not `plot_overlap_matrix` is set. `plot_overlap_matrix` adds a heatmap PDF for each leg. 1.1.3
  wrote only a complex heatmap and no raw matrix.
- `restraint_reference: pose_medoid` anchors the complex restraints to the medoid of the dominant
  ligand pose cluster instead of the endstate last frame (0aa0541). The last frame is a single
  random draw from the pose ensemble. On MCL-1 ligand-3 it landed in a 2.9% cluster while 69.7% of
  frames sat in one basin, which inflates restraint work and broadens dU at the endstate seam. This
  reduces variance only: the endstate is the full unrestrained ensemble either way, so the total ΔG
  is unchanged. Default `lastframe`.
- `restraint_reference` also accepts a path to a single-frame complex restart, such as a crystal
  pose or `scripts/relax_binding_pose.py` output (d53b529). The same pose then restrains and seeds
  every replicate.
- `pose_cluster_cutoff`, `pose_pocket_radius` and `pose_cluster_stride` tune the clustering.
  Stride 0 picks a stride that caps the O(n²) distance matrix at about 2000 frames.
  `pose_cluster_skip` drops the first N endstate frames from the clustering only (burn-in); the
  endstate is still scored on every frame (8bf2790).
- `implicit_solvent_ddm/pose_clusters.py`: pose clustering with pytraj and scipy, so it runs in the
  workflow env without MDAnalysis or scikit-learn. It is a port of an earlier MDAnalysis analysis
  script that is not part of the repository, and it reproduces that script's result on MCL-1
  ligand-3 (26 clusters, dominant 69.7%, medoid frame 415, second 17.4%). It also runs on
  host-guest systems: with no backbone to align on, the pocket falls back to the receptor heavy
  atoms. When `pose_medoid` is on, `summary.txt`, `assignments.csv`, `pose_analysis.png` and
  `dendrogram.png` are written to the endstate output directory.
- `unrestrained_receptor_mask` leaves receptor atoms out of the conformational restraint network
  (f69f2e0). The receptor network is written into both the receptor and complex restraint files,
  so the two legs stay consistent. `receptor_ramp_mask` and `receptor_ramp_slope` ramp a
  selection's restraints instead of dropping them: free near the endstate, at full network
  strength at the top of the ladder (d53b529). The two can act on different atoms, for example a
  ramped `':22-32'` with `'@H*'` dropped, which gives an H-free receptor network (6ab9434).
- `lj_scale_windows`: linear ligand-receptor LJ windows on the complex leg between
  `no_interactions` (0) and `interactions` (1) (d53b529).
- `workflow_jobs.md_only` runs the intermediate MD and stops, with no scoring and no MBAR
  (037f2fa). Running again with it off against the same `output_directory_name` scores the
  windows without re-running MD. It needs `ignore_unique_naming: True`.
- `make_mdin_file(score_igb=...)` rewrites `igb=` in a generated mdin for single-point GB
  re-scoring (68ae683). No workflow code calls it yet.
- Timing tools (ed995f1, b339f81, 8b780ea). MD and post-analysis subprocesses, warm cache loads
  and MBAR solves write `[TIMING]` records to the Toil leader log, which is again written at INFO
  level. `scripts/timing_report.py` turns a log into a per-phase wall-clock and CPU-h/GPU-h
  breakdown, and `scripts/_run_cb7_gpu.py` is a CB7 GPU+CPU timing driver.
- `scripts/_run_cb7_als.py` runs the CB7 workflow with post-production ALS: production on the
  config's windows, then insertion rounds, with the threshold, round count and cycle set through
  environment variables (7a509c9).
- Analysis and diagnostic scripts in `scripts/`, indexed in `scripts/README.md` (4993a32 and
  others). They cover restraint, overlap and replicate audits, binding-pose relaxation, REMD
  temperature extraction and mixing, BAR vs MBAR, the in-solvent state checks, the GPU packing
  benchmark and the apo receptor endstate MD. `_run_cb7_als.py`, `_run_cb7_gpu.py` and
  `timing_report.py` were first committed at the repository root and now live in `scripts/`
  (2ac3667). One-off paper probes and lab-path scripts are in `scripts/paper_2026/`, which is
  provenance only and not maintained (bb33351).

### Fixed
- The receptor endstate scoring was given the ligand trajectory (wrong tuple index), so the
  receptor endstate had no valid frames and was tied to the cycle only through the weakest
  restraint window (61014a3). Present since 1.1.1. The effect is largest for flexible protein
  receptors.
- The receptor post-processing leg has its own starting coordinate (9875b2f). When one receptor
  REMD run is reused across ligands, `receptor_coordinate_filename` is a multi-frame trajectory,
  and sander could not read it as `-c`. `ignore_unique_naming` is now honoured where file names
  are made.
- Endstate launch (272573f, a21b957). A single-replica GPU job ran as `mpirun pmemd.cuda.MPI` with
  no `-np`; it now runs the serial build with no MPI launcher. A non-zero AMBER exit raises with the
  command and AMBER's stderr, instead of failing two jobs later in the REMD groupfile writer as
  `KeyError: 'incrd'`. REMD uses the `.MPI` build of `executable` and checks it is on `PATH`. Its
  `-n` comes from `nthreads` instead of the Toil core reservation, and is kept a whole multiple of
  `-ng`. A `pmemd.cuda.MPI` executable no longer puts the ligand REMD leg on the GPU; it runs on
  `sander.MPI`.
- `rem.log` is kept in REMD output outside debug mode (0aa0541). Exchange acceptance and replica
  round-trip counts are the only record of whether a ladder mixed. Without them, a non-mixing
  ladder went unnoticed across a whole benchmark.
- Window restraint files no longer have the flat-bottom FileID path appended (2a57f19). AMBER
  ignored that line, so no energy changes.
- AMBER failures report the tail of the mdout (d53b529).
- GPU placement, merged to `main` after 1.1.3 (#99). Toil places alchemical windows with one
  window per allocated GPU, and `num_accelerators` is a per-job request clamped to 1.
- Basic-MD endstate on the GPU (#100, 8d599cb). Under Toil `single_machine` the endstate
  `pmemd.cuda` jobs got an empty `CUDA_VISIBLE_DEVICES`, saw no device and wrote no restart. The
  `basic_md` endstate jobs now request a GPU, and a GPU job left with an empty
  `CUDA_VISIBLE_DEVICES` runs with the variable unset, so CUDA uses the GPUs the allocation exposes.
- The example configs `script_examples/config_files/remd_config.yaml` and
  `user_defined_endstate.yaml` failed to load in 1.1.3: they used the old keys `system_parameters`,
  `top_directory_name`, `intermidate_states_arguments` and `mdin_intermidate_config`. They now use
  `hardware_parameters`, `output_directory_name`, `intermediate_states_arguments` and
  `mdin_intermediate_file`, and the walkthroughs in `docs/examples/` include the config files
  instead of copying them.

### Removed
- The pre-production ALS pilot (50 ps windows scheduled before production) and its schedule merge
  step (7a509c9). Its keys are dropped with a warning when a config loads: `pilot_ps`,
  `pilot_frames`, `pilot_nstlim`, `max_adaptive_iterations`, `candidate_pool_step` and
  `redistribution_mode`, which `main` had, and `batch_insertion`, `prune_after_convergence`,
  `prune_margin`, `charge_pool_step` and `dielectric_pool_step`, which existed only in feature
  builds.
- The pilot drivers `_run_cb7_pilot.py` and `_run_cb7_pilot_real_production.py`, the tests
  `test_adaptive_insertion.py` and `test_pilot_mdin.py` (7a509c9), and the unused
  `real_production_run.yaml` test input (cdc8993).

### Experimental
- RMSD collective-variable restraints (2a57f19, 2b2a94b), opt-in with a non-empty
  `rmsd_tolerance_windows`. The pairwise conformational network is replaced by an NFE
  `MULTI_RMSD` flat-bottom restraint whose tolerance (r3, in Å) is laddered across windows at a
  fixed `rmsd_force_constant`. Post-analysis computes the restraint term analytically
  (`nfe_energy.py`). Other keys: `rmsd_reference_source` (default `restraint_reference`),
  `rmsd_mask`, `rmsd_restrain_ligand`, `rmsd_ligand_tolerance_scale`, `rmsd_site_cutoff`,
  `rmsd_site_tolerance_scale`, `rmsd_site_atoms`, `rmsd_r1`/`rmsd_r2`/`rmsd_r4`,
  `rmsd_output_freq`. It cannot be combined with `receptor_ramp_mask` or
  `unrestrained_receptor_mask`, and ALS drops its restraint band in this mode, so
  `candidate_rmsd_pool` and `rmsd_pool_step` are accepted but have no effect. Example:
  `script_examples/cb7_rmsd_restraints.md`. Not validated for production.

## [1.1.3] - 2025-10-23

COM flat-bottom restraint atom selection fix. See `RELEASE_NOTES_v1.1.3.md`.

## [1.1.2] - 2025-10-23

GPU distribution fixes. See `RELEASE_NOTES_v1.1.2.md`.

## [1.1.1] - 2025-10-15

GPU support and the phase-based workflow. See `RELEASE_NOTES_v1.1.1.md`.

## [1.0.0] - 2024-12-19

**Note**: This version represents the code used for the paper submission and publication. This release ensures reproducibility of the published results.

### Added
- Comprehensive documentation with Sphinx
- API documentation for all modules
- Example configurations and workflows
- Support for multiple AMBER engines (sander, sander.MPI, pmemd, pmemd.MPI, pmemd.CUDA)
- Automated restraint generation for Boresch restraints
- Temperature replica exchange molecular dynamics (TREMD) support
- Intermediate state calculations with configurable lambda windows
- Post-processing and MBAR analysis capabilities
- SLURM batch job submission support
- YAML configuration system for workflow parameters

### Features
- **Core Workflow**: Automated absolute binding free energy calculations using implicit solvent models
- **DDM Implementation**: Double decoupling method with Boresch restraints and GB solvent
- **Multi-Engine Support**: Compatible with various AMBER MD engines
- **Parallel Processing**: Support for multi-core and multi-node calculations
- **Flexible Configuration**: YAML-based configuration system
- **Analysis Tools**: Built-in trajectory analysis and MBAR post-processing

### Technical Details
- Python command-line interface
- Toil workflow management integration
- Support for generalized Born surface area (GBSA) models
- Configurable restraint types (CoM-CoM, CoM-Heavy_Atom, Heavy_Atom-Heavy_Atom)
- Temperature and lambda parameter sweeps
- Automated file management and job scheduling

### Documentation
- Complete API documentation
- Installation and setup instructions
- Configuration file examples
- Workflow tutorials and examples
- SLURM batch job templates

---

## Release Notes for v1.0.0

This release represents the first stable version of the Implicit Solvent DDM package suitable for publication and production use. The package provides a complete workflow for performing absolute binding free energy calculations using molecular dynamics with implicit solvent models.

### Key Capabilities
1. **Automated Workflow**: Complete automation of the thermodynamic cycle for binding free energy calculations
2. **Implicit Solvent Models**: Efficient calculations using generalized Born surface area methods
3. **Restraint Implementation**: Automated generation and application of Boresch restraints
4. **Parallel Computing**: Support for high-performance computing environments
5. **Analysis Pipeline**: Integrated post-processing and statistical analysis

### System Requirements
- Python 3.5+
- AMBER molecular dynamics package
- SLURM or PBS job scheduler (optional)
- Toil workflow management system

### Installation
```bash
git clone https://github.com/LuchkoLab/Impicit-Solvent-DDM.git
cd Impicit-Solvent-DDM
conda env create -f devtools/conda-envs/test_env.yaml
conda activate mol_ddm_env
python setup.py sdist
pip install dist/*
```

### Usage
```bash
run_implicit_ddm.py file:/path/to/job-store --config_file config.yaml --workDir /path/to/working/directory
```

This version is ready for publication and provides a robust foundation for binding free energy calculations in molecular dynamics simulations.
