# In-solvent decoupling cycle (`decouple_in_solvent`)

The ligand is decoupled inside GB water instead of in the gas phase. The cycle has no gas
state and no GB-dielectric (`extdiel`) band, so every complex and receptor window runs on
`pmemd.cuda`. The end states are the same as in the gas-phase cycle (bound complex in water,
free ligand in water, non-interacting decoupled ligand), so the result is still an absolute
binding free energy.

Why: `pmemd.cuda` ignores `extdiel` in GB MD, so a GPU GB-dielectric window samples water.
When the band is sampled correctly on CPU, the protein reorganises and overlap collapses. With
ligand charges 0, ligand GB `SCREEN` 0 and ligand-receptor LJ 0, the complex reproduces the
apo receptor exactly on every engine (`scripts/ingb_decouple_check.py`).

When the switch is off (the default), the gas-phase cycle is unchanged.

## Turning it on

```yaml
workflow:
  intermediate_states_arguments:
    decouple_in_solvent: True
    lj_scale_windows: [0.2]
    screening_windows: [0.5]
    gb_extdiel_windows: []          # must be empty
    charges_lambda_window: [0.5]    # 0 and 1 are always added
```

Validation at config load:
- `gb_extdiel_windows` must be empty when `decouple_in_solvent` is on.
- `screening_windows` must lie strictly in (0, 1) and requires `decouple_in_solvent`.
- The intermediate mdin template must have `gbsa=0`, or be missing `gbsa`. A decoupled ligand
  would otherwise still change the receptor's surface area.
- `workflow.remove_GB_solvent_receptor` is forced off, so there is no `no_gb` state.

## Cycle

All states use `igb = igb_solvent`, `extdiel = 78.5` and the user's `saltcon`, and are scored
with `post_mdin`.

Complex leg (decoupled end first, as `CycleSteps.complex_order` lists it):

| state | ligand q | ligand SCREEN | lig-rec LJ | restraints |
|---|---|---|---|---|
| `no_interactions` | 0 | 0 | 0 | anchor |
| `lj_{s}` (`lj_scale_windows`) | 0 | 0 | s | anchor |
| `interactions` | 0 | 0 | 1 | anchor |
| `screening_{s}` (`screening_windows`) | 0 | s | 1 | anchor |
| `electrostatics` q = 0 ... 1 | q | 1 | 1 | anchor |
| `lambda_window` release | 1 | 1 | 1 | ladder |
| endstate | 1 | 1 | 1 | none |

The fully decoupled state uses LJ scale 0 (the edit `ingb_decouple_check.py` validated), not
`addExclusions`.

Ligand leg: endstate → restraint windows → charges 1 → 0 in water. The full-charge state is
the restraint anchor window, so no separate q = 1 charge window is run.

Receptor leg: endstate → restraint windows → anchor. There is no GB band and no `no_gb`.

State keys `(label, extdiel, charge, restraint)` carry `extdiel = "78.5"` for the decoupled
states. They can never match the gas cycle's `"0.0"` keys or directories, so a gas-cycle
output tree is never reused as an in-solvent one.

## ΔG assembly

`ConsolidateData(decouple_in_solvent=True)` uses these leg endpoints:

| leg | from | to |
|---|---|---|
| complex | `("no_interactions", "78.5", "0.0", "<con>_<orient>")` | endstate |
| ligand | endstate | `("electrostatics", "78.5", "0.0", "<con>")` |
| receptor | endstate | `("lambda_window", "78.5", "1.0", "<con>")` |

The Boresch and flat-bottom terms are unchanged.

## ALS

- The pilot skips the dielectric pass and runs charge → restraint on the complex.
- `CycleSteps.start_gb_extdiel_matrix` and `start_receptor_gb_matrix` raise in this mode.
- The complex charge band is still `[start_complex_charge_matrix : halo_restraint_matrix + 1]`,
  i.e. uncharged → charged electrostatics. The screening windows are not an ALS band yet.
- `start_ligand_charge_matrix` points at the full-charge anchor, so the ligand charge band
  keeps its q = 1 end.
- ALS-inserted ligand charge windows stay in water.

## Code map

| file | change |
|---|---|
| `config.py` | `decouple_in_solvent`, `screening_windows`, validation, `gbsa` check |
| `alchemical.py` | `alter_topology(screen_scale=...)` scales ligand `SCREEN` |
| `matrix_order.py` | `CycleSteps(in_solvent, screening_windows)`, `screening` states, in-solvent orders and indices, `cycle_steps_from_config` |
| `setup_simulations.py` | `_decoupled_mdin` moves decoupled windows into water; `setup_screening_simulation`; ligand charge windows in water |
| `workflow_phases.py` | in-solvent topologies (LJ 0 + SCREEN 0), screening loop, no ligand q = 1 window, pilot without the dielectric pass |
| `runner.py`, `adaptive_restraints.py` | build `CycleSteps` through `cycle_steps_from_config`; ALS ligand charge windows in water |
| `postTreatment.py` | leg endpoints for the in-solvent cycle |

## Validation

1. `scripts/in_solvent_states_check.py`: a single point on each state's topology, built by
   `alter_topology`, on `pmemd.cuda`, `pmemd` and `sander`.
2. Ligand 17 ×3, pairwise restraint scheme, 10 ns windows. Compare the replicate SD and the
   per-step cancellation (`scripts/cycle_audit.py cancel`) with `mcl1_ljdih_10ns_rep{1,2,3}`.
   Check the `interactions → screening → electrostatics q=0` overlaps with
   `scripts/overlap_audit.py`.
3. The RMSD restraint scheme on the in-solvent cycle.
