# Running cb7 with RMSD conformational restraints

Add these keys to `intermediate_states_arguments`. The mode is off unless
`rmsd_tolerance_windows` is non-empty, so every other config is untouched.

```yaml
    # --- RMSD conformational restraint ---
    rmsd_tolerance_windows: [ ... ]           # r3 in Angstrom, WEAKEST FIRST (descending)
    rmsd_force_constant: 10.0                 # k, kcal/mol/A^2, fixed across the ladder
    rmsd_reference_source: ensemble_average   # | endstate_lastframe | minimized | <path>
```

## Why `exponent_conformational_forces` is still required

Because the six Boresch terms are **split across two ladders**, and always have been. In
`templates/restraints/orientational.template` the distance term takes `$drest` while the two angles
and three torsions take `$arest`/`$trest` -- and `write_restraint_forces` feeds `$drest` from the
*conformational* list, `$arest`/`$trest` from the orientational one. The reason is units: the distance
spring is kcal/mol/A^2 and cannot share the angular ladder's kcal/mol/rad^2 magnitudes. With the
production `+4` offset the anchor lands at 16 kcal/mol/A^2 and 256 kcal/mol/rad^2, which are ordinary
Boresch values.

So RMSD mode removes one of that list's two jobs, not both. The pairwise network is gone; the Boresch
distance ladder is not, and it still has to be released to reach the unrestrained endstate. It is also
what sets `Kr` in the analytic standard-state correction.

The name is misleading once the conformational network is gone, so there is an explicit alias:

```yaml
    exponent_boresch_distance_forces: [-8.0, -2.0, 4.0]   # takes precedence; same length as orientational
```

Give it and `exponent_conformational_forces` becomes inert (the dataclass still requires the field, so
it has to be present, but nothing reads it). Omit it and the conformational list is used, which is what
the two shared historically -- so an existing config keeps its exact Boresch schedule with no edit.

## Sizing the ladder against the real production configs

The production cb7 configs
(`different_gbs/igb_2/cb7_re_parameterization/rep1/config_files/cb7-mol??.yaml`, one per guest;
adenine is `cb7-mol14.yaml`) run **33 restraint windows**, not the 3 in the test fixture, with a
constant `orient - con` offset of exactly +4.0.

`rmsd_tolerance_windows` must be **the same length as `exponent_orientational_forces`** and stored
**descending**: the ladders are paired by index, index 0 is the weakest restraint everywhere in the
codebase, and the last entry is the pinned anchor.

**Run the matched 33-window ladder first.** It changes exactly one variable -- the form of the
conformational restraint -- while the window count and the entire Boresch release schedule stay as
they were, so the overlap matrix answers "does one RMSD coordinate give better overlap per window
than 2,833 pairwise springs?" on its own terms. Shortening the ladder at the same time changes two
things at once, and a poor result would be ambiguous between them. Prune afterwards, once the
per-window overlaps say how much room there is.

Linear from 1.5 to 0.0 over 33 points is the simple default. The principled version is equal
quantiles of the endstate RMSD distribution, which is approximately equal-overlap spacing by
construction -- available once a run has written `pmd.txt`.

## Three things to check in these configs before reusing them

* `gb_extdiel_windows: [0.1]` -- these are now **explicit epsilon values**, clamped to
  `[gb_extdiel_floor, 78.5]`, so `0.1` becomes `1.0` and warns. That semantics change came in with
  commit `5e7d0ef`, well after these configs were written, and it turns the GB band into a single
  vacuum window rather than "10% of 78.5". Unrelated to the RMSD work, but it will bite silently.
* `min_degree_overlap: 0.00` -- the ALS false-convergence trap (the test `overlap < 0` never fires).
  Harmless while `adaptive_lambda` is off, which it must be here anyway.
* `executable: pmemd.MPI` with `CUDA:` empty -- so this runs on CPU. NFE has been probed on `sander`
  and `pmemd.cuda`, not on `pmemd.MPI`. Same module, but unverified.

The top rung has to clear the free ensemble's own fluctuation, or the ladder never reaches the
unrestrained endstate. The probe run measured the cb7 complex endstate at **0.45-0.94 A** (95 heavy
atoms, against the minimized structure), so `r3 = 1.5` is comfortably past it and that window's
restraint energy is identically zero on every frame. Against an `ensemble_average` reference the
RMSDs will be smaller still, since the centroid sits inside the cloud rather than off to one side.

`r3 = 0.0` at the anchor makes the restraint a pure harmonic `0.5*k*R^2`. With `k = 10` and an
ensemble near 0.9 A that is about 4 kcal/mol, roughly 7 kT -- firm without being a hard freeze.

cb7 has no `@CA`, so the selection falls back to all heavy atoms. That is the intended host-guest
path and it is what the probe exercised.

## What to look for

* Every window's directory now carries an Angstrom tolerance where it used to carry a log2 exponent
  (`.../lambda_window/1.0/78.5/0.6/`). Same path depth, so warm-resume and `fromPath2Dict` are
  unaffected -- but a tree written in one mode must not be mixed with the other.
* `pmd.txt` appears beside each window's `mdout`, carrying the CV per frame. It is the quickest check
  that the restraint is doing something: the CV should sit under `r3` for the tighter windows.
* `restraint.RST` for the apo legs is now effectively empty, and the complex file holds only the six
  Boresch terms. The 2,833-record pairwise block is gone.
* The adjacent-window overlaps across the restraint band are the point of the exercise. The band's
  top window should overlap the endstate at ~1.0, because at `r3 = 1.5` they are the same
  Hamiltonian.

## Not yet wired

`adaptive_lambda` must stay **off**. The ALS band for the RMSD coordinate (`plan_rmsd_insertion`,
`pair_orientational`) is not implemented, and `derive_offset` would raise on a tolerance ladder.
