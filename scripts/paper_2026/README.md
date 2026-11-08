# `scripts/paper_2026/` — provenance only

Provenance only: one-time probes and lab-path scripts from the 2026 paper work. Kept for the record;
not maintained; paths are specific to the original cluster.

Run them from the repo root as `python scripts/paper_2026/<name>.py`, in the same environment as the
scripts in `scripts/` (see `../README.md`). `gb_endpoint_probe.py` and `gb_endpoint_analyze.py`
import `scripts/gb_ddG_bar.py` and put `scripts/` on `sys.path` themselves.

| file | what it does | needs AMBER? |
|---|---|---|
| `gb_ddG_ign2igb8.yaml` | `scripts/gb_ddG_bar.py` manifest for cb7/adenine: igb=2 → igb=8 (OBC → GBn2), with per-model `parm_A`/`parm_B`. Every path points into the original cluster's run tree | (manifest) |
| `gb_endpoint_probe.py` | Re-scores a whole GB ladder under both scoring salt concentrations plus the gas endpoint (the full N×N version of `scripts/gb_junction_overlap.py`) | yes |
| `gb_endpoint_analyze.py` | Builds two MBAR problems from `gb_endpoint_probe.py` output that differ only in scoring `saltcon`, and compares their overlap matrices | no |
| `bound_vs_apo_receptor_rmsd.py` | **How far does the receptor reorganize on binding?** — strips the ligand from the complex endstate trajectory and compares it with the apo receptor endstate directly (within-ensemble spread vs average-to-average separation), which removes the window-convergence confound in `scripts/endstate_window_rmsd.py` | yes (cpptraj) |
| `remd_ensemble_clusters.py` | **Do the complex and apo receptor REMD ensembles share basins?** — strips the ligand, clusters both ensembles together, and cross-tabulates cluster membership by origin. Answers whether a REMD floor window could cross the holo↔apo barrier, and emits apo seed frames for a multi-seeded window. On MCL-1: 14 clusters, **0 mixed** | yes |
| `pocket_compression_check.py` | **Why does the complex LJ-off → LJ-on step have no overlap even when the ligand stays put?** Compares the `no_interactions` and `interactions` windows (same restraints, only ligand–protein LJ differs): where the conformational-network pairs around the pocket sit in their ±0.5 Å flat-bottom well, how far pocket atoms close on the decoupled ligand, and the restraint energy a narrower well would add. Prints a provisional verdict: A (pocket closes inside the free play → narrow the well), B (walls engaged but soft → force constant), C (network doesn't see it) | no (pytraj) |
| `gb_half_check.py` | **Does CPU pmemd fix the GB band?** `pmemd.cuda` ignores `extdiel`, so a GPU GB-dielectric window samples water. `setup` rebuilds one GB window (default ε=2, "GB half on") as a CPU `pmemd.MPI` job with the same mdin, restraint file and a `cv.in` reconstructed from the mdout NFE echo; `analyze` scores gas, CPU half, GPU half and water under the gas/half/water Hamiltonians with sander and prints BAR for gas→half→water with the CPU and the GPU half window; `setup-band` writes one CPU batch job (`submit.sh`) per GB-dielectric window of each leg plus a `submit_all.sh` | yes |
| `nfe_rmsd_probe.py` | **Can AMBER give us a GPU RMSD restraint, and is it best-fit?** — stands up `infe=1` + `&pmd` + `MULTI_RMSD` from scratch and checks that it runs, that the CV is rotation/translation invariant and equals cpptraj `rms … mass`, that it applies force rather than only monitoring, and that `cv_ni` does not silently truncate | yes |

---

## `gb_ddG_ign2igb8.yaml` — the igb=2 → igb=8 manifest

```bash
module load amber                                       # sander + cpptraj on PATH
python scripts/gb_ddG_bar.py --manifest scripts/paper_2026/gb_ddG_ign2igb8.yaml --out ddG_gb2togb8.csv
```

See the `gb_ddG_bar.py` section of `../README.md` for replica handling and per-model radii. The
annotated template for a new manifest is `scripts/gb_ddG_manifest.example.yaml`.

---

## `pocket_compression_check.py` — is the pocket closing inside the restraints' free play?

**Why.** With the Boresch restraints holding (MCL-1 ligands 17 and 4), the complex `no_interactions → interactions`
step still has overlap 0.002–0.03: with LJ off, ligand–protein contacts close by ~0.35 Å, and the r⁻¹² wall turns that
into a +27 kcal/mol median penalty (0.1% of LJ-off frames resemble LJ-on ones). The conformational network is a
flat-bottom well — zero force within d₀ ± 0.5 Å — so a stronger force constant cannot stop motion inside it. This
script tells whether that free play is where the pocket closes, before any restraint change.

    python scripts/paper_2026/pocket_compression_check.py fragment-opt-abfe-benchmark/isdmm_runs/mcl1_rampbb_10ns_rep1 \
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

---

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
    python scripts/paper_2026/nfe_rmsd_probe.py --probe 1,2 --executable sander --show-pmd \
        --prmtop implicit_solvent_ddm/tests/structs/cb7-mol01.parm7 \
        --inpcrd implicit_solvent_ddm/tests/structs/cb7-mol01.ncrst \
        --traj  implicit_solvent_ddm/tests/structs/cb7-mol01_300K.nc \
        --workdir nfe_probe_cb7

    # then the GPU question, and the force check
    python scripts/paper_2026/nfe_rmsd_probe.py --probe gate --executable pmemd.cuda \
        --prmtop ... --inpcrd ... --traj ... --workdir nfe_probe_cb7_gpu
    python scripts/paper_2026/nfe_rmsd_probe.py --probe 3 --nstlim 25000 --k 50 --prmtop ... --inpcrd ...

Probe 3 always runs new MD — you cannot show that a force is applied without integrating it.
