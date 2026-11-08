# Implicit Solvent DDM

**Automated, efficient absolute binding free energy calculations using implicit solvent models.**

Implicit Solvent DDM is a Python package for performing fully automated binding free energy calculations (ABFEs) using Molecular Dynamics. The package implements the double decoupling method (DDM) with Boresch restraints and generalized Born surface area (GBSA) models to reduce computational costs while maintaining accuracy.

**Paper**: [Automated Workflow for Absolute Binding Free Energy Calculations with Implicit Solvent and Double Decoupling](https://arxiv.org/abs/2509.21808) - *arXiv:2509.21808*

## Documentation

**[Read the documentation](https://impicit-solvent-ddm.readthedocs.io/en/latest/)**

It covers installation, the YAML config, the DDM workflow, how to run the in-solvent protocol and GPU window packing, worked examples, the tests and CI, and the API reference.

## Quick start

```bash
git clone https://github.com/LuchkoLab/Implicit-Solvent-DDM.git
cd Implicit-Solvent-DDM

# Environment from the lock file: exact packages, no solve, about 30 s
micromamba create -n isddm_env -f devtools/conda-envs/test_env.conda-lock.yml
micromamba activate isddm_env
# or let conda solve the env file (a cold solve can take 10+ minutes):
#   conda env create -f devtools/conda-envs/test_env.yaml
#   conda activate isddm_env

pip install -e .

# The paper's in-solvent protocol on the CB7/M01 example. Run from the repository root.
run_implicit_ddm.py file:jobstore_cb7 \
    --config_file script_examples/config_files/in_solvent_paper.yaml \
    --workDir /path/to/scratch
```

You need Python 3.10 or newer. The conda-forge AmberTools in the env gives you `sander`, `cpptraj` and `tleap`, but not `pmemd`, `pmemd.MPI`, `pmemd.cuda` or `sander.MPI`. The example configs use `pmemd.MPI`, and GPU runs and REMD need those binaries too, so put a full AMBER install on your `PATH` (on a cluster, usually a module). The example above is the full protocol, with REMD at the end states and 10 ns per window, so start it on a cluster node.

## Running

Every run is the same command: a Toil job store, a config file and a scratch directory, optionally with `--maxCores N`. Start it from the repository root, since the example configs use relative paths. If a run stops part way, run the same command again with `--restart` added.

| Page | What it covers |
|---|---|
| [Installation](docs/Installation.rst) | Install, AMBER engines, the run command and the example configs |
| [In-solvent protocol](docs/in_solvent_protocol.rst) | Setting up and running the in-solvent decoupling cycle |
| [GPU window packing](docs/gpu_packing.rst) | Running several `pmemd.cuda` windows on each GPU under CUDA MPS |
| [Config reference](docs/config.rst) | The YAML config, section by section |
| [Examples](docs/examples/) | Basic MD, REMD and user-defined end states |

## What's new in 2.0.0

2.0.0 is the first release since 1.1.3. The new methods are opt-in, but several defaults changed in ways that alter the computed ΔG, and output trees from older versions cannot be re-analysed. Read the breaking changes in [RELEASE_NOTES_v2.0.0.md](RELEASE_NOTES_v2.0.0.md) before you upgrade. Every change, with commit references, is in [CHANGELOG.md](CHANGELOG.md).

- **In-solvent protocol** (`decouple_in_solvent: True`). The ligand is decoupled inside GB water instead of in the gas phase. There is no gas state and no GB-dielectric band, so every complex and receptor window can run on `pmemd.cuda`.
- **Post-production ALS** (`workflow_jobs.adaptive_lambda`). After the run finishes on the configured windows, ALS adds windows in the gaps where neighbouring windows overlap poorly and computes only the new windows and energy cells. It repeats for up to three rounds by default.
- **GPU window packing** (`hardware_parameters.windows_per_gpu`). Several `pmemd.cuda` windows share one GPU under CUDA MPS. On one MCL-1 ligand-17 window (2,471 atoms, RTX 5000 Ada), going from 1 to 4 windows per GPU raised total GPU throughput from 1,735 to 3,327 ns/day, with identical energies. The numbers are below.
- **Block-MBAR** (`post_analysis_block_size`). Each leg is solved as a chain of K-state MBAR blocks, and only the energy evaluations those blocks need are run. K = 2 is the adjacent-window BAR chain: 3N - 2 cells instead of N².
- **Fast CPU-only CI**. The tests are split into unit, workflow and hpc tiers. GitHub CI runs the first two in parallel from a conda lock file in about 2-3 minutes.

GPU packing benchmark (`scripts/gpu_packing_benchmark.py`, MPS on, one MCL-1 ligand-17 window, energies identical across copies):

| Windows per GPU | ns/day per window | Total ns/day | vs 1 window | GPU memory peak (MiB) |
|---:|---:|---:|---:|---:|
| 1 | 1,735 | 1,735 | 1.00x | 436 |
| 2 | 1,363 | 2,726 | 1.57x | 835 |
| 4 | 832 | 3,327 | 1.92x | 1,632 |
| 8 | 435 | 3,480 | 2.01x | 3,227 |
| 12 | 290 | 3,480 | 2.01x | 4,823 |
| 16 | 218 | 3,489 | 2.01x | 6,418 |

Four windows per GPU already reach about 95% of the plateau, which is why the production runs use `windows_per_gpu: 4`. Without MPS the windows time-slice and the total stays around 1,400-1,750 ns/day however many you pack.

## Key features

- **Automated workflow**: the whole DDM cycle runs from one YAML config
- **Implicit solvent**: GB/GBSA models keep the cost down
- **AMBER support**: `sander`, `pmemd`, their MPI builds and `pmemd.cuda`
- **GPU acceleration**: CUDA runs, with several windows packed on each GPU
- **HPC ready**: SLURM/PBS job scheduling through Toil
- **MBAR analysis**: free energies from MBAR, or block-MBAR to cut the energy evaluations

## Tests

```bash
pytest               # unit tier: 257 pure-logic tests, about 4 s
pytest -m workflow   # one small CB7 Toil + sander run checked by 14 tests, about 65-90 s
```

Tests that need a GPU, an MPI build of AMBER or data outside the repository carry the `hpc` marker. They skip themselves when what they need is missing, and GitHub CI never runs them; on the cluster, run them with `pytest -m hpc`. [docs/testing.rst](docs/testing.rst) explains the tiers, what GitHub CI runs and how to update the lock file.

---

**For detailed information, examples and the API reference, see the [complete documentation](https://impicit-solvent-ddm.readthedocs.io/en/latest/).**

*Last updated: November 2026*

## Citation

If you use this software in your research, please cite:

```bibtex
@misc{ayoub2025automated,
  title={Automated Workflow for Absolute Binding Free Energy Calculations with Implicit Solvent and Double Decoupling},
  author={Steven Ayoub and Michael Barton and David A. Case and Tyler Luchko},
  year={2025},
  eprint={2509.21808},
  archivePrefix={arXiv},
  primaryClass={physics.chem-ph},
  url={https://arxiv.org/abs/2509.21808}
}
```

## License

See [LICENSE](LICENSE).
