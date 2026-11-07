# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [Unreleased]

### Added
- `restraint_reference: pose_medoid` anchors the complex restraints to the medoid of the
  dominant ligand pose cluster instead of the endstate last frame. The last frame is a
  single random draw from the pose ensemble -- on MCL-1 ligand-3 it landed in a 2.9%
  cluster while 69.7% of frames sat in one basin -- which inflates restraint work and
  broadens dU at the endstate seam. Variance fix only: the endstate is the full
  unrestrained ensemble either way, so the total dG is unchanged. Default off.
- `pose_cluster_cutoff`, `pose_pocket_radius` and `pose_cluster_stride` tune the
  clustering. Stride 0 auto-sizes to cap the O(n^2) distance matrix at ~2000 frames.
- `implicit_solvent_ddm/pose_clusters.py`, a pytraj + scipy port of
  `scripts/analyze_clusters.py`. The script is now a CLI over it, so there is one
  implementation. Reproduces the MDAnalysis result exactly on MCL-1 ligand-3 (26 clusters,
  dominant 69.7%, medoid frame 415, second 17.4%) without needing MDAnalysis or
  scikit-learn.
- The clustering runs on host-guest systems, which the MDAnalysis version refused: the
  pocket falls back to receptor heavy atoms when there is no backbone to align on.
- Cluster artifacts (`summary.txt`, `assignments.csv`, `pose_analysis.png`,
  `dendrogram.png`) are written to the endstate output directory when the flag is on.

### Fixed
- `rem.log` is no longer dropped from REMD output outside debug mode. Exchange acceptance
  and replica round-trip counts are the only record of whether a ladder mixed, and losing
  them is how a non-mixing ladder went unnoticed across a whole benchmark.
- `scripts/analyze_clusters.py` no longer segfaults writing medoid PDBs; it strips the
  topology once rather than handing `pt.write_traj` a mask and a trajectory slice.

### Planned
- GPU acceleration support
- Enhanced error handling and recovery
- Additional analysis methods
- Improved user interface
- Extended documentation

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

## [Unreleased]

### Planned for v1.1.0
- GPU acceleration support
- Enhanced error handling and recovery
- Additional analysis methods
- Improved user interface
- Extended documentation

### Planned for v2.0.0
- Major refactoring from gpu_enable_merge branch
- Significant performance improvements
- Enhanced workflow management
- Advanced GPU support
