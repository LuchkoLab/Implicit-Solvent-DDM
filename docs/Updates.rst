.. _updates-label:

Updates and Changelog
======================

This section documents the major updates and changes made to the Implicit Solvent DDM package across different versions.

Version 2.0.0 - In-Solvent Protocol and Post-Production ALS
============================================================

**Release Date**: November 2026
**Status**: Latest Release

Version 2.0.0 brings the in-solvent protocol used in the paper, adaptive lambda scheduling (ALS)
after production, block-MBAR and GPU window packing, plus new restraint options for
protein-ligand systems. It also fixes a bug, present since 1.1.1, that handed the receptor
end-state scoring the ligand trajectory.

Some defaults changed, and they change the computed ΔG. Output directories written by older
versions cannot be re-analysed with 2.0.0, so read :ref:`v2-breaking-changes` before you upgrade.
The complete list, with commit references, is in ``CHANGELOG.md`` and ``RELEASE_NOTES_v2.0.0.md``
in the repository.

The new methods stay off unless you turn them on:

.. list-table::
   :header-rows: 1
   :widths: 38 37 25

   * - Feature
     - Switch
     - Default
   * - In-solvent decoupling cycle
     - ``decouple_in_solvent``
     - off (gas-phase cycle)
   * - Post-production ALS
     - ``workflow_jobs.adaptive_lambda``
     - off (fixed windows)
   * - Block-MBAR
     - ``post_analysis_block_size``
     - ``None`` (full MBAR)
   * - GPU window packing
     - ``windows_per_gpu``
     - 1 (no packing)
   * - RMSD restraints (experimental)
     - ``rmsd_tolerance_windows``
     - empty (pairwise network)

Major Features
--------------

In-Solvent Decoupling Cycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``decouple_in_solvent: True`` the ligand is decoupled while it sits in GB water, rather than
after a move to the gas phase. On the complex leg its charges go first, then its GB ``SCREEN``,
then its LJ interactions with the receptor. The cycle has no gas state and no GB-dielectric band,
so every complex and receptor window can run on ``pmemd.cuda``. Its end states are those of the
gas-phase cycle, which means the answer is still an absolute binding free energy.

``script_examples/config_files/in_solvent_paper.yaml`` is a complete config for this cycle, and
:doc:`in_solvent_protocol` walks through a run.

Post-Production ALS
~~~~~~~~~~~~~~~~~~~

Set ``workflow_jobs.adaptive_lambda: True`` and the run first finishes on the windows in your
config. ALS then computes the two-state MBAR overlap of every pair of neighbouring windows in each
leg, and wherever it falls below ``min_degree_overlap`` (0.04) a new full-length window goes into
the gap. Only the new windows and the energies they add get computed. The loop ends when no weak
gap is left or after ``als_max_rounds`` rounds (3 by default), and each round is logged in
``.cache/<complex>/als/``. This replaces the 50 ps pilot that ``adaptive_lambda`` ran on ``main``
after 1.1.3. :doc:`DDM_Workflow` shows where the rounds sit in a run.

Block-MBAR
~~~~~~~~~~

``post_analysis_block_size: K`` solves each leg as a chain of K-state MBAR blocks and runs only the
``sander imin=5`` energy evaluations those blocks need. For K = 2, the BAR chain between
neighbouring windows, that is 3N - 2 evaluations instead of N². See :ref:`block-mbar`.

GPU Window Packing
~~~~~~~~~~~~~~~~~~

``hardware_parameters.windows_per_gpu`` puts several ``pmemd.cuda`` windows on each GPU at once.
It only pays off when the CUDA MPS daemon is started before the workflow. On an RTX 5000 Ada, four
windows per GPU gave about 1.9 times the throughput of one. The Slurm script and the full
benchmark are in :doc:`gpu_packing`.

Other GPU changes:

- ``remd_accelerators`` lets the complex and receptor REMD jobs reserve GPUs. ``remd_executable``
  sets a separate AMBER binary for them, for example a CPU build when your AMBER has no
  ``pmemd.cuda.MPI``.
- MBAR jobs no longer hold a GPU while pymbar runs on the CPU (``mbar_accelerators`` defaults
  to 0).
- ``gb_dielectric_cpu_cores`` runs the GB-dielectric windows of the gas-phase cycle on CPU
  ``pmemd.MPI``. ``pmemd.cuda`` ignores ``extdiel`` in GB MD, so on the GPU those windows would
  sample full water.

Running and Re-running
~~~~~~~~~~~~~~~~~~~~~~

- A window is scored as soon as its own MD ends, so CPU scoring overlaps with the GPU MD instead
  of waiting for all of it.
- ``workflow_jobs.md_only: True`` runs the MD and stops. Run again without it, against the same
  ``output_directory_name``, to score the windows without repeating the MD.
- With ``workflow_jobs.run_endstate_method: False`` a run takes the end-state trajectories of a
  finished run instead of running REMD again.
- ``hardware_parameters.export_intermediate_files: False`` keeps intermediate files in the Toil job
  store and out of the output directory.
- Phase 7 writes the MBAR overlap matrix of every leg (``*_O_MBAR.h5``), not only the complex one.
- ``[TIMING]`` lines in the Toil log feed ``scripts/timing_report.py``, which breaks a run down by
  phase into wall-clock time and CPU and GPU hours.

Restraint Options
~~~~~~~~~~~~~~~~~

- ``restraint_reference`` picks the frame the restraints are built from: the last end-state frame
  (``lastframe``, the default), the medoid of the main ligand pose cluster (``pose_medoid``), or a
  coordinate file you supply.
- ``boresch_anchors`` takes the receptor anchor atoms from the pocket backbone
  (``pocket_backbone``, the new default) or keeps the old rule (``legacy``).
- ``unrestrained_receptor_mask`` leaves receptor atoms out of the conformational restraint network;
  ``receptor_ramp_mask`` ramps their restraints in along the ladder instead.
- ``lj_scale_windows`` adds ligand-receptor LJ windows to the complex leg, in either cycle.
- ``exponent_boresch_distance_forces`` gives the Boresch distance term a ladder of its own.
- ``rmsd_tolerance_windows`` replaces the pairwise network with an RMSD restraint. It is
  experimental and not validated for production.

Example Configuration
---------------------

The 2.0 keys in place, for an in-solvent run with GPU packing, ALS and block-MBAR switched on:

.. code-block:: yaml

    hardware_parameters:
      executable: pmemd.cuda
      CUDA: True
      num_accelerators: 1
      windows_per_gpu: 4              # needs CUDA MPS
      remd_accelerators: 2

    workflow_jobs:
      adaptive_lambda: True           # post-production ALS

    workflow:
      intermediate_states_arguments:
        decouple_in_solvent: True
        gb_extdiel_windows: []        # must be empty in the in-solvent cycle
        charges_lambda_window: [0.5]
        screening_windows: [0.5]
        lj_scale_windows: [0.2]
        post_analysis_block_size: 3   # block-MBAR
        min_degree_overlap: 0.04      # ALS threshold
        als_max_rounds: 3

.. _v2-breaking-changes:

Breaking Changes
----------------

A 1.1.x config that sets none of the keys below still loads, but its result can differ.

1. The flat-bottom term is no longer added to the total ΔG. The complex leg already ends at the
   flat-bottom-restrained end state, so adding it counted the confinement twice. It is still
   written out as ``flat_bottom_contribution``.
2. When ``restrained_receptor_atoms`` is not set, the flat-bottom receptor group is now the
   binding site: the CA atoms of residues within ``pocket_radius`` (8 Å) of the ligand, or the
   host heavy atoms within that distance for a host-guest system. In 1.1.x it was every receptor
   CA atom, or every host heavy atom.
3. Boresch anchors come from the pocket backbone by default. For a protein, the receptor has to
   come before the ligand in the complex topology, or the Boresch job raises ``ValueError``. The
   three dihedral restraints are now bounded at their target ± 180° instead of 0° to 360°.
4. The receptor leg's lowest restraint window runs for 10 ns (``long_restraint_window_ns: 10.0``).
5. The GB-dielectric band runs and is scored with ``saltcon=0``, so it reaches vacuum at ε = 1.
6. ``gb_extdiel_windows`` takes dielectric constants, not fractions of 78.5. An old fractional
   list still loads but collapses, with a warning, to a single ε = 1.0 window.
7. ``adaptive_lambda`` now means post-production ALS, and the pilot keys from ``main`` are dropped
   with a warning.
8. End-state labels name their method (``remd_endstate``, ``vanillaMD_endstate`` or
   ``user_provided_endstate``) instead of plain ``endstate``. Re-analysing an older output tree
   raises ``KeyError``.
9. The tests are split into tiers by pytest markers; see :doc:`testing`.
10. Python 3.10 or newer is required.

Migration Guide
---------------

- Start every 2.0.0 run in a new output directory, and analyse old output with the version that
  wrote it.
- Convert ``gb_extdiel_windows`` to dielectric constants: multiply the old values by 78.5 and drop
  any 0 or 1 entries, so ``[0.25, 0.5, 0.75]`` becomes ``[19.625, 39.25, 58.875]``. The in-solvent
  cycle needs the list empty.
- Remove the pilot keys ``pilot_ps``, ``pilot_frames``, ``pilot_nstlim``,
  ``max_adaptive_iterations``, ``candidate_pool_step`` and ``redistribution_mode``.
  ``als_max_rounds`` takes over from ``max_adaptive_iterations``.
- To compare an older total ΔG with a 2.0.0 one, subtract the ``flat_bottom_contribution`` stored
  with the older result from its total.
- To keep the old behaviour where you need it:

  .. code-block:: yaml

      workflow:
        restraints:
          restrained_receptor_atoms: ":REC & @CA"       # protein: every receptor CA atom
          # restrained_receptor_atoms: ":REC & !@H*"    # host: every host heavy atom
        intermediate_states_arguments:
          boresch_anchors: legacy                       # old anchor choice
          long_restraint_window_ns: null                # every window at the template length

  Replace ``:REC`` with your ``receptor_mask``.
- For a gas-phase run on GPUs, set ``gb_dielectric_cpu_cores`` so the GB-dielectric windows run on
  the CPU.

Installation
------------

.. code-block:: bash

    git clone https://github.com/LuchkoLab/Implicit-Solvent-DDM.git
    cd Implicit-Solvent-DDM
    git checkout 2.0.0
    micromamba create -n isddm_env -f devtools/conda-envs/test_env.conda-lock.yml
    micromamba activate isddm_env
    pip install -e .

The lock file installs in about 30 s, with no solve. ``pmemd``, ``pmemd.MPI`` and ``pmemd.cuda``
are not in the conda env and come from your own AMBER install. See
:ref:`Installation <installation-label>`.

Fixes
-----

- Since 1.1.1 the receptor end-state scoring had been given the ligand trajectory, so the receptor
  end state had no valid frames. Flexible protein receptors are hit hardest.
- When one receptor REMD run is reused across ligands, the receptor post-processing leg now gets a
  single-frame starting coordinate of its own.
- A failed AMBER run raises an error that shows the command and AMBER's stderr. Single-replica GPU
  jobs run the serial build without ``mpirun``, and the ligand REMD leg stays on the CPU.
- ``rem.log`` is kept outside debug mode, so REMD mixing can be checked after a run.
- Basic-MD end states on the GPU no longer fail under Toil ``single_machine`` when
  ``CUDA_VISIBLE_DEVICES`` is left empty.
- The example configs ``remd_config.yaml`` and ``user_defined_endstate.yaml`` load again.

Known Issues and Limitations
----------------------------

- Nothing stops a gas-phase run from putting its ``gb_extdiel_windows`` windows on
  ``pmemd.cuda``, where they sample full water. On GPUs, set ``gb_dielectric_cpu_cores`` or use the
  in-solvent cycle.
- In the API reference, the ``IntermediateStateArgs`` docstring is out of date for
  ``gb_extdiel_windows`` (it takes dielectric constants) and ``min_degree_overlap`` (the ALS
  threshold, default 0.04). :doc:`config` has the correct descriptions.
- ``scripts/als_census.py`` and ``scripts/prepare_refine.py``, which apply ALS to finished runs,
  assume the directory layout of the lab's host-guest campaign.
- The RMSD restraints are experimental.

Version 1.1.1 - GPU Acceleration Support
=========================================

**Release Date**: October 2025
**Status**: Superseded

Major Features
--------------

GPU Acceleration Support
~~~~~~~~~~~~~~~~~~~~~~~~~

The package now includes comprehensive GPU acceleration support for molecular dynamics simulations using CUDA-enabled AMBER executables.

**Key Features:**
- **Automatic GPU Detection**: The system automatically detects available GPUs when ``CUDA: True`` is set
- **Flexible GPU Allocation**: Support for specifying the number of GPUs per simulation
- **CUDA-Aware Job Scheduling**: Intelligent job distribution across available GPU resources
- **Fallback Support**: Graceful fallback to CPU execution when GPUs are unavailable

**Supported GPU Executables:**
- ``pmemd.cuda`` - GPU-accelerated PMEMD engine
- Custom CUDA-enabled AMBER executables

Configuration Changes
~~~~~~~~~~~~~~~~~~~~~

New Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following new parameters have been added to the configuration system:

**System Settings:**
- ``CUDA`` (bool): Enable/disable GPU acceleration (default: ``False``)
- ``num_accelerators`` (int): Number of GPUs to request (default: ``0`` - auto-detect)

**Example Configuration:**

.. code-block:: yaml

    system_parameters:
      executable: "pmemd.cuda"  # GPU-enabled executable
      mpi_command: "srun"       # or "mpirun"/"mpiexec"
      CUDA: True                # Enable GPU acceleration
      num_accelerators: 2       # Number of GPUs (0 = auto-detect)
      output_directory_name: "gpu_simulation"

**Complete CUDA Configuration Example:**

.. code-block:: yaml

    # CUDA-enabled configuration example
    system_parameters:
      executable: "pmemd.cuda"
      mpi_command: "srun"
      CUDA: True
      num_accelerators: 2       # Use 2 GPUs
      memory: "10G"            # Increased memory for GPU jobs
      disk: "20G"              # Increased disk space
      output_directory_name: "cuda_ddm_run"

    endstate_parameter_files:
      complex_parameter_filename: "path/to/complex.parm7"
      complex_coordinate_filename: "path/to/complex.rst7"

    number_of_cores_per_system:
      complex_ncores: 8        # CPU cores for complex simulation
      ligand_ncores: 4         # CPU cores for ligand simulation  
      receptor_ncores: 4       # CPU cores for receptor simulation

    AMBER_masks:
      receptor_mask: ":RECEPTOR"
      ligand_mask: ":LIGAND"

    workflow:
      endstate_method: "basic_md"
      endstate_arguments:
        md_template_mdin: "path/to/template.mdin"
      intermediate_states_arguments:
        mdin_intermediate_file: "path/to/intermediate.mdin"
        igb_solvent: 2
        temperature: 300
        exponent_conformational_forces: [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        exponent_orientational_forces: [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        restraint_type: 2

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

**SystemSettings Class Updates:**
- Added ``CUDA`` boolean field with automatic GPU detection
- Added ``num_accelerators`` field with intelligent defaults
- Enhanced ``__post_init__`` method for GPU environment validation

**Simulation Class Enhancements:**
- Updated command-line argument generation for CUDA executables
- Enhanced GPU-aware job scheduling logic
- Improved environment variable handling for CUDA_VISIBLE_DEVICES

**Runner Class Improvements:**
- Added GPU job batching and resource management
- Implemented intelligent GPU allocation across simulation batches
- Enhanced error handling for GPU resource conflicts

**Key Code Changes:**

In ``config.py``:

.. code-block:: python

    @dataclass
    class SystemSettings:
        CUDA: bool = field(default=False)
        num_accelerators: int = field(default=0)
        
        def __post_init__(self):
            if self.CUDA and self.num_accelerators == 0:
                try:
                    from numba import cuda
                    self.num_accelerators = len(cuda.gpus)
                except ImportError:
                    raise RuntimeError("CUDA requested but 'cuda' module not available.")

In ``simulations.py``:

.. code-block:: python

    def setup(self):
        if self.CUDA and self.system_type in ["complex", "receptor"]:
            self.exec_list.append(self.executable)
        # ... rest of setup logic

Performance Improvements
~~~~~~~~~~~~~~~~~~~~~~~~

- **GPU Acceleration**: Significant speedup for large system simulations
- **Resource Optimization**: Better utilization of available computational resources
- **Memory Management**: Enhanced memory handling for GPU-accelerated simulations
- **Job Scheduling**: Improved parallel execution with GPU-aware scheduling

Backward Compatibility
~~~~~~~~~~~~~~~~~~~~~~

- All existing CPU-only configurations remain fully functional
- Default behavior unchanged (CPU execution)
- No breaking changes to existing API or configuration format
- Seamless upgrade path from previous versions

Migration Guide
~~~~~~~~~~~~~~~

**For Existing Users:**

To enable GPU acceleration, simply add the following to your configuration:

.. code-block:: yaml

    system_parameters:
      executable: "pmemd.cuda"  # Change from "pmemd.MPI" to "pmemd.cuda"
      CUDA: True                # Add this line
      num_accelerators: 1       # Optional: specify number of GPUs

**Hardware Requirements:**
- CUDA-enabled GPU with sufficient memory
- CUDA-compatible AMBER installation
- Appropriate CUDA drivers and runtime

**Software Dependencies:**
- AMBER with CUDA support
- CUDA toolkit (version 10.0 or higher recommended)
- Python packages: numba (for GPU detection)

GPU-Enabled PyMBAR Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For GPU-accelerated free energy analysis using PyMBAR, you can enable JAX CUDA support to leverage GPU computing for MBAR calculations.

**Installation Requirements:**

Follow the JAX installation guide for NVIDIA GPU support with CUDA 12:

.. code-block:: bash

    # Install JAX with CUDA 12 support
    pip install -U "jax[cuda12]"

**Verification:**

Check your CUDA version:

.. code-block:: bash

    nvcc --version

**Configuration:**

Once JAX with CUDA support is installed, PyMBAR will automatically detect and use GPU acceleration when available. The analysis will be performed using JAX's GPU-accelerated operations, significantly speeding up MBAR calculations for large datasets.

**GPU Allocation:**

Note that only one GPU will be used per simulation job. This means each individual simulation (such as complex lambda windows, endstate simulations, charge lambda windows, etc.) will utilize a single GPU for acceleration. Multiple simulations can run in parallel across different GPUs when available.

**Benefits:**
- Accelerated MBAR free energy calculations
- Faster convergence for large simulation datasets
- Reduced analysis time for complex systems
- Automatic GPU detection and utilization

**References:**
- `JAX Installation Guide <https://docs.jax.dev/en/latest/installation.html>`_
- `JAX CUDA Support <https://docs.jax.dev/en/latest/installation.html#nvidia-gpu>`_

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- GPU memory requirements may be higher than CPU simulations
- Some small systems may not benefit significantly from GPU acceleration
- CUDA_VISIBLE_DEVICES environment variable management requires careful configuration in multi-GPU setups



Version 1.0.0 - Initial Stable Release
======================================

**Release Date**: December 19, 2024  
**Status**: First Stable Release for Publication

This version represents the first stable release of the Implicit Solvent DDM package, containing the exact code used for the paper submission and publication.

**Key Features:**
- Complete DDM workflow implementation
- Implicit solvent support (GBSA models)
- Multi-engine compatibility (AMBER executables)
- Parallel computing support (SLURM/PBS)
- Automated restraint generation
- Temperature replica exchange (TREMD)
- Integrated MBAR analysis

For detailed information about v1.0.0 features, see :ref:`Installation <installation-label>` and :ref:`Implementation Details <ddm_cycle-label>`.

Related Documentation
--------------------

- :ref:`Installation <installation-label>` - Setup instructions including GPU requirements
- :ref:`Configuration <config-label>` - Complete configuration reference
- :ref:`Examples <examples-label>` - Usage examples including GPU configurations
- :ref:`API Documentation <api-label>` - Technical API reference
