.. _ddm_workflow-label:

DDM Workflow Structure
======================

This page shows the order in which the workflow runs its jobs. The entry point is
``ddm_workflow()`` in ``implicit_ddm_workflow.py``; the phases themselves live in
``workflow_phases.py``. Every step is a Toil job in one job store, so a stopped run can be continued
with ``--restart``.

Overview
--------

A run goes through seven phases. Since 2.0, Phases 5 and 6 are one step: each window's energies
are scored as soon as that window's MD finishes, instead of after all MD is done.

.. code-block:: text

                         run_implicit_ddm.py
                                 │
                                 ▼
    ┌───────────────────────────────────────────────┐
    │  PHASE 1: SETUP                               │
    │  setup_workflow_components()                  │
    │  • MD input files for the intermediate states │
    │  • empty restraint templates, flat-bottom     │
    └───────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌───────────────────────────────────────────────┐
    │  PHASE 2: END STATES                          │
    │  run_endstate_simulations()                   │
    │  • REMD, basic MD, or your own trajectories   │
    │    for the complex, receptor and ligand       │
    │  • optional restraint reference frame         │
    └───────────────────────────────────────────────┘
                                 │
                                 │  adaptive_lambda: True → Phases 3-7 run as ALS rounds (below)
                                 ▼
    ┌───────────────────────────────────────────────┐
    │  PHASE 3: DECOMPOSITION AND RESTRAINTS        │
    │  decompose_system_and_generate_restraints()   │
    │  • split the complex into receptor + ligand   │
    │  • Boresch and conformational restraints      │
    └───────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌───────────────────────────────────────────────┐
    │  PHASE 4: INTERMEDIATE STATES                 │
    │  setup_intermediate_simulations()             │
    │  • every window of the complex, receptor and  │
    │    ligand legs, plus the flat-bottom pair     │
    └───────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌───────────────────────────────────────────────┐
    │  PHASES 5 + 6: MD AND SCORING                 │
    │  run_intermediate_and_post()                  │
    │  • all MD windows are queued first            │
    │  • each window is re-scored as soon as its    │
    │    own MD finishes                            │
    └───────────────────────────────────────────────┘
                                 │
                                 │  md_only: True → stop here
                                 ▼
    ┌───────────────────────────────────────────────┐
    │  AGGREGATION                                  │
    │  _aggregate_post_output()                     │
    │  • collect the scored rows of each leg        │
    └───────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌───────────────────────────────────────────────┐
    │  PHASE 7: FREE ENERGY                         │
    │  compute_free_energy_and_consolidate()        │
    │  • MBAR (or block-MBAR) for each leg          │
    │  • flat-bottom term by exponential averaging  │
    │  • binding free energy written to .cache/     │
    └───────────────────────────────────────────────┘

The phases
----------

**Phase 1: Setup.**
   Writes the MD input files for the intermediate states from your mdin template, creates empty
   restraint templates, and prepares the flat-bottom restraint.

**Phase 2: End states.**
   Samples the bound complex, the free receptor and the free ligand with the method set by
   ``endstate_method``: temperature REMD (``remd``), one MD run per state (``basic_md``), or
   trajectories you supply (``0``). A finished run's end-state trajectories can also be reused, so
   a re-run does not repeat REMD. If a restraint reference is configured, the frame the restraints
   are built from is picked here (last frame, pose medoid, or a coordinate file you give).

**Phase 3: Decomposition and restraints.**
   Splits the complex into receptor and ligand, chooses the Boresch anchor atoms (pocket backbone
   by default), and builds the conformational restraint network and the restraint files the
   windows use.

**Phase 4: Intermediate states.**
   Builds every window of the three legs. Which windows exist depends on the cycle: the gas-phase
   cycle has the GB-dielectric band, the in-solvent cycle (``decouple_in_solvent``) has the
   screening and LJ steps instead (see :doc:`in_solvent_protocol`). Both have the restraint ladder
   and the ligand charge windows, and by default a 10 ns floor window on the receptor leg.

**Phases 5 and 6: MD and scoring.**
   All MD windows are queued first, in the order complex, receptor, ligand, flat-bottom. Each
   window's trajectory is then re-scored at every state of its leg, and that scoring job starts
   as soon as the window's own MD ends. CPU scoring therefore fills the cores that sit idle while
   the GPU windows run. With ``post_analysis_block_size`` set, only the energies that block-MBAR
   needs are computed. Windows and energies already on disk are reused. GPU window packing
   (:doc:`gpu_packing`) applies to the MD in this step. With ``md_only: True`` the run stops after
   the MD; run it again without ``md_only`` and the same ``output_directory_name`` to score it.

**Aggregation.**
   Collects the scored rows of each leg into the tables Phase 7 reads.

**Phase 7: Free energy.**
   Solves each leg with MBAR, either one solve over all windows or a chain of blocks (block-MBAR,
   see :ref:`block-mbar`). The flat-bottom term comes from exponential averaging; it is written out
   but not added to the binding free energy. The results go to ``.cache/<complex>/``:
   ``deltaG_<complex>.h5`` holds the binding free energy and its terms, next to the per-leg free
   energy (``*_fe.h5``), error (``*_error.h5``) and overlap (``*_O_MBAR.h5``) tables.

Post-production ALS
-------------------

With ``workflow_jobs.adaptive_lambda: True``, Phases 3 to 7 run as rounds:

.. code-block:: text

    PHASE 2 (end states, shared by every round)
         │
         ▼
    als_round(0)        Phases 3-7 on the windows in your config (ordinary production)
         │
         ▼
    refine_schedule()   two-state overlap of every pair of neighbouring windows;
         │              a new window goes into each gap below min_degree_overlap
         │
         ├── no weak gap left, or als_max_rounds reached → stop
         │
         ▼
    als_round(1)        Phases 3-7 again; windows and energies already on disk are
         │              reused, so only the new windows and new energies are computed
         ▼
        ...

Each round checks the windows already on disk against the restraints and topologies it rebuilt
before reusing them. The round history goes to ``.cache/<complex>/als/history.csv`` and the
outcome to ``.cache/<complex>/als/final.json``. ``docs/post_production_als.md`` in the repository
has the details.

Modules
-------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Module
     - What it does
   * - ``implicit_ddm_workflow.py``
     - Entry point: wires the phases together and assigns GPU slots for window packing.
   * - ``workflow_phases.py``
     - The phase functions, the MD-and-scoring step, and the ALS rounds.
   * - ``run_endstate.py``
     - REMD, basic MD and user-supplied end states.
   * - ``restraints.py``, ``restraint_helper.py``
     - Boresch, flat-bottom and conformational restraints.
   * - ``setup_simulations.py``
     - Builds the windows of each leg.
   * - ``matrix_order.py``
     - The order of the states in each leg (``CycleSteps``).
   * - ``alchemical.py``
     - Topology edits: ligand charges, ligand-receptor LJ and ligand GB ``SCREEN``.
   * - ``simulations.py``, ``runner.py``
     - Run an MD window or a scoring job; one scoring row per window.
   * - ``mdin.py``, ``mdout.py``
     - Write AMBER input files and parse AMBER output.
   * - ``adaptive_restraints.py``
     - MBAR and exponential averaging for each leg.
   * - ``pandasmbar.py``, ``block_mbar.py``
     - pymbar on pandas tables, and block-chained MBAR.
   * - ``postTreatment.py``
     - Puts the legs together into the binding free energy (``ConsolidateData``).
   * - ``als_refine.py``
     - Plans the windows post-production ALS adds.
   * - ``pose_clusters.py``
     - Ligand pose clustering for the pose-medoid restraint reference.
   * - ``nfe_energy.py``
     - Energy of AMBER's NFE flat-bottom restraint (used by the experimental RMSD restraints).
   * - ``config.py``
     - Reads and checks the YAML config (:doc:`config`).

Related pages
-------------

- :ref:`Implementation Details <ddm_cycle-label>`: the thermodynamic cycle.
- :doc:`in_solvent_protocol`: the cycle used in the paper.
- :doc:`api`: every module, generated from the code.
