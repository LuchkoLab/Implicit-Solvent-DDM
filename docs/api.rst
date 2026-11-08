API Documentation
=================

The Implicit Solvent DDM (Double Decoupling Method) package provides a comprehensive Python API for performing absolute binding free energy calculations using implicit solvent models. The API is organized into several key modules that handle different aspects of the workflow:

**Core Workflow Modules:**

- :mod:`implicit_solvent_ddm.implicit_ddm_workflow` - Main workflow orchestration and execution
- :mod:`implicit_solvent_ddm.workflow_phases` - Defines the different phases of the DDM cycle
- :mod:`implicit_solvent_ddm.setup_simulations` - Handles simulation setup and configuration
- :mod:`implicit_solvent_ddm.als_refine` - Post-production adaptive lambda scheduling (ALS)

**Simulation Management:**

- :mod:`implicit_solvent_ddm.runner.IntermidateRunner` - Manages intermediate state simulations
- :mod:`implicit_solvent_ddm.run_endstate` - Handles end-state simulation execution
- :mod:`implicit_solvent_ddm.simulations` - Core simulation utilities and functions

**Configuration and Control:**

- :mod:`implicit_solvent_ddm.config` - Configuration file parsing and validation
- :mod:`implicit_solvent_ddm.matrix_order.CycleSteps` - Defines the thermodynamic cycle steps

**Restraints and Alchemical Transformations:**

- :mod:`implicit_solvent_ddm.adaptive_restraints` - Adaptive restraint application and management
- :mod:`implicit_solvent_ddm.alchemical` - Alchemical transformation protocols
- :mod:`implicit_solvent_ddm.restraints` - Restraint definitions and utilities
- :mod:`implicit_solvent_ddm.restraint_helper` - Helper functions for restraint calculations
- :mod:`implicit_solvent_ddm.pose_clusters` - Ligand pose clustering on a pocket-aligned trajectory
- :mod:`implicit_solvent_ddm.nfe_energy` - Analytic energy of AMBER's NFE ``&pmd`` flat-bottom umbrella

**File I/O and Analysis:**

- :mod:`implicit_solvent_ddm.mdin` - AMBER input file generation and manipulation
- :mod:`implicit_solvent_ddm.mdout` - AMBER output file parsing and analysis
- :mod:`implicit_solvent_ddm.pandasmbar` - MBAR analysis integration with pymbar
- :mod:`implicit_solvent_ddm.block_mbar` - Block-chained MBAR
- :mod:`implicit_solvent_ddm.postTreatment` - Post-processing and analysis utilities

This API enables researchers to programmatically set up, execute, and analyze implicit solvent binding free energy calculations with full control over the thermodynamic cycle parameters, restraint schemes, and analysis protocols.

.. autosummary::
   :toctree: autosummary
   :recursive:

   implicit_solvent_ddm.implicit_ddm_workflow
   implicit_solvent_ddm.workflow_phases
   implicit_solvent_ddm.setup_simulations
   implicit_solvent_ddm.als_refine
   implicit_solvent_ddm.runner.IntermidateRunner
   implicit_solvent_ddm.run_endstate
   implicit_solvent_ddm.config
   implicit_solvent_ddm.matrix_order.CycleSteps
   implicit_solvent_ddm.adaptive_restraints
   implicit_solvent_ddm.alchemical
   implicit_solvent_ddm.mdin
   implicit_solvent_ddm.mdout
   implicit_solvent_ddm.pandasmbar
   implicit_solvent_ddm.block_mbar
   implicit_solvent_ddm.postTreatment
   implicit_solvent_ddm.restraint_helper
   implicit_solvent_ddm.restraints
   implicit_solvent_ddm.pose_clusters
   implicit_solvent_ddm.nfe_energy
   implicit_solvent_ddm.simulations
