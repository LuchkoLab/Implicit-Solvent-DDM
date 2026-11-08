Implicit Solvent DDM
====================

Implicit Solvent DDM computes absolute binding free energies with the double decoupling method
in generalized Born implicit solvent. One YAML file describes the system and the protocol. The
workflow then runs the end-state sampling, every alchemical window and the MBAR analysis through
Toil, on CPUs or GPUs.

Version 2.0 adds the in-solvent protocol used in the paper, post-production adaptive lambda
scheduling (ALS), GPU window packing and block-MBAR.

Paper: `Automated Workflow for Absolute Binding Free Energy Calculations with Implicit Solvent and
Double Decoupling <https://arxiv.org/abs/2509.21808>`_ (arXiv:2509.21808).

Get started
-----------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`download;1.2em;sd-mr-1` Install
      :link: Installation
      :link-type: doc

      Create the environment from the lock file, add a full AMBER install, and check it with
      the unit tests.

   .. grid-item-card:: :octicon:`workflow;1.2em;sd-mr-1` The workflow
      :link: DDM_Workflow
      :link-type: doc

      How the cycle is built: the three legs, their states, and the order the jobs run in.

   .. grid-item-card:: :octicon:`gear;1.2em;sd-mr-1` Config reference
      :link: config
      :link-type: doc

      Every YAML key, section by section, with its default.

Run
---

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`beaker;1.2em;sd-mr-1` In-solvent protocol
      :link: in_solvent_protocol
      :link-type: doc

      The paper protocol: the ligand is switched off inside GB water, charges first, then
      screening, then LJ.

   .. grid-item-card:: :octicon:`zap;1.2em;sd-mr-1` GPU window packing
      :link: gpu_packing
      :link-type: doc

      Run several windows on each GPU under CUDA MPS. Four per GPU nearly doubles throughput.

   .. grid-item-card:: :octicon:`file-code;1.2em;sd-mr-1` Examples
      :link: examples/basic_md_example
      :link-type: doc

      Basic MD, REMD and user-supplied end states on the CB7/M01 host-guest system.

Reference
---------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`code;1.2em;sd-mr-1` API reference
      :link: api
      :link-type: doc

      The Python modules behind the workflow.

   .. grid-item-card:: :octicon:`checklist;1.2em;sd-mr-1` Tests and CI
      :link: testing
      :link-type: doc

      The unit, workflow and hpc test tiers, and what GitHub runs on every pull request.

   .. grid-item-card:: :octicon:`megaphone;1.2em;sd-mr-1` What's new in 2.0
      :link: https://github.com/LuchkoLab/Implicit-Solvent-DDM/blob/main/RELEASE_NOTES_v2.0.0.md

      The 2.0.0 release notes, including the changes that alter computed free energies.

.. toctree::
   :hidden:
   :caption: Get started

   Installation
   Implementation
   DDM_Workflow
   config

.. toctree::
   :hidden:
   :caption: Run

   in_solvent_protocol
   gpu_packing

.. toctree::
   :hidden:
   :caption: Examples

   examples/basic_md_example
   examples/remd_example
   examples/user_defined_example

.. toctree::
   :hidden:
   :caption: Reference

   api
   testing
   Updates
   about
