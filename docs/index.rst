:hide-toc:

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

.. rubric:: Get started
   :class: landing-section

.. grid:: 1 2 3 3
   :gutter: 2
   :class-container: landing-grid

   .. grid-item-card:: :octicon:`download;1.1em;sd-mr-1` Installation
      :link: Installation
      :link-type: doc

   .. grid-item-card:: :octicon:`gear;1.1em;sd-mr-1` YAML Config Syntax
      :link: config
      :link-type: doc

   .. grid-item-card:: :octicon:`file-code;1.1em;sd-mr-1` Examples
      :link: examples/basic_md_example
      :link-type: doc

.. rubric:: Method
   :class: landing-section

.. grid:: 1 2 3 3
   :gutter: 2
   :class-container: landing-grid

   .. grid-item-card:: :octicon:`book;1.1em;sd-mr-1` Implementation Details
      :link: Implementation
      :link-type: doc

   .. grid-item-card:: :octicon:`git-branch;1.1em;sd-mr-1` Conformationally restrained Alchemical-Pathway
      :link: alchemical-pathway
      :link-type: ref

   .. grid-item-card:: :octicon:`workflow;1.1em;sd-mr-1` DDM Workflow Structure
      :link: DDM_Workflow
      :link-type: doc

.. rubric:: Run
   :class: landing-section

.. grid:: 1 2 3 3
   :gutter: 2
   :class-container: landing-grid

   .. grid-item-card:: :octicon:`beaker;1.1em;sd-mr-1` Running the in-solvent (paper) protocol
      :link: in_solvent_protocol
      :link-type: doc

   .. grid-item-card:: :octicon:`zap;1.1em;sd-mr-1` Running on GPUs with window packing
      :link: gpu_packing
      :link-type: doc

.. rubric:: Reference
   :class: landing-section

.. grid:: 1 2 3 3
   :gutter: 2
   :class-container: landing-grid

   .. grid-item-card:: :octicon:`code;1.1em;sd-mr-1` API Documentation
      :link: api
      :link-type: doc

   .. grid-item-card:: :octicon:`checklist;1.1em;sd-mr-1` Tests and continuous integration
      :link: testing
      :link-type: doc

   .. grid-item-card:: :octicon:`megaphone;1.1em;sd-mr-1` What's new in 2.0
      :link: https://github.com/LuchkoLab/Implicit-Solvent-DDM/blob/main/RELEASE_NOTES_v2.0.0.md

.. toctree::
   :hidden:
   :caption: Get started

   Installation
   config

.. toctree::
   :hidden:
   :caption: Method

   Implementation
   DDM_Workflow

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
