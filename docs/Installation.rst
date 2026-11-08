.. _installation-label:

Installation
============

Implicit_Solvent_DDM 2.0.0 needs Linux, Python 3.10 or newer, and conda or micromamba. The
install has two parts: a conda environment called ``isddm_env`` that holds the scientific stack
and AmberTools, and the package itself, installed into that environment with pip.

1. Get the code
---------------

.. code-block:: bash

   git clone https://github.com/LuchkoLab/Implicit-Solvent-DDM.git
   cd Implicit-Solvent-DDM

2. Create the environment
-------------------------

You can build ``isddm_env`` two ways. Both give Python 3.10, AmberTools 24.8, pymbar-core 4.2.0,
alchemlyb 2.4.1, numpy below 2 and pandas 2.x.

**From the lock file.** ``devtools/conda-envs/test_env.conda-lock.yml`` pins every conda-forge
package for linux-64, so micromamba installs it without solving anything. It takes about 30
seconds, and you get the same environment CI tests against.

.. code-block:: bash

   micromamba create -n isddm_env -f devtools/conda-envs/test_env.conda-lock.yml
   micromamba activate isddm_env

**From the environment file.** ``devtools/conda-envs/test_env.yaml`` lists the same packages
with version constraints and lets conda (or micromamba) solve them. A cold solve can take ten
minutes or more.

.. code-block:: bash

   conda env create -f devtools/conda-envs/test_env.yaml
   conda activate isddm_env

3. Install the package
----------------------

From the repository root, with ``isddm_env`` active:

.. code-block:: bash

   pip install -e .

This pulls in Toil 8.2.0 and puts ``run_implicit_ddm.py`` on your ``PATH``. To check the
install, run the unit tests. They take a few seconds (see :doc:`testing`).

.. code-block:: bash

   pytest

4. AMBER engines
----------------

The conda-forge ``ambertools`` package gives you ``sander``, ``cpptraj`` and ``tleap``. It does
not have ``pmemd``, ``pmemd.MPI``, ``pmemd.cuda`` or ``sander.MPI``. You need a full AMBER
install for:

* GPU runs (``pmemd.cuda``),
* REMD at the end states, which needs an MPI build,
* the shipped example configs, which use ``pmemd.MPI``.

On a cluster this is usually a module. Load it in the shell or Slurm script that starts the
workflow, then check that the executables your config names are found:

.. code-block:: bash

   which pmemd.MPI pmemd.cuda sander.MPI

Usage
-----

Start runs from the repository root, because the example configs use paths relative to it.

.. code-block:: bash

   run_implicit_ddm.py file:<jobstore> \
       --config_file <config.yaml> \
       --workDir <scratch dir> \
       [--maxCores N]

``file:<jobstore>`` is the Toil job store, a directory Toil creates to keep track of the run.
It is deleted when the run succeeds. If a run stops part way, run the same command again with
``--restart`` added and Toil carries on from where it stopped. ``--workDir`` is where Toil keeps
temporary files, and ``--maxCores`` caps how many cores it uses.

Here is the paper's in-solvent protocol on the CB7/M01 host-guest system that ships in
``script_examples/``:

.. code-block:: bash

   mkdir -p /path/to/scratch
   run_implicit_ddm.py file:jobstore_cb7 \
       --config_file script_examples/config_files/in_solvent_paper.yaml \
       --workDir /path/to/scratch

That config is the full protocol, with REMD at the three end states and 10 ns per window, so
run it on a cluster node that has a full AMBER install. Set ``mpi_command`` in the config to
your system's launcher (``mpirun``, ``mpiexec`` or ``srun``).

``script_examples/config_files/`` also has ``basic_md_config.yaml``, ``remd_config.yaml`` and
``user_defined_endstate.yaml``. The Examples pages walk through them:
:doc:`examples/basic_md_example`, :doc:`examples/remd_example` and
:doc:`examples/user_defined_example`. For the in-solvent cycle in detail, and for packing
several windows on one GPU, see :doc:`in_solvent_protocol` and :doc:`gpu_packing`.
