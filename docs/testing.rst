.. _testing-label:

Tests and continuous integration
================================

The tests live in ``implicit_solvent_ddm/tests`` and are split into three tiers by pytest
marker. ``setup.cfg`` sets ``addopts = --strict-markers -m "not workflow"``, so a bare
``pytest`` leaves the workflow tier out. Run pytest from the repository root: the test
configs use paths relative to it.

.. list-table::
   :header-rows: 1
   :widths: 10 38 20 10 22

   * - Tier
     - What it checks
     - Command
     - Time
     - Needs
   * - unit
     - 257 tests of pure logic (config checks, mdin and restraint files, ALS, block-MBAR, GPU
       slot counts). No MD is run.
     - ``pytest``
     - ~4 s
     - ``isddm_env``
   * - workflow
     - One small CB7 run through Toil and ``sander`` (``endstate_method: 0``, no REMD,
       0.2 ps windows), checked by 14 tests.
     - ``pytest -m workflow``
     - 65-90 s
     - ``isddm_env`` (``sander`` comes with AmberTools)
   * - hpc
     - 2 tests: the MCL-1 Boresch anchor test and an ``mpirun`` + ``sander.MPI`` REMD launch
       test.
     - ``pytest -m hpc``
     - seconds
     - the MCL-1 data files, or an MPI build of AMBER

A bare ``pytest`` also collects the two hpc tests. Each one has a ``skipif`` and skips itself
when its files or ``mpirun``/``sander.MPI`` are missing, so on a laptop you see
``257 passed, 2 skipped``. On a cluster node with the AMBER module loaded the REMD launch test
will run.

The workflow tier runs the whole pipeline in the current directory and leaves its output there
(``mdgb/`` and ``.cache/``).

GitHub CI
---------

``.github/workflows/CI.yaml`` runs two CPU jobs in parallel:

``unit``
   ``pytest implicit_solvent_ddm/tests -m "not workflow and not hpc"`` with coverage. The
   report goes to Codecov; an upload failure does not fail the job.

``workflow``
   ``pytest implicit_solvent_ddm/tests -m "workflow and not hpc"``. When it fails, the
   ``mdgb/`` directory is uploaded as the ``workflow-logs`` artifact so you can read the AMBER
   output.

CI runs on every pull request to ``main``, on every push to ``main``, once a week (Monday 07:30
UTC, to catch conda-forge drift) and by hand from the Actions tab (``workflow_dispatch``). A
change that only touches ``docs/``, ``scripts/``, Markdown files or ``docs.yml`` does not start
it. A new push to a pull request cancels the run still going for the older commit. Both jobs
build the env from the lock file and then run ``pip install -e .``; the whole run takes about
2-3 minutes.

GitHub runners have no GPU and no MPI build of AMBER, so the hpc tests never run there. Run them
yourself on the cluster.

The Docs workflow (``.github/workflows/docs.yml``) builds the Sphinx site in the same
environment. It also runs on pull requests that touch the docs or ``README.md``, but it deploys
only from ``main``.

Running the CI checks locally
-----------------------------

This is what CI does, step by step:

.. code-block:: bash

   micromamba create -n isddm_env -f devtools/conda-envs/test_env.conda-lock.yml
   micromamba activate isddm_env
   python -m pip install -e .

   pytest implicit_solvent_ddm/tests -m "not workflow and not hpc"
   pytest implicit_solvent_ddm/tests -m "workflow and not hpc"

For the hpc tests, get an allocation, load your AMBER module so ``mpirun`` and ``sander.MPI``
are on ``PATH``, and run:

.. code-block:: bash

   pytest -m hpc

The MCL-1 anchor test looks for the benchmark data in ``fragment-opt-abfe-benchmark/`` at the
repository root and skips if it is not there.

The lock file
-------------

``devtools/conda-envs/test_env.conda-lock.yml`` lists the exact conda-forge packages (linux-64)
solved from ``devtools/conda-envs/test_env.yaml``. CI and the Docs workflow install from it, so
there is no solve: the env is ready in about 30 s instead of 10 minutes or more.

The lock file is not updated for you. If you change ``test_env.yaml``, regenerate the lock and
commit both files together, or CI keeps testing the old package set:

.. code-block:: bash

   conda-lock -f devtools/conda-envs/test_env.yaml -p linux-64 \
       --lockfile devtools/conda-envs/test_env.conda-lock.yml

``conda-lock`` is not part of ``isddm_env``. Install it separately (it is on conda-forge and
PyPI).

Writing new tests
-----------------

Keep the unit tier pure logic. A unit test must not run an MD engine, need a GPU or MPI, or read
data that is not in the repository. That is what keeps it at a few seconds and lets it run on
every pull request.

A test that needs a GPU, ``mpirun``, an MPI or CUDA build of AMBER, or data files outside the
repository gets the ``hpc`` marker. Give it a ``skipif`` that says what is missing, like the two
that exist now:

.. code-block:: python

   @pytest.mark.hpc
   @pytest.mark.skipif(
       shutil.which("mpirun") is None or shutil.which("sander.MPI") is None,
       reason="needs mpirun and an AMBER MPI build; run locally with the AMBER module loaded",
   )
   def test_mpirun_can_launch_a_replica_ladder():
       ...

A test that needs the CB7 Toil run is a ``workflow`` test. Use the session fixture
``run_workflow`` from ``conftest.py`` so that the pipeline still runs only once per session.

Markers are strict (``--strict-markers``). A new marker has to be registered under ``markers``
in ``setup.cfg`` before pytest accepts it.
