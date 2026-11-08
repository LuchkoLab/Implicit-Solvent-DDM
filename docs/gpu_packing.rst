.. _gpu-packing:

Running on GPUs with window packing
===================================

A single GB window of a small system does not keep a modern GPU busy. One MCL-1 window
(2,471 atoms) on an RTX 5000 Ada uses about half the card. Window packing runs several
``pmemd.cuda`` windows on the same GPU at once, and CUDA MPS (Multi-Process Service) lets them
share the card at the same time instead of taking turns. On that window, four per GPU nearly
doubles the throughput.

When to use it
--------------

Use packing with the in-solvent cycle (:doc:`in_solvent_protocol`). Every complex and receptor
window in that cycle can run on ``pmemd.cuda``.

Do not put a gas-phase cycle with ``gb_extdiel_windows`` on the GPU. ``pmemd.cuda`` ignores
``extdiel`` in GB MD, so those windows sample water and give a wrong answer. Nothing in the
code catches this yet.

Config
------

All of this goes under ``hardware_parameters``:

.. code-block:: yaml

    hardware_parameters:
      executable: pmemd.cuda     # the windows run serial pmemd.cuda
      CUDA: True
      num_accelerators: 1        # GPUs per window; anything above 1 is clamped to 1
      windows_per_gpu: 4         # windows sharing each GPU (default 1 = no packing)
      remd_accelerators: 2       # GPUs per REMD job; the replicas share them under MPS
      mpi_command: mpirun

The REMD end states then run on ``pmemd.cuda.MPI``, which the workflow derives from
``executable``. If your AMBER build has no ``pmemd.cuda.MPI``, or you would rather keep the
replica ladder on CPU, run it there and leave the windows on the GPU:

.. code-block:: yaml

    hardware_parameters:
      remd_executable: pmemd.MPI
      remd_accelerators: 0

``windows_per_gpu`` only works when the whole workflow runs inside one Slurm job, which is Toil's
``single_machine`` batch system and the normal way to start a run. With any other batch system it
is reset to 1.

The Slurm job
-------------

Three things have to be right in the job script: ask Slurm for the GPUs, start an MPS daemon
before the workflow, and ask for enough CPU cores. This is the script the MCL-1 production runs
used, with the site-specific parts made generic:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:2
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=90
    #SBATCH --mem=128G
    #SBATCH --time=48:00:00

    conda activate isddm_env
    # Some AMBER modules set PYTHONPATH, which can hide the env's pytraj. Keep the old value.
    _PP_BEFORE=${PYTHONPATH-}
    module load amber            # must provide pmemd.cuda (and pmemd.cuda.MPI for GPU REMD)
    if [ -n "$_PP_BEFORE" ]; then export PYTHONPATH="$_PP_BEFORE"; else unset PYTHONPATH; fi

    for exe in pmemd.cuda nvidia-cuda-mps-control; do
        command -v "$exe" >/dev/null || { echo "ERROR: $exe not on PATH"; exit 1; }
    done

    # A private MPS daemon for this job, so two jobs on one node do not share one.
    # Toil's workers inherit the pipe directory from the environment.
    mkdir -p /tmp/slurm.$SLURM_JOB_ID
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/slurm.$SLURM_JOB_ID/mps_pipe
    export CUDA_MPS_LOG_DIRECTORY=/tmp/slurm.$SLURM_JOB_ID/mps_log
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d || { echo "ERROR: could not start MPS"; exit 1; }
    trap 'echo quit | nvidia-cuda-mps-control' EXIT

    run_implicit_ddm.py "file:/tmp/slurm.$SLURM_JOB_ID/jobstore" \
        --config_file my_config.yaml \
        --workDir /tmp/slurm.$SLURM_JOB_ID/ \
        --maxCores 90

Start the script from the directory your config's relative paths assume (the repository root
for the shipped examples). The ``trap`` line shuts the daemon down when the job ends, even if the
workflow fails.

The job store above sits on the node's local ``/tmp``, which is fast but disappears with the job.
If you want to continue a failed run with ``--restart``, put the job store on shared storage
instead.

CPU cores
---------

Each packed window keeps one CPU core busy: the ``pmemd.cuda`` host thread launches kernels and
waits on the GPU. The workflow sets those cores aside so that CPU work, such as the ligand leg
and the energy re-scoring, does not land on them. It takes ``GPUs x windows_per_gpu`` cores off
``--maxCores``. If ``--maxCores`` is not given, it starts from ``SLURM_CPUS_PER_TASK``, or from
the CPUs the process is allowed to use.

With 2 GPUs and ``windows_per_gpu: 4``, 8 cores are held for the GPU windows and the other 82 of
the 90 do the CPU work. If there are fewer cores than window slots, the run stops at the start
with:

.. code-block:: text

    [GPU] 8 GPU window slots need one core each, but only 4 core(s) are available;
    lower windows_per_gpu or request more CPUs.

REMD asks for ``remd_accelerators x windows_per_gpu`` slots, so a REMD job still gets whole
GPUs to itself and its replicas share them under MPS.

Checking that it works
----------------------

The job's ``.out`` file shows what the workflow found. A good start looks like this:

.. code-block:: text

    [GPU] allocation detect: ... -> allocated_gpus=['0', '1'] (via ...)
    [GPU] maxCores lowered to 82 for 8 GPU slot(s)

Two lines mean something is wrong:

``[GPU] WARNING: no GPUs detected -> single_machine GPU pinning DISABLED``
    Slurm gave the job no GPU. Check ``#SBATCH --gres=gpu:N``.

``[GPU] WARNING: CUDA MPS daemon not found.``
    The windows will take turns on the GPU and run about ``windows_per_gpu`` times slower each,
    with no gain overall. Check that ``nvidia-cuda-mps-control -d`` ran in the job script.

While the windows run, ``nvidia-smi`` on the node should list several ``pmemd.cuda`` processes
on each GPU.

Benchmark
---------

Measured with ``scripts/gpu_packing_benchmark.py`` on one MCL-1 ligand-17 window (2,471 atoms)
on an NVIDIA RTX 5000 Ada, with MPS on. Every copy ran the same input, and the energies of all
copies were identical (largest difference in EPtot: 0).

.. list-table::
   :header-rows: 1
   :widths: 18 22 18 16 26

   * - Windows per GPU
     - ns/day per window
     - Total ns/day
     - vs 1 window
     - GPU memory peak (MiB)
   * - 1
     - 1,735
     - 1,735
     - 1.00x
     - 436
   * - 2
     - 1,363
     - 2,726
     - 1.57x
     - 835
   * - 4
     - 832
     - 3,327
     - 1.92x
     - 1,632
   * - 8
     - 435
     - 3,480
     - 2.01x
     - 3,227
   * - 12
     - 290
     - 3,480
     - 2.01x
     - 4,823
   * - 16
     - 218
     - 3,489
     - 2.01x
     - 6,418

Throughput levels off at about twice the single-window rate. Four windows per GPU already get
about 95% of that, which is why the production runs use ``windows_per_gpu: 4``; more copies only
make each window slower. Without MPS the total stays around 1,400 to 1,750 ns/day however many
windows you pack.

The production runs held up the same way. MCL-1 ligand 4 with 10 ns windows ran 124 GPU windows
on 2 GPUs in 4.42 hours of MD, about 3,450 ns/day per GPU, close to twice the rate with one window
per GPU. Packing did not change the physics either: the MD-averaged energies of every window in
packed and unpacked runs agree within the scatter between replicates.
