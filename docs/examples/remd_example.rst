
Running REMD at the endstate simulation
#####################################################


Running host-guest (CB7) absolute free energy calculation example:
==================================================================
All the example input files are in the ``script_examples`` directory. The config file for this
example is ``script_examples/config_files/remd_config.yaml``:

.. literalinclude:: ../../script_examples/config_files/remd_config.yaml
   :language: yaml

The ``endstate_method`` is set to ``remd`` which will run Replica Exchange Molecular Dynamics simulation at the endstates (states 1 and 8) :ref:`ddm_cycle-label`.
Every key within the YAML config file has been covered in previous section :ref:`my-reference-label`.


Running entire workflow
-----------------------
The recommended way to run this workflow is to submit a batch file, so one job with the right
resources runs the whole workflow. Example batch file for the SLURM scheduler:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --partition=main
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=16
    #SBATCH --time=09:30:00
    #SBATCH --job-name=ex_01
    #SBATCH --export=all

    pwd

    mkdir /scratch/username/

    run_implicit_ddm.py file:jobstore_example_02 --config_file script_examples/config_files/remd_config.yaml --workDir /scratch/username/

Each complex and host REMD job in ``remd_config.yaml`` starts 16 MPI ranks (``nthreads_complex``,
``nthreads_receptor``) for 8 replicas, because ``pmemd.MPI`` needs at least two ranks per replica.
``srun -n 16`` cannot start inside a smaller allocation, so ask for at least 16 tasks. To run the
complex (16 ranks), host (16 ranks) and guest (8 ranks on ``sander.MPI``) REMD at the same time,
ask for 40.

