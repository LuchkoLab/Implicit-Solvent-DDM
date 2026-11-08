User Provided Endstate Simulation 
#################################


Running host-guest (CB7) absolute free energy calculation example:
==================================================================
If the user wishes to run there own endstate simulation they just need to provide there own ``.parm7`` and ``.nc`` files. 
Please remove all solvent molecules from the ``.parm7`` and ``.nc`` files if used prior. As-well make sure the ``.parm7`` file has the recommened bondi radii for specified GB solvent model (check AMBER manual). 
All the example input files are in the ``script_examples`` directory. The config file for this
example is ``script_examples/config_files/user_defined_endstate.yaml``:

.. literalinclude:: ../../script_examples/config_files/user_defined_endstate.yaml
   :language: yaml

The ``endstate_method`` is set to ``0`` which assumes the user provided there own endstate simulation. 
Please note if ``endstate_method`` is set to ``0`` the user must provided the following keys values for ``ligand_parameter_filename``, ``ligand_coordinate_filename``, ``receptor_parameter_filename``, and ``receptor_coordinate_filename``.
Every key within the YAML config file has been covered in previous section :ref:`my-reference-label`.

Running entire workflow
-----------------------
The recommend way for running this workflow is by submitting an batch file. This ensures the correct number of resrources and the submission of a single job for entire workflow. Example of batch file for SLURM schedular: 

.. code-block:: bash 

    #!/bin/bash
    #SBATCH --partition=main
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=12
    #SBATCH --time=09:30:00
    #SBATCH --job-name=ex_01
    #SBATCH --export=all

    pwd

    mkdir /scratch/username/

    run_implicit_ddm.py file:jobstore_example_03 --config_file script_examples/config_files/user_defined_endstate.yaml --workDir /scratch/username/

