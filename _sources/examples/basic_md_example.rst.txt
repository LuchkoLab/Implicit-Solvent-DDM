Running Basic MD at the endstate simulation 
#####################################################


Running host-guest (CB7) absolute free energy calculation example:
==================================================================
All the example input files are in the ``script_examples`` directory. The config file for this
example is ``script_examples/config_files/basic_md_config.yaml``:

.. literalinclude:: ../../script_examples/config_files/basic_md_config.yaml
   :language: yaml

The ``endstate_method`` is set to ``basic_md`` which will run a basic MD simulation at the endstates (states 1 and 8) :ref:`ddm_cycle-label`.
Every key within the YAML config file has been covered in previous section :ref:`my-reference-label`.


Basic MD mdin template (``md_template_mdin``) 
---------------------------------------------
The template must contain ``DISANG=$restraint``; the workflow writes the restraint file and fills
in the path. This is the template the example uses
(``script_examples/endstate_templates_required/basic_md.template``):

.. literalinclude:: ../../script_examples/endstate_templates_required/basic_md.template
   :language: text

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

    run_implicit_ddm.py file:jobstore_example_01 --config_file script_examples/config_files/basic_md_config.yaml --workDir /scratch/username/

