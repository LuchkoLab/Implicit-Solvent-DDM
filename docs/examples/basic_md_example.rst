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
Only requirment is to set ``DISANG=$restraint`` the restraint file will be loaded in during the workflow.

.. code-block:: bash 

 Relaxation 
 &cntrl
   irest=0, ntx=1, 
   nstlim=25000, dt=0.004,
   ntt=3, gamma_ln=1.0,
   temp0=300, ig=-1,
   ntc=2, ntf=2, nscm=1000,
   ntb=0, igb=2,
   cut=999.0, rgbmax=999.0,
   ntpr=1, ntwx=1,
   nmropt=1, ioutfm=1,
   saltcon=0.3, gbsa = 0
 /
 &wt TYPE='END'
 /
 DISANG=$restraint  

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

