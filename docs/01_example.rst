
Running Implicit Solvent Double Decouling Simulations
#####################################################


Running host-guest (CB7) absolute free energy calculation example:
==================================================================
All the example inputs scripts can be found in ``script_examples`` directrory. Lets first take a look at our example YAML config file ``config_script1.yaml``. 

.. code-block:: yaml

    hardware_parameters:
        working_directory: working_directory
        executable: "pmemd.MPI" # executable machine for MD choice : [sander, sander.MPI, pmemd, pmemd.MPI, pmeded.CUDA]
        mpi_command: srun # system dependent /
        CUDA: False # run on GPU
        output_directory_name: cb7_examples
        cache_directory_output: path/to/cache_directory

    endstate_parameter_files:
        complex_parameter_filename: script_examples/structs/cb7-mol01_Hmass.parm7 # path to topology file; ["path/to/complex.parm7"]
        complex_coordinate_filename: script_examples/structs/cb7-mol01.rst7 # path to coordinate ["path/to/complex.ncrst"]list of coordinate file of a complex

    number_of_cores_per_system:
        complex_ncores: 2 #total number of cores per job
        ligand_ncores: 1 #total number of cores per ligand simulation
        receptor_ncores: 1 #total number of cores per ligand simulation

    AMBER_masks:
        receptor_mask: ":CB7" # list of Amber masks denoting receptor atoms in respected complex file
        ligand_mask: ":M01" # list of Amber masks denoting ligand atoms in respected complex file

    workflow:
        endstate_method: remd #options REMD or 0 (meaning no endstate simulation will be performed just intermidates)endstate_method: REMD #options REMD, MD or 0 (meaning no endstate simulation will be performed just intermidates) 
        endstate_arguments:
            nthreads_complex: 16
            nthreads_receptor: 16
            nthreads_ligand: 16
            ngroups: 8 
            target_temperature: 300 # list of temperatures or temp.dat file
            remd_template_mdin: script_examples/endstate_templates_required/remd.template
            equilibrate_mdin_template: script_examples/endstate_templates_required/equil.template
            temperatures: [300.00, 327.32, 356.62, 388.05, 421.77, 457.91, 496.70, 500.00]

        intermediate_states_arguments:
            mdin_intermediate_file: script_examples/four_fento_time_step_mdin_template.mdin # AMBER mdin template for every intermediate window
            igb_solvent: 2 #igb [1,2,3,7,8]
            temperature: 300
            exponent_conformational_forces: [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.584963, 3, 3.584963, 4]  # list exponent values 2**p 
            exponent_orientational_forces: [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.584963, 7, 7.584963, 8]  # list exponent values 2**p 
            restraint_type: 2 # choices: [ 1: CoM-CoM, 2: CoM-Heavy_Atom, 3: Heavy_Atom-Heavy_Atom, must be 1, 2 or 3 ]

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

    run_implicit_ddm.py file:jobstore_example_01 --config_file script_examples/config_script1.yaml --workDir /scratch/username/

Submitting via batch file will ensure a sinlge job will be queued rather spawning multiple jobs to be queued. This can also be ran with mutipled nodes but the ``--workDir`` command flag must point to a location that is shared among all the nodes and has high storage capactity.  


User provides there own end-state simulation run:
=================================================
In the case the user wishes to provided there own endstate simulation run, for example running with explicit solvent model. The user must strip out all the water 
