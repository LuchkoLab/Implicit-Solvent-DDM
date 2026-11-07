.. _my-reference-label:

YAML Config Syntax 
******************
This assumes you have install ISDDM.py successfully. 


We will work through an example of yaml config file for absolute binding free energy calculations (ABFE) in implicit solvent.

ABFE example:
=============
We will go line-by-line through the sections of the following YAML config file that sets up and run ABFE for host-guest system.
    
.. code-block:: yaml

    hardware_parameters:
        working_directory: /nas0/ayoub/Impicit-Solvent-DDM/
        executable: pmemd.MPI  #executable machine for MD choice : [sander, sander.MPI, pmemd, pmemd.MPI, pmeded.CUDA]
        mpi_command: srun # system dependent /
        output_directory_name: isddm_outputdir
        CUDA: False # run on CPUs
        cache_directory_output: path/to/cache_directory

    number_of_cores_per_system:
        complex_ncores: 2 #total number of cores per job
        ligand_ncores: 1 #total number of cores per ligand simulation
        receptor_ncores: 1 #total number of cores per ligand simulation
    
    endstate_parameter_files:
        complex_parameter_filename: script_examples/structs/cb7-mol01_Hmass.parm7 # path to topology file; ["path/to/complex.parm7"]
        complex_coordinate_filename: script_examples/structs/cb7-mol01.rst7 # path to coordinate ["path/to/complex.ncrst"]list of coordinate file of a complex

    AMBER_masks:
        receptor_mask: :CB7 # list of Amber masks denoting receptor atoms in respected complex file
        ligand_mask: :M01 # list of Amber masks denoting ligand atoms in respected complex file

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
            mdin_intermidate_config: script_examples/user_intermidate_mdin_args.yaml #intermidate mdins required states 3-8
            igb_solvent: 2 #igb [1,2,3,7,8]
            temperature: 300
            exponent_conformational_forces: [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.584963, 3, 3.584963, 4]  # list exponent values 2**p 
            exponent_orientational_forces: [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.584963, 7, 7.584963, 8]  # list exponent values 2**p 
            restraint_type: 2 # choices: [ 1: CoM-CoM, 2: CoM-Heavy_Atom, 3: Heavy_Atom-Heavy_Atom, must be 1, 2 or 3 ]
            charges_lambda_window:  [0.5, 1.0]  # 50% 
            charges_lambda_window:  [0.5, 1.0]  # 50% 100%

Hardware Parameters
"""""""""""""""""""
The hardware parameters denote the software and hardware resources there user wishes to use.

.. code-block:: yaml

    #user input values
    hardware_parameters:
        working_directory: /nas0/ayoub/Impicit-Solvent-DDM/
        executable: pmemd.MPI  #executable machine for MD choice : [sander, sander.MPI, pmemd, pmemd.MPI, pmeded.CUDA]
        mpi_command: srun # system dependent /
        output_directory_name: isddm_outputdir
        CUDA: False # run on CPUs
        

The ``working_directory`` keyword is the working directory for exporting output data. Following ``executable`` TheAmber MD engine the user wishes to execute(``sander``, ``sander.MPI``, ``pmemd``,
``pmemd.MPI``, ``pmemd.cuda`` or ``pmemd.cuda.MPI``) for the the end-state and intermediate MD simulations. The ``mpi_command`` used to run MPI programs on the computer system;
e.g., ``mpirun``, ``mpiexec``, or ``srun``. The ``output_directory_name`` keyword is the name of output directory. The ``CUDA`` keyword specifies whether to use GPU acceleration: ``False`` means it will run on CPUs, while ``True`` runs on GPUs.

Packing GPU windows
'''''''''''''''''''

.. code-block:: yaml

    hardware_parameters:
        CUDA: True
        executable: pmemd.cuda
        windows_per_gpu: 4

``windows_per_gpu`` (default ``1``) sets how many ``pmemd.cuda`` windows share each allocated GPU under
Toil's ``single_machine`` batch system. It is ignored (reset to ``1``) for other batch systems.

- **CUDA MPS is required.** Start the daemon on the node before the workflow
  (``nvidia-cuda-mps-control -d``). Without it the windows time-slice and gain nothing; the leader
  prints a warning if the daemon is not found.
- **One core per slot.** Each packed window holds about one CPU core, so ``maxCores`` is lowered by
  ``n_gpus * windows_per_gpu`` to keep CPU post-analysis off those cores. With ``--maxCores`` unset,
  the core count comes from ``SLURM_CPUS_PER_TASK``, else the process CPU affinity.
- **REMD** requests ``remd_accelerators * windows_per_gpu`` slots, so it still claims whole devices.

Benchmark (MCL-1 ligand-17 window, 2,471 atoms, RTX 5000 Ada, ``scripts/gpu_packing_benchmark.py``):
the GPU total is 1,735 / 2,726 / 3,327 / 3,480 ns/day at 1 / 2 / 4 / 8 copies under MPS, with identical
energies. Without MPS it stays flat at ~1,400-1,750 ns/day. ``4`` reaches ~95% of the plateau.

Specify number of performance cores  
"""""""""""""""""""""""""""""""""""
Number of processor to be used for each individual ligand, receptor and complex system.

.. code-block:: yaml

    number_of_cores_per_system:
        complex_ncores: 2 #total number of cores per job
        ligand_ncores: 1 #total number of cores per ligand simulation
        receptor_ncores: 1 #total number of cores per ligand simulation
    
The following keywords ``complex_ncores``, ``ligand_ncores`` and ``receptor_ncores`` denote the of processors to be used for a single intermediate molecular dynamics simulation for respective system. For example every intermidate MD simulation for the complex will request 2 cores. 

Parameterize endstate complex 
"""""""""""""""""""""""""""""
Users need to specify a path to AMBER ``parm7`` and ``rst7`` files to designated bound receptor-ligand-complex system. The user must first parameterize the complex with there desired force fields and charge model. 

.. code-block:: yaml

    endstate_parameter_files:
        complex_parameter_filename: script_examples/structs/cb7-mol01_Hmass.parm7 # path to topology file; ["path/to/complex.parm7"]
        complex_coordinate_filename: script_examples/structs/cb7-mol01.rst7 # path to coordinate ["path/to/complex.ncrst"]list of coordinate file of a complex

The ``complex_parameter_filename`` keyword is a path to an AMBER ``parm7`` topology file, which defines which atoms are bonded to each other. Following ``complex_coordinate_filename`` keyword is a path to an AMBER ``rst7`` file, which defines where each atom is located on a 3-dimensional coordinate plane.

Denote Receptor and Ligand Atoms (AMBER masks)
""""""""""""""""""""""""""""""""""""""""""""""
Amber masks are used to denote ligand and receptor atoms from the complex parameter file.

.. code-block:: yaml

    AMBER_masks:
        receptor_mask: :CB7 # list of Amber masks denoting receptor atoms in respected complex file
        ligand_mask: :M01 # list of Amber masks denoting ligand atoms in respected complex file
    
``receptor_mask`` and ``ligand_mask`` are Amber mask syntax to select receptor and ligand atoms respectivly. 

Workflow 
""""""""
The general workflow to perform implicit solvent ABFE calculations.

.. code-block:: yaml
    
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
            mdin_intermidate_config: script_examples/user_intermidate_mdin_args.yaml #intermidate mdins required states 3-8
            igb_solvent: 2 #igb [1,2,3,7,8]
            temperature: 300
            exponent_conformational_forces: [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.584963, 3, 3.584963, 4]  # list exponent values 2**p 
            exponent_orientational_forces: [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.584963, 7, 7.584963, 8]  # list exponent values 2**p 
            restraint_type: 2 # choices: [ 1: CoM-CoM, 2: CoM-Heavy_Atom, 3: Heavy_Atom-Heavy_Atom, must be 1, 2 or 3 ]
            charges_lambda_window:  [0.5, 1.0]  # 50% 
            gb_extdiel_windows: [0.1, 0.2, 0.5]

Endstate method and arugments:
------------------------------
.. code-block:: yaml
    
    workflow:
        endstate_method: remd #options REMD or 0 (meaning no endstate simulation will be performed just intermidates)endstate_method: REMD #options REMD, MD or 0 (meaning no endstate simulation will be performed just intermidates) 
        endstate_arguments:
            nthreads_complex: 16
            nthreads_receptor: 16
            nthreads_ligand: 16
            temperatures: [300.00, 327.32, 356.62, 388.05, 421.77, 457.91, 496.70, 500.00]
            remd_template_mdin: script_examples/endstate_templates_required/remd.template
            equilibrate_mdin_template: script_examples/endstate_templates_required/equil.template
    
The nested ``endstate_method`` is the type of end-state simulation. The default setting, ``remd``, end-state simulation runs replica exchange molecular dynamics (REMD). If the user wishes to run a standard MD simulation, will specify ``basic_MD``. The user can specify their own pre-calculated end-state simulation by setting value to 0 . If the user denotes ``remd`` for the ``endstate_method`` user must supply ``endstate_arguments``. The nested keys ``nthreads_complex``, ``nthreads_receptor`` and ``nthreads_ligand`` are the total number of processes to run for replica exchange MD for each respective system. As example the ``nthreads_complex`` will request 16 cores to run REMD simulation for the complex system. ``temperatures`` key specify a list of temperature to be ran for REMD. ``remd_template_mdin`` and ``equilibrate_mdin_template`` will be explained down below. 

Equilibrate mdin template example 
---------------------------------
.. code-block:: bash 

 Relaxation 
 &cntrl
   irest=0, ntx=1, 
   nstlim=25000, dt=0.004,
   ntt=3, gamma_ln=1.0,
   temp0=$temp, ig=$ig,
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

REMD mdin template example 
--------------------------
.. code-block:: bash 

 TREMD
 &cntrl
   irest=1, ntx=5, 
   nstlim = 1, dt=0.004, 
   numexchg = 250000.0,
   ntt=3, gamma_ln=1.0,
   temp0=$temp, ig=$ig,
   ntc=2, ntf=2, nscm=0,
   ntb=0, igb=2,
   cut=999.0, rgbmax=999.0,
   ntpr=25, ntwx=25,
   ioutfm=1, nmropt=1,
   saltcon=0.3, gbsa = 0
 /
 &wt TYPE='END'
 /
 DISANG=$restraint

Both the equilibration(relaxtion) and remd template files are AMBER style mdin format. The user can specify any desired parameters for respective run. The only requirment is to leave ``temp0=$temp``, ``ig=$ig`` and ``DISANG=$restraint`` as is, these parameters will be substituted for user during the workflow. 

Intermidate arugments
---------------------
.. code-block:: yaml
    
    workflow:

        intermediate_states_arguments:
            mdin_intermidate_config: script_examples/user_intermidate_mdin_args.yaml #intermidate mdins required states 3-8
            igb_solvent: 2 #igb [1,2,3,7,8]
            temperature: 300
            exponent_conformational_forces: [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.584963, 3, 3.584963, 4]  # list exponent values 2**p 
            exponent_orientational_forces: [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.584963, 7, 7.584963, 8]  # list exponent values 2**p 
            restraint_type: 2 # choices: [ 1: CoM-CoM, 2: CoM-Heavy_Atom, 3: Heavy_Atom-Heavy_Atom, must be 1, 2 or 3 ]
            charges_lambda_window:  [0.5, 1.0]  # 50% 100%
            gb_extdiel_windows: [0.1, 0.2, 0.5] # 10%, 20%, 50%
            restraint_reference: lastframe # lastframe (default) or pose_medoid
            pose_cluster_cutoff: 2.0 # A, only read when restraint_reference is pose_medoid
            pose_pocket_radius: 8.0 # A around the ligand
            pose_cluster_stride: 0 # 0 -> auto
            pose_cluster_skip: 0 # frames dropped from the start before clustering

Lets skip ``mdin_intermidate_config`` for now and look at the following keys. ``igb_solvent`` GB version for all calculation: GB-OBC (igb=2). ``temperature`` Temperature to run intermediate simulations if complete end-state simulations are provided. Note this temperature will be used for target temperature extraction if user wishes to run REMD at the end-states. ``exponent_conformational_forces`` Strength of harmonic conformational restraints are specified by a list of integers to calculate powers of 2, for example -8 gives a restraint coefficient of 2^-8 kcal/mol = 0.00390625 kcal/mol. ``exponent_orientational_forces`` Strength of harmonic orientational restraints are specified by a list of integers to calculate powers of 2. ``restraint_type`` Orientational restraint type to generate orientational based on center of mass of ligand and receptor. default:2. ``charges_lambda_window`` Values to use for linear scaling the ligand electrostatics such as 50% orginal ligand net charge to 100% fully charge. Lastly. ``gb_extdiel_windows`` Values to use when scaling the GB external dielectric.

``restraint_reference`` picks the endstate frame the complex restraints are built from. The default
``lastframe`` keeps the previous behaviour, taking the final frame of the endstate trajectory. That is
a single random draw from the pose ensemble, so on a system whose ligand visits more than one basin it
often lands outside the populated one, which costs restraint work and broadens the energy overlap at
the endstate seam. Set it to ``pose_medoid`` to superpose every frame on the binding site, cluster the
ligand pose, and anchor to the medoid of the most populated cluster instead. This is a variance fix,
not a bias one -- the endstate is the full unrestrained ensemble either way, so the total free energy
is correct under both settings and no multi-pose correction is applied. It affects the complex leg
only. The remaining three keys are read only when ``pose_medoid`` is set. ``pose_cluster_cutoff`` is
the average-linkage distance cutoff in Angstrom on the no-fit ligand RMSD. ``pose_pocket_radius`` is
the radius in Angstrom around the ligand defining the pocket atoms every frame is superposed on:
receptor C-alpha atoms when the receptor has a backbone, all receptor heavy atoms otherwise, so
host-guest systems work too. ``pose_cluster_stride`` is the frame stride, where ``0`` sizes it
automatically to keep the pairwise distance matrix near 2000 frames. ``pose_cluster_skip`` drops that
many frames from the start of the endstate trajectory before clustering, so a burn-in cannot pick the
reference; it does not change which frames the endstate is scored on. When the flag is on the run also
writes ``summary.txt``, ``assignments.csv``, ``pose_analysis.png`` and ``dendrogram.png`` next to the
endstate trajectory, and logs a warning if no cluster holds a majority of the frames.

Mdin intermidate config key
---------------------------
.. code-block:: yaml

    #mdin required input parameters for intermidates 
    nstlim: 2500000
    dt: 0.004
    igb: 2
    saltcon: 0.3
    rgbmax: 999.0
    gbsa: 0
    temp0: 300
    ntpr: 250
    ntwx: 250
    cut: 999
    ntc: 2

These key-value pairs will be used for as input mdin parameters for all intermidate simulation runs. 




