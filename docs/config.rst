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
            mdin_intermediate_file: script_examples/four_fento_time_step_mdin_template.mdin # AMBER mdin template for every intermediate window
            igb_solvent: 2 #igb [1,2,3,7,8]
            temperature: 300
            exponent_conformational_forces: [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.584963, 3, 3.584963, 4]  # list exponent values 2**p 
            exponent_orientational_forces: [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.584963, 7, 7.584963, 8]  # list exponent values 2**p 
            restraint_type: 2 # choices: [ 1: CoM-CoM, 2: CoM-Heavy_Atom, 3: Heavy_Atom-Heavy_Atom, must be 1, 2 or 3 ]
            charges_lambda_window:  [0.5, 1.0]  # 50% 100%

A complete config for the paper protocol (in-solvent decoupling cycle with pairwise and Boresch
restraints) on the same CB7/M01 system is in
``script_examples/config_files/in_solvent_paper.yaml``.

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
-------------------

.. code-block:: yaml

    hardware_parameters:
        CUDA: True
        executable: pmemd.cuda
        windows_per_gpu: 4

``windows_per_gpu`` (default ``1``) sets how many ``pmemd.cuda`` windows share each allocated GPU under
Toil's ``single_machine`` batch system. It is ignored (reset to ``1``) for other batch systems.

- **CUDA MPS is required.** Start the daemon in the job before the workflow
  (``nvidia-cuda-mps-control -d``). Without it the windows time-slice and gain nothing; the leader
  prints a warning if the daemon is not found. The lines to put in the Slurm script are in
  :ref:`gpu-mps`.
- **One core per slot.** Each packed window holds about one CPU core, so ``maxCores`` is lowered by
  ``n_gpus * windows_per_gpu`` to keep CPU post-analysis off those cores. With ``--maxCores`` unset,
  the core count comes from ``SLURM_CPUS_PER_TASK``, else the process CPU affinity.
- **REMD** requests ``remd_accelerators * windows_per_gpu`` slots, so it still claims whole devices.

Four windows per GPU nearly double the throughput of one (1,735 to 3,327 ns/day on an MCL-1
window). The Slurm script, the MPS setup and the full benchmark table are in :doc:`gpu_packing`.

Other hardware keys
-------------------
All of these keys go under ``hardware_parameters`` and are optional.

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Key
     - Type
     - Default
     - Meaning
   * - ``cache_directory_output``
     - str
     - current directory
     - Where the ``.cache`` results (deltaG, overlap matrices, ALS history) are written.
   * - ``num_accelerators``
     - int
     - ``0``
     - GPUs per intermediate MD window. Becomes ``1`` when ``CUDA`` is ``True`` and it is left at
       ``0``; values above ``1`` are clamped to ``1``, since ``pmemd.cuda`` uses one GPU per
       simulation.
   * - ``mbar_accelerators``
     - int
     - ``0``
     - GPUs per MBAR job. pymbar runs on CPU, so leave it at ``0``; values above ``1`` are clamped.
   * - ``remd_executable``
     - str or null
     - ``null``
     - AMBER binary for the REMD endstate legs only. REMD needs an MPI build; set it to
       ``pmemd.MPI`` (with ``remd_accelerators: 0``) when the install has ``pmemd.cuda`` but no
       ``pmemd.cuda.MPI``. ``null`` derives it from ``executable``.
   * - ``remd_accelerators``
     - int
     - ``0``
     - GPUs per REMD job. Not clamped: one job runs every replica, so the natural value is
       ``len(temperatures)``. Fewer GPUs than replicas needs CUDA MPS. ``0`` requests no GPU.
   * - ``export_intermediate_files``
     - bool
     - ``True``
     - ``False`` keeps the intermediate MD and post-analysis outputs in the Toil jobStore only and
       writes just the final results. Post-production ALS needs ``True``.
   * - ``memory``
     - int or str
     - ``"5G"``
     - Memory requested per job.
   * - ``disk``
     - int or str
     - ``"5G"``
     - Disk requested per job.

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

Reusing finished endstate trajectories
--------------------------------------

.. code-block:: yaml

    workflow_jobs:
        run_endstate_method: False

    endstate_parameter_files:
        complex_parameter_filename: path/to/complex.parm7
        complex_coordinate_filename: path/to/complex.rst7      # the run's starting structure
        complex_endstate_trajectory: path/to/complex_298K.nc
        ligand_parameter_filename: path/to/ligand.parm7
        ligand_coordinate_filename: path/to/ligand.rst7
        ligand_endstate_trajectory: path/to/ligand_298K.nc
        ignore_unique_naming: True

``complex_endstate_trajectory`` and ``ligand_endstate_trajectory`` (default ``null``) reuse the
endstate ensembles of a finished run instead of simulating them again. They are read when
``workflow_jobs.run_endstate_method`` is ``False`` and ``endstate_method`` is not ``0``. The
endstate keeps its method's label (for example ``remd_endstate``), so the endstate cells the
finished run already scored are reused. ``*_coordinate_filename`` stays the run's starting
structure. Both trajectories and both parameter files must be given, and each trajectory must have
at least two frames for its topology. ``scripts/prepare_refine.py`` writes configs of this form.

``ignore_unique_naming`` (default ``False``) keeps the ligand and receptor topology names as given.
When it is ``False`` a random three-letter suffix is added, which changes the output directory
names on every submission; set it to ``True`` to resume or re-score a run in the same output
directory.

Denote Receptor and Ligand Atoms (AMBER masks)
""""""""""""""""""""""""""""""""""""""""""""""
Amber masks are used to denote ligand and receptor atoms from the complex parameter file.

.. code-block:: yaml

    AMBER_masks:
        receptor_mask: :CB7 # list of Amber masks denoting receptor atoms in respected complex file
        ligand_mask: :M01 # list of Amber masks denoting ligand atoms in respected complex file
    
``receptor_mask`` and ``ligand_mask`` are Amber mask syntax to select receptor and ligand atoms respectivly. 

Workflow jobs
"""""""""""""
The optional ``workflow_jobs`` block switches parts of the workflow on and off. Every key is a
boolean.

.. code-block:: yaml

    workflow_jobs:
        md_only: False
        adaptive_lambda: False
        run_endstate_method: True
        plot_overlap_matrix: False

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Key
     - Default
     - Meaning
   * - ``md_only``
     - ``False``
     - Run the intermediate MD and stop: no ``imin=5`` scoring and no MBAR. Re-run with it off,
       against the same ``output_directory_name``, to score the trajectories without repeating the
       MD. Set ``ignore_unique_naming: True`` so the second run finds the same directories.
   * - ``adaptive_lambda``
     - ``False``
     - Post-production adaptive lambda scheduling (ALS). After production, every adjacent pair of
       states whose two-state overlap is below ``min_degree_overlap`` gets a new full-length window
       and the run re-scores with warm resume. See `Post-production adaptive lambda scheduling (ALS)`_.
       Off keeps the schedule in the config. This key no longer runs the removed pre-production
       pilot.
   * - ``run_endstate_method``
     - ``True``
     - ``False`` skips the endstate simulations. With ``endstate_method`` other than ``0`` it needs
       ``complex_endstate_trajectory`` and ``ligand_endstate_trajectory``.
   * - ``plot_overlap_matrix``
     - ``False``
     - Write an overlap-matrix heatmap PDF for each leg. The raw overlap matrices are written as
       ``.h5`` either way.

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
            mdin_intermediate_file: script_examples/four_fento_time_step_mdin_template.mdin # AMBER mdin template for every intermediate window
            igb_solvent: 2 #igb [1,2,3,7,8]
            temperature: 300
            exponent_conformational_forces: [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.584963, 3, 3.584963, 4]  # list exponent values 2**p 
            exponent_orientational_forces: [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.584963, 7, 7.584963, 8]  # list exponent values 2**p 
            restraint_type: 2 # choices: [ 1: CoM-CoM, 2: CoM-Heavy_Atom, 3: Heavy_Atom-Heavy_Atom, must be 1, 2 or 3 ]
            charges_lambda_window:  [0.5, 1.0]  # 50% 
            gb_extdiel_windows: [1.25, 1.67, 2.5, 5.0] # explicit epsilon values, lambda = 1 - 1/eps of about 0.2, 0.4, 0.6, 0.8

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
            mdin_intermediate_file: script_examples/four_fento_time_step_mdin_template.mdin # AMBER mdin template for every intermediate window
            igb_solvent: 2 #igb [1,2,3,7,8]
            temperature: 300
            exponent_conformational_forces: [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.584963, 3, 3.584963, 4]  # list exponent values 2**p 
            exponent_orientational_forces: [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.584963, 7, 7.584963, 8]  # list exponent values 2**p 
            restraint_type: 2 # choices: [ 1: CoM-CoM, 2: CoM-Heavy_Atom, 3: Heavy_Atom-Heavy_Atom, must be 1, 2 or 3 ]
            charges_lambda_window:  [0.5, 1.0]  # 50% 100%
            gb_extdiel_windows: [1.25, 1.67, 2.5, 5.0] # explicit epsilon values (not fractions of 78.5)
            restraint_reference: lastframe # lastframe (default), pose_medoid, or a path to a single-frame complex restart
            pose_cluster_cutoff: 2.0 # A, only read when restraint_reference is pose_medoid
            pose_pocket_radius: 8.0 # A around the ligand
            pose_cluster_stride: 0 # 0 -> auto
            pose_cluster_skip: 0 # frames dropped from the start before clustering

``mdin_intermediate_file`` is covered in `Intermediate mdin template`_ below. ``igb_solvent`` GB version for all calculation: GB-OBC (igb=2). ``temperature`` Temperature to run intermediate simulations if complete end-state simulations are provided. Note this temperature will be used for target temperature extraction if user wishes to run REMD at the end-states. ``exponent_conformational_forces`` Strength of harmonic conformational restraints are specified by a list of integers to calculate powers of 2, for example -8 gives a restraint coefficient of 2^-8 kcal/mol = 0.00390625 kcal/mol. ``exponent_orientational_forces`` Strength of harmonic orientational restraints are specified by a list of integers to calculate powers of 2. ``restraint_type`` Orientational restraint type to generate orientational based on center of mass of ligand and receptor. default:2. ``charges_lambda_window`` Values to use for linear scaling the ligand electrostatics such as 50% orginal ligand net charge to 100% fully charge; ``0.0`` and ``1.0`` are always added. ``gb_extdiel_windows`` lists explicit external dielectric constants for the GB-dielectric band of the gas-phase cycle; see `GB-dielectric band (gas-phase cycle)`_.

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
automatically to keep the pairwise distance matrix near 2000 frames. ``pose_cluster_skip`` (int,
default ``0``, must be ``>= 0``) drops that many frames from the start of the endstate trajectory
before clustering, so a burn-in cannot pick the reference; it does not change which frames the
endstate is scored on. When the flag is on the run also writes ``summary.txt``, ``assignments.csv``,
``pose_analysis.png`` and ``dendrogram.png`` next to the endstate trajectory, and logs a warning if
no cluster holds a majority of the frames.

``restraint_reference`` also accepts a path to a single-frame complex restart, such as a crystal
pose or the output of ``scripts/relax_binding_pose.py``. The file must hold exactly one frame that
matches the complex topology. Every leg is then restrained to that structure, every window starts
from it, and the flat-bottom binding site is defined from it. A REMD or basic-MD complex endstate
also starts from it; with ``endstate_method: 0`` the supplied complex ensemble is kept. Unlike the
two keywords, which draw the reference from each run's own sampling, a path gives every replicate
the same reference.

Flat-bottom restraint
---------------------
The ``restraints`` block sits under ``workflow`` and sets the flat-bottom distance restraint that
keeps the ligand in the binding site during endstate sampling. Every key is optional.

.. code-block:: yaml

    workflow:
        restraints:
            flat_bottom_width: 5.0
            harmonic_distance: 10.0
            spring_constant: 1.0
            pocket_radius: 8.0
            restrained_receptor_atoms: ":CB7 & !@H*"   # AMBER mask or 0-based atom indices
            restrained_ligand_atoms: null

The restraint acts on the distance between the centres of mass of the receptor group and the
ligand group. ``flat_bottom_width`` (float, default ``5.0`` Angstrom) sets the half-width of the
flat well around the starting distance, ``harmonic_distance`` (float, default ``10.0`` Angstrom) how far
the harmonic walls extend before the force becomes linear, and ``spring_constant`` (float, default
``1.0``) the wall force constant (``rk2`` and ``rk3``). ``flat_bottom_restraints`` (dict, default
``null``) sets ``r1``, ``r2``, ``r3``, ``r4``, ``rk2`` and ``rk3`` directly instead.

``restrained_receptor_atoms`` and ``restrained_ligand_atoms`` (default ``null``) take a list of
0-based atom indices or an AMBER mask; a mask that selects no atoms is an error. When the receptor
side is ``null`` it becomes the binding site: the C-alpha atoms of residues within
``pocket_radius`` (float, default ``8.0`` Angstrom) of the ligand, or the contacting heavy atoms
when the receptor has no backbone (host-guest). When the ligand side is ``null`` it becomes the
ligand heavy atoms. The site is taken from the starting structure, or from ``restraint_reference``
when that is a file.

The flat-bottom term is no longer added to the binding free energy, because the complex leg already
ends at the flat-bottom-restrained endstate. ``flat_bottom_contribution`` is still reported in the
deltaG output as a wall-contact diagnostic.

GB-dielectric band (gas-phase cycle)
------------------------------------

.. code-block:: yaml

    workflow:
        intermediate_states_arguments:
            gb_extdiel_windows: [1.25, 1.67, 2.5, 5.0]
            gb_extdiel_floor: 1.0
            gb_dielectric_cpu_cores: 8

In the gas-phase cycle the decharged ligand moves between the gas phase and GB water through a band
of GB windows with a reduced external dielectric (``extdiel``). ``gb_extdiel_windows`` (list of
float, default ``[]``) lists those windows as explicit dielectric constants (epsilon). They are not
fractions of 78.5. At load time every value is clamped to [``gb_extdiel_floor``, 78.5] with a
warning, duplicates are removed and the list is sorted. ``gb_extdiel_floor`` (float, default
``1.0``) is the lowest allowed value; epsilon 1 corresponds to vacuum (lambda 0). An older config
written with fractions, such as ``[0.1, 0.2, 0.5]``, therefore loads as ``[1.0]``.

The band is scheduled on lambda = 1 - 1/epsilon, so windows evenly spaced in lambda sit at
epsilon = 1/(1 - lambda). For lambda 0.2, 0.4, 0.6 and 0.8 that is ``[1.25, 1.67, 2.5, 5.0]``. An
empty list turns the band off. The list must be empty when ``decouple_in_solvent`` is ``True``.

``pmemd.cuda`` ignores ``extdiel`` in GB MD, so a GPU GB-dielectric window samples full water.
``gb_dielectric_cpu_cores`` (int or null, default ``null``) runs only these windows on CPU
``pmemd.MPI`` with that many MPI ranks; it must be ``>= 1``. The band's MD and scoring both use
``saltcon=0`` so the GB polar term stays linear in lambda.

LJ windows
----------
``lj_scale_windows`` (list of float, default ``[]``) adds complex windows with the
ligand-receptor Lennard-Jones interaction scaled between ``no_interactions`` (scale 0) and
``interactions`` (scale 1). Values must lie strictly between 0 and 1. The windows are built only
when ``workflow_jobs.complex_ligand_exclusions`` and ``workflow_jobs.complex_turn_off_exclusions``
are on (the default). They work in both cycles.

In-solvent decoupling cycle
---------------------------

.. code-block:: yaml

    workflow:
        intermediate_states_arguments:
            decouple_in_solvent: True
            gb_extdiel_windows: []          # must be empty
            lj_scale_windows: [0.2]
            screening_windows: [0.5]
            charges_lambda_window: [0.5, 1.0]

``decouple_in_solvent`` (bool, default ``False``) decouples the ligand inside GB water instead of
the gas phase. The cycle has no gas state and no GB-dielectric band, so every complex and receptor
window can run on ``pmemd.cuda``. The end states are the same as in the gas-phase cycle, so the
result is still an absolute binding free energy. With the switch off, the gas-phase cycle is
unchanged.

``screening_windows`` (list of float, default ``[]``) adds complex windows that scale the ligand GB
``SCREEN`` parameters between ``interactions`` (scale 0) and the uncharged ``electrostatics`` state
(scale 1). Values must lie strictly between 0 and 1.

Validation at load time:

- ``gb_extdiel_windows`` must be empty when ``decouple_in_solvent`` is ``True``.
- ``screening_windows`` requires ``decouple_in_solvent: True``.
- The intermediate mdin template must have ``gbsa=0`` (or no ``gbsa``), because a decoupled
  ligand would otherwise still change the receptor's surface area.
- ``workflow_jobs.remove_GB_solvent_receptor`` is forced off; there is no gas-phase receptor state.

The full state tables and the free energy assembly are in ``docs/in_solvent_cycle.md``. A complete
config is in ``script_examples/config_files/in_solvent_paper.yaml``.

Receptor restraint selection
----------------------------

.. code-block:: yaml

    workflow:
        intermediate_states_arguments:
            unrestrained_receptor_mask: ":22-32"
            receptor_ramp_mask: null
            receptor_ramp_slope: 2.0
            long_restraint_window_ns: 10.0

``unrestrained_receptor_mask`` (str or null, default ``null``) is an AMBER mask of receptor atoms
left out of the conformational restraint network. The receptor network is written into both the
receptor and the complex restraint files, so the two legs stay consistent and the cycle still
closes. Use it to free floppy regions whose restraint energy dominates the endstate seam without
bearing on binding. ``null`` restrains every receptor atom.

``receptor_ramp_mask`` (str or null, default ``null``) is an AMBER mask of receptor atoms whose
network restraints are ramped instead of dropped: free near the endstate and held at full strength
at the top of the ladder, so the complex and receptor legs stay identical in the GB, charge and LJ
windows. The ramp exponent is ``E_max - receptor_ramp_slope * (E_max - E_con)``, which equals the
network exponent at the top of the ladder. ``receptor_ramp_slope`` (float, default ``2.0``) must be
``>= 1``. The two masks can be combined on different atoms (pairs are dropped first, then the rest
of the selection is ramped); do not ramp and drop the same region. Atoms in either mask are never
used as Boresch anchors. Neither mask can be used with ``rmsd_tolerance_windows``.

``long_restraint_window_ns`` (float or null, default ``10.0``) is on by default. It runs the
receptor leg's weakest restraint window for that many nanoseconds, because the link between the
receptor endstate and that window is the one measured to lose overlap at short run lengths. The
window keeps the user's timestep and Hamiltonian and the same number of frames (``ntwx`` and
``ntpr`` are scaled up), and the setting never shortens a window: if the mdin template already runs
that long, nothing changes. ``null`` turns it off; a value ``<= 0`` is an error. On its own it does
not repair the endstate seam under a whole-receptor network; it helps once
``unrestrained_receptor_mask`` frees the floppy regions.

Boresch anchors and distance ladder
-----------------------------------

.. code-block:: yaml

    workflow:
        intermediate_states_arguments:
            boresch_anchors: pocket_backbone
            boresch_pocket_radius: 10.0
            exponent_boresch_distance_forces: null
            boresch_distance_force_constant: null
            boresch_distance_offset: -4.0

``boresch_anchors`` (str, default ``pocket_backbone``) chooses the receptor anchor atoms of the
Boresch restraint. ``pocket_backbone`` picks R1-R3 from the backbone CA, C and N atoms of residues
within ``boresch_pocket_radius`` (float, default ``10.0`` Angstrom) of the ligand, excluding any
freed or ramped atoms. ``legacy`` allows any receptor heavy atom and takes the longest lever arm,
which on a protein can pick flexible side-chain tips. A receptor with no backbone (host-guest)
always uses ``legacy``.

The Boresch distance term (kcal/mol/Angstrom^2) is released on a different ladder from the angles
and dihedrals, which use ``exponent_orientational_forces`` (kcal/mol/rad^2).
``exponent_boresch_distance_forces`` (list of float or null, default ``null``) sets the distance
ladder as log2 exponents; it must have the same length as ``exponent_orientational_forces``.
``null`` reuses ``exponent_conformational_forces``, which is what every earlier config did. When
neither list is given (RMSD mode only), the distance ladder is the orientational one plus
``boresch_distance_offset`` (float, default ``-4.0``). ``boresch_distance_force_constant`` (float
or null, default ``null``) sets the distance force constant written at the anchor and used in the
analytic standard-state correction; ``null`` takes the largest value of the distance ladder.

.. _block-mbar:

Block-chained post-analysis
---------------------------

.. code-block:: yaml

    workflow:
        intermediate_states_arguments:
            post_analysis_block_size: 3

``post_analysis_block_size`` (int or null, default ``null``) solves each leg as a chain of small
MBAR blocks instead of one dense N x N solve, and runs only the ``sander`` ``imin=5`` evaluations
those blocks need. ``null`` keeps the full N x N post-analysis. ``2`` is the adjacent-window BAR
chain (3N-2 evaluations); larger blocks cost more evaluations and pool more states per solve. The
value must be ``>= 2``. Consecutive blocks share one state, and a transition that crosses a
sub-leg boundary gets its own two-state block. With ALS, the first insertion fixes each leg's block
ends (``block_anchor_states``, set by the workflow; do not set it by hand), so a new window joins
the block spanning its neighbours and no scored cell is lost.

Post-production adaptive lambda scheduling (ALS)
------------------------------------------------

.. code-block:: yaml

    workflow_jobs:
        adaptive_lambda: True

    workflow:
        intermediate_states_arguments:
            min_degree_overlap: 0.04
            als_max_rounds: 3
            als_bands: [restraint, charge, lj, screening]
            als_max_new_windows_per_round: null
            als_min_step: {lj: 0.01}

ALS runs after production. The run first finishes on the windows in the config (round 0). Then
every adjacent pair of states whose two-state overlap ``min(O_ab, O_ba)`` is below
``min_degree_overlap`` gets a full-length window at the midpoint of its gap, and the workflow
re-runs setup, MD and post-analysis with warm resume, so only the new windows and the new energy
cells are computed. This repeats until no pair is weak or ``als_max_rounds`` is reached. The
overlap comes from a two-state MBAR of just those two states (scale 0 to 0.5), not from the full
N-state overlap matrix, whose superdiagonal is about 1/N in a dense band. Pairs that touch the
unrestrained endstate are reported but never filled, and the ends of each band never move.

All keys below go under ``intermediate_states_arguments`` except ``adaptive_lambda``, which goes
under ``workflow_jobs``. They are read only when ``adaptive_lambda`` is ``True``.

.. list-table::
   :header-rows: 1
   :widths: 30 17 18 35

   * - Key
     - Type
     - Default
     - Meaning
   * - ``min_degree_overlap``
     - float
     - ``0.04``
     - Two-state overlap below which a gap is filled (0 to 0.5).
   * - ``als_max_rounds``
     - int
     - ``3``
     - Insertion rounds after production. ``0`` reports the overlaps and inserts nothing.
   * - ``als_bands``
     - list of str
     - ``[restraint, charge, lj, screening]``
     - Bands ALS may fill. ``dielectric`` adds the gas cycle's GB band. Bands the cycle does not
       have are dropped with a warning.
   * - ``als_max_new_windows_per_round``
     - int or null
     - ``null``
     - MD-window budget per round, weakest gaps first. One insertion costs 3 windows (restraint),
       2 (charge, dielectric) or 1 (lj, screening). ``null`` means no limit.
   * - ``als_min_step``
     - dict
     - ``{}``
     - Smallest half-gap a split may create, per band. Missing bands use restraint ``0.05``
       (log2), charge ``0.02``, lj ``0.005``, screening ``0.02``, dielectric ``0.01``.
   * - ``candidate_conformational_pool``
     - list of float
     - ``[]``
     - Explicit restraint candidates (log2 exponents). Empty splits each gap at its midpoint.
   * - ``candidate_charge_pool``
     - list of float
     - ``[]``
     - Explicit charge-fraction candidates, strictly in (0, 1).
   * - ``candidate_lj_pool``
     - list of float
     - ``[]``
     - Explicit LJ-scale candidates, strictly in (0, 1).
   * - ``candidate_screening_pool``
     - list of float
     - ``[]``
     - Explicit SCREEN-scale candidates, strictly in (0, 1).
   * - ``candidate_dielectric_pool``
     - list of float
     - ``[]``
     - Explicit GB-band candidates on the lambda = 1 - 1/epsilon axis.
   * - ``refine_expected_deltaG``
     - float or null
     - ``null``
     - Refine mode: round 0 must reproduce this binding free energy (kcal/mol) within 1e-6 before
       anything is inserted.

With a candidate pool, the candidate nearest the gap midpoint is used. A gap whose split would land
on an existing value, or create a half-gap narrower than ``als_min_step``, is reported as
``weak_pool_exhausted`` instead of being split.

Validation at load time:

- ``hardware_parameters.export_intermediate_files`` must be ``True``; each round resumes from the
  window outputs and scored parquets on disk.
- ``run_post_analysis`` and ``consolidate_output`` must be on and ``md_only`` off.
- Restraints must be generated by the workflow. User restraint files and templates are indexed by
  position, so an inserted window would have none.
- ``als_max_rounds`` must be ``>= 0`` and every band name must be known.
- ``lj`` is dropped without the complex exclusion steps, ``screening`` without
  ``decouple_in_solvent``, ``dielectric`` with ``decouple_in_solvent``, and ``restraint`` in RMSD
  mode.

The keys of the removed pre-production pilot (``pilot_ps``, ``pilot_frames``, ``pilot_nstlim``,
``max_adaptive_iterations``, ``batch_insertion``, ``prune_after_convergence``, ``prune_margin``,
``redistribution_mode``, ``candidate_pool_step``, ``charge_pool_step``, ``dielectric_pool_step``)
are ignored with a warning, so older configs still load.

Results of the last round are written to the normal ``.cache/<complex>/`` files; the per-round
history is in ``.cache/<complex>/als/``. ``scripts/prepare_refine.py`` refines a finished run with
ALS without rerunning REMD. More detail is in ``docs/post_production_als.md``.

RMSD collective-variable restraints (Experimental)
--------------------------------------------------

.. warning::

   Experimental. The RMSD collective-variable restraint and every ``rmsd_*`` key below may change
   in later releases. The mode is off unless ``rmsd_tolerance_windows`` is non-empty.

.. code-block:: yaml

    workflow:
        intermediate_states_arguments:
            rmsd_tolerance_windows: [1.5, 1.0, 0.5, 0.0]   # Angstrom, weakest first
            exponent_orientational_forces: [-4, 0, 4, 8]   # same length, weakest first
            exponent_conformational_forces: []             # still required; may be empty here
            rmsd_force_constant: 10.0
            rmsd_reference_source: restraint_reference

This mode replaces the pairwise conformational network with one best-fit RMSD collective variable
(AMBER NFE ``&pmd`` / ``MULTI_RMSD``) under a flat-bottom umbrella. The ladder moves the upper edge
``r3`` of the flat bottom at a fixed force constant. Once ``r3`` exceeds the free ensemble's own
fluctuation the restraint energy is zero, so the loosest window is the unrestrained endstate. The
restraint energies across the band are computed analytically from the per-frame RMSD
(``implicit_solvent_ddm.nfe_energy``). A worked CB7 setup is in
``script_examples/cb7_rmsd_restraints.md``.

.. list-table::
   :header-rows: 1
   :widths: 30 15 20 35

   * - Key (Experimental)
     - Type
     - Default
     - Meaning
   * - ``rmsd_tolerance_windows``
     - list of float
     - ``[]`` (off)
     - ``r3`` ladder in Angstrom. Needs at least two distinct values, none negative, and the same
       length as ``exponent_orientational_forces``, which must be non-decreasing. Stored
       weakest first (descending); the smallest value is the anchor.
   * - ``rmsd_force_constant``
     - float
     - ``10.0``
     - Force constant in kcal/mol/Angstrom^2, the same in every window.
   * - ``rmsd_reference_source``
     - str
     - ``restraint_reference``
     - Reference positions: ``restraint_reference`` (the binding-mode frame), ``ensemble_average``,
       ``endstate_lastframe``, ``minimized``, or a path to a coordinate file.
   * - ``rmsd_mask``
     - str or null
     - ``null``
     - Restrained receptor atoms. ``null`` uses ``@CA``, falling back to heavy atoms for
       host-guest systems.
   * - ``rmsd_restrain_ligand``
     - bool
     - ``True``
     - Add a second collective variable over the ligand's own conformation.
   * - ``rmsd_ligand_tolerance_scale``
     - float
     - ``1.0``
     - Ligand ``r3`` = scale x receptor ``r3``.
   * - ``rmsd_site_cutoff``
     - float or null
     - ``null``
     - Distance in Angstrom around the ligand for a binding-site collective variable. ``null``
       adds none.
   * - ``rmsd_site_tolerance_scale``
     - float
     - ``1.0``
     - Site ``r3`` = scale x global ``r3``.
   * - ``rmsd_site_atoms``
     - str
     - ``heavy``
     - Site atoms: ``heavy`` (every heavy atom of the site residues) or ``global`` (the global
       group's atom class).
   * - ``rmsd_r1``
     - float
     - ``-10.0``
     - Lower anchor ``r1``; unreachable, so only the upper wall acts.
   * - ``rmsd_r2``
     - float
     - ``0.0``
     - Lower anchor ``r2``.
   * - ``rmsd_r4``
     - float
     - ``1000.0``
     - Upper anchor ``r4``, matching the ``r4=1000`` convention of the restraint templates.
   * - ``rmsd_output_freq``
     - int
     - ``250``
     - ``&pmd`` ``output_freq`` (``pmd.txt``), MD only.
   * - ``rmsd_pool_step``
     - float or null
     - ``null`` (0.25)
     - Reserved ALS pool step in Angstrom. ALS does not insert restraint windows in RMSD mode, so
       it has no effect.
   * - ``candidate_rmsd_pool``
     - list of float
     - ``[]``
     - Reserved ALS candidate pool in Angstrom; no effect for the same reason.

In RMSD mode ``exponent_conformational_forces`` no longer drives a conformational network but must
still be present; it may be an empty list. The Boresch distance ladder then comes from
``exponent_boresch_distance_forces``, else ``exponent_conformational_forces``, else the
orientational ladder plus ``boresch_distance_offset`` (see
`Boresch anchors and distance ladder`_). ``long_restraint_window_ns`` applies to the widest
tolerance. ``unrestrained_receptor_mask`` and ``receptor_ramp_mask`` cannot be used, and ALS drops
the ``restraint`` band.

Intermediate mdin template
--------------------------
``mdin_intermediate_file`` is the AMBER mdin template every intermediate window runs. The workflow
rewrites it per state (``imin``, ``ntx``, ``irest``, ``extdiel``, and ``igb=6`` for gas-phase
states) and builds the ``imin=5`` scoring inputs from it. The example template
``script_examples/four_fento_time_step_mdin_template.mdin`` runs 10 ns per window with a 4 fs
timestep:

.. code-block:: bash

 MDIN for intermidate states
 &cntrl
     imin=0            ! MD
     nstlim = 2500000     ! # of steps
     nscm = 0             ! don't removed COM translations
     nrespa = 1           ! don't use MTS
     ntx = 1           ! read in velocities or not
     irest = 0       ! restart
     dt = 0.004
     ioutfm = 1
     ntb = 0

     igb = 2, saltcon=0.3
     gbsa = 0
     rgbmax= 999.0
     cut = 999

     ntt = 3
     ig = -1
     gamma_ln=1
     temp0 = 298

     ntc = 2              ! SHAKE all hydrogen bonds
     ntf = 2

     nmropt = 1

     ntxo=2,ioutfm=1    ! binary restart and trajectories
     ntpr=250         ! print energies
     ntwx=250        ! write to trajectory

     extdiel = $extdiel

 /


 &wt type='END'

 /

 DISANG = $restraint

The template must contain ``igb``, ``nmropt``, ``extdiel = $extdiel`` and ``DISANG = $restraint``;
the workflow fills in the two placeholders. With ``decouple_in_solvent`` it must also have
``gbsa=0`` or no ``gbsa``.

There is no ``score_igb`` config key. ``implicit_solvent_ddm.mdin.make_mdin_file`` takes an optional
``score_igb`` argument that rewrites ``igb`` in a generated mdin so a trajectory can be re-scored
under another GB model. The workflow never passes it; it is meant for analysis scripts that call
``make_mdin_file`` directly.
