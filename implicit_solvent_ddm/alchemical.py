"""
Functions that perform alchemical transformations. 
"""
import os

import parmed as pmd
import pytraj as pt


def split_complex_system(
    job,
    endstate_complex_parameter_filename,
    endstate_coordiante_complex_filename,
    ligand_mask,
    receptor_mask,
):
    """
    Splitting a restart coordinate file of an complex into individual receptor and ligand topology and coordinate files

    Parameters
    ----------
    job: toil.job.FunctionWrappingJob
        context manager that represents a Toil workflow
    endstate_complex_parameter_filename: str
        complex parameter topology file absolute path
    endstate_coordiante_complex_filename: str
        basename for complex parameter topology file
    ligand_mask: str
        Amber mask which select ligand atoms from the complex
    receptor_mask: str
        Amber mask which selects receptor atoms from the complex

    Returns
    -------
    receptor_ID: str
        A jobStoreFileID to access receptor_ID globally during the workflow
    ligand_ID: str
        A jobStoreFileID to access ligand_ID globally during the workflow
    """
    temp_dir = job.fileStore.getLocalTempDir()
    # read files into temporary directory
    solute_file = job.fileStore.readGlobalFile(
        endstate_complex_parameter_filename,
        userPath=os.path.join(
            temp_dir, os.path.basename(endstate_complex_parameter_filename)
        ),
    )
    coordinate_file = job.fileStore.readGlobalFile(
        endstate_coordiante_complex_filename,
        userPath=os.path.join(
            temp_dir, os.path.basename(endstate_coordiante_complex_filename)
        ),
    )

    job.fileStore.logToMaster(f"splitting receptor/ligand from {endstate_coordiante_complex_filename}")
    # load in endstate lastframe coordinate of complex file
    traj = pt.load(coordinate_file, solute_file)
    # strip into receptor and ligand respectivlity
    receptor = pt.strip(traj, ligand_mask)
    ligand = pt.strip(traj, receptor_mask)

    receptor_name = "receptor_system" #receptor_mask.strip(":")
    ligand_name = "ligand_system" #ligand_mask.strip(":")
    # write files into temporary directory
    pt.write_traj(f"{temp_dir}/split_{receptor_name}.ncrst", receptor, overwrite=True)
    pt.write_traj(f"{temp_dir}/split_{ligand_name}.ncrst", ligand, overwrite=True)
    # write global copies of receptor and ligand coordiante files into job-store
    receptor_ID = job.fileStore.writeGlobalFile(
        f"{temp_dir}/split_{receptor_name}.ncrst.1"
    )
    ligand_ID = job.fileStore.writeGlobalFile(f"{temp_dir}/split_{ligand_name}.ncrst.1")

    return receptor_ID, ligand_ID


def scale_ligand_receptor_lj(parm, ligand_atoms, receptor_atoms, lj_scale):
    """
    Scale only the ligand-receptor Lennard-Jones coefficients of a topology, in place.

    Parameters
    ----------
    parm : parmed.amber.AmberParm
        Topology to modify.
    ligand_atoms, receptor_atoms : array_like of int
        0-based atom indices.
    lj_scale : float
        Factor on every ligand-receptor A and B coefficient.

    Raises
    ------
    ValueError
        If the ligand and receptor still share an LJ type after the split.
    """
    type_index = parm.parm_data["ATOM_TYPE_INDEX"]
    # addLJType gives the whole mask one type, so split once per original ligand type
    for original in sorted({type_index[i] for i in ligand_atoms}):
        atoms = [int(i) + 1 for i in ligand_atoms if type_index[i] == original]
        pmd.tools.actions.addLJType(parm, "@" + ",".join(map(str, atoms))).execute()
    type_index = parm.parm_data["ATOM_TYPE_INDEX"]
    ligand_types = {type_index[i] for i in ligand_atoms}
    receptor_types = {type_index[i] for i in receptor_atoms}
    if ligand_types & receptor_types:
        raise ValueError("ligand and receptor still share LJ types after addLJType")
    ntypes = parm.ptr("ntypes")
    nb_index = parm.parm_data["NONBONDED_PARM_INDEX"]
    acoef = parm.parm_data["LENNARD_JONES_ACOEF"]
    bcoef = parm.parm_data["LENNARD_JONES_BCOEF"]
    for li in ligand_types:
        for rj in receptor_types:
            slot = nb_index[ntypes * (li - 1) + (rj - 1)]
            if slot > 0:
                acoef[slot - 1] *= lj_scale
                bcoef[slot - 1] *= lj_scale


def alter_topology(
    job,
    solute_amber_parm,
    solute_amber_coordinate,
    ligand_mask,
    receptor_mask,
    exculsions=False,
    set_charge=0.0,
    lj_scale=None,
) -> str:
    """
    Simple command to alter the ligand charge and/or turn off non-bonded interactions with receptor w/ligand.

    Parameters
    ----------
    solute_parm: str
        AMBER parameter file format
    solute_amber_coordinate: str
        AMBER coordinate file format
    ligand_mask: str
        AMBER mask which selects all ligand atoms from the complex
    receptor_mask: str
        AMBER mask which selects all recetor atoms from the complex
    exculsions: bool
        If True no non-bonded interactions between the ligand and receptor will be computed
    set_charge: float
        Specify the percent charge of all ligand atoms. 0.2 -> 20% total charges.
    lj_scale: float, optional
        Scale on the ligand-receptor Lennard-Jones interaction; None leaves it unchanged.
    Returns
    -------
    toil.fileStores.FileID
        Upload a file (as a path) to the job store.
    """
    temp_dir = job.fileStore.getLocalTempDir()
    read_solute_prmtop = job.fileStore.readGlobalFile(
        solute_amber_parm,
        userPath=os.path.join(temp_dir, os.path.basename(solute_amber_parm)),
    )
    read_solute_coordiate = job.fileStore.readGlobalFile(
        solute_amber_coordinate,
        userPath=os.path.join(temp_dir, os.path.basename(solute_amber_coordinate)),
    )

    # load pytraj
    traj = pt.iterload(read_solute_coordiate, read_solute_prmtop)
    ligand_atoms = traj.topology.select(ligand_mask)

    # load parmed
    solute_parm = pmd.load_file(read_solute_prmtop, read_solute_coordiate)
    # before the charge edit: addLJType reloads atoms from parm_data, discarding atom-level changes
    if lj_scale is not None:
        if exculsions:
            raise ValueError("lj_scale and exculsions both set: exclusions already remove ligand-receptor LJ")
        scale_ligand_receptor_lj(solute_parm, ligand_atoms, traj.topology.select(receptor_mask), lj_scale)

    saved_filename = ""

    # alter ligand charge
    reduce_charge = [
        set_charge * charge for charge in solute_parm[ligand_mask].parm_data["CHARGE"]
    ]
    for index, new_charge in enumerate(reduce_charge):
        solute_parm.atoms[ligand_atoms[index]].charge = new_charge
    saved_filename += f"charge_{set_charge}_"

    if lj_scale is not None:
        saved_filename += f"lj_{lj_scale}_"

    if exculsions:
        pmd.tools.actions.addExclusions(
            solute_parm, ligand_mask, receptor_mask
        ).execute()
        saved_filename += "exclusions_"

    if solute_parm.name is not None:
        saved_filename += os.path.basename(solute_parm.name)
    else:
        saved_filename += f"{ligand_mask.strip(':')}.parm7"
    solute_parm.save(saved_filename)

    return job.fileStore.writeGlobalFile(saved_filename)
