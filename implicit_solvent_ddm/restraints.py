"""
Classes for Boresch, FlatBottom and conformational restraints. 
"""
import itertools
import math
import os
import re
import time
from string import Template
from typing import Optional, Union

import numpy as np
import pandas as pd
import parmed as pmd
import pytraj as pt
from toil.common import FileID, Toil
from toil.job import Job
import subprocess
import tempfile
import warnings
from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.restraint_helper import (
    compute_dihedral_angle,
    count_heavy_neighbors,
    create_atom_neighbor_array,
    distance_calculator,
    find_angle,
    norm_distance,
    ramped_force,
    refactor_find_heavy_bonds,
    screen_for_distance_restraints,
    shortest_distance_between_molecules,
)


class FlatBottom(Job):
    """A receptor-ligand restraint using a flat potential well with harmonic walls.

    A receptor-ligand restraint that uses flat potential inside the
    host/protein volume with harmonic restrainting walls. It will
    prevent the ligand drifting too far from the receptor during
    implicit solvent calculations. The ligand will be allow
    for free movement in the "bound" region and sample still different
    binding modes. The restriant will be applied between the groups of
    atoms that belong to the receptor and ligand respectively.

    Parameters
    ----------
    config : Config
        The config is an configuration file containing
        user input values

    Attributes
    ----------
    well_radius : simtk.unit.Quantity, optional
        The distance r0 (see energy expression above) at which the harmonic
        restraint is imposed in units of distance (default is None).
    restrained_receptor_atoms : iterable of int, int, or str, optional
        The indices of the receptor atoms to restrain, an
        This can temporarily be left undefined, but ``_missing_parameters()``
        will be called which will define receptor atoms by the provided AMBER masks.
    restrained_ligand_atoms : iterable of int, int, or str, optional
        The indices of the ligand atoms to restrain.
        This can temporarily be left undefined, but ``_missing_parameters()``
        will be called which will define ligand atoms by the provided AMBER masks.
    flat_bottom_width: float, optional
        The distance r0  at which the harmonic restraint is imposed.
        The well with a square bottom between r2 and r3, with parabolic sides out
        to a defined distance. This has an default value of 5 Å if not provided.
    harmonic_restraint: float, optional
        The upper bound parabolic sides out to define distance
        (r1 and r4 for lower and upper bounds, respectively),
        and linear sides beyond that distance. This has an default
        value of 10 Å, if not provided.
    spring_constant: float
        The spring constant K in units compatible
        with kJ/mol*nm^2 f (default is 1 kJ/mol*nm^2).
    flat_bottom_restraints: dict, optional
        User provided {r1, r2, r3, r4, rk2, rk3} restraint
        parameters. This can be temporily left undefined, but
        ``_missing_parameters()`` will be called which which would
        define all the restraint parameters. See example down below.
    receptor_mask: str
        An AMBER mask which denotes all receptor atoms.
    ligand_mask: str
        An AMBER mask which denotes all ligand atoms.
    complex_topology: toil.fileStores.FileID
        The complex paramter (.parm7) filepath.
    complex_coordinate: toil.fileStores.FileID
        The complex coordinate (.ncrst, rst7, ect) filepath.
    """

    def __init__(
        self,
        config: Config,
        memory: Optional[Union[int, str]] = None,
        cores: Optional[Union[int, float, str]] = None,
        disk: Optional[Union[int, str]] = None,
        preemptable: Optional[Union[bool, int, str]] = None,
        unitName: Optional[str] = "",
        checkpoint: Optional[bool] = False,
        displayName: Optional[str] = "",
        descriptionClass: Optional[str] = None,
    ) -> None:
        super().__init__(
            memory,
            cores,
            disk,
            accelerators=None,
            preemptible="false",
            unitName=unitName,
            checkpoint=checkpoint,
            displayName=displayName,
        )

        # restraint parameters
        self.restrained_receptor_atoms = (
            config.endstate_method.flat_bottom.restrained_receptor_atoms
        )
        self.restrained_ligand_atoms = (
            config.endstate_method.flat_bottom.restrained_ligand_atoms
        )
        self.flat_bottom_width = config.endstate_method.flat_bottom.flat_bottom_width
        self.harmonic_restraint = config.endstate_method.flat_bottom.harmonic_distance
        self.spring_constant = config.endstate_method.flat_bottom.spring_constant
        self.flat_bottom_restraints = (
            config.endstate_method.flat_bottom.flat_bottom_restraints
        )
        # amber masks
        self.receptor_mask = config.amber_masks.receptor_mask
        self.ligand_mask = config.amber_masks.ligand_mask
        # topology parameters
        self.complex_topology = config.endstate_files.complex_parameter_filename
        # the fixed restraint reference when given (same file as the Boresch and conformational
        # restraints); with endstate_method 0 complex_coordinate_filename is the ensemble
        self.complex_coordinate = (
            config.endstate_files.restraint_reference_coordinate
            or config.endstate_files.complex_coordinate_filename
        )

        self.readfiles = {}

    @property
    def _restrained_atoms_given(self) -> bool:
        """Check if the atoms were defined for ligand and receptor"""

        for atoms in [self.restrained_receptor_atoms, self.restrained_ligand_atoms]:
            if atoms is None or not (isinstance(atoms, list)) and len(atoms) > 0:
                return False
        return True

    @property
    def _restraints_parameters_given(self) -> bool:
        """Check if the AMBER restraint parameters were given"""
        for parameters in [self.flat_bottom_restraints]:
            if (
                parameters is None
                or not (isinstance(parameters, dict))
                and len(parameters) > 0
            ):
                return False
        return True

    @property
    def _com_ligand(self) -> np.ndarray:
        """Compute ligand center of mass"""

        # frame 0, matching the pocket selection; identical for a single-frame restart
        return pt.center_of_mass(self.complex_traj, mask=self.restrained_ligand_atoms)[0]  # type: ignore

    @property
    def _com_receptor(self) -> np.ndarray:
        """Compute receptor center of mass"""

        return pt.center_of_mass(self.complex_traj, mask=self.restrained_receptor_atoms)[0]  # type: ignore

    @property
    def _center_of_mass_difference(self) -> float:
        """The center of mass difference between the receptor and ligand

        Returns:
            float:
        """
        return float(
            abs(np.linalg.norm(np.subtract(self._com_receptor, self._com_ligand)))
        )

    @property
    def _r1(self):
        """Compute lower bound linear response region"""

        return max(0, self._r2 - self.harmonic_restraint)

    @property
    def _r2(self):
        """Compute lower bounds of the flat well"""

        return max(0, self._center_of_mass_difference - self.flat_bottom_width)

    @property
    def _r3(self):
        """Compute the upper bound of the flat well"""

        return self._r2 + min(
            self._center_of_mass_difference + self.flat_bottom_width,
            2 * self.flat_bottom_width,
        )

    @property
    def _r4(self):
        """Compute upper bound linear response region"""

        return self._r3 + self.harmonic_restraint

    @property
    def _flat_bottom_restraints_template(self):
        """Parse in flat bottom restraint template.


        Returns:
            _type_: str
            return an string template with specified restraint
            parameters for AMBER (center of mass) flatbottom restraint file.
        """
        restraint_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "templates/restraints/COM.restraint",
            )
        )
        string_template = ""

        with open(restraint_path) as f:
            template = Template(f.read())

            restraint_template = template.substitute(   
                host_atom_numbers=",".join(
                    [str(atom_index + 1) for atom_index in self.restrained_receptor_atoms], 
                ),
                guest_atom_numbers=",".join(
                    [str(atom_index + 1) for atom_index in self.restrained_ligand_atoms], 
                ),
                r1=self.flat_bottom_restraints["r1"],
                r2=self.flat_bottom_restraints["r2"],
                r3=self.flat_bottom_restraints["r3"],
                r4=self.flat_bottom_restraints["r4"],
                rk2=self.flat_bottom_restraints["rk2"],
                rk3=self.flat_bottom_restraints["rk3"],
            )

            string_template += restraint_template

        return string_template

    @property
    def _protein_like(self) -> bool:
        """Check if the system is protein-like.
        
        Returns:
            bool: True if the system is protein-like, False otherwise.
        """
        ca_sel = self.topology.select(f"{self.receptor_mask} & @CA")
        if ca_sel is not None and ca_sel.size > 0:
            # Protein-like system: restrain only backbone Cα atoms
            return True 
        else:
            # Non-protein (e.g., host–guest): restrain all heavy atoms
            return False 
    
    def _missing_parameters(self):
        """ "Automatically determine missing parameters"""
                
        if not self._restrained_atoms_given:
            self._determine_restraint_atoms()
        if not self._restraints_parameters_given:
            self._determine_restraint_parameters()

    def _determine_restraint_atoms(self):
        """Select receptor and ligand atoms for restraints.

        - Ligand: always use heavy atoms (!@H*)
        - Receptor:
            * If receptor contains Cα atoms (@CA), use those (protein case)
            * Otherwise, use all heavy atoms (host–guest or non-protein)
        """
        self.restrained_ligand_atoms, _ = select_ligand_restraint_atoms(
            self.topology, self.ligand_mask
        )
        self.restrained_receptor_atoms, _ = select_receptor_restraint_atoms(
            self.topology, self.receptor_mask
        )

        
    def _determine_restraint_parameters(self):
        """Define distance, harmonic and linear restraint values."""

        self.flat_bottom_restraints = {}
        self.flat_bottom_restraints["r1"] = self._r1  # type: ignore
        self.flat_bottom_restraints["r2"] = self._r2  # type: ignore
        self.flat_bottom_restraints["r3"] = self._r3  # type: ignore
        self.flat_bottom_restraints["r4"] = self._r4  # type: ignore

        self.flat_bottom_restraints["rk2"] = self.spring_constant  # type: ignore
        self.flat_bottom_restraints["rk3"] = self.spring_constant  # type: ignore

    def run(self, fileStore):
        fileStore.logToMaster("Creating FlatBottom Harmonic Restraints")

        self.filestore = fileStore
        tempDir = fileStore.getLocalTempDir()

        self.readfiles["prmtop"] = fileStore.readGlobalFile(
            self.complex_topology,
            userPath=os.path.join(tempDir, os.path.basename(self.complex_topology)),
        )
        self.readfiles["incrd"] = fileStore.readGlobalFile(
            self.complex_coordinate,
            userPath=os.path.join(tempDir, os.path.basename(self.complex_coordinate)),
        )
        fileStore.logToMaster(f"prmtop: {self.readfiles['prmtop']}")
        # load pytraj object
        self.complex_traj = pt.iterload(
            self.readfiles["incrd"], self.readfiles["prmtop"]
        )

        self.topology = self.complex_traj.top

        self._missing_parameters()
        fileStore.logToMaster("Setting restraint parameters:")
        fileStore.logToMaster(f"self.r1: {self._r1}")
        fileStore.logToMaster(f"self.r2: {self._r2}")
        fileStore.logToMaster(f"self.r3: {self._r3}")
        fileStore.logToMaster(f"self.r4: {self._r4}")
        fileStore.logToMaster(f"receptor atoms: {self.restrained_receptor_atoms}")
        fileStore.logToMaster(f"ligand atoms: {self.restrained_ligand_atoms}")

        temp_file = fileStore.getLocalTempFile()
        # one path for both: the protein / host-guest split already lives in the selection
        # (C-alphas when the receptor has a backbone, heavy atoms otherwise), and only this
        # template writes igr1/igr2 from the selected atoms
        restraint = self._flat_bottom_restraints_template
        fileStore.logToMaster(
            f"Flat-bottom groups: {len(self.restrained_receptor_atoms)} receptor "
            f"{'CA' if self._protein_like else 'heavy'} atoms, "
            f"{len(self.restrained_ligand_atoms)} ligand atoms"
        )
        with open(temp_file, "w") as fH:
            fH.write(restraint)

        return fileStore.writeGlobalFile(temp_file), restraint

class BoreschRestraints(Job):
    """
    Impose Boresch orientational restraints on host-guest system.

    Conformations are restrained by applying a harmonic distance restraint
    between every atom and each neighbor within 6 Å. Three heavy atoms from
    the ligand and receptor are selected by constraining 1 distance, 2
    angles and 3 dihedrals. Then the script selects the best suited
    heavy atoms, b and B, to create angles ✓A, ✓a, AB , aA, and ba between 80 and 100.

    Parameters
    ----------
    complex_prmtop: toil.fileStores.FileID
        The complex paramter (.parm7) filepath.
    complex_coordinate: toil.fileStores.FileID
        The complex coordinate (.nc, ncrst, .rst, ect) filepath.
    restraint_type: int
        restaint_type = 1: Find atom closest to ligand's CoM and relevand information.
        restaint_type = 2: Distance restraints between CoM Ligand and closest heavy atom in receptor.
        restraint_type = 3: Distance restraints between the two closest heavy atoms in the ligand and the receptor.
    ligand_mask: str
        AMBER type mask to denote ligand atoms
    receptor_mask: str
        AMBER type mask to denote receptor atoms
    K_r: float
        The spring constant for the restrained distance
    K_thetaA: float
        The spring constants for angle(r2, r3, l1).
    K_thetaB: float
        The spring constants for angle(r3, l1, l2).
    K_phiA, K_phiB, K_phiC: float
        The spring constants for ``dihedral(r1, r2, r3, l1)``,
        ``dihedral(r2, r3, l1, l2)`` and ``dihedral(r3,l1,l2,l3)``
    r_aA0: float
        The equilibrium distance between r3 and l1 (units of length).
    theta_A0, theat_B0: float
        The equilibrium angles of ``angle(r2, r3, l1)`` and ``angle(r3, l1, l2)``
        (units compatible with radians).
    phi_A0, phi_B0, phi_C0: float
        The equilibrium torsion of ``dihedral(r1,r2,r3,l1)``, ``dihedral(r2,r3,l1,l2)``
        and ``dihedral(r3,l1,l2,l3)`` (units compatible with radians).
    restrained_receptor_atoms: list[int]
        A list index restrained receptor atoms. Parmed index
    restrained_ligand_atoms: list[int]
        A list index restrained ligand atoms. Parmed index
    ligand_heavy_atom_distance_parm_index: int
        The selected l1 atom to construct the distance restraint
    ligand_heavy_atom_2_parm_index: int
        The selected l2 atom to constuct angle and dihedral angles
    ligand_heavy_atom_3_parm_index: int
        The selected l3 atom to constuct angle and dihedral angles
    receptor_heavy_atom_distance_parm_index: int
        The selected r1 atom to construct the distance restraint
    receptor_heavy_atom_2_parm_index: int
        The selected l2 atom to constuct angle and dihedral angles
    receptor_heavy_atom_3_parm_index: int
        The selected l3 atom to constuct angle and dihedral angles
    boresch_template: str
        Orientational restraint written .RST file.
    anchor_pool: str
        ``"pocket_backbone"`` restricts R1-R3 to backbone CA/C/N of residues within
        ``pocket_radius`` of the ligand, minus ``excluded_receptor_mask``; ``"legacy"`` screens
        every receptor heavy atom. A receptor without backbone atoms always uses legacy.
    pocket_radius: float
        Contact distance (A) defining the pocket residues for ``pocket_backbone``.
    excluded_receptor_mask: str
        AMBER mask of receptor atoms that are not held by the network in every window.

    References
    ----------
    [1] Boresch S, Tettinger F, Leitgeb M, Karplus M. J Phys Chem B. 107:9535, 2003.
        http://dx.doi.org/10.1021/jp0217839
    [2] Mobley DL, Chodera JD, and Dill KA. J Chem Phys 125:084902, 2006.
        https://dx.doi.org/10.1063%2F1.2221683


    """

    def __init__(
        self,
        complex_prmtop,
        complex_coordinate,
        restraint_type,
        ligand_mask,
        receptor_mask,
        K_r,  # max distance
        K_thetaA: float,  # max_torsional
        K_thetaB: float,  # max_torisional
        K_phiA: float,  # max torisional
        K_phiB: float,  # max torisional
        K_phiC: float,  # max torisional
        r_aA0: Optional[float] = None,  # computed distance
        theta_A0=None,  # ligand angle
        theta_B0: Optional[float] = None,  # receptor angle
        phi_A0: Optional[float] = None,  # ligand computed phi
        phi_B0: Optional[float] = None,  # recepotr idk
        phi_C0: Optional[float] = None,  # compute central
        restrained_receptor_atoms: Optional[list] = None,
        restrained_ligand_atoms: Optional[list] = None,
        ligand_heavy_atom_distance_parm_index: Optional[int] = None,
        ligand_heavy_atom_2_parm_index: Optional[int] = None,
        ligand_heavy_atom_3_parm_index: Optional[int] = None,
        receptor_heavy_atom_distance_parm_index: Optional[int] = None,
        receptor_heavy_atom_2_parm_index: Optional[int] = None,
        receptor_heavy_atom_3_parm_index: Optional[int] = None,
        boresch_template: Optional[str] = None,
        anchor_pool: str = "legacy",
        pocket_radius: float = 10.0,
        excluded_receptor_mask: Optional[str] = None,
        memory: Optional[Union[int, str]] = None,
        cores: Optional[Union[int, float, str]] = None,
        disk: Optional[Union[int, str]] = None,
        preemptable: Optional[Union[bool, int, str]] = None,
        unitName: Optional[str] = "",
        checkpoint: Optional[bool] = False,
        displayName: Optional[str] = "",
        descriptionClass: Optional[str] = None,
    ) -> None:
        super().__init__(
            memory,
            cores,
            disk,
            accelerators=None,
            preemptible="false",
            unitName=unitName,
            checkpoint=checkpoint,
            displayName=displayName,
        )
        self.complex_prmtop = complex_prmtop
        self.complex_coordinate = complex_coordinate
        self.restraint_type = restraint_type
        self.ligand_mask = ligand_mask
        self.receptor_mask = receptor_mask
        self.restrained_receptor_atoms = restrained_receptor_atoms
        self.restrained_ligand_atoms = restrained_ligand_atoms
        self.lig_dist_atom = ligand_heavy_atom_distance_parm_index
        self.rec_dist_atom = receptor_heavy_atom_distance_parm_index
        self.ligand_atom2 = ligand_heavy_atom_2_parm_index
        self.ligand_atom3 = ligand_heavy_atom_3_parm_index
        self.receptor_atom2 = receptor_heavy_atom_2_parm_index
        self.receptor_atom3 = receptor_heavy_atom_3_parm_index
        self.K_r = K_r  # max distance
        self.r_aA0 = r_aA0  # computed distance
        self.K_thetaA = K_thetaA  # max_torsional
        self.theta_A0 = theta_A0  # ligand angle
        self.K_thetaB = K_thetaB  # max_torisional
        self.theta_B0 = theta_B0  # receptor angle
        self.K_phiA = K_phiA  # max torisional
        self.phi_A0 = phi_A0  # ligand computed phi
        self.K_phiB = K_phiB  # max torisional
        self.phi_B0 = phi_B0  # recepotr idk
        self.K_phiC = K_phiC  # max torisional
        self.phi_C0 = phi_C0  # compute central
        self.boresch_template = boresch_template
        self.anchor_pool = anchor_pool
        self.pocket_radius = pocket_radius
        self.excluded_receptor_mask = excluded_receptor_mask
        self._pool = None

    def _receptor_anchor_pool(self):
        """Receptor atoms eligible as R1-R3 under ``pocket_backbone``.

        Returns
        -------
        numpy.ndarray or None
            0-based receptor atom indices (the complex indices, receptor first), or None for
            the legacy screen (requested, or no backbone atoms, i.e. host-guest).

        Raises
        ------
        ValueError
            If the receptor is not first in the complex, or fewer than three atoms qualify.
        """
        if self.anchor_pool == "legacy":
            return None
        from implicit_solvent_ddm.pose_clusters import build_pocket_atoms

        top = self.complex_traj.top
        receptor = pt.select_atoms(self.receptor_mask, top)
        # a host can have atoms named C or N; three C-alphas mean a protein backbone
        if pt.select_atoms(f"({self.receptor_mask})&@CA", top).size < 3:
            return None
        backbone = pt.select_atoms(f"({self.receptor_mask})&@CA,C,N", top)
        if not np.array_equal(receptor, np.arange(receptor.size)):
            raise ValueError("pocket_backbone Boresch anchors need the receptor first in the complex topology")
        pocket_ca, _, _ = build_pocket_atoms(
            top, self.complex_traj.xyz[0], self.ligand_mask, self.receptor_mask, radius=self.pocket_radius
        )
        residues = {top.atom(int(i)).resid for i in pocket_ca}
        excluded = set()
        if self.excluded_receptor_mask:
            excluded = set(pt.select_atoms(self.excluded_receptor_mask, top).tolist())
        pool = np.array([int(i) for i in backbone
                         if top.atom(int(i)).resid in residues and int(i) not in excluded])
        if pool.size < 3:
            raise ValueError(f"only {pool.size} pocket backbone atoms available for Boresch anchors")
        return pool

    def _anchor_pairs(self, atoms, offset):
        """Ordered (a2, a3) candidate pairs whose a3 has more than one heavy neighbour.

        Parameters
        ----------
        atoms : list[pytraj.core.topology_objects.Atom]
            Candidate atoms, indexed within their molecule's sub-topology.
        offset : int
            Index of the molecule's first atom in the complex, to reach its parmed atom.

        Returns
        -------
        list[tuple]
            Candidate pairs in the form ``_check_suitable_restraints`` expects.
        """
        branched = [a for a in atoms if count_heavy_neighbors(self.complex_parm.atoms[offset + a.index]) > 1]
        return [(a, b) for a in atoms for b in branched if a.index != b.index]

    def _anchor_summary(self):
        """R1-R3 and L1-L3 as residue:atom, for the job log."""
        top = self.complex_traj.top
        name = lambda i: f"{top.atom(int(i) - 1).resname}{top.atom(int(i) - 1).resid + 1}:{top.atom(int(i) - 1).name}"
        receptor = [self.rec_dist_atom, self.receptor_atom2, self.receptor_atom3]
        ligand = [self.lig_dist_atom, self.ligand_atom2, self.ligand_atom3]
        return f"R1-R3 {[name(i) for i in receptor]}  L1-L3 {[name(i) for i in ligand]}"

    @property
    def receptor_heavy_atoms(self):
        """Determine receptor heavy atoms only.

        Returns
        -------
        receptor_heavy_atoms: list[pytraj.core.topology_objects.Atom]
            List of receptor heavy atoms.
        """
        return self._determine_heavy_atoms(self.complex_traj[self.receptor_mask])

    @property
    def ligand_heavy_atoms(self):
        """Determine ligand heavy atoms only.

        Returns
        -------
        ligand_heavy_atoms: list[pytraj.core.topology_objects.Atom]
            List of ligand heavy atoms.
        """
        return self._determine_heavy_atoms(self.complex_traj[self.ligand_mask])

        # We determine automatically only the parameters that have been left undefined.

    @property
    def receptor_heavy_atom_pairs(self):
        """Search for all possible combinations of receptor heavy atom pairs.
        Return pairs with more than one heavy atom convalent bond.

        Returns
        -------
        receptor_heavy_atom_pairs: tuple(pytraj.core.topology_objects.Atom, pytraj.core.topology_objects.Atom)
            Receptor heavy atom pairs with more than one heavy atom convalent bonds.
        """
        no_proton_pairs = list(itertools.permutations(self.receptor_heavy_atoms, r=2))
        # get parmed infomation of the second atom
        parmed_atoms = [
            self.complex_parm.atoms[atom[1].index] for atom in no_proton_pairs
        ]

        # if the atom have more than 1 heavy atom bond
        heavy_atoms = list(map(refactor_find_heavy_bonds, parmed_atoms))

        return [x for x, y in zip(no_proton_pairs, heavy_atoms) if y]

    @property
    def ligand_heavy_atom_pairs(self):
        """Search for all possible combinations of ligand heavy atom pairs.
        Return pairs with more than one heavy atom convalent bond.

        Returns
        -------
        ligand_heavy_atom_pairs: tuple(pytraj.core.topology_objects.Atom, pytraj.core.topology_objects.Atom)
            Ligand heavy atom pairs with more than one heavy atom convalent bonds.
        """

        no_proton_pairs = list(itertools.permutations(self.ligand_heavy_atoms, r=2))
        # get parmed infomation of the second atom
        parmed_atoms = [
            self.complex_parm.atoms[atom[1].index] for atom in no_proton_pairs
        ]

        # if the atom have more than 1 heavy atom bond
        heavy_atoms = list(map(refactor_find_heavy_bonds, parmed_atoms))

        return [x for x, y in zip(no_proton_pairs, heavy_atoms) if y]

    @property
    def _write_orientational_template(self):
        """
        Write the orientational restraint template with each dihedral centred on its target.

        Returns
        -------
        str
            Path of the written ``<complex>_orientational_template.RST``.
        """

        string_template = ""
        restraint_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "templates/restraints/orientational.template",
            )
        )
        # restraint_path = "/nas0/ayoub/Impicit-Solvent-DDM/implicit_solvent_ddm/templates/restraint.RST"
        with open(restraint_path) as t:
            template = Template(t.read())
            restraint_template = template.substitute(
                atom_R3=self.receptor_atom3,
                atom_R2=self.receptor_atom2,
                atom_R1=self.rec_dist_atom,
                atom_L1=self.lig_dist_atom,
                atom_L2=self.ligand_atom2,
                atom_L3=self.ligand_atom3,
                dist_rest=self.r_aA0,
                lig_angrest=self.theta_A0,
                rec_angrest=self.theta_B0,
                central_torres=self.phi_C0,
                rec_torres=self.phi_B0,
                lig_torres=self.phi_A0,
                # +/-180 around the target keeps the restraint harmonic in both directions
                central_tor_lo=float(self.phi_C0) - 180.0,
                central_tor_hi=float(self.phi_C0) + 180.0,
                rec_tor_lo=float(self.phi_B0) - 180.0,
                rec_tor_hi=float(self.phi_B0) + 180.0,
                lig_tor_lo=float(self.phi_A0) - 180.0,
                lig_tor_hi=float(self.phi_A0) + 180.0,
                drest="$drest",
                arest="$arest",
                trest="$trest",
            )
            string_template += restraint_template

        complex_name = re.sub(r"\..*", "", os.path.basename(self.complex_prmtop))
        with open(
            f"{complex_name}_orientational_template.RST", "w"
        ) as restraint_string:
            restraint_string.write(string_template)

        return f"{complex_name}_orientational_template.RST"

    @property
    def compute_boresch_restraints(self):
        """Analytically calculate the DeltaG of Boresch restraints contribution.

        Returns
        -------
        df: pd.DataFrame
            A dataframe containing all variables to computed standard state.
        """

        Rgas = 8.31446261815324  # Ideal Gas constant (J)/(mol*K)
        kB = (
            Rgas / 4184
        )  # Converting from Joules to kcal units (kB = Rgas when using molar units)
        T = 298  # Value read from mdin file, assume this is natural units for the similatuion(Kelvin)
        V = 1660  # Angstrom Cubed
        # k = 1.38*(10**(-23))
        print(self.K_thetaA)
        theta_1_rad = math.radians(self.theta_B0)
        theta_2_rad = math.radians(self.theta_A0)
        K_numerator = math.sqrt(
            self.K_r
            * self.K_thetaA
            * self.K_thetaB
            * self.K_phiA
            * self.K_phiB
            * self.K_phiC
        )
        K_denom = (2 * (math.pi) * kB * T) ** 3
        left_numerator = 8 * ((math.pi) ** 2) * V
        left_denom = (
            (self.r_aA0**2) * (math.sin(theta_1_rad)) * (math.sin(theta_2_rad))
        )
        log_argument = (left_numerator / left_denom) * (K_numerator / K_denom)
        result = kB * T * np.log(log_argument)

        df = pd.DataFrame()
        df["r"] = [self.r_aA0]
        df["theta_1"] = [self.theta_A0]
        df["theta_2"] = [self.theta_B0]
        df["Kr"] = [self.K_r]
        df["Ktheta_1"] = [self.K_thetaA]
        df["Ktheta_2"] = [self.K_thetaB]
        df["Kphi1"] = [self.K_phiA]
        df["Kphi2"] = [self.K_phiB]
        df["Kphi3"] = [self.K_phiC]
        df["DeltaG"] = [result]

        return df

    def _assign_if_undefined(self, attr_name, attr_value):
        """Assign value to self.name only if it is None."""

        if getattr(self, attr_name) is None:
            setattr(self, attr_name, attr_value)

    def _determine_atoms_specified_restraints(self, receptor, ligand):
        """Determines the ligand and receptor atom for distance restraints

        Depending on user type of restraint selected:
        restraint_type=1 Determine the heavy atoms closest between
        receptor and ligand center of mass (COM)
        restraint_type=2: Using ligand-COM heavy atom searches the shortest distance for
        any receptor heavy atom
        restraint_type=3: Selects the shortest distance between any receptor and ligand atoms

        Parameters
        ----------
        receptor: pytraj.trajectory
            Pytraj trajectory of the receptor molecule
        ligand: pytraj.trajectory
            Pytraj trajectory of the ligand molecule

        Returns
        -------
        lig_a1_coords: numpy.ndarry
            Array of coordiante of selected L1 position
        rec_a1_coords: numpy.ndarry
            Array of coordiante of selected R1 position
        """
        ligand_com = pt.center_of_mass(ligand)
        receptor_com = pt.center_of_mass(receptor)

        if self.restraint_type == 1:
            # find atom closest to ligand's CoM and relevand information
            (
                ligand_suba1,
                lig_a1_coords,
                dist_liga1_com,
            ) = screen_for_distance_restraints(ligand.n_atoms, ligand_com, ligand)
            ligand_a1 = receptor.n_atoms + ligand_suba1
            dist_liga1_com = distance_calculator(lig_a1_coords, ligand_com)
            receptor_a1, rec_a1_coords, dist_reca1_com = screen_for_distance_restraints(
                receptor.n_atoms, receptor_com, receptor
            )
            dist_rest = distance_calculator(lig_a1_coords, rec_a1_coords)

        elif self.restraint_type == 3:
            (
                ligand_suba1,
                lig_a1_coords,
                receptor_a1,
                rec_a1_coords,
                dist_rest,
            ) = shortest_distance_between_molecules(receptor, ligand)
            ligand_a1 = receptor.n_atoms + ligand_suba1

        else:
            # find atom closest to ligand's CoM and relevand information
            (
                ligand_suba1,
                lig_a1_coords,
                dist_liga1_com,
            ) = screen_for_distance_restraints(ligand.n_atoms, ligand_com, ligand)
            ligand_a1 = receptor.n_atoms + ligand_suba1
            receptor_a1, rec_a1_coords, dist_rest = screen_for_distance_restraints(
                receptor.n_atoms, lig_a1_coords, receptor
            )

        if self._pool is not None:
            # R1 = pocket backbone atom closest to L1, not the receptor COM atom
            xyz = receptor.xyz[0]
            distances = np.linalg.norm(xyz[self._pool] - lig_a1_coords, axis=1)
            nearest = int(self._pool[int(np.argmin(distances))])
            receptor_a1, rec_a1_coords, dist_rest = nearest + 1, xyz[nearest], float(distances.min())

        # set attributes
        self._assign_if_undefined("lig_dist_atom", ligand_a1)
        self._assign_if_undefined("rec_dist_atom", receptor_a1)
        self._assign_if_undefined("r_aA0", dist_rest)

        return lig_a1_coords, rec_a1_coords

    def _determine_heavy_atoms(self, molecule):
        """Returns a list of only heavy atoms

        Returns:
        _determine_heavy_atoms: list[pytraj.core.topology_objects.Atom]
            A list of heavy atoms only.
        """
        # ignore protons return a list of heavy atoms
        return list(
            itertools.filterfalse(
                lambda atom: atom.name.startswith("H"), molecule.topology.atoms
            )
        )

    @staticmethod
    def _check_suitable_restraints(
        atom1_position, atomx_position, only_heavy_pairs, atom_coords
    ):
        """Tries to find the best suited for atoms L2, L3,
        R2, R3.
            self._check_suitable_restraints(
            ligand_atom_A_coord, receptor_atom_a_coord, self.ligand_heavy_atom_pairs,self.complex_traj[self.ligand_mask]
            )
        Parameters
        ----------
        atom1_position: numpy.ndarry
            Coordinates to either ligand atom A or receptor atom a.
        atomx_position: numpy.ndarry
            Coordinates to either ligand atom A or receptor atom a.
        only_heavy_pairs: list[pytraj.core.topology_objects.Atom]
            Heavy atom pairs of the ligand or receptor.
        atom_coords: pytraj.trajectory
            Pytraj of receptor or ligand.

        Returns
        -------
        selected_atom2: int
            Parm index of selected atom 2
        saved_atom2_position: numpy.ndarry
            Coordinate position of selected atom 2
        saved_angle_a1a2: ndarray of floats
            computed theta angle between atom 1 and atom 2
        saved_torsion_angle: ndarray of floats
            Computed torisonal angle between atomx_position, atom1_position, atom2_position and atom3_position
        selected_atom3: int
            Parm index of selected atom 3
        """
        atom_coords = atom_coords.xyz[0]
        min_angle = 80
        max_angle = 100
        saved_average_distance_value = 0
        suitable_restraint = False

        while not suitable_restraint:
            for atom_index, atom in enumerate(only_heavy_pairs):
                atom2_position = atom_coords[
                    atom[0].index
                ]  # [position_data[i][1],position_data[i][2],position_data[i][3]]
                atom3_position = atom_coords[
                    atom[1].index
                ]  # [position_data[i][1],position_data[i][2],position_data[i][3]]
                angle_a1a2 = find_angle(atom2_position, atom1_position, atomx_position)
                angle_a2a3 = find_angle(atom3_position, atom2_position, atom1_position)

                if (
                    angle_a1a2 > min_angle
                    and angle_a1a2 < max_angle
                    and angle_a2a3 > min_angle
                    and angle_a2a3 < max_angle
                ):
                    torsion_angle = compute_dihedral_angle(
                        atomx_position, atom1_position, atom2_position, atom3_position
                    )
                    new_distance_a1a2 = distance_calculator(
                        atom1_position, atom2_position
                    )
                    new_distance_a3_norm_a1a2 = norm_distance(
                        atom1_position, atom2_position, atom3_position
                    )

                    if (
                        new_distance_a1a2 + new_distance_a3_norm_a1a2
                    ) / 2 > saved_average_distance_value:
                        saved_average_distance_value = (
                            new_distance_a1a2 + new_distance_a3_norm_a1a2
                        ) / 2
                        saved_angle_a1a2 = angle_a1a2
                        saved_torsion_angle = torsion_angle
                        saved_atom2_position = atom2_position
                        selected_atom2 = atom[0].index + 1
                        selected_atom3 = atom[1].index + 1
                        suitable_restraint = True
                        print(
                            f"print(saved_average_distance_value) refactor {saved_average_distance_value}"
                        )
            if not suitable_restraint:
                if min_angle > 10:
                    print(min_angle)
                    min_angle -= 1
                    max_angle += 1
                else:
                    import sys

                    sys.exit(
                        "no suitable restraint atom found that fit all parameters!!"
                    )
        return (
            selected_atom2,
            saved_atom2_position,
            saved_angle_a1a2,
            saved_torsion_angle,
            selected_atom3,
        )

    def run(self, fileStore):
        fileStore.logToMaster(f"Creating Boresch Restraints from {self.complex_coordinate}")
        temp_dir = fileStore.getLocalTempDir()
        complex_prmtop_ID = fileStore.readGlobalFile(
            self.complex_prmtop,
            userPath=os.path.join(temp_dir, os.path.basename(self.complex_prmtop)),
        )
        complex_coordinate_ID = fileStore.readGlobalFile(
            self.complex_coordinate,
            userPath=os.path.join(
                temp_dir, os.path.basename(self.complex_coordinate[0])
            ),
        )
        # self.complex_traj = pt.load(self.complex_coordinate, self.complex_prmtop)
        # self.complex_parm = pmd.load_file(self.complex_prmtop)
        self.complex_traj = pt.load(complex_coordinate_ID, complex_prmtop_ID)
        self.complex_parm = pmd.load_file(complex_prmtop_ID)
        self._pool = self._receptor_anchor_pool()
        receptor_traj = self.complex_traj[self.receptor_mask]
        if self._pool is None:
            ligand_pairs, receptor_pairs = self.ligand_heavy_atom_pairs, None
        else:
            ligand_pairs = self._anchor_pairs(self.ligand_heavy_atoms, offset=receptor_traj.n_atoms)
            # iterate topology.atoms: topology.atom(i) returns an Atom whose .index is always 0
            pool = set(self._pool.tolist())
            receptor_pairs = self._anchor_pairs(
                [a for k, a in enumerate(receptor_traj.topology.atoms) if k in pool], offset=0
            )

        (
            ligand_atom_A_coord,
            receptor_atom_a_coord,
        ) = self._determine_atoms_specified_restraints(
            receptor=self.complex_traj[self.receptor_mask],
            ligand=self.complex_traj[self.ligand_mask],
        )

        (
            ligand_suba2,
            ligand_a2_coords,
            ligand_angle1,
            ligand_torsion,
            ligand_suba3,
        ) = self._check_suitable_restraints(
            ligand_atom_A_coord,
            receptor_atom_a_coord,
            ligand_pairs,
            self.complex_traj[self.ligand_mask],
        )
        ligand_a2 = self.complex_traj[self.receptor_mask].n_atoms + ligand_suba2
        ligand_a3 = self.complex_traj[self.receptor_mask].n_atoms + ligand_suba3
        # set ligand angles and heavy atom attributes
        self._assign_if_undefined("ligand_atom2", ligand_a2)
        self._assign_if_undefined("ligand_atom3", ligand_a3)
        self._assign_if_undefined("theta_A0", ligand_angle1)
        self._assign_if_undefined("phi_A0", ligand_torsion)

        (
            receptor_a2,
            receptor_a2_coords,
            receptor_angle1,
            receptor_torsion,
            receptor_a3,
        ) = self._check_suitable_restraints(
            receptor_atom_a_coord,
            ligand_atom_A_coord,
            self.receptor_heavy_atom_pairs if receptor_pairs is None else receptor_pairs,
            self.complex_traj[self.receptor_mask],
        )

        # set receptor angles and heavy atom attributes
        self._assign_if_undefined("receptor_atom2", receptor_a2)
        self._assign_if_undefined("receptor_atom3", receptor_a3)
        self._assign_if_undefined("theta_B0", receptor_angle1)
        self._assign_if_undefined("phi_B0", receptor_torsion)

        central_torsion = compute_dihedral_angle(
            receptor_a2_coords,
            receptor_atom_a_coord,
            ligand_atom_A_coord,
            ligand_a2_coords,
        )

        # set central torsion phi ange
        self._assign_if_undefined("phi_C0", central_torsion)
        mode = "legacy" if self._pool is None else "pocket_backbone"
        fileStore.logToMaster(f"Boresch anchors ({mode}): {self._anchor_summary()}")
        # assign boresch template
        self._assign_if_undefined(
            "boresch_template",
            fileStore.writeGlobalFile(self._write_orientational_template),
        )

        return self


class RestraintMaker(Job):
    def __init__(
        self,
        config: Config,
        boresch_restraints: BoreschRestraints,
        complex_binding_mode: FileID,
        flat_bottom: FileID,
        endstate_trajectory: Optional[FileID] = None,
        conformational_template=None,
        orientational_template=None,
        memory: Optional[Union[int, str]] = None,
        cores: Optional[Union[int, float, str]] = None,
        disk: Optional[Union[int, str]] = None,
        preemptable: Optional[Union[bool, int, str]] = None,
        unitName: Optional[str] = "",
        checkpoint: Optional[bool] = False,
        displayName: Optional[str] = "",
        descriptionClass: Optional[str] = None,
    ) -> None:
        super().__init__(
            memory,
            cores,
            disk,
            accelerators=None,
            preemptible="false",
            unitName=unitName,
            checkpoint=checkpoint,
            displayName=displayName,
        )
        self.complex_binding_mode = complex_binding_mode
        self.flat_bottom = flat_bottom
        self.complex_restraint_file = config.intermediate_args.complex_restraint_files
        self.ligand_restraint_file = config.intermediate_args.guest_restraint_files
        self.receptor_restraint_file = config.intermediate_args.receptor_restraint_files
        # The pinned anchor: max spring constant in force mode, MINIMUM tolerance in rmsd mode.
        self.max_con_force = config.intermediate_args.strongest_conformational_restraint
        self.max_orient_force = config.intermediate_args.max_orientational_restraint

        self.conformational_forces = (
            config.intermediate_args.conformational_restraints_forces
        )
        self.orientational_forces = (
            config.intermediate_args.orientational_restraint_forces
        )
        self.config = config
        self.boresch = boresch_restraints
        # every write_restraint_forces call for a receptor/complex window must carry these
        self.ramp = config.intermediate_args.ramp_kwargs
        self.flat_bottom = flat_bottom
        self.restraints = {}
        self.cv_files = {}          # rmsd mode only; keyed like restraints with an _cv suffix
        self.cv_spec = None         # FileID of the shared group/reference specification
        self.endstate_trajectory = endstate_trajectory
        self.ligand_conformational_restraints = None
        self.receptor_conformational_restraints = None
        self.complex_conformational_restraints = None

    def _assign_if_undefined(self, attr_name, attr_value):
        """Assign value to self.name only if it is None."""

        if getattr(self, attr_name) is None:
            setattr(self, attr_name, attr_value)

    @property
    def max_ligand_conformational_restraint(self):
        return self.restraints[f"ligand_{self.max_con_force}_rst"]

    @property
    def max_receptor_conformational_restraint(self):
        return self.restraints[f"receptor_{self.max_con_force}_rst"]

    @property
    def max_complex_restraint(self):
        return self.restraints[
            f"complex_{self.max_con_force}_{self.max_orient_force}_rst"
        ]

    def run(self, fileStore):
        fileStore.logToMaster(f"Creating Restraints: conformational network from {self.complex_binding_mode}")
        rmsd_mode = self.config.intermediate_args.rmsd_mode
        conformational_restraints = self.addChildJobFn(
            get_conformational_restraints,
            self.config.endstate_files.complex_parameter_filename,
            self.complex_binding_mode,
            self.config.amber_masks.receptor_mask,
            self.config.amber_masks.ligand_mask,
            self.config.intermediate_args.unrestrained_receptor_mask,
            self.config.intermediate_args.receptor_ramp_mask,
            rmsd_mode=rmsd_mode,
        )
        if rmsd_mode:
            self.cv_spec = self.addChildJobFn(
                get_rmsd_cv_templates,
                self.config,
                self.config.endstate_files.complex_parameter_filename,
                self.complex_binding_mode,
                self.endstate_trajectory,
            ).rv()
        self.conformational_restraints = conformational_restraints

        # just added remove
        self._assign_if_undefined(
            "complex_conformational_restraints", conformational_restraints.rv(0)
        )
        self._assign_if_undefined(
            "ligand_conformational_restraints", conformational_restraints.rv(1)
        )
        self._assign_if_undefined(
            "receptor_conformational_restraints", conformational_restraints.rv(2)
        )

        # self.ligand_conformational_restraints = conformational_restraints.rv(1)
        # self.complex_conformational_restraints = conformational_restraints.rv(0)
        # self.receptor_conformational_restraints = conformational_restraints.rv(2)

        self.boresch_deltaG = self.boresch.compute_boresch_restraints

        boresch_forces = self.config.intermediate_args.boresch_distance_forces
        for index, (conformational_force, orientational_force) in enumerate(
            zip(
                self.config.intermediate_args.conformational_restraints_forces,
                self.config.intermediate_args.orientational_restraint_forces,
            )
        ):
            if rmsd_mode:
                for system, key in (
                    ("ligand", f"ligand_{conformational_force}_cv"),
                    ("receptor", f"receptor_{conformational_force}_cv"),
                    (
                        "complex",
                        f"complex_{conformational_force}_{orientational_force}_cv",
                    ),
                ):
                    self.cv_files[key] = self.addFollowOnJobFn(
                        write_cv_file,
                        self.cv_spec,
                        system,
                        conformational_force,
                        self.config.intermediate_args.rmsd_force_constant,
                        r1=self.config.intermediate_args.rmsd_r1,
                        r2=self.config.intermediate_args.rmsd_r2,
                        r4=self.config.intermediate_args.rmsd_r4,
                    ).rv()

            if len(self.ligand_restraint_file) == 0:
                self.restraints[
                    f"ligand_{conformational_force}_rst"
                ] = self.addFollowOnJobFn(
                    write_restraint_forces,
                    conformational_restraints.rv(1),
                    conformational_force=conformational_force,
                ).rv()

                self.restraints[
                    f"receptor_{conformational_force}_rst"
                ] = self.addFollowOnJobFn(
                    write_restraint_forces,
                    conformational_restraints.rv(2),
                    conformational_force=conformational_force,
                    **self.ramp,
                ).rv()

                self.restraints[
                    f"complex_{conformational_force}_{orientational_force}_rst"
                ] = self.addFollowOnJobFn(
                    write_restraint_forces,
                    conformational_restraints.rv(0),
                    self.boresch.boresch_template,
                    conformational_force=conformational_force,
                    orientational_force=orientational_force,
                    boresch_distance_force=float(boresch_forces[index]),
                    **self.ramp,
                ).rv()
            else:
                self.restraints[
                    f"ligand_{conformational_force}_rst"
                ] = self.ligand_restraint_file[index]

                self.restraints[
                    f"receptor_{conformational_force}_rst"
                ] = self.receptor_restraint_file[index]

                self.restraints[
                    f"complex_{conformational_force}_{orientational_force}_rst"
                ] = self.complex_restraint_file[index]

        # if self.complex_restraint_file:
        #     self.boresch_deltaG = self._get_boresch_parameters(
        #         fileStore.readGlobalFile(
        #             self.complex_restraint_file[-1],
        #             userPath=os.path.join(
        #                 self.tempDir, os.path.basename(self.complex_restraint_file[-1])
        #             ),
        #         )
        #     )

        return self

    def add_complex_window(self, conformational_force, orientational_force):
        return self.addChildJobFn(
            write_restraint_forces,
            self.conformational_restraints.rv(0),
            self.boresch.boresch_template,
            conformational_force=conformational_force,
            orientational_force=orientational_force,
            **self.ramp,
        )

    def add_ligand_window(self, conformational_force):
        return self.addChildJobFn(
            write_restraint_forces,
            self.conformational_restraints.rv(1),
            conformational_force=conformational_force,
        ).rv()

    def add_receptor_window(self, conformational_force):
        return self.addFollowOnJobFn(
            write_restraint_forces,
            self.conformational_restraints.rv(2),
            conformational_force=conformational_force,
            **self.ramp,
        ).rv()

    def _get_boresch_parameters(self, restraint_filename):
        """
        Get the Boresch parameter from user provided restraint files.

        The purpose is to read an .RST file. This script assumes the user
        formated the .RST file correctly with the 6 orientational restraints
        at the top of the file. The method will scan each line and find a
        floating point number ignoring integers (atom numbers).
        The order list follows:
        1. The first floating point number should corresponds to distance(r) restraint
            between the select host/guest atom.
        2. The second is the conformational restraint force constant

        Returns
        -------
        boresch_parameter: pd.DataFrame()
        """
        # read in the orientational file with the MAX restraint forces

        with open(restraint_filename) as f:
            restraints = f.readlines()

        values = []
        for line in restraints:
            # if number is floating point number
            current_line = re.search(r"\d*\.\d*", line)
            if current_line is not None:
                values.append(float(current_line[0]))
        return
        # return compute_boresch_restraints(
        #     dist_restraint_r=values[0],
        #     angle1_rest_val=values[2],
        #     angle2_rest_val=values[4],
        #     dist_rest_Kr=values[1],
        #     angle1_rest_Ktheta1=values[3],
        #     angle2_rest_Ktheta2=values[3],
        #     torsion1_rest_Kphi1=values[3],
        #     torsion2_rest_Kphi2=values[3],
        #     torsion3_rest_Kphi3=values[3],
        # )

    @staticmethod
    def get_cv_file(cv_obj, system, conformational_force, orientational_force=None):
        """Look up one window's ``cv.in`` by the same key scheme the restraints use.

        Parameters
        ----------
        cv_obj : dict
            ``RestraintMaker.cv_files``.
        system : {"ligand", "receptor", "complex"}
            Which leg the window belongs to.
        conformational_force : float
            The window's RMSD tolerance, which is also its state label.
        orientational_force : float, optional
            Required for the complex, whose key carries both coordinates.

        Returns
        -------
        toil.fileStore.FileID or None
            The rendered ``cv.in``, or None when the workflow is not in RMSD mode.
        """
        if not cv_obj:
            return None
        if system == "ligand":
            return cv_obj.get(f"ligand_{conformational_force}_cv")
        if system == "receptor":
            return cv_obj.get(f"receptor_{conformational_force}_cv")
        return cv_obj.get(f"complex_{conformational_force}_{orientational_force}_cv")

    @staticmethod
    def get_restraint_file(
        restraint_obj, system, conformational_force, orientational_force=None
    ):
        if system == "ligand":
            return restraint_obj[f"ligand_{conformational_force}_rst"]

        elif system == "receptor":
            return restraint_obj[f"receptor_{conformational_force}_rst"]
        else:
            return restraint_obj[
                f"complex_{conformational_force}_{orientational_force}_rst"
            ]


def get_conformational_restraints(
    job,
    complex_prmtop,
    complex_coordinate,
    receptor_mask,
    ligand_mask,
    unrestrained_receptor_mask=None,
    receptor_ramp_mask=None,
    rmsd_mode=False,
):
    """
     Purpose to create a series of conformational restraint template files.

    The orentational restraints will returns 6 atoms best suited for NMR restraints, based on specific restraint type the user specified.

     Parameters
     ----------
     job: toil.job.FunctionWrappingJob
         A context manager that represents a Toil workflow
     complex_coordinate: toil.fileStore.FileID
         The jobStoreFileID of the imported file. The file being an coordinate file (.ncrst, .nc) of a complex
     complex_restrt_filename: srt
         Name of the coordinate complex file
     unrestrained_receptor_mask: str, optional
         Receptor atoms dropped from the network in every window.
     receptor_ramp_mask: str, optional
         Receptor atoms whose restraints are kept but written with the ``$frest_ramp``
         placeholder, which ``write_restraint_forces`` fills from the ramp schedule.


     Returns
     -------
     restraint_complex_ID: str
         Conformational restraint template for the complex
     restraint_ligand_ID: str
         Conformational restraint template for the ligand only
     restriant_receptor_ID: str
         Conformational restraint template for the receptor only

     Notes
     -----
     ``rmsd_mode=True`` returns three empty templates: the RMSD collective variable replaces the
     pairwise network wholesale, so the cKDTree neighbour search is skipped entirely. That search is
     what produces 82,077 ``&rst`` records on MCL-1, and it is the reason the restraint energy is
     extensive in contact count rather than a property of the restraint itself.
    """
    tempDir = job.fileStore.getLocalTempDir()
    if rmsd_mode:
        empty = job.fileStore.getLocalTempFile()
        with open(empty, "w") as handle:
            handle.write("&end")
        shared = job.fileStore.writeGlobalFile(empty)
        return (shared, shared, shared)

    complex_prmtop_ID = job.fileStore.readGlobalFile(
        complex_prmtop, userPath=os.path.join(tempDir, os.path.basename(complex_prmtop))
    )
    complex_coordinate_ID = job.fileStore.readGlobalFile(
        complex_coordinate,
        userPath=os.path.join(tempDir, os.path.basename(complex_coordinate[0])),
    )

    traj_complex = pt.load(complex_coordinate_ID, complex_prmtop_ID)
    ligand = traj_complex[ligand_mask]
    receptor = traj_complex[receptor_mask]

    receptor_atom_neighbor_index = create_atom_neighbor_array(receptor.xyz[0])
    ligand_atom_neighbor_index = create_atom_neighbor_array(ligand.xyz[0])

    if unrestrained_receptor_mask:
        # create_atom_neighbor_array emits 1-based indices into the RECEPTOR topology, the same
        # numbering the iat records carry, so the mask is selected against `receptor` not the complex.
        freed = set((receptor.top.select(unrestrained_receptor_mask) + 1).tolist())
        if not freed:
            raise ValueError(
                f"unrestrained_receptor_mask {unrestrained_receptor_mask!r} selected no receptor atoms"
            )
        before = len(receptor_atom_neighbor_index)
        receptor_atom_neighbor_index = [
            pair
            for pair in receptor_atom_neighbor_index
            if pair[0] not in freed and pair[1] not in freed
        ]
        job.fileStore.logToMaster(
            f"unrestrained_receptor_mask {unrestrained_receptor_mask}: freed {len(freed)} atoms, "
            f"dropped {before - len(receptor_atom_neighbor_index)} of {before} receptor restraints"
        )

    ramped_pairs = []
    if receptor_ramp_mask:
        ramped = set((receptor.top.select(receptor_ramp_mask) + 1).tolist())
        if not ramped:
            raise ValueError(f"receptor_ramp_mask {receptor_ramp_mask!r} selected no receptor atoms")
        ramped_pairs = [p for p in receptor_atom_neighbor_index if p[0] in ramped or p[1] in ramped]
        receptor_atom_neighbor_index = [
            p for p in receptor_atom_neighbor_index if p[0] not in ramped and p[1] not in ramped
        ]
        job.fileStore.logToMaster(
            f"receptor_ramp_mask {receptor_ramp_mask}: {len(ramped)} atoms, "
            f"{len(ramped_pairs)} ramped of {len(ramped_pairs) + len(receptor_atom_neighbor_index)} receptor restraints"
        )

    ligand_template = conformational_restraints_template(ligand_atom_neighbor_index)
    receptor_template = conformational_restraints_template(
        receptor_atom_neighbor_index
    ) + conformational_restraints_template(ramped_pairs, placeholder="$frest_ramp")
    complex_template = conformational_restraints_template(
        ligand_atom_neighbor_index, num_receptor_atoms=receptor.n_atoms
    )

    # Create a local temporary file.

    ligand_scratchFile = job.fileStore.getLocalTempFile()
    receptor_scratchFile = job.fileStore.getLocalTempFile()
    complex_scratchFile = job.fileStore.getLocalTempFile()
    # job.log(f"ligand_template {ligand_template}")
    with open(complex_scratchFile, "w") as fH:
        fH.write(complex_template)
        fH.write(receptor_template)
        fH.write("&end")
    with open(ligand_scratchFile, "w") as fH:
        fH.write(ligand_template)
        fH.write("&end")
    with open(receptor_scratchFile, "w") as fH:
        fH.write(receptor_template)
        fH.write("&end")

    restraint_complex_ID = job.fileStore.writeGlobalFile(complex_scratchFile)
    restraint_ligand_ID = job.fileStore.writeGlobalFile(ligand_scratchFile)
    restriant_receptor_ID = job.fileStore.writeGlobalFile(receptor_scratchFile)

    # job.fileStore.export_file(restraint_complex_ID, "file://" + os.path.abspath(os.path.join("/home/ayoub/nas0/Impicit-Solvent-DDM/output_directory", os.path.basename(restraint_complex_ID))))
    # toil.exportFile(outputFileID, "file://" + os.path.abspath(os.path.join(ioFileDirectory, "out.txt")))

    return (restraint_complex_ID, restraint_ligand_ID, restriant_receptor_ID)


def write_restraint_forces(
    job,
    conformational_template,
    orientational_template=None,
    conformational_force=0.0,
    orientational_force=0.0,
    ramp_slope=None,
    ramp_max_force=None,
    boresch_distance_force=None,
):
    """Compose one window's ``restraint.RST`` from the cached templates.

    The flat-bottom COM term is deliberately absent: it is an ENDSTATE-only restraint, applied by
    ``setup_simulations.setup_post_endstate_simulation`` as the ``flat_bottom``/``no_flat_bottom``
    pair whose exponential average is reported as ``_flat_bottom_contribution`` (not added to the
    total, since the complex endstate already carries the restraint). It was previously passed in
    here, but a Toil ``FileID`` subclasses ``str``, so what got concatenated was a jobStore path
    rather than restraint text -- AMBER ignored the stray line and the term was never applied. Every
    result to date is therefore correct, and dropping it changes no energy.

    Parameters
    ----------
    job : toil.job.Job
        Owning job, for fileStore access.
    conformational_template : toil.fileStore.FileID
        Template with a ``$frest`` placeholder. Empty in RMSD mode, where the collective variable
        replaces the pairwise network.
    orientational_template : toil.fileStore.FileID, optional
        Boresch template with ``$drest``/``$arest``/``$trest`` placeholders.
    conformational_force : float, optional
        Spring constant substituted into the pairwise network.
    orientational_force : float, optional
        Spring constant for the Boresch angles and torsions.
    boresch_distance_force : float, optional
        Spring constant for the Boresch *distance* term. Defaults to ``conformational_force``, which
        is the legacy behaviour; RMSD mode must pass its own value, since the conformational slot
        then carries an Angstrom tolerance rather than a force constant.
    ramp_slope, ramp_max_force : float, optional
        Ramp schedule for ``$frest_ramp`` (``RestraintMaker.ramp``); required when the template
        carries that placeholder.

    Returns
    -------
    toil.fileStore.FileID
        The composed ``restraint.RST``.

    Raises
    ------
    ValueError
        If the template has ramped restraints but no ramp schedule was passed.
    """
    temp_dir = job.fileStore.getLocalTempDir()

    read_conformational_template = job.fileStore.readGlobalFile(
        conformational_template,
        userPath=os.path.join(temp_dir, os.path.basename(conformational_template)),
    )
    string_template = ""

    if orientational_template is not None:
        read_orientational_template = job.fileStore.readGlobalFile(
            orientational_template,
            userPath=os.path.join(temp_dir, os.path.basename(orientational_template)),
        )
        with open(read_orientational_template) as oren_temp:
            template = Template(oren_temp.read())
            orientational_temp = template.substitute(
                drest=(
                    conformational_force
                    if boresch_distance_force is None
                    else boresch_distance_force
                ),
                arest=orientational_force,
                trest=orientational_force,
            )
            string_template += orientational_temp

    with open(read_conformational_template) as temp:
        text = temp.read()
    forces = dict(frest=conformational_force)
    if "$frest_ramp" in text:
        if ramp_slope is None or ramp_max_force is None:
            raise ValueError("restraint template has ramped restraints but no ramp schedule was passed")
        forces["frest_ramp"] = ramped_force(conformational_force, ramp_max_force, ramp_slope)
    string_template += Template(text).substitute(**forces)

    string_template = string_template.replace("&end", "")

    with open("restraint.RST", "w") as fn:
        fn.write(string_template)
        fn.write("&end")

    return job.fileStore.writeGlobalFile("restraint.RST")


def conformational_restraints_template(
    solute_conformational_restraint, num_receptor_atoms=0, delta_flat = 0.5, placeholder="$frest"):
    restraint_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "templates/restraints/conformational_restraints.template",
        )
    )
    string_template = ""
    for index in range(len(solute_conformational_restraint)):
        with open(restraint_path) as f:
            template = Template(f.read())

            restraint_template = template.substitute(
                solute_primary_atom=solute_conformational_restraint[index][0]
                + num_receptor_atoms,
                solute_sec_atom=solute_conformational_restraint[index][1]
                + num_receptor_atoms,
                distance_lower=solute_conformational_restraint[index][2] - delta_flat,
                distance_upper=solute_conformational_restraint[index][2] + delta_flat,
                frest=placeholder,
            )
            string_template += restraint_template
    return string_template


def write_empty_restraint(job):
    temp_dir = job.fileStore.getLocalTempDir()
    with open("empty.restraint", "w") as fn:
        fn.write("")
    return job.fileStore.writeGlobalFile("empty.restraint")


def export_restraint(job, restraints: RestraintMaker):
    # f"complex_{conformational_force}_{orientational_force}_rst"
    tempDir = job.fileStore.getLocalTempDir()

    job.log(f"restraints keys {restraints.restraints.keys()}")
    restraint_file = job.fileStore.readGlobalFile(
        restraints.restraints["complex_0.00390625_0.0625_rst"],
        userPath=os.path.join(
            tempDir,
            os.path.basename(restraints.restraints["complex_0.00390625_0.0625_rst"]),
        ),
    )
    job.fileStore.export_file(
        restraint_file,
        "file://"
        + os.path.abspath(
            os.path.join(
                "/nas0/ayoub/Impicit-Solvent-DDM/restraint_check",
                os.path.basename(restraint_file),
            )
        ),
    )
    job.log(
        f"""\n
            r: {restraints.boresch_deltaG["r"]}\n 
            theta_1: {restraints.boresch_deltaG["theta_1"]}\n
            theta_2: {restraints.boresch_deltaG["theta_2"]}\n
            Kr: {restraints.boresch_deltaG["Kr"]}\n 
            Ktheta_1: {restraints.boresch_deltaG["Ktheta_1"]}\n 
            Ktheta_2: {restraints.boresch_deltaG["Ktheta_2"]}\n 
            Kphi1: {restraints.boresch_deltaG["Kphi1"]}\n 
            Kphi2: {restraints.boresch_deltaG["Kphi2"]}\n 
            Kphi3: {restraints.boresch_deltaG["Kphi3"]}\n 
            DeltaG: {restraints.boresch_deltaG["DeltaG"]}\n 
            """
    )


def hello_job(job, config: Config):
    # get_orientational_restraints(job, complex_prmtop, complex_coordinate, receptor_mask, ligand_mask, restraint_type):
    # get_orientational_restraints(job, complex_prmtop, complex_coordinate, receptor_mask, ligand_mask, restraint_type):
    # output = job.addChildJobFn(get_orientational_restraints, prmtop, coordinate, ":CB7", ":M01", 1)

    boresch = job.addChild(
        BoreschRestraints(
            complex_prmtop=config.endstate_files.complex_parameter_filename,
            complex_coordinate=config.endstate_files.complex_coordinate_filename,
            restraint_type=2,
            ligand_mask=config.amber_masks.ligand_mask,
            receptor_mask=config.amber_masks.receptor_mask,
            K_r=16,
            K_thetaA=256,
            K_thetaB=256,
            K_phiA=256,
            K_phiB=256,
            K_phiC=256,
        )
    )
    config.inputs[
        "endstate_complex_lastframe"
    ] = config.endstate_files.complex_coordinate_filename
    a = boresch.addFollowOn(
        RestraintMaker(config=config, boresch_restraints=boresch.rv())
    )

    b = job.addFollowOnJobFn(export_restraint, a.rv())

    return a


if __name__ == "__main__":
    # traj = pt.load("/home/ayoub/nas0/Impicit-Solvent-DDM/success_postprocess/mdgb/split_complex_folder/ligand/split_M01_000.ncrst.1", "/home/ayoub/nas0/Impicit-Solvent-DDM/success_postprocess/mdgb/M01_000/4/4.0/M01_000.parm7")
    complex_coord = "/nas0/ayoub/sampl4_cb7/sampl4_cb7/cb7-mol01_Hmass/lambda_window/1.0/78.5/-8.0/-4.0/cb7-mol01_Hmass_300K_lastframe.ncrst"
    complex_parm = "/nas0/ayoub/sampl4_cb7/sampl4_cb7/cb7-mol01_Hmass/lambda_window/1.0/78.5/-8.0/-4.0/cb7-mol01_Hmass.parm7"

    options = Job.Runner.getDefaultOptions("./toilWorkflowRun")
    options.logLevel = "INFO"
    options.clean = "always"
    with Toil(options) as toil:
        import yaml

        with open(
            "/nas0/ayoub/Impicit-Solvent-DDM/config_files/no_restraints.yaml"
        ) as fH:
            yaml_config = yaml.safe_load(fH)

        config = Config.from_config(yaml_config)
        if not toil.options.restart:
            config.endstate_files.toil_import_parmeters(toil=toil)

            if config.endstate_method.endstate_method_type == "remd":
                config.endstate_method.remd_args.toil_import_replica_mdin(toil=toil)

            if config.intermediate_args.guest_restraint_files is not None:
                config.intermediate_args.toil_import_user_restriants(toil=toil)

            config.inputs["min_mdin"] = str(
                toil.import_file(
                    "file://"
                    + os.path.abspath(
                        os.path.dirname(os.path.realpath(__file__))
                        + "/templates/min.mdin"
                    )
                )
            )
        # complex_coord = toil.import_file(
        #     "file://" + os.path.abspath(complex_coord)
        # )
        # complex_parm = toil.import_file(
        #     "file://" + os.path.abspath(complex_parm)
        # )
        output = toil.start(Job.wrapJobFn(hello_job, config))

    # boresch = BoreschRestraints(complex_prmtop=complex_parm,
    #                             complex_coordinate=complex_coord,
    #                             restraint_type=2, ligand_mask=":G3",
    #                             receptor_mask=":WP6", K_r=4,
    #                             K_thetaA=8.0, K_thetaB=8.0,
    #                             K_phiA=8.0, K_phiB=8.0, K_phiC=8.0)
    # print('K_r', boresch.K_r)
    # a = boresch.run()
    # print(type(a.theta_A0))
    # print(boresch.compute_boresch_restraints)

    # print('-'*20)

    # get_orientational_restraints_no_toil(complex_prmtop=complex_parm, complex_coordinate=complex_coord,
    #                                      receptor_mask=":WP6", ligand_mask=":G3",
    #                                      restraint_type=2, max_torisonal_rest=8.0,
    #                                      max_distance_rest=4.0)
    # screen_for_distance_restraints(num_atoms, com, mol)
    # traj = pt.load(complex_coord, complex_parm)
    # parmed_traj = pmd.load_file(complex_parm)
    # receptor = traj[":CB7"]
    # ligand = traj[":M02"]
    # ligand_com = pt.center_of_mass(ligand)
    # receptor_com = pt.center_of_mass(receptor)

    # # find atom closest to ligand's CoM and relevand information
    # # ligand_suba1, lig_a1_coords, dist_liga1_com = distance_btw_center_of_mass(

    # receptor_atom_neighbor_index = create_atom_neighbor_array(receptor.xyz[0])
    # ligand_atom_neighbor_index = create_atom_neighbor_array(ligand.xyz[0])

    # print("receptor_atom_neighbor_index")
    # print(receptor_atom_neighbor_index)
    # print("-" * 80)
    # print("ligand_atom_neighbor_index")
    # print(ligand_atom_neighbor_index)
    # ligand_template = conformational_restraints_template(ligand_atom_neighbor_index)
    # receptor_template = conformational_restraints_template(receptor_atom_neighbor_index)
    # complex_template = conformational_restraints_template(
    #     ligand_atom_neighbor_index, num_receptor_atoms=receptor.n_atoms
    # )

    # # Create a local temporary file.

    # # ligand_scratchFile = job.fileStore.getLocalTempFile()
    # # receptor_scratchFile = job.fileStore.getLocalTempFile()
    # # complex_scratchFile = job.fileStore.getLocalTempFile()
    # # # job.log(f"ligand_template {ligand_template}")
    # with open("complex_conformational.RST", "w") as fH:
    #     fH.write(complex_template)
    #     fH.write(receptor_template)
    #     fH.write("&end")
    # with open("ligand_conformational.RST", "w") as fH:
    #     fH.write(ligand_template)
    #     fH.write("&end")
    # with open("receptor_conformational", "w") as fH:
    #     fH.write(receptor_template)
    #     fH.write("&end")

    # restraint_complex_ID = job.fileStore.writeGlobalFile(complex_scratchFile)
    # restraint_ligand_ID = job.fileStore.writeGlobalFile(ligand_scratchFile)
    # restriant_receptor_ID = job.fileStore.writeGlobalFile(receptor_scratchFile)

    # # job.fileStore.export_file(restraint_complex_ID, "file://" + os.path.abspath(os.path.join("/home/ayoub/nas0/Impicit-Solvent-DDM/output_directory", os.path.basename(restraint_complex_ID))))
    # # toil.exportFile(outputFileID, "file://" + os.path.abspath(os.path.join(ioFileDirectory, "out.txt")))

    # return (restraint_complex_ID, restraint_ligand_ID, restriant_receptor_ID)


# --------------------------------------------------------------------------------------------------
# RMSD collective-variable restraint (AMBER NFE &pmd / MULTI_RMSD)
# --------------------------------------------------------------------------------------------------
def select_receptor_restraint_atoms(topology, receptor_mask):
    """Choose the receptor atoms an RMSD or COM restraint acts on.

    Protein-like systems are restrained through their backbone alpha carbons; anything without a
    backbone -- a cucurbituril host, say -- falls back to all heavy atoms. This is the same rule
    ``FlatBottom`` has always applied, lifted out so the RMSD path cannot drift from it.

    Parameters
    ----------
    topology : pytraj.Topology
        Topology of the complex.
    receptor_mask : str
        AMBER mask selecting the receptor.

    Returns
    -------
    indices : numpy.ndarray
        Zero-based atom indices, in topology order.
    mask : str
        The mask that was actually resolved, for logging.
    """
    ca_sel = topology.select(f"{receptor_mask} & @CA")
    if ca_sel is not None and np.size(ca_sel) > 0:
        return np.asarray(ca_sel), f"{receptor_mask} & @CA"
    return np.asarray(topology.select(f"{receptor_mask} & !@H*")), f"{receptor_mask} & !@H*"


def select_ligand_restraint_atoms(topology, ligand_mask):
    """Choose the ligand atoms a restraint acts on -- always heavy atoms.

    Parameters
    ----------
    topology : pytraj.Topology
        Topology of the complex.
    ligand_mask : str
        AMBER mask selecting the ligand.

    Returns
    -------
    indices : numpy.ndarray
        Zero-based atom indices, in topology order.
    mask : str
        The mask that was actually resolved.
    """
    return np.asarray(topology.select(f"{ligand_mask} & !@H*")), f"{ligand_mask} & !@H*"


def select_site_restraint_atoms(topology, coordinates, receptor_mask, ligand_mask, cutoff, atom_class="heavy"):
    """Choose the binding-site residues for a second, tighter RMSD group.

    A single global CA leash does not hold the pocket: a loop can collapse into an empty site while
    moving global CA RMSD by a few tenths of an Angstrom, and every LJ and charge decoupling window
    runs at the anchor. Restraining the site separately, at its own tolerance, covers that.

    Parameters
    ----------
    topology : pytraj.Topology
        Topology of the complex.
    coordinates : numpy.ndarray, shape (n_atoms, 3)
        Complex coordinates the site is measured on (the binding mode).
    receptor_mask, ligand_mask : str
        AMBER masks selecting the receptor and the ligand.
    cutoff : float
        Distance in Angstrom; whole residues with any atom within this of the ligand are selected.
    atom_class : {"heavy", "global"}
        ``"heavy"`` restrains every heavy atom of the site residues; ``"global"`` restricts the site
        to the atom class of the global group (CA on a protein).

    Returns
    -------
    indices : numpy.ndarray
        Zero-based atom indices.
    mask : str
        An equivalent residue mask, for logging.
    """
    ligand = np.asarray(topology.select(ligand_mask))
    receptor = np.asarray(topology.select(receptor_mask))
    xyz = np.asarray(coordinates)
    nearest = np.linalg.norm(xyz[receptor][:, None, :] - xyz[ligand][None, :, :], axis=2).min(axis=1)
    resids = sorted({topology.atom(int(i)).resid for i in receptor[nearest <= cutoff]})
    site = np.array([i for i in receptor if topology.atom(int(i)).resid in set(resids)], dtype=int)
    if atom_class == "heavy":
        base, suffix = np.asarray(topology.select(f"({receptor_mask}) & !@H*")), "&!@H*"
    else:
        base, resolved = select_receptor_restraint_atoms(topology, receptor_mask)
        suffix = "&@CA" if "@CA" in resolved else "&!@H*"
    mask = ":" + ",".join(str(r + 1) for r in resids) + suffix
    return np.intersect1d(site, np.asarray(base)), mask


def map_complex_to_subsystem(subsystem_indices, group_indices):
    """Translate complex-frame atom indices into their apo-topology equivalents.

    ``conformational_restraints_template`` offsets ligand indices by the receptor atom count, which
    silently assumes the receptor occupies the first ``n_receptor`` slots. That happens to hold today
    -- ``pt.strip`` and ``complex[mask]`` both preserve order -- but computing the mapping exactly
    costs microseconds and fails loudly rather than restraining the wrong atoms.

    Parameters
    ----------
    subsystem_indices : numpy.ndarray
        Sorted zero-based complex indices of every atom in the sub-system.
    group_indices : numpy.ndarray
        Zero-based complex indices of the restrained group, a subset of ``subsystem_indices``.

    Returns
    -------
    numpy.ndarray
        Zero-based indices of the same atoms within the sub-system topology.

    Raises
    ------
    ValueError
        If any group atom is absent from the sub-system.
    """
    subsystem = np.asarray(subsystem_indices)
    group = np.asarray(group_indices)
    position = np.searchsorted(subsystem, group)
    if np.any(position >= subsystem.size) or np.any(subsystem[np.minimum(position, subsystem.size - 1)] != group):
        missing = group[[p >= subsystem.size or subsystem[min(p, subsystem.size - 1)] != g
                         for p, g in zip(position, group)]]
        raise ValueError(f"restrained atoms {missing.tolist()} are not in the sub-system selection")
    return position


def render_colvar(atom_indices, reference, tolerance, force_constant, r1=-10.0, r2=0.0, r4=1000.0):
    """Render one single-group ``MULTI_RMSD`` ``&colvar`` block.

    Single-group on purpose: the multi-group form of ``MULTI_RMSD`` combines groups as
    ``sqrt(sum_g M_g/M R_g**2)``, whereas separate blocks give ``U = sum_g U_g``. The additive form
    matches how the pairwise restraint file already composes ligand and receptor terms, so
    cancellation against the apo legs is structurally unchanged, and it lets each group carry its own
    tolerance and map 1:1 onto a single cpptraj ``rms ... mass`` series.

    Parameters
    ----------
    atom_indices : numpy.ndarray
        Zero-based atom indices in the topology this ``cv.in`` will be used with; written one-based.
    reference : numpy.ndarray, shape (n, 3)
        Reference positions for those atoms, written flattened as ``cv_r``.
    tolerance : float
        Flat-bottom upper edge ``r3``, in Angstrom.
    force_constant : float
        ``anchor_strength`` for both walls.
    r1, r2, r4 : float, optional
        Remaining anchor positions.

    Returns
    -------
    str
        The ``&colvar`` block, newline-terminated.
    """
    one_based = [int(index) + 1 for index in atom_indices]
    flat = np.asarray(reference, dtype=float).reshape(-1)
    index_block = ",\n         ".join(
        ", ".join(str(value) for value in one_based[start : start + 10])
        for start in range(0, len(one_based), 10)
    )
    reference_block = ",\n         ".join(
        ", ".join(f"{value:.6f}" for value in flat[start : start + 6])
        for start in range(0, flat.size, 6)
    )
    return (
        "&colvar\n"
        "  cv_type = 'MULTI_RMSD'\n"
        f"  cv_ni = {len(one_based)},\n"
        f"  cv_i = {index_block},\n"
        f"  cv_nr = {flat.size},\n"
        f"  cv_r = {reference_block},\n"
        f"  anchor_position = {r1}, {r2}, {tolerance}, {r4}\n"
        f"  anchor_strength = {force_constant}, {force_constant}\n"
        "/\n"
    )


def build_rmsd_reference(topology_path, binding_mode_path, indices, source, trajectory_path=None):
    """Resolve the reference positions the RMSD collective variable measures against.

    The reference is a *waypoint*: the receptor and ligand legs pay ``free -> held at X`` while the
    complex leg pays ``held at X -> free bound``, so ``X`` cancels from the cycle. What it changes is
    path length -- both ends walk *to* ``X``, so a reference off the apo-to-holo segment makes every
    leg walk further. A single snapshot sits about one thermal RMSF off that segment (measured here:
    1.76 A on the receptor endstate, against a 2.16 A reorganization), which is why the ensemble
    average is the default and a lastframe is not.

    Parameters
    ----------
    topology_path : str
        Complex topology.
    binding_mode_path : str
        Complex binding-mode coordinates -- the endstate last frame, and the legacy reference.
    indices : numpy.ndarray
        Zero-based complex indices of the restrained group.
    source : str
        ``restraint_reference`` (the binding-mode frame, i.e. the shared restraint reference when one
        is configured), ``ensemble_average``, ``endstate_lastframe``, ``minimized``, or a path to a
        coordinate file.
    trajectory_path : str, optional
        Holo trajectory to average. Required by ``ensemble_average``; without it the call falls back
        to the binding mode and warns.

    Returns
    -------
    numpy.ndarray, shape (n, 3)
        Reference positions for ``indices``, in complex-frame order.
    """
    if source == "ensemble_average":
        if trajectory_path is None:
            warnings.warn(
                "rmsd_reference_source='ensemble_average' but no endstate trajectory was supplied; "
                "falling back to the binding-mode frame, which carries a full thermal fluctuation"
            )
        else:
            mask = "@" + ",".join(str(int(i) + 1) for i in indices)
            traj = pt.load(trajectory_path, topology_path, mask=mask)
            pt.superpose(traj, ref=0)
            return np.asarray(traj.xyz).mean(axis=0)
    elif source not in ("restraint_reference", "endstate_lastframe", "minimized"):
        return np.asarray(pt.load(source, topology_path).xyz[0][indices])
    return np.asarray(pt.load(binding_mode_path, topology_path).xyz[0][indices])


def get_rmsd_cv_templates(job, config, complex_prmtop, complex_coordinate, endstate_trajectory=None):
    """Build the per-leg collective-variable specification for the RMSD restraint.

    Produces one ``.npz`` in the jobStore holding, per leg and per group, the atom indices in that
    leg's own topology plus the shared reference positions. Passing a FileID rather than the raw
    arrays keeps large objects out of Toil promises, which is the same reason ``HamiltonianSpec``
    exists.

    Parameters
    ----------
    job : toil.job.Job
        Owning job, for fileStore access.
    config : Config
        Supplies the masks and every ``rmsd_*`` knob.
    complex_prmtop : toil.fileStore.FileID
        Complex topology.
    complex_coordinate : toil.fileStore.FileID or list
        Complex binding-mode coordinates.
    endstate_trajectory : toil.fileStore.FileID or list, optional
        Holo trajectory used by ``rmsd_reference_source='ensemble_average'``.

    Returns
    -------
    toil.fileStore.FileID
        The ``.npz`` specification consumed by :func:`write_cv_file`.
    """
    temp_dir = job.fileStore.getLocalTempDir()

    def _read(handle):
        target = handle[0] if isinstance(handle, (list, tuple)) else handle
        return job.fileStore.readGlobalFile(
            target, userPath=os.path.join(temp_dir, os.path.basename(str(target)))
        )

    top_path = _read(complex_prmtop)
    mode_path = _read(complex_coordinate)
    traj_path = _read(endstate_trajectory) if endstate_trajectory is not None else None

    args = config.intermediate_args
    receptor_mask = config.amber_masks.receptor_mask
    ligand_mask = config.amber_masks.ligand_mask
    topology = pt.load_topology(top_path)

    groups = {}
    if args.rmsd_mask:
        groups["receptor"] = (np.asarray(topology.select(args.rmsd_mask)), 1.0)
    else:
        indices, resolved = select_receptor_restraint_atoms(topology, receptor_mask)
        job.fileStore.logToMaster(f"[rmsd] receptor group: {resolved} -> {indices.size} atoms")
        groups["receptor"] = (indices, 1.0)
    if args.rmsd_site_cutoff:
        coordinates = pt.load(mode_path, top_path).xyz[0]
        indices, resolved = select_site_restraint_atoms(
            topology, coordinates, receptor_mask, ligand_mask, args.rmsd_site_cutoff,
            atom_class=args.rmsd_site_atoms,
        )
        job.fileStore.logToMaster(f"[rmsd] site group: {resolved} -> {indices.size} atoms")
        groups["site"] = (indices, args.rmsd_site_tolerance_scale)
    if args.rmsd_restrain_ligand:
        indices, resolved = select_ligand_restraint_atoms(topology, ligand_mask)
        job.fileStore.logToMaster(f"[rmsd] ligand group: {resolved} -> {indices.size} atoms")
        groups["ligand"] = (indices, args.rmsd_ligand_tolerance_scale)

    receptor_atoms = np.asarray(topology.select(receptor_mask))
    ligand_atoms = np.asarray(topology.select(ligand_mask))
    # Which groups each leg's cv.in carries. The complex holds every group so its bias is exactly the
    # sum of the apo legs', which is what makes the cycle cancel.
    membership = {
        "complex": [name for name in groups],
        "receptor": [name for name in groups if name != "ligand"],
        "ligand": ["ligand"] if "ligand" in groups else [],
    }
    subsystem = {"complex": None, "receptor": receptor_atoms, "ligand": ligand_atoms}

    payload = {"groups": np.asarray(list(groups), dtype=object)}
    for name, (indices, scale) in groups.items():
        payload[f"ref:{name}"] = build_rmsd_reference(
            top_path, mode_path, indices, args.rmsd_reference_source, traj_path
        )
        payload[f"scale:{name}"] = np.asarray(float(scale))
        for system, members in membership.items():
            if name not in members:
                continue
            mapped = (
                indices if subsystem[system] is None
                else map_complex_to_subsystem(subsystem[system], indices)
            )
            payload[f"idx:{system}:{name}"] = np.asarray(mapped)
    for system, members in membership.items():
        payload[f"members:{system}"] = np.asarray(members, dtype=object)

    spec_path = os.path.join(temp_dir, "rmsd_cv_spec.npz")
    np.savez(spec_path, **payload)
    return job.fileStore.writeGlobalFile(spec_path)


def write_cv_file(job, spec_id, system_type, tolerance, force_constant, r1=-10.0, r2=0.0, r4=1000.0):
    """Render one window's ``cv.in`` from the shared specification.

    Parameters
    ----------
    job : toil.job.Job
        Owning job, for fileStore access.
    spec_id : toil.fileStore.FileID
        Output of :func:`get_rmsd_cv_templates`.
    system_type : {"complex", "receptor", "ligand"}
        Which leg's group membership and index mapping to use.
    tolerance : float
        This window's ``r3``, in Angstrom, before each group's scale is applied.
    force_constant : float
        ``anchor_strength`` shared by every group.
    r1, r2, r4 : float, optional
        Remaining anchor positions.

    Returns
    -------
    toil.fileStore.FileID
        The rendered ``cv.in``.
    """
    temp_dir = job.fileStore.getLocalTempDir()
    spec_path = job.fileStore.readGlobalFile(
        spec_id, userPath=os.path.join(temp_dir, "rmsd_cv_spec.npz")
    )
    spec = np.load(spec_path, allow_pickle=True)

    blocks = ""
    for name in spec[f"members:{system_type}"]:
        name = str(name)
        blocks += render_colvar(
            spec[f"idx:{system_type}:{name}"],
            spec[f"ref:{name}"],
            round(float(tolerance) * float(spec[f"scale:{name}"]), 4),
            force_constant,
            r1=r1,
            r2=r2,
            r4=r4,
        )

    out_path = os.path.join(temp_dir, "cv.in")
    with open(out_path, "w") as handle:
        handle.write(blocks)
    return job.fileStore.writeGlobalFile(out_path)


def rmsd_series_for_trajectory(job, spec_id, system_type, prmtop, trajectory):
    """Measure each collective-variable group's RMSD over one trajectory.

    Called once per completed trajectory rather than once per matrix cell: every window in the
    restraint band shares this series, and only the tolerance ``r3`` differs between them.

    Uses the same mass-weighted best-fit RMSD the restraint itself applies -- ``MULTI_RMSD`` was
    measured equal to cpptraj ``rms ... mass`` to 3e-5 Angstrom, and invariant to a rigid transform
    of the reference to 3e-8 Angstrom -- so the energy rebuilt from this series is exact rather than
    an approximation.

    Parameters
    ----------
    job : toil.job.Job
        Owning job, for fileStore access.
    spec_id : toil.fileStore.FileID
        Output of :func:`get_rmsd_cv_templates`.
    system_type : {"complex", "receptor", "ligand"}
        Which leg's group membership and index mapping to use.
    prmtop : toil.fileStore.FileID
        Topology matching ``trajectory``.
    trajectory : toil.fileStore.FileID or list
        The completed MD trajectory to measure.

    Returns
    -------
    dict of str to numpy.ndarray
        Per-frame RMSD in Angstrom, keyed by group name.
    """
    temp_dir = job.fileStore.getLocalTempDir()

    def _read(handle):
        target = handle[0] if isinstance(handle, (list, tuple)) else handle
        return job.fileStore.readGlobalFile(
            target, userPath=os.path.join(temp_dir, os.path.basename(str(target)))
        )

    spec = np.load(_read(spec_id), allow_pickle=True)
    top_path = _read(prmtop)
    traj_path = _read(trajectory)

    series = {}
    for name in spec[f"members:{system_type}"]:
        name = str(name)
        indices = np.asarray(spec[f"idx:{system_type}:{name}"])
        reference = np.asarray(spec[f"ref:{name}"])
        mask = "@" + ",".join(str(int(i) + 1) for i in indices)
        traj = pt.load(traj_path, top_path, mask=mask)
        reference_frame = pt.Frame(traj[0])
        reference_frame.xyz[:] = reference
        series[name] = np.asarray(
            pt.rmsd(traj, mask="", ref=reference_frame, mass=True), dtype=float
        )
    return series
