"""
Helper functions used to construct conformational and orientational restraints.
"""
import itertools
import time
from scipy.spatial import cKDTree
import numpy as np


def compute_dihedral_angle(atom1, atom2, atom3, atom4):

    """
    Function is one of two options to calculate dihedral angles

    Code for torsion restraints from:
    https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    WIKIPEDIA EXAMPLE
    """
    p0 = np.array(atom1, dtype=float)
    p1 = np.array(atom2, dtype=float)
    p2 = np.array(atom3, dtype=float)
    p3 = np.array(atom4, dtype=float)

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)

    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    y = np.dot(b0xb1_x_b1xb2, b1) * (1.0 / np.linalg.norm(b1))
    x = np.dot(b0xb1, b1xb2)

    initial_angle = np.degrees(np.arctan2(y, x))
    if initial_angle < 0:
        final_angle = 360 + initial_angle
    else:
        final_angle = initial_angle

    return final_angle


# barton source code
def norm_distance(point_a, point_b, point_c):
    """Compute normal distance between 3 atom positions

    Parameters
    ----------
    point_a: numpy.ndarray
        The selected distance restraint atom position. 
    point_b: 
        Potential selection of heavy atom
    point_c: 
        Potential selection of heavy atom
    
    Returns 
    -------
    norm_distance: ndarray
        The computed norm distance between 3 atoms. 

    Returns:
        _type_: _description_
    """
    p1 = np.array(point_a)
    p2 = np.array(point_b)
    p3 = np.array(point_c)
    # x = p1-p2
    # dotproduct = np.dot(p3, x)/np.dot(x,x)
    # norm_distance = np.linalg.norm(dotproduct*(x)+p2-p3)

    # norm_distance = np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)

    # normalized tangent vector
    d = np.divide(p2 - p1, np.linalg.norm(p2 - p1))

    # signed parallel distance components
    s = np.dot(p1 - p3, d)
    t = np.dot(p3 - p2, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = np.cross(p3 - p1, d)

    norm_distance = np.hypot(h, np.linalg.norm(c))

    # print('norm_distance: ', norm_distance)

    return norm_distance


def create_atom_neighbor_array(atom_coordinates, cutoff=6.0):
    """
    Efficiently find all atom pairs within a given cutoff using cKDTree.

    Parameters
    ----------
    atom_coordinates : np.ndarray
        An (N, 3) array of atomic positions
    cutoff : float
        Distance cutoff in angstroms

    Returns
    -------
    atom_neighbor_array : list
        A list of (i, j, distance) tuples where distance <= cutoff
    """
    start_time = time.time()

    tree = cKDTree(atom_coordinates)
    pairs = tree.query_pairs(r=cutoff)

    atom_neighbor_array = []
    for i, j in pairs:
        dist = np.linalg.norm(atom_coordinates[i] - atom_coordinates[j])
        atom_neighbor_array.append((i + 1, j + 1, dist))  # optional: 1-based index

    print("--- %s seconds ---" % (time.time() - start_time))
    return atom_neighbor_array

# def create_atom_neighbor_array(atom_coordinates):
# Older slower function that utilizes itertools
#     """
#      Conformational restraints will be applied by creating harmonic distance restraints between every atom and all neighbors within 6 Å that are part of the same molecule

#      Parameters
#      ----------
#      atom_coordinates: numpy.ndarray
#         An array of atomic coordiantes
#     Returns
#     -------
#     atom_neighbor_array: list
#          A list containing the nearest neighbor atoms within 6 angstroms
#     """

#     start_time = time.time()

#     atom_neighbor_array = []
#     atoms = [_ for _ in range(1, len(atom_coordinates) + 1)]
#     atom_pairs = list(itertools.combinations(atoms, r=2))

#     coordinate_pairs = list(itertools.combinations(atom_coordinates, r=2))
#     atom_distances = list(itertools.starmap(distance_calculator, coordinate_pairs))
#     closest_neighbor = list(map(distance_filter, atom_distances))

#     for atom_pair_index, neighbors in enumerate(closest_neighbor):
#         if neighbors:
#             atom_neighbor_array.append(
#                 [
#                     atom_pairs[atom_pair_index][0],
#                     atom_pairs[atom_pair_index][1],
#                     atom_distances[atom_pair_index],
#                 ]
#             )
#     print("--- %s seconds ---" % (time.time() - start_time))
#     return atom_neighbor_array


def distance_calculator(point_a, point_b):
    """
    Function calcualtes the distance between two points
    """

    a_array = np.asarray(point_a, dtype=float)
    b_array = np.asarray(point_b, dtype=float)
    difference = np.subtract(a_array, b_array)
    return np.linalg.norm(difference)


def distance_filter(distance) -> bool:
    """
    Returns a boolean whether the atoms pairs are within 6 angstroms

    Returns
    -------
        Returns True: if the distance is less than or equal to 6
        Returns False: if the distance is greater than 6
    """
    return distance <= 6



def shortest_distance_between_molecules(receptor_mol, ligand_mol):
    """
        Finds the shortest distance between host and guest atoms.

        Takes in array of coordinates of host and guest atoms and computes the distance only taking into account heavy atoms, ignoring protons
    Parameters
    ----------
        receptor_mol: numpy.ndarry
            An array of x,y,z positions of each atom within the host
        ligand_mol: numpy.ndarry
            An array of x,y,z positions of each atom within the guest
    Returns
    -------
        ligand_selected_atom_parmindex: int
            The selected atom index value (ligand system)
        ligand_selected_atom_position: numpy.ndarry
            The selected atom position (x,y,z) coordinate for the guest
        receptor_selected_atom_parmindex: int
            The selected atom index value (receptor system)
        receptor_selected_atom_position: numpy.ndarry
            The selected atom position (x,y,z) coordinate for the host
        shortest_distance: float
            The distance between the atoms chosen
    """

    start_time = time.time()

    mole_a_all_atoms = ligand_mol.xyz[0]
    mole_a_no_protons = list(
        itertools.filterfalse(
            lambda atom: atom.name.startswith("H"), ligand_mol.topology.atoms
        )
    )
    mole_a_coordinates = [mole_a_all_atoms[atom.index] for atom in mole_a_no_protons]

    # access molecule coordiantes array
    receptor_mol_all_atoms = receptor_mol.xyz[0]
    # ignore any protons
    mole_b_no_protons = list(
        itertools.filterfalse(
            lambda atom: atom.name.startswith("H"), receptor_mol.topology.atoms
        )
    )
    # coordinates with only heavy atoms
    mole_b_coordinates = [
        receptor_mol_all_atoms[atom.index] for atom in mole_b_no_protons
    ]

    # all possible coordinate combinations between molecule a and b
    atom_coordinate_combindations = list(
        itertools.product(mole_a_coordinates, mole_b_coordinates)
    )
    # all possible atom combinations between molecule a and b
    atom_combindations = list(itertools.product(mole_a_no_protons, mole_b_no_protons))

    # calculate distances
    distances = list(
        itertools.starmap(distance_calculator, atom_coordinate_combindations)
    )

    # shortest distance between systems
    shortest_distance = np.min(distances)
    index_position = np.argmin(distances)

    # get the atom chosen for the ligand and receptor
    ligand_atom_chosen = atom_combindations[index_position][0]
    receptor_atom_chosen = atom_combindations[index_position][1]

    ligand_selected_atom_parmindex = ligand_atom_chosen.index + 1
    ligand_selected_atom_position = atom_coordinate_combindations[index_position][0]
    receptor_selected_atom_parmindex = receptor_atom_chosen.index + 1
    receptor_selected_atom_position = atom_coordinate_combindations[index_position][1]
    print("--- %s seconds ---" % (time.time() - start_time))
    print("refactor_screen_between_molecules\n")
    print(f"ligand_selected_atom_parmindex: {ligand_selected_atom_parmindex}")
    print(f"ligand_selected_atom_position: {ligand_selected_atom_position}")
    print(f"receptor_selected_atom_parmindex: {receptor_selected_atom_parmindex}")
    print(f"receptor_selected_atom_position: {receptor_selected_atom_position}")
    print(f"shortest_distance: {shortest_distance}")
    return (
        ligand_selected_atom_parmindex,
        ligand_selected_atom_position,
        receptor_selected_atom_parmindex,
        receptor_selected_atom_position,
        shortest_distance,
    )


def find_angle(point_a, point_b, point_c):
    """
    Function calculates the angle created by two lines connectin 3 points, where point_b is the vertex.

    """

    a = np.array(point_a, dtype=float)
    b = np.array(point_b, dtype=float)
    c = np.array(point_c, dtype=float)
    # difference
    # ba = a - b
    # bc = c - b
    ba = np.subtract(a, b)
    bc = np.subtract(c, b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    angle = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle)

    return angle_degrees



def refactor_find_heavy_bonds(parmed_traj_bonds):
    """Find heavy atoms that are covalently bonded to another heavy atom (i.e. NOT hydrogens)
    Iterate through list of parmed Atom descriptions and return a list of boolean expression 
    whether the atom has more than one heavy atom bounded to it.
    parmed_traj_bonds: list[parmed.topologyobjects.Atom]
        AtomList instances.

    Returns: 
    --------
        Checks whether a heavy atom has more than one heavy atom bounded to it.
    """

    num_total_bonds = len(parmed_traj_bonds.bonds)

    num_heavy_bonds = 0

    for bond_index in range(num_total_bonds):
        bonded_atom_name = parmed_traj_bonds.bonds[bond_index].atom2.name

        if not bonded_atom_name.startswith("H"):
            num_heavy_bonds += 1

    return num_heavy_bonds > 1

def ramped_force(conformational_force, max_force, slope):
    """
    Force constant of a ramped restraint subset in one window.

    Parameters
    ----------
    conformational_force : float
        The window's network force constant, kcal/mol/A^2.
    max_force : float
        The network force at the top of the ladder.
    slope : float
        How many times faster than the network the subset falls, in log2.

    Returns
    -------
    float
        ``2**(E_max - slope*(E_max - E_con))``: equal to ``max_force`` at the top of the ladder
        and effectively zero at the floor (2^-32 at E_con = -14 for E_max = 4, slope 2).
    """
    e_max, e_con = np.log2(max_force), np.log2(conformational_force)
    return float(2.0 ** (e_max - slope * (e_max - e_con)))


def count_heavy_neighbors(atom):
    """
    Number of heavy atoms covalently bonded to a parmed atom.

    Unlike ``refactor_find_heavy_bonds``, which reads ``bond.atom2`` whatever end the query atom
    sits on (so a terminal N can count itself), this counts the partner of each bond. The legacy
    function is left as is so legacy Boresch anchors stay reproducible.

    Parameters
    ----------
    atom : parmed.topologyobjects.Atom
        Atom to inspect.

    Returns
    -------
    int
        Bonded partners whose name does not start with ``H``.
    """
    partners = (bond.atom2 if bond.atom1 is atom else bond.atom1 for bond in atom.bonds)
    return sum(not partner.name.startswith("H") for partner in partners)


def screen_for_distance_restraints(num_atoms, com, mol):
    """
    This function combs a molecule object (num_atoms) to find the atom closest to the coordinates at (com).
    This (com) atom is traditionally an atom closest to the center of mass used as a distance restraint in NMR calculations in AMBER.
    It should however be re_named as it sometimes is another atom altogther.

    Parameters
    ----------
    num_atoms: int
        Total number of atoms in specified system
    com: numpy.ndarry
        Center of mass of complex
    mol: pytraj.trajectory.trajectory.Trajectory
        Pytraj trajectory file of HOST or guest system

    Returns
    -------
    ligand_selected_atom_parmindex: int
        Index of the ligand atom
    selected_atom_position: numpy.ndarry
        Array of coordiante of selected atom position
    shorest_distance: float
        distance of shortest
    """
    start_time = time.time()
    atom_coords = mol.xyz
    # calculate the distance betwen indivdual atoms and center of mass of solute
    distances = list(
        map(
            distance_calculator,
            atom_coords[0],
            itertools.repeat(com, len(atom_coords[0])),
        )
    )
    # initialize from the initial atom
    selected_atom_parmindex = 1
    shorest_distance = distances[0]
    selected_atom_name = mol.topology.atom(0).name
    selected_atom_position = atom_coords[0][0]
    # cycle through all computed distances
    for atom_index in range(num_atoms):
        atom_name = mol.topology.atom(atom_index).name
        if not atom_name.startswith("H"):
            if distances[atom_index] < shorest_distance:
                shorest_distance = distances[atom_index]
                selected_atom_parmindex = (
                    atom_index + 1
                )  # Plus one alters the index intialized at 0 to match with parm id initialized at 1
                selected_atom_name = atom_name
                selected_atom_position = atom_coords[0][atom_index]
    print("--- %s seconds ---" % (time.time() - start_time))
    print(
        "Success!! Selected distance atom: Type:",
        selected_atom_name,
        "At coordinates: ",
        selected_atom_position,
    )
    return selected_atom_parmindex, selected_atom_position, shorest_distance
