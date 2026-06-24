from dataclasses import dataclass, field
from typing import Union


@dataclass
class CycleSteps:
    """
    Simple dataclass that arranges all the thermodyamic steps in the correct chronological order for the given system (i.e. complex, ligand, receptor).
    """

    conformation_forces: list[Union[int, float]]
    orientational_forces: list[Union[int, float]]
    charges_windows: list[float]
    external_dielectic: list[float]
    endstate: list[tuple[str, str, str, str]] = field(default_factory=list, init=False)
    no_gb: list[tuple[str, str, str, str]] = field(init=False)
    no_interactions: list[tuple[str, str, str, str]] = field(init=False)
    interations: list[tuple[str, str, str, str]] = field(init=False)

    def __post_init__(self):
        # self.external_dielectic = [
        #     78.5 * extdeil for extdeil in self.external_dielectic
        # ]

        self.conformation_forces = [float(force) for force in self.conformation_forces]

        self.orientational_forces = [
            float(force) for force in self.orientational_forces
        ]

        self.endstate = [("endstate", "78.5", "1.0", "0.0")]

        self.no_gb = [("no_gb", "0.0", "1.0", f"{max(self.conformation_forces)}")]

        self.interations = [
            (
                "interactions",
                "0.0",
                "0.0",
                f"{max(self.conformation_forces)}_{max(self.orientational_forces)}",
            )
        ]

        self.no_interactions = [
            (
                "no_interactions",
                "0.0",
                "0.0",
                f"{max(self.conformation_forces)}_{max(self.orientational_forces)}",
            )
        ]

    def round(self, dec):
        self.conformation_forces = [
            round(float(force), dec) for force in self.conformation_forces
        ]
        self.orientational_forces = [
            round(float(force), dec) for force in self.orientational_forces
        ]

    @property
    def complex_GB_exl_windows(self):
        # Sorted by epsilon ASCENDING so the band runs gas-anchor -> full-water monotonically and each
        # adjacent pair is a single small step in lambda = (1 - 1/eps) (see adaptive_restraints).
        return [
            (
                "gb_dielectric",
                f"{dielectric}",
                "0.0",
                f"{max(self.conformation_forces)}_{max(self.orientational_forces)}",
            )
            for dielectric in sorted(self.external_dielectic)
        ]

    @property
    def receptor_GB_exl_windows(self):
        # Receptor (apo-host) GB-dielectric band. Same epsilon schedule as the complex band so the host
        # desolvation cancels between the two legs, but at FULL host charge (q=1.0; the ligand is not in
        # the apo system) and a single conformational restraint label (apo systems have no orientational
        # restraint). Sorted epsilon DESCENDING (water -> gas) so it splices monotonically between
        # apply_restraints (ends at max restraint, eps=78.5) and no_gb (gas).
        return [
            (
                "gb_dielectric",
                f"{dielectric}",
                "1.0",
                f"{max(self.conformation_forces)}",
            )
            for dielectric in sorted(self.external_dielectic, reverse=True)
        ]

    @property
    def remove_restraints(self):
        temp_con_rest = self.conformation_forces.copy()
        temp_orient_rest = self.orientational_forces.copy()
        # reverse order
        temp_con_rest.sort(reverse=True)
        temp_orient_rest.sort(reverse=True)

        return [
            ("lambda_window", "78.5", "1.0", f"{con_rest}_{orien_rest}")
            for con_rest, orien_rest in zip(temp_con_rest, temp_orient_rest)
        ][1:]

    @property
    def complex_charges(self):
        charges_sorted = sorted(self.charges_windows)
        return [
            (
                "electrostatics",
                "78.5",
                f"{charge}",
                f"{max(self.conformation_forces)}_{max(self.orientational_forces)}",
            )
            for charge in charges_sorted
        ]

    @property
    def apply_restraints(self):
        # In ascending order
        sorted_conformation_forces = sorted(self.conformation_forces)
        return [
            ("lambda_window", "78.5", "1.0", f"{con_rest}")
            for con_rest in sorted_conformation_forces
        ]

    @property
    def ligand_charges(self):
        temp_charges = self.charges_windows.copy()
        temp_charges.sort(reverse=True)
        return [
            ("electrostatics", "0.0", f"{charge}", f"{max(self.conformation_forces)}")
            for charge in temp_charges
        ]

    @property
    def ligand_order(self) -> list:
        return self.endstate + self.apply_restraints + self.ligand_charges

    @property
    def receptor_order(self) -> list:
        # GB-dielectric band (when external_dielectic is non-empty) splices between the restraint windows
        # and the gas no_gb anchor: endstate -> restraints(ASC) -> GB(eps DESC) -> no_gb(gas).
        return (
            self.endstate
            + self.apply_restraints
            + self.receptor_GB_exl_windows
            + self.no_gb
        )

    @property
    def complex_order(self) -> list:
        return (
            self.no_interactions
            + self.interations
            + self.complex_GB_exl_windows
            + self.complex_charges
            + self.remove_restraints
            + [("endstate", "78.5", "1.0", "0.0_0.0")]
        )

    @property
    def start_ligand_charge_matrix(self):
        return len(self.endstate) + len(self.apply_restraints)

    @property
    def start_complex_charge_matrix(self):
        return (
            len(self.no_interactions)
            + len(self.interations)
            + len(self.external_dielectic)
        )

    @property
    def start_gb_extdiel_matrix(self):
        return len(self.no_interactions)

    @property
    def start_receptor_gb_matrix(self):
        # Index in receptor_order of the FIRST GB-dielectric window (just after endstate + restraints).
        # The receptor dielectric BAND (for ALS) is [start_receptor_gb_matrix - 1 : start + G + 1], i.e.
        # the max-restraint water anchor -> G GB windows -> the gas no_gb anchor (inclusive).
        return len(self.endstate) + len(self.apply_restraints)

    @property
    def apo_end_restraint_matrix(self) -> int:
        return len(self.endstate) + len(self.apply_restraints) - 1

    @property
    def halo_restraint_matrix(self) -> int:
        return (
            len(self.no_interactions)
            + len(self.interations)
            + len(self.complex_GB_exl_windows)
            + len(self.complex_charges)
            - 1
        )

    def construct_order(self):
        return []
