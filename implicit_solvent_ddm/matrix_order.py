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
    # Must match setup_simulations.apo_endstate_dirstruct's state_label: that value is the
    # MBAR index key, so a mismatch makes every endstate .loc[] lookup raise KeyError.
    endstate_label: str = "endstate"
    lj_windows: list[float] = field(default_factory=list)
    # "force"  -> conformation_forces are 2^p spring constants, so BIGGER is stronger.
    # "rmsd"   -> they are flat-bottom tolerances r3 in Angstrom, so SMALLER is stronger.
    # Only the ordering and the anchor selector change; every length, and therefore every band
    # index below, is identical in both modes.
    restraint_coordinate: str = "force"
    screening_windows: list[float] = field(default_factory=list)
    in_solvent: bool = False
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

        if self.in_solvent and self.external_dielectic:
            raise ValueError("in_solvent cycles have no GB-dielectric band; external_dielectic must be empty")

        self.endstate = [(self.endstate_label, "78.5", "1.0", "0.0")]

        self.no_gb = [("no_gb", "0.0", "1.0", f"{self._strongest()}")]

        self.interations = [
            (
                "interactions",
                self._decoupled_extdiel,
                "0.0",
                f"{self._strongest()}_{max(self.orientational_forces)}",
            )
        ]

        self.no_interactions = [
            (
                "no_interactions",
                self._decoupled_extdiel,
                "0.0",
                f"{self._strongest()}_{max(self.orientational_forces)}",
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
    def lj_scaled(self):
        """Intermediate ligand-receptor LJ windows, ascending, between no_interactions and interactions."""
        return [
            (
                f"lj_{float(lj_scale)}",
                self._decoupled_extdiel,
                "0.0",
                f"{self._strongest()}_{max(self.orientational_forces)}",
            )
            for lj_scale in sorted(self.lj_windows)
        ]

    @property
    def screening(self):
        """Ligand GB-screening windows, ascending, between interactions and uncharged electrostatics."""
        return [
            (
                f"screening_{float(screen_scale)}",
                "78.5",
                "0.0",
                f"{self._strongest()}_{max(self.orientational_forces)}",
            )
            for screen_scale in sorted(self.screening_windows)
        ]

    @property
    def _decoupled_extdiel(self) -> str:
        """External dielectric of the decoupled-ligand states: water in solvent, gas otherwise."""
        return "78.5" if self.in_solvent else "0.0"

    @property
    def _rmsd(self) -> bool:
        return self.restraint_coordinate == "rmsd"

    def _strongest(self) -> float:
        """Return the pinned max-restraint anchor value.

        Returns
        -------
        float
            ``max(conformation_forces)`` in force mode, ``min(...)`` in rmsd mode, where the
            tightest flat bottom is the smallest tolerance.
        """
        return (min if self._rmsd else max)(self.conformation_forces)

    def _con_weak_to_strong(self) -> list:
        """Conformational coordinates ordered from unrestrained-most to most restrained.

        Returns
        -------
        list of float
            Ascending force constants, or descending RMSD tolerances.
        """
        return sorted(self.conformation_forces, reverse=self._rmsd)

    def _paired_strong_to_weak(self) -> list:
        """Index-paired ``(conformational, orientational)`` ordered most restrained first.

        Sorting the two lists independently, as this used to, only agrees with the index pairing
        because ``derive_offset`` guarantees a constant offset; in rmsd mode the two carry different
        units and opposite senses, so the pairing has to be carried through the sort.

        Returns
        -------
        list of tuple of float
            ``(con, orient)`` pairs, strongest restraint first.
        """
        pairs = list(zip(self.conformation_forces, self.orientational_forces))
        return sorted(pairs, key=lambda pair: pair[0], reverse=not self._rmsd)

    @property
    def complex_GB_exl_windows(self):
        # Sorted by epsilon ASCENDING so the band runs gas-anchor -> full-water monotonically and each
        # adjacent pair is a single small step in lambda = (1 - 1/eps) (see adaptive_restraints).
        return [
            (
                "gb_dielectric",
                f"{dielectric}",
                "0.0",
                f"{self._strongest()}_{max(self.orientational_forces)}",
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
                f"{self._strongest()}",
            )
            for dielectric in sorted(self.external_dielectic, reverse=True)
        ]

    @property
    def remove_restraints(self):
        # Strongest first; [1:] drops the pinned anchor, which complex_charges already carries.
        return [
            ("lambda_window", "78.5", "1.0", f"{con_rest}_{orien_rest}")
            for con_rest, orien_rest in self._paired_strong_to_weak()
        ][1:]

    @property
    def complex_charges(self):
        charges_sorted = sorted(self.charges_windows)
        return [
            (
                "electrostatics",
                "78.5",
                f"{charge}",
                f"{self._strongest()}_{max(self.orientational_forces)}",
            )
            for charge in charges_sorted
        ]

    @property
    def apply_restraints(self):
        # Weakest restraint first, so the band runs endstate -> pinned anchor.
        sorted_conformation_forces = self._con_weak_to_strong()
        return [
            ("lambda_window", "78.5", "1.0", f"{con_rest}")
            for con_rest in sorted_conformation_forces
        ]

    @property
    def ligand_charges(self):
        # in solvent the q=1 window is the restraint anchor, so it is not repeated
        temp_charges = [c for c in self.charges_windows if not (self.in_solvent and c == 1.0)]
        temp_charges.sort(reverse=True)
        return [
            ("electrostatics", self._decoupled_extdiel, f"{charge}", f"{self._strongest()}")
            for charge in temp_charges
        ]

    @property
    def ligand_order(self) -> list:
        return self.endstate + self.apply_restraints + self.ligand_charges

    @property
    def receptor_order(self) -> list:
        # GB-dielectric band (when external_dielectic is non-empty) splices between the restraint windows
        # and the gas no_gb anchor: endstate -> restraints(ASC) -> GB(eps DESC) -> no_gb(gas).
        if self.in_solvent:
            return self.endstate + self.apply_restraints
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
            + self.lj_scaled
            + self.interations
            + self.screening
            + self.complex_GB_exl_windows
            + self.complex_charges
            + self.remove_restraints
            + [(self.endstate_label, "78.5", "1.0", "0.0_0.0")]
        )

    @property
    def start_ligand_charge_matrix(self):
        """Index in ``ligand_order`` of the full-charge state the ligand charge band starts from."""
        start = len(self.endstate) + len(self.apply_restraints)
        return start - 1 if self.in_solvent else start

    @property
    def start_complex_charge_matrix(self):
        return (
            len(self.no_interactions)
            + len(self.lj_scaled)
            + len(self.interations)
            + len(self.screening)
            + len(self.external_dielectic)
        )

    @property
    def start_gb_extdiel_matrix(self):
        # index of the interactions state, the gas anchor the GB band starts from
        self._require_gas_cycle("start_gb_extdiel_matrix")
        return len(self.no_interactions) + len(self.lj_scaled)

    @property
    def start_receptor_gb_matrix(self):
        # Index in receptor_order of the FIRST GB-dielectric window (just after endstate + restraints).
        # The receptor dielectric BAND (for ALS) is [start_receptor_gb_matrix - 1 : start + G + 1], i.e.
        # the max-restraint water anchor -> G GB windows -> the gas no_gb anchor (inclusive).
        self._require_gas_cycle("start_receptor_gb_matrix")
        return len(self.endstate) + len(self.apply_restraints)

    @property
    def apo_end_restraint_matrix(self) -> int:
        return len(self.endstate) + len(self.apply_restraints) - 1

    @property
    def halo_restraint_matrix(self) -> int:
        return (
            len(self.no_interactions)
            + len(self.lj_scaled)
            + len(self.interations)
            + len(self.screening)
            + len(self.complex_GB_exl_windows)
            + len(self.complex_charges)
            - 1
        )

    def _require_gas_cycle(self, name: str):
        """Raise for an index that only exists in the gas-phase cycle.

        Parameters
        ----------
        name : str
            Property being accessed.

        Raises
        ------
        ValueError
            If the cycle decouples in solvent.
        """
        if self.in_solvent:
            raise ValueError(f"{name} is undefined for an in_solvent cycle (no GB-dielectric band)")

    def construct_order(self):
        return []


def cycle_steps_from_config(config) -> CycleSteps:
    """Build the rounded ``CycleSteps`` for a workflow config.

    Parameters
    ----------
    config : Config
        Workflow configuration.

    Returns
    -------
    CycleSteps
        Cycle order with restraint values rounded to 3 decimals.
    """
    args = config.intermediate_args
    cycle_steps = CycleSteps(
        conformation_forces=args.exponent_conformational_forces_list,
        orientational_forces=args.exponent_orientational_forces_list,
        charges_windows=args.charges_lambda_window,
        external_dielectic=args.gb_extdiel_windows,
        endstate_label=config.endstate_method.endstate_state_label,
        lj_windows=args.lj_scale_windows,
        restraint_coordinate="rmsd" if args.rmsd_mode else "force",
        screening_windows=args.screening_windows,
        in_solvent=args.decouple_in_solvent,
    )
    cycle_steps.round(3)
    return cycle_steps
