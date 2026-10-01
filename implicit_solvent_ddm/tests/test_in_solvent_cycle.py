"""Tests for the in-solvent decoupling cycle."""
from types import SimpleNamespace

import parmed as pmd
import pytest

from implicit_solvent_ddm import block_mbar, setup_simulations
from implicit_solvent_ddm.matrix_order import CycleSteps
from implicit_solvent_ddm.setup_simulations import SimulationSetup
from implicit_solvent_ddm.tests.test_lj_window_and_dihedral_bounds import PARM, _cb7_topology

CON = [-8.0, 4.0]
ORIENT = [-4.0, 8.0]
CHARGES = [0.0, 0.5, 1.0]


def _cycle():
    return CycleSteps(conformation_forces=CON, orientational_forces=ORIENT, charges_windows=CHARGES,
                      external_dielectic=[], lj_windows=[0.2], screening_windows=[0.5], in_solvent=True)


def test_complex_order_and_band_indices():
    cs = _cycle()
    order = cs.complex_order
    assert [state[0] for state in order] == [
        "no_interactions", "lj_0.2", "interactions", "screening_0.5",
        "electrostatics", "electrostatics", "electrostatics", "lambda_window", "endstate",
    ]
    assert all(state[1] == "78.5" for state in order)
    assert order[cs.start_complex_charge_matrix] == ("electrostatics", "78.5", "0.0", "4.0_8.0")
    assert order[cs.halo_restraint_matrix] == ("electrostatics", "78.5", "1.0", "4.0_8.0")


def test_ligand_and_receptor_orders():
    cs = _cycle()
    assert cs.ligand_order == [
        ("endstate", "78.5", "1.0", "0.0"),
        ("lambda_window", "78.5", "1.0", "-8.0"),
        ("lambda_window", "78.5", "1.0", "4.0"),
        ("electrostatics", "78.5", "0.5", "4.0"),
        ("electrostatics", "78.5", "0.0", "4.0"),
    ]
    assert cs.ligand_order[cs.start_ligand_charge_matrix] == ("lambda_window", "78.5", "1.0", "4.0")
    assert cs.receptor_order == cs.endstate + cs.apply_restraints
    with pytest.raises(ValueError):
        cs.start_receptor_gb_matrix


def _setup(system_type, recorded):
    setup = object.__new__(SimulationSetup)
    setup.system_type = system_type
    setup.topology = "top.parm7"
    setup.max_conformational_exponent = 4.0
    setup.max_orientational_exponent = 8.0
    setup.num_cores = 1
    setup.binding_mode = "bm.rst7"
    setup.restraints = "restraints"
    setup.dirstruct = "dirstruct_halo" if system_type == "complex" else "dirstruct_apo"
    setup.simulations = recorded
    setup.config = SimpleNamespace(
        inputs={"default_mdin": "WATER", "no_solvent_mdin": "GAS"},
        intermediate_args=SimpleNamespace(decouple_in_solvent=True, igb_solvent=2),
        system_settings=SimpleNamespace(executable="pmemd.cuda", mpi_command="mpirun", CUDA=True,
                                        working_directory=".", memory="1G", disk="1G",
                                        num_accelerators=1, top_directory_path="top"),
        workflow=SimpleNamespace(debug=False),
    )
    return setup


def test_setup_windows_key_into_the_cycle_order(monkeypatch):
    recorded = []
    monkeypatch.setattr(setup_simulations, "Simulation", lambda **kwargs: SimpleNamespace(**kwargs))
    cs = _cycle()
    complex_setup = _setup("complex", recorded)
    complex_setup.setup_remove_gb_solvent_simulation("key", "parm")
    complex_setup.setup_lj_scaled_simulation("key", "parm", 0.2)
    complex_setup.setup_lj_interations_simulation("key", "parm")
    complex_setup.setup_screening_simulation("key", "parm", 0.5)
    for charge in CHARGES:
        complex_setup.setup_ligand_charge_simulation("parm", charge, "key")
    complex_states = set(cs.complex_order)
    for sim in recorded:
        assert block_mbar.canonical_state_key(sim.directory_args, complex_states) is not None
        assert sim.input_file == "WATER"
        assert sim.directory_args["igb_value"] == 2

    recorded.clear()
    ligand_setup = _setup("ligand", recorded)
    for charge in (0.0, 0.5):
        ligand_setup.setup_ligand_charge_simulation("parm", charge, "key")
    for sim in recorded:
        assert block_mbar.canonical_state_key(sim.directory_args, set(cs.ligand_order)) is not None
        assert sim.input_file == "WATER"


def test_screen_scale_edits_only_ligand_screen(tmp_path, monkeypatch):
    original = pmd.load_file(PARM)
    scaled = _cb7_topology(tmp_path, monkeypatch, screen_scale=0.5)
    for before, after in zip(original.atoms, scaled.atoms):
        expected = 0.5 * before.screen if before.residue.name == "M01" else before.screen
        assert after.screen == pytest.approx(expected)
