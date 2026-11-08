"""Pure unit tests for a fixed restraint reference file (``restraint_reference: <path>``).

A reference drawn from each run's own sampling (``lastframe`` / ``pose_medoid``) differs per
replicate, and the two ~2,700 kcal/mol protein legs amplify that into ~10-20 kcal/mol of scatter
(MCL-1/ligand-17: rep2's medoid reference alone tripled the receptor seam's restraint energy for
every endstate). A path makes every replicate restrain to, and start from, the same structure.
No Toil job store, no AMBER. Run with::

    pytest implicit_solvent_ddm/tests/test_restraint_reference.py -v
"""
import os
from types import SimpleNamespace

import pytest

from implicit_solvent_ddm.config import Config, IntermediateStateArgs
from implicit_solvent_ddm.workflow_phases import use_reference_frame

HERE = os.path.dirname(os.path.realpath(__file__))
MDIN = os.path.join(HERE, "input_files", "intermediate.mdin")
STRUCTS = os.path.join(HERE, "structs")
PARM = os.path.join(STRUCTS, "cb7-mol01.parm7")
SINGLE = os.path.join(STRUCTS, "cb7-mol01.ncrst")
MULTI = os.path.join(STRUCTS, "cb7-mol01_300K.nc")  # 50 frames
OTHER_PARM = os.path.join(STRUCTS, "CB7.parm7")  # 126 atoms vs the complex's 150


def _args(reference):
    return IntermediateStateArgs(
        exponent_conformational_forces=[-8.0, 4.0],
        exponent_orientational_forces=[-4.0, 8.0],
        restraint_type=1,
        igb_solvent=2,
        mdin_intermediate_file=MDIN,
        temperature=298,
        restraint_reference=reference,
    )


def _config(reference, method="remd", coordinate="original.rst7", parm=PARM):
    """Just the fields _resolve_restraint_reference reads."""
    return SimpleNamespace(
        intermediate_args=_args(reference),
        endstate_files=SimpleNamespace(
            complex_parameter_filename=parm,
            complex_coordinate_filename=coordinate,
            restraint_reference_coordinate=None,
        ),
        endstate_method=SimpleNamespace(endstate_method_type=method),
    )


def test_path_resolves_to_absolute_and_keywords_stay_sampling_based():
    rel = os.path.relpath(SINGLE)
    assert _args(rel).restraint_reference_file == os.path.abspath(SINGLE)
    assert _args("pose_medoid").restraint_reference_file is None


def test_typo_is_rejected_not_treated_as_a_path():
    with pytest.raises(ValueError, match="restraint_reference"):
        _args("pose_mediod")


def test_remd_restrains_to_and_starts_from_the_reference():
    config = _config(SINGLE)
    Config._resolve_restraint_reference(config)
    assert config.endstate_files.restraint_reference_coordinate == SINGLE
    # the complex REMD starts from the same structure the restraints use
    assert config.endstate_files.complex_coordinate_filename == SINGLE


def test_user_endstate_keeps_its_ensemble():
    # with method 0 the complex coordinate IS the sampled ensemble; only the reference changes
    config = _config(SINGLE, method=0, coordinate=MULTI)
    Config._resolve_restraint_reference(config)
    assert config.endstate_files.restraint_reference_coordinate == SINGLE
    assert config.endstate_files.complex_coordinate_filename == MULTI


def test_multi_frame_reference_is_rejected():
    with pytest.raises(ValueError, match="50 frames"):
        Config._resolve_restraint_reference(_config(MULTI))


def test_reference_from_a_different_system_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        Config._resolve_restraint_reference(_config(SINGLE, parm=OTHER_PARM))


def test_swap_replaces_only_the_binding_mode():
    # the receptor trajectory must stay at index 2 (see 9ec10b2: [2] vs [3] was a real bug)
    job = SimpleNamespace(fileStore=SimpleNamespace(logToMaster=lambda msg: None))
    endstate = ("medoid.ncrst", "complex.nc", "receptor.nc", "ligand.nc")
    assert use_reference_frame(job, endstate, "ref.rst7") == ("ref.rst7", "complex.nc", "receptor.nc", "ligand.nc")


class _Job:
    """Stand-in Toil job that records what a phase function schedules, without running it."""

    def __init__(self, fn=None, args=(), obj=None):
        self.fn, self.args, self.obj = fn, args, obj
        self.children, self.followons = [], []
        self.fileStore = SimpleNamespace(logToMaster=lambda msg: None)

    def addChildJobFn(self, fn, *args, **kwargs):
        self.children.append(_Job(fn, args))
        return self.children[-1]

    def addFollowOnJobFn(self, fn, *args, **kwargs):
        self.followons.append(_Job(fn, args))
        return self.followons[-1]

    def addChild(self, obj):
        self.children.append(_Job(obj=obj))
        return self.children[-1]

    def rv(self, *index):
        return ("promise", self, index)


def _flat_bottom_settings():
    return SimpleNamespace(restrained_receptor_atoms=[0], restrained_ligand_atoms=[1],
                           flat_bottom_width=8.0, harmonic_distance=10.0, spring_constant=1.0,
                           flat_bottom_restraints=None)


def test_user_endstate_builds_every_restraint_from_the_reference():
    # endstate_method 0: Phase 2 must swap the reference in, and Phase 3 must build the
    # receptor/ligand split, the Boresch restraints and the conformational network from it --
    # not from the supplied ensemble's last frame.
    from implicit_solvent_ddm.run_endstate import user_defined_endstate
    from implicit_solvent_ddm.workflow_phases import (
        decompose_system_and_generate_restraints,
        run_endstate_simulations,
    )

    config = SimpleNamespace(
        workflow=SimpleNamespace(run_endstate_method=False),
        endstate_method=SimpleNamespace(endstate_method_type=0, flat_bottom=_flat_bottom_settings()),
        endstate_files=SimpleNamespace(complex_parameter_filename=PARM, complex_coordinate_filename=MULTI,
                                       restraint_reference_coordinate=SINGLE),
        amber_masks=SimpleNamespace(ligand_mask=":M01", receptor_mask=":CB7"),
        intermediate_args=_args(SINGLE),
        inputs={"flat_bottom_restraint": "flat_bottom.RST"},
    )

    phase2 = _Job()
    run_endstate_simulations(phase2, config)
    (endstate,) = phase2.children
    assert endstate.fn is user_defined_endstate
    (swap,) = endstate.followons
    assert swap.fn is use_reference_frame and swap.args[1] == SINGLE

    # what Phase 3 receives once the swap has run
    resolved = use_reference_frame(phase2, ("ensemble_lastframe.ncrst", "complex.nc", "receptor.nc", "ligand.nc"), SINGLE)
    phase3 = _Job()
    decompose_system_and_generate_restraints(phase3, resolved, config)
    (split,) = phase3.children
    (boresch,) = split.children
    (maker,) = boresch.children
    assert split.args[1] == SINGLE
    assert boresch.obj.complex_coordinate == SINGLE
    assert maker.obj.complex_binding_mode == SINGLE


def test_flat_bottom_uses_the_reference_not_the_ensemble():
    from implicit_solvent_ddm.restraints import FlatBottom

    config = SimpleNamespace(
        endstate_method=SimpleNamespace(flat_bottom=_flat_bottom_settings()),
        amber_masks=SimpleNamespace(ligand_mask=":M01", receptor_mask=":CB7"),
        endstate_files=SimpleNamespace(complex_parameter_filename=PARM, complex_coordinate_filename=MULTI,
                                       restraint_reference_coordinate=SINGLE),
    )
    assert FlatBottom(config).complex_coordinate == SINGLE
