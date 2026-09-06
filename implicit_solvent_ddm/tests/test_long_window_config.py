"""Pure unit tests for resolving WHICH window runs long (IntermediateStateArgs).

The exponent is resolved once in ``__post_init__`` rather than at dispatch time, because
``workflow_phases`` clears and rebuilds ``exponent_conformational_forces_list`` inside the window
loop -- taking ``min()`` there would depend on how far the loop had got. It must also round exactly
the way that loop rounds, or the float compare in ``SimulationSetup`` never matches. No Toil, no
AMBER. Run with::

    pytest implicit_solvent_ddm/tests/test_long_window_config.py -v
"""
import os

import numpy as np
import pytest

from implicit_solvent_ddm.config import IntermediateStateArgs

MDIN = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "input_files", "intermediate.mdin"
)

LADDER = [-14.0, -13.0, -12.0, -8.0, -2.0, 4.0]


def _args(conformational=None, **overrides):
    """An IntermediateStateArgs carrying only what the long-window resolution reads."""
    conformational = LADDER if conformational is None else conformational
    kwargs = dict(
        exponent_conformational_forces=list(conformational),
        exponent_orientational_forces=[e + 4.0 for e in conformational],
        restraint_type=1,
        igb_solvent=2,
        mdin_intermediate_file=MDIN,
        temperature=298,
        # The knob defaults OFF, so the resolution tests below have to switch it on explicitly.
        long_restraint_window_ns=10.0,
    )
    kwargs.update(overrides)
    return IntermediateStateArgs(**kwargs)


def test_default_is_off():
    # Built to rescue the endstate seam, and measured NOT to: a 10 ns floor window still had zero
    # shared support with the endstate, because the window is seeded holo and the endstate is apo.
    args = IntermediateStateArgs(
        exponent_conformational_forces=list(LADDER),
        exponent_orientational_forces=[e + 4.0 for e in LADDER],
        restraint_type=1,
        igb_solvent=2,
        mdin_intermediate_file=MDIN,
        temperature=298,
    )
    assert args.long_restraint_window_ns is None
    assert args.long_restraint_window_exponent is None


def test_resolves_the_ladder_floor_when_enabled():
    assert _args().long_restraint_window_exponent == -14.0


def test_picks_the_minimum_not_a_hardcoded_value():
    # the floor is expected to move to -15/-16 as the ladder is extended
    assert _args([-16.0, -14.0, 4.0]).long_restraint_window_exponent == -16.0


def test_unsorted_exponent_list_still_picks_the_lowest():
    assert _args([4.0, -14.0, -2.0, -8.0]).long_restraint_window_exponent == -14.0


def test_source_list_is_not_mutated():
    # exponent_conformational_forces is zipped POSITIONALLY against the orientational ladder,
    # so reordering it would silently re-pair the two.
    shuffled = [4.0, -14.0, -2.0, -8.0]
    args = _args(shuffled)
    assert args.exponent_conformational_forces == shuffled


def test_none_disables():
    assert _args(long_restraint_window_ns=None).long_restraint_window_exponent is None


@pytest.mark.parametrize("bad", [0, 0.0, -1.0])
def test_rejects_non_positive_lengths(bad):
    with pytest.raises(ValueError):
        _args(long_restraint_window_ns=bad)


def test_exponent_rounded_like_the_workflow_loop():
    # workflow_phases computes round(np.log2(force), 3) off np.exp2(exponent); the stored value must
    # be that exact float or the dispatch's == never fires.
    ladder = [2.584963, 4.0]
    expected = round(float(np.log2(np.exp2(2.584963))), 3)
    assert _args(ladder).long_restraint_window_exponent == expected
