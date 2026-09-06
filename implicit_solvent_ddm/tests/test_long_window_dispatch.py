"""Pure unit tests for handing ONE window the stretched mdin (SimulationSetup).

``setup_apply_restraint_windows`` builds the restraint windows for all three legs off a single
shared ``default_mdin``. ``_restraint_window_mdin`` is the seam that redirects exactly one of them:
the lowest exponent, on a leg in ``LONG_WINDOW_LEGS``, outside an ALS pilot. Every other window,
leg, and mode must come back with ``default_mdin``. No Toil, no AMBER. Run with::

    pytest implicit_solvent_ddm/tests/test_long_window_dispatch.py -v
"""
import pytest

from implicit_solvent_ddm.setup_simulations import LONG_WINDOW_LEGS, SimulationSetup

FLOOR = -14.0


def _setup(system_type="receptor", long_exponent=FLOOR, **inputs):
    """A SimulationSetup carrying only what _restraint_window_mdin reads."""
    setup = object.__new__(SimulationSetup)
    setup.system_type = system_type
    setup.long_window_exponent = (
        long_exponent if system_type in LONG_WINDOW_LEGS else None
    )
    resolved = {"default_mdin": "DEFAULT", "long_window_mdin": "LONG"}
    resolved.update(inputs)
    setup.config = type("Cfg", (), {"inputs": resolved})()
    return setup


def test_receptor_is_the_only_leg_wired_up():
    assert LONG_WINDOW_LEGS == ("receptor",)


def test_lowest_receptor_window_gets_the_long_mdin():
    assert _setup()._restraint_window_mdin(FLOOR) == "LONG"


@pytest.mark.parametrize("exponent", [-13.0, -8.0, 0.0, 4.0])
def test_every_other_window_gets_the_default_mdin(exponent):
    assert _setup()._restraint_window_mdin(exponent) == "DEFAULT"


@pytest.mark.parametrize("leg", ["complex", "ligand"])
def test_legs_outside_long_window_legs_are_untouched(leg):
    assert _setup(system_type=leg)._restraint_window_mdin(FLOOR) == "DEFAULT"


def test_als_pilot_never_gets_the_long_mdin():
    # the pilot swaps default_mdin for the 50 ps pilot mdin; it must stay uniformly short
    assert _setup(als_pilot=True)._restraint_window_mdin(FLOOR) == "DEFAULT"


def test_missing_key_falls_back_to_default():
    # a jobstore/config predating the feature must resume without KeyError
    setup = _setup()
    del setup.config.inputs["long_window_mdin"]
    assert setup._restraint_window_mdin(FLOOR) == "DEFAULT"


def test_knob_off_falls_back_to_default():
    assert _setup(long_exponent=None)._restraint_window_mdin(FLOOR) == "DEFAULT"


def test_dispatch_tolerates_float_noise():
    # the loop passes round(np.log2(np.exp2(x)), 3); both sides round to 3 places
    assert _setup()._restraint_window_mdin(-13.9999999999) == "LONG"
