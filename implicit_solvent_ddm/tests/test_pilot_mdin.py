"""Pure unit tests for the ALS pilot MD-length / frame math (mdin.pilot_md_steps).

The pilot is always 50 ps (the paper's value), with the step count derived from the user mdin's
timestep so it is correct regardless of dt (2 fs vs the paper's 4 fs), and ntwx set so the pilot
writes ~pilot_frames frames for MBAR. These tests do NOT touch Toil/AMBER. Run with::

    pytest implicit_solvent_ddm/tests/test_pilot_mdin.py -v
"""
from implicit_solvent_ddm.mdin import pilot_md_steps

MDIN_2FS = "  dt = 0.002\n  nstlim = 100\n  ntwx=10\n"
MDIN_4FS = "  dt = 0.004\n  nstlim = 2000000\n  ntwx=5000\n"   # paper: 4 fs, ns-scale production
MDIN_NO_DT = "  nstlim = 100\n  ntwx=10\n"


# --- 50 ps is derived from the user's timestep -----------------------------------------------------
def test_50ps_at_2fs_is_25000_steps():
    nstlim, _ = pilot_md_steps(MDIN_2FS, pilot_ps=50.0, pilot_frames=100)
    assert nstlim == 25000  # 50 ps / 0.002 ps


def test_50ps_at_4fs_is_12500_steps():
    # same 50 ps wall-time, fewer steps because the timestep is larger (the paper's setup)
    nstlim, _ = pilot_md_steps(MDIN_4FS, pilot_ps=50.0, pilot_frames=100)
    assert nstlim == 12500  # 50 ps / 0.004 ps


def test_pilot_ps_scales():
    assert pilot_md_steps(MDIN_2FS, pilot_ps=100.0, pilot_frames=100)[0] == 50000


# --- explicit step override wins (tiny test systems where 50 ps is absurd) -------------------------
def test_pilot_nstlim_override_wins():
    nstlim, _ = pilot_md_steps(MDIN_2FS, pilot_ps=50.0, pilot_frames=100, pilot_nstlim=1000)
    assert nstlim == 1000  # ignores pilot_ps/dt


# --- ntwx is set for ~pilot_frames frames, independent of dt and production ntwx -------------------
def test_ntwx_targets_pilot_frames():
    # 25000 steps / 100 frames -> write every 250 steps
    nstlim, ntwx = pilot_md_steps(MDIN_2FS, pilot_ps=50.0, pilot_frames=100)
    assert ntwx == 250
    assert nstlim // ntwx == 100  # ~100 frames


def test_ntwx_independent_of_production_ntwx():
    # the production mdin's ntwx=5000 (tuned for ns-scale) must NOT leak into the pilot
    _, ntwx = pilot_md_steps(MDIN_4FS, pilot_ps=50.0, pilot_frames=100)
    assert ntwx == 125  # 12500 / 100, not 5000


def test_ntwx_floor_is_one():
    # very short override with many requested frames must never produce ntwx=0
    nstlim, ntwx = pilot_md_steps(MDIN_2FS, pilot_ps=50.0, pilot_frames=100, pilot_nstlim=50)
    assert nstlim == 50 and ntwx == 1  # round(50/100)=0 -> floored to 1 -> 50 frames


def test_frame_count_for_cb7_smoke():
    # cb7 driver default: 1000 steps, 100 frames -> ntwx=10 -> 100 frames (no more 5-frame thin data)
    nstlim, ntwx = pilot_md_steps(MDIN_2FS, pilot_ps=50.0, pilot_frames=100, pilot_nstlim=1000)
    assert nstlim == 1000 and ntwx == 10


# --- dt fallback when the mdin has no explicit dt --------------------------------------------------
def test_dt_default_when_absent():
    # AMBER's own default dt is 0.001 ps -> 50 ps = 50000 steps
    nstlim, _ = pilot_md_steps(MDIN_NO_DT, pilot_ps=50.0, pilot_frames=100)
    assert nstlim == 50000
