"""Pure unit tests for the long low-restraint window math (mdin.long_window_md_steps).

The lowest conformational-restraint window neighbours the endstate, whose supplied ensemble is ~10x
longer and does not shrink when the ladder is shortened, so that link is the one whose overlap
collapses (0.0157 at 2 ns vs 0.0590 at 10 ns). That one window is stretched to a target length while
holding its FRAME COUNT fixed -- post-analysis stores one rectangular frames x states matrix per leg
(postTreatment.py), so a differing count gives a ragged column. The user's timestep is read, never
rewritten. These tests do NOT touch Toil/AMBER. Run with::

    pytest implicit_solvent_ddm/tests/test_long_window_mdin.py -v
"""
from implicit_solvent_ddm.mdin import long_window_md_steps

# the production template: 2 ns at 4 fs, 10,000 frames
MDIN_PROD = "  dt = 0.004\n  nstlim = 500000\n  ntpr=50\n  ntwx=50\n"
# tests/input_files/intermediate.mdin: 0.2 ps at 2 fs, 10 frames
MDIN_2FS = "  dt = 0.002\n  nstlim = 100\n  ntpr=10\n  ntwx=10\n"
MDIN_ALREADY_LONG = "  dt = 0.004\n  nstlim = 2500000\n  ntwx=250\n"
MDIN_NO_NTWX = "  dt = 0.004\n  nstlim = 500000\n"
MDIN_NO_NSTLIM = "  dt = 0.004\n  ntwx=50\n"
MDIN_NO_DT = "  nstlim = 500000\n  ntwx=50\n"


# --- the production case ---------------------------------------------------------------------------
def test_ten_ns_from_the_two_ns_template():
    # 10 ns / 0.004 ps = 2,500,000 steps; ntwx 50 -> 250 to keep 10,000 frames
    assert long_window_md_steps(MDIN_PROD, 10.0) == (2500000, 250, 250)


def test_frame_count_is_preserved():
    # the invariant the whole feature exists to protect
    nstlim, ntwx, _ = long_window_md_steps(MDIN_PROD, 10.0)
    assert nstlim // ntwx == 500000 // 50 == 10000


def test_ntpr_tracks_ntwx():
    nstlim, ntwx, ntpr = long_window_md_steps(MDIN_PROD, 10.0)
    assert ntpr == ntwx


def test_reads_the_users_timestep():
    # same 10 ns, twice the steps because the timestep is half the size -- and still 10 frames
    nstlim, ntwx, _ = long_window_md_steps(MDIN_2FS, 10.0)
    assert nstlim == 5000000  # 10 ns / 0.002 ps
    assert nstlim // ntwx == 100 // 10 == 10


def test_dt_default_when_absent():
    # AMBER's own default, 0.001 ps
    assert long_window_md_steps(MDIN_NO_DT, 10.0)[0] == 10000000


# --- never shorten ---------------------------------------------------------------------------------
def test_never_shortens_when_already_long_enough():
    assert long_window_md_steps(MDIN_ALREADY_LONG, 5.0) is None


def test_equal_length_is_a_noop():
    assert long_window_md_steps(MDIN_ALREADY_LONG, 10.0) is None


def test_disabled_when_target_is_none():
    assert long_window_md_steps(MDIN_PROD, None) is None


# --- edge cases ------------------------------------------------------------------------------------
def test_snaps_up_to_a_whole_number_of_frames():
    # 7.3 ns = 1,825,000 steps, not divisible by 10,000 frames -> ceil ntwx, run slightly longer
    nstlim, ntwx, _ = long_window_md_steps(MDIN_PROD, 7.3)
    assert ntwx == 183  # ceil(1825000 / 10000)
    assert nstlim == 1830000 >= 1825000
    assert nstlim // ntwx == 10000  # the frame count is still exact


def test_no_ntwx_lengthens_without_inventing_a_write_interval():
    assert long_window_md_steps(MDIN_NO_NTWX, 10.0) == (2500000, None, None)


def test_returns_none_when_nstlim_absent():
    # make_mdin_file's regex needs an integer nstlim already present; nothing to rewrite
    assert long_window_md_steps(MDIN_NO_NSTLIM, 10.0) is None
