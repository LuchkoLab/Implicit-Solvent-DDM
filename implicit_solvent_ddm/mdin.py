"""
Simple functions to create all the required MD input files (i.e. 'mdin').
"""

import itertools
import os
import random
from dataclasses import dataclass
from string import Template
from tkinter import Y
import re
import yaml
from toil.common import FileID


def generate_extdiel_mdin(
    job,
    user_mdin_ID: FileID,
    gb_extdiel: float,
    nstlim=None,
    ntwx=None,
    saltcon=0.0,
) -> FileID:
    """Write an mdin with a unique external dielectric for the generalized-Born solvent.

    ``nstlim``/``ntwx`` shorten the run for the ALS pilot (no-ops when ``None`` -> production length).
    ``saltcon`` defaults to 0 across the GB-dielectric band: with salt the Debye screening makes the GB
    polar term no longer exactly linear in ``lambda = 1 - 1/eps``, which is the property the band relies
    on for uniform overlap. Pass ``saltcon=None`` to leave the user's value untouched.
    """
    mdin_global = job.fileStore.readGlobalFile(user_mdin_ID)

    return job.fileStore.writeGlobalFile(
        make_mdin_file(
            mdin_global,
            "gb_extdiel_mdin",
            gb_extdiel=gb_extdiel,
            nstlim=nstlim,
            ntwx=ntwx,
            saltcon=saltcon,
        )
    )


def get_mdins(job, user_mdin_ID: FileID):
    """Writes all mdins for intermidate states

    Parameters
    ----------
    user_mdin_args: str
        A user specified yaml file containing mdin arguments

    Returns
    -------
    default_mdin : FileID
        Solvated MD.
    no_solvent_mdin : FileID
        Gas-phase MD (igb=6).
    post_mdin : FileID
        Single-point scoring (imin=5), user's saltcon.
    post_nosolv : FileID
        Single-point scoring for the gas-phase states.
    post_saltfree : FileID
        Single-point scoring at saltcon=0, for the GB-dielectric band.
    """

    mdin_global = job.fileStore.readGlobalFile(user_mdin_ID)

    default_mdin = job.fileStore.writeGlobalFile(make_mdin_file(mdin_global, "_mdin"))
    no_solvent_mdin = job.fileStore.writeGlobalFile(
        make_mdin_file(mdin_global, "no_solv_mdin", turn_off_solvent=True)
    )
    post_mdin = job.fileStore.writeGlobalFile(
        make_mdin_file(mdin_global, "post_mdin", post_process=True)
    )
    post_nosolv = job.fileStore.writeGlobalFile(
        make_mdin_file(
            mdin_global, "post_nosolv_mdin", turn_off_solvent=True, post_process=True
        )
    )
    # GB-dielectric band: its MD is written salt-free by generate_extdiel_mdin, so score it the same
    # way. With salt the prefactor (1/intdiel - exp(-kappa*f)/extdiel) does not vanish at extdiel=1,
    # so the band never reaches vacuum and leaves a ~350 kcal/mol cliff against the igb=6 gas anchor.
    post_saltfree = job.fileStore.writeGlobalFile(
        make_mdin_file(mdin_global, "post_saltfree_mdin", post_process=True, saltcon=0.0)
    )

    return (default_mdin, no_solvent_mdin, post_mdin, post_nosolv, post_saltfree)


def pilot_md_steps(mdin_text, pilot_ps, pilot_frames, pilot_nstlim=None, dt_default=0.001):
    """Return ``(nstlim, ntwx)`` for the ALS pilot from the user mdin's timestep (pure, testable).

    The pilot is **always 50 ps** (the paper's value, ``pilot_ps``) regardless of the user's timestep:
    ``nstlim = round(pilot_ps / dt)`` where ``dt`` is read from the user mdin (e.g. dt=0.002 ps ->
    25000 steps for 50 ps). ``pilot_nstlim``, if given, overrides this (for tiny test systems where
    50 ps is absurd). ``ntwx`` is chosen so ~``pilot_frames`` trajectory frames are written over the
    window, making the MBAR sample count independent of ``dt`` and of the user's production ``ntwx``
    (which is tuned for a much longer run and would otherwise leave a 50 ps pilot far too thin).

    ``dt_default`` (AMBER's own default, 0.001 ps) is used only if no ``dt`` is found in the mdin.
    """
    match = re.search(r"\bdt\s*=\s*([0-9.eE+-]+)", mdin_text)
    dt = float(match.group(1)) if match else dt_default
    nstlim = int(pilot_nstlim) if pilot_nstlim is not None else max(1, round(pilot_ps / dt))
    ntwx = max(1, round(nstlim / max(1, int(pilot_frames))))
    return nstlim, ntwx


def long_window_md_steps(mdin_text, target_ns, dt_default=0.001):
    """Return ``(nstlim, ntwx, ntpr)`` running ONE window for ``target_ns`` ns at the user's frame
    count, or ``None`` when nothing should change (pure, testable).

    The lowest conformational-restraint window neighbours the endstate, whose supplied ensemble is
    ~10x longer and does not shrink when the ladder is shortened, so that link is the one that
    collapses (overlap 0.0157 at 2 ns vs 0.0590 at 10 ns). This stretches that window alone::

        target_steps = round(target_ns * 1000 / dt)   # dt is READ from the mdin, never rewritten
        frames       = user_nstlim // user_ntwx       # what the user's mdin already writes
        ntwx         = ceil(target_steps / frames)    # snapped UP, so the count is exact
        nstlim       = ntwx * frames                  # >= target_steps, EXACTLY `frames` frames

    On the production template (``dt=0.004, nstlim=500000, ntwx=50``) at 10 ns this gives
    ``(2500000, 250, 250)`` -- 10 ns, still 10,000 frames, same 4 fs timestep. Holding the frame
    count is required: post-analysis stores one rectangular frames x states matrix per leg
    (``postTreatment.py:296``). ``ntpr`` only keeps the mdout's record count in step; the MD mdout is
    never parsed for energies, so it cannot move a number.

    The long window's samples are spaced further apart in time than the rest of the ladder. MBAR is
    unaffected -- each column only needs samples from its own Hamiltonian -- but the
    statistical-inefficiency estimate for that window is not comparable to a uniform ladder's.

    Parameters
    ----------
    mdin_text: str
        Contents of the user's intermediate mdin.
    target_ns: float or None
        Target MD length in nanoseconds. ``None`` disables (returns ``None``).
    dt_default: float
        Timestep in ps assumed when the mdin has no ``dt`` (AMBER's own default, 0.001).

    Returns
    -------
    tuple or None
        ``(nstlim, ntwx, ntpr)``, or ``None`` when the window should be left alone: no target, no
        integer ``nstlim`` for ``make_mdin_file`` to rewrite, or the user's run already reaches
        ``target_ns`` (this NEVER shortens a window). ``ntwx``/``ntpr`` come back ``None`` when the
        mdin writes no trajectory -- the length is extended without inventing a write interval.
    """
    if target_ns is None:
        return None

    dt_match = re.search(r"\bdt\s*=\s*([0-9.eE+-]+)", mdin_text)
    dt = float(dt_match.group(1)) if dt_match else dt_default

    nstlim_match = re.search(r"\bnstlim\s*=\s*(\d+)", mdin_text)
    if nstlim_match is None:
        return None
    user_nstlim = int(nstlim_match.group(1))

    target_steps = round(float(target_ns) * 1000.0 / dt)
    if target_steps <= user_nstlim:
        return None

    ntwx_match = re.search(r"\bntwx\s*=\s*(\d+)", mdin_text)
    user_ntwx = int(ntwx_match.group(1)) if ntwx_match else 0
    frames = user_nstlim // user_ntwx if user_ntwx > 0 else 0
    if frames < 1:
        return target_steps, None, None

    ntwx = -(-target_steps // frames)  # ceil, so nstlim lands on a whole number of frames
    nstlim = ntwx * frames
    if nstlim <= user_nstlim:
        return None
    return nstlim, ntwx, ntwx


def get_long_window_mdin(job, user_mdin_ID: FileID, target_ns: float = None):
    """Write the long-run mdin for the lowest restraint window. Mirrors ``get_pilot_mdin``.

    Returns ``(long_mdin, nstlim, ntwx)``, stored at ``config.inputs["long_window_mdin"]`` and
    consumed only by ``SimulationSetup._restraint_window_mdin``.

    The file is ALWAYS written: whether the stretch applies depends on ``dt``/``nstlim``/``ntwx``
    inside the user's mdin, readable only here, long after the DAG handed every ``Simulation`` its
    ``input_file`` promise. When ``long_window_md_steps`` declines, the overrides go in as ``None``
    and the result is byte-identical to ``default_mdin``, so the DAG needs no conditional edge.
    """
    mdin_global = job.fileStore.readGlobalFile(user_mdin_ID)
    with open(mdin_global) as fh:
        steps = long_window_md_steps(fh.read(), target_ns)
    nstlim, ntwx, ntpr = steps if steps is not None else (None, None, None)
    if steps is None:
        job.fileStore.logToMaster(
            f"[long-window] target {target_ns} ns needs no change to the user mdin (already that "
            "long, or no nstlim to rewrite); the lowest window runs the ladder length."
        )
    else:
        frames = nstlim // ntwx if ntwx else "n/a"
        job.fileStore.logToMaster(
            f"[long-window] target {target_ns} ns -> nstlim={nstlim} ntwx={ntwx} ntpr={ntpr} "
            f"({frames} frames, unchanged from the user mdin)"
        )
    long_mdin = job.fileStore.writeGlobalFile(
        make_mdin_file(mdin_global, "long_window_mdin", nstlim=nstlim, ntwx=ntwx, ntpr=ntpr)
    )
    return long_mdin, nstlim, ntwx


def get_pilot_mdin(
    job,
    user_mdin_ID: FileID,
    pilot_ps: float = 50.0,
    pilot_frames: int = 100,
    pilot_nstlim: int = None,
):
    """Write the short-pilot intermediate mdins for the ALS pilot (always 50 ps by default).

    Returns a ``(pilot_default, pilot_no_solvent)`` pair, both shortened to ``pilot_ps`` (50 ps,
    converted to steps via the user mdin's ``dt``) with ``ntwx`` set for ~``pilot_frames`` frames:

    * ``pilot_default``    — the solvated/default mdin (restraint windows, solvated charge states).
    * ``pilot_no_solvent`` — the ``igb=6`` gas-phase mdin (no_interactions / interactions / igb=6
      charge states). This MUST be shortened too: the pilot runs the FULL cycle, and those gas-phase
      states use ``no_solvent_mdin``; if only the default mdin is shortened they run at full production
      length (the bug this fixes). Stored at ``config.inputs["pilot_mdin"]`` /
      ``config.inputs["pilot_no_solvent_mdin"]``. Used ONLY when ``workflow.adaptive_lambda`` is set.

    NOTE: GB-external-dielectric pilot states are shortened separately by ``generate_extdiel_mdin`` (it
    takes the same ``nstlim``/``ntwx`` derived here via ``pilot_md_steps``); see
    ``workflow_phases.adaptive_restraint_pilot``.
    """
    mdin_global = job.fileStore.readGlobalFile(user_mdin_ID)
    with open(mdin_global) as fh:
        nstlim, ntwx = pilot_md_steps(fh.read(), pilot_ps, pilot_frames, pilot_nstlim)
    pilot_default = job.fileStore.writeGlobalFile(
        make_mdin_file(mdin_global, "pilot_mdin", nstlim=nstlim, ntwx=ntwx)
    )
    pilot_no_solvent = job.fileStore.writeGlobalFile(
        make_mdin_file(
            mdin_global, "pilot_no_solv_mdin", turn_off_solvent=True, nstlim=nstlim, ntwx=ntwx
        )
    )
    # nstlim/ntwx are returned too so the GB-dielectric band can shorten its per-epsilon mdins
    # (generate_extdiel_mdin) to the same pilot length when an ALS insertion adds a new dielectric window.
    return pilot_default, pilot_no_solvent, nstlim, ntwx


def make_mdin_file(
    user_mdin_file,
    mdin_name,
    gb_extdiel=78.5,
    turn_off_solvent=False,
    post_process=False,
    nstlim=None,
    ntwx=None,
    ntpr=None,
    saltcon=None,
    score_igb=None,
):
    """Rewrite users AMBER mdin file for specific thermodynamic states

    Parameters
    ----------
    user_mdin_file: FileID
        User provided mdin for thermodyamic states
    mdin_name: str
        A unique mdin filename
    turn_off_solvent: bool
        Set igb=6 if turn_off_solvent=True
    post_process: bool
        Set imin=5 and ntx=5 if post_process=True
    score_igb: int, optional
        If provided, override the GB model (``igb=<score_igb>``) in the mdin. No-op when ``None`` so
        the four production mdins stay byte-identical; used only by the standalone cross-GB-model
        re-scoring analysis (gb_ddG_bar.py) to score a trajectory under an explicit igb.
    nstlim: int, optional
        If provided, override the MD step count (``nstlim``) in the mdin. No-op when ``None`` so the
        production mdins stay byte-identical; set only for the short ALS pilot.
    ntwx: int, optional
        If provided, override the trajectory write interval. No-op when ``None``.
    ntpr: int, optional
        If provided, override the energy print interval. No-op when ``None``. Set alongside ``ntwx``
        by the long low-restraint window; cannot affect any energy, since the MD mdout is never
        parsed (only the ``imin=5`` scoring mdout is).
    Returns
    -------
    mdin: str
        Absolute path where the MD input file was created.
    """
    # with open(yaml_args) as fH:
    #     mdin_args = yaml.safe_load(fH)

    # general setting
    imin = "imin = 0"
    ioutfm = "ioutfm = 1"
    ntx = "ntx=1"
    irest = "irest=0"
    extdiel = f"extdiel={gb_extdiel}"
    # arguments for post-analysis
    if post_process:
        imin = "imin = 5"
        ioutfm = "ioutfm = 1"

    with open(user_mdin_file, "r") as output:
        data = output.readlines()

    new_mdin = ""
    for line in data:

        # Vacuum state
        if turn_off_solvent:
            line = re.sub(r"saltcon\s*=\s*\d+\.?\d+", "saltcon=0.0", line)
            line = re.sub(r"igb\s*=\s*\d+", "igb=6", line)
            line = re.sub(r"extdiel\s*=\s*\$extdiel", "extdiel=0.0", line)
        if "imin" in line:
            line = re.sub(r"imin\s*=\s*\d+", imin, line)
        if "ioutfm" in line:
            line = re.sub(r"ioutfm\s*=\s*\d+", ioutfm, line)
        if "irest" in line:
            line = re.sub(r"irest\s*=\s*\d+", irest, line)
        if "ntx" in line:
            line = re.sub(r"ntx\s*=\s*\d+", ntx, line)
        if not post_process:
            line = re.sub(r"extdiel\s*=\s*\$extdiel", extdiel, line)
        # ALS short-pilot: override MD length (nstlim) and trajectory write interval (ntwx). Both are
        # no-ops when None, so the four production intermediate mdins remain byte-identical (flag-off
        # regression). ntwx is set so the 50 ps pilot writes ~pilot_frames frames for MBAR.
        if nstlim is not None:
            line = re.sub(r"nstlim\s*=\s*\d+", f"nstlim = {nstlim}", line)
        if ntwx is not None:
            line = re.sub(r"ntwx\s*=\s*\d+", f"ntwx = {ntwx}", line)
        # Tracks ntwx on the long window, so a longer run does not leave a bigger mdout.
        if ntpr is not None:
            line = re.sub(r"ntpr\s*=\s*\d+", f"ntpr = {ntpr}", line)
        # GB-dielectric band: pin saltcon (default 0) so the GB polar term stays exactly linear in
        # lambda = 1 - 1/eps. No-op when None (production/other mdins keep the user's saltcon).
        if saltcon is not None:
            line = re.sub(r"saltcon\s*=\s*[0-9.]+", f"saltcon={saltcon}", line)
        # Cross-GB-model re-scoring: override igb with an explicit model (OBC=2, OBC2=5, GBn=7,
        # GBn2=8, ...). No-op when None so production mdins are unaffected. Mirrors the gas igb=6
        # swap above; applied last so an explicit score_igb wins.
        if score_igb is not None:
            line = re.sub(r"igb\s*=\s*\d+", f"igb={score_igb}", line)
        new_mdin += line

    with open(mdin_name, "w") as output:
        output.write(new_mdin)
    return os.path.abspath(mdin_name)


def generate_replica_mdin(
    job, mdin_input: FileID, temperatures: list, runtype="remd"
) -> list[FileID]:
    """Writes a series of equilibration/relaxtions and production/remd AMBER mdin files.

    Parameters:
    ----------
    job: toil.job
    mdin_input: FileID
        mdin template for REMD simulation
    temperatures: list[int]
        a list of temperatures for each replica mdin
    Returns:
        list[FileID]: _description_
    """
    tempdir = job.fileStore.getLocalTempDir()

    # read in template mdin
    read_replica_mdin = job.fileStore.readGlobalFile(
        mdin_input, userPath=os.path.join(tempdir, os.path.basename(mdin_input))
    )

    replica_mdin_IDs = []
    generated_seeds = []

    # read replica template in temporary directory
    with open(read_replica_mdin) as temp:
        template = Template(temp.read())

    for index, temperature in enumerate(temperatures, start=1):
        # get unique random seed
        ig = generate_random_seeds(generated_seeds)
        # append to exisiting random seeds
        generated_seeds.append(ig)

        # update the temperature and ig seed
        replica_mdin = template.substitute(
            temp=temperature,
            ig=ig,
            restraint="$restraint",
        )
        # write a replica mdin
        mdin_filename = f"mdin.{runtype}.{index:03}"
        with open(mdin_filename, "w") as mdin:
            mdin.write(replica_mdin)

        replica_mdin_IDs.append(
            job.fileStore.writeGlobalFile(os.path.abspath(mdin_filename))
        )
    job.fileStore.logToMaster(f"replica mdins {replica_mdin_IDs}")

    return replica_mdin_IDs


def generate_random_seeds(seeds: list):
    """Random seed generator
    Generates unique random integer to used used in replica exchange MDIN.

    Parameters:
    -----------
    list_seeds: list[int]
        A list of unique generated intger values
    Returns:
        new_seed: int
        A unique random generated integer value.
    """
    new_seed = random.randrange(0, 32767)

    while new_seed in seeds:
        new_seed = random.randrange(0, 32767)

    return new_seed


if __name__ == "__main__":
    replica_mdin = "/nas0/ayoub/Impicit-Solvent-DDM/new_replicas/mdin.temp"
    mdins = generate_replica_mdin(
        mdin_input=replica_mdin, temperatures=[269.5, 300.0, 334.0], runtype="equil"
    )
    print(mdins)
