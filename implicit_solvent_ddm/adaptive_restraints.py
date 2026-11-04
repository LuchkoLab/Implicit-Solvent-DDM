"""
MBAR and exponential-averaging analysis of the post-processed intermediate states.

The window-insertion logic that used to live here (the pre-production ALS pilot) was replaced by
post-production ALS in ``als_refine.py``.
"""

import time
from typing import Optional

import pandas as pd

import implicit_solvent_ddm.pandasmbar as pdmbar
from implicit_solvent_ddm import block_mbar
from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.matrix_order import CycleSteps, cycle_steps_from_config

AVAGADRO = 6.0221367e23
BOLTZMAN = 1.380658e-23
JOULES_PER_KCAL = 4184

_INDEX = [
    "solute",
    "parm_state",
    "extdiel",
    "charge",
    "parm_restraints",
    "traj_state",
    "traj_extdiel",
    "traj_charge",
    "traj_restraints",
    "Frames",
]


def kcals_per_kt(temperature: float) -> float:
    """kcal/mol per kT at ``temperature``."""
    return ((BOLTZMAN * AVAGADRO) / JOULES_PER_KCAL) * float(temperature)


def assemble_df_mbar(simulation_data: list, temperature: float) -> pd.DataFrame:
    """Reduced-energy matrix from the post-analysis rows of one leg.

    Parameters
    ----------
    simulation_data : list of pandas.DataFrame
        One dataframe per (trajectory, Hamiltonian) cell, as ``create_mdout_dataframe`` builds them.
    temperature : float
        Temperature (K) the energies are reduced at.

    Returns
    -------
    pandas.DataFrame
        Rows indexed by the sampled state ``(traj_state, traj_extdiel, traj_charge, traj_restraints)``,
        one column per evaluated state ``(state, extdiel, charge, restraints)``, values in kT. Column
        order is whatever the data gives; ``compute_mbar`` reorders it.
    """
    df = pd.concat(simulation_data, axis=0, ignore_index=True)
    df = df.set_index(_INDEX, drop=True)
    df = df[["ENERGY"]]
    df = df.unstack(["parm_state", "extdiel", "charge", "parm_restraints"])  # type: ignore
    df = df.reset_index(["Frames", "solute"], drop=True)
    df.columns = [tuple(column[1:]) for column in df.columns]  # type: ignore
    print(f"Created Unique MBAR dataframe {df.index.unique()}\n")
    return df / kcals_per_kt(temperature)


def compute_mbar(
    simulation_data: list[pd.DataFrame],
    temperature: float,
    matrix_order: Optional[CycleSteps],
    system: str,
    memory="2G",
    cores=1,
    disk="3G",
    block_size: Optional[int] = None,
    block_anchors: Optional[list] = None,
    log=print,
):
    """Execute MBAR analysis.

    Arrange and structure DataFrames to perform MBAR analysis.

    Parameters
    ----------
    simulation_data: list[pd.DataFrame]
        A completed mdout output that contains system information including timestep energies, and temperature.
    temperature: float
        Specified thermostat temperature used in MD simulations.
    matrix_order: CycleSteps
        Arranges the MBAR matrix in chronological order depending on the system.
    system: str
        Denoting the chronogical order of the matrix (i.e. complex, receptor or ligand).
    block_size: int, optional
        When set, solve the full cycle as a chain of ``block_size``-state MBAR blocks rather
        than one dense solve (see block_mbar.py). Required when post-analysis was run banded,
        because ``df_mbar`` is then sparse.
    block_anchors: list of tuple, optional
        Shared endpoint states of the blocks, fixed when ALS first inserts a window (see
        ``block_mbar.leg_blocks``). None cuts the blocks by position.

    Returns
    -------
    pdmbar.mbar(df_subsampled): tuple[DataFrame, DataFrame, MBAR]
        DataFrames for the free energies differences (Deltaf_ij), error estimates in free energy difference (dDeltaf_ij), and the pyMBAR object.
    df_mbar: pd.DataFrame
        An formated and chronological arrange DataFrame before any MBAR analysis was performed. (Which can be used to create pdfs of MBAR matrix).
    """
    df_mbar = assemble_df_mbar(simulation_data, temperature)

    print(f"df.index.unique :\n {df_mbar.index.unique()}\n")
    print(f"df.columns.unique :\n {df_mbar.columns.unique()}\n")

    def _ordered(order):
        # Every CycleSteps state must exist as an MBAR-dataframe column. A mismatch means the
        # schedule and the scored trajectories have drifted apart -- fail loud with the diff instead
        # of the bare pandas KeyError the reindex would otherwise raise.
        missing = [column for column in order if column not in df_mbar.columns]
        if missing:
            raise ValueError(
                f"compute_mbar[{system}]: {len(missing)} CycleSteps state(s) are absent from the "
                f"MBAR dataframe columns — schedule/data mismatch.\n"
                f"  missing (CycleSteps order, not in df): {missing}\n"
                f"  present df columns: {list(df_mbar.columns)}"
            )
        return df_mbar[order]

    order = None
    if matrix_order is not None:
        order = {
            "complex": matrix_order.complex_order,
            "ligand": matrix_order.ligand_order,
        }.get(system, matrix_order.receptor_order)
        df_mbar = _ordered(order)

    equil_info = pdmbar.detect_equilibration(df_mbar)

    df_subsampled = pdmbar.subsample_correlated_data(df_mbar, equil_info=equil_info)

    log("performing MBAR")
    # MBAR timing record (parsed by timing_report.py). Every Toil call site passes job.log so the
    # record reaches the LEADER log; a raw print() goes only to the worker's stdout.
    _mbar_t0 = time.perf_counter()
    if block_size and order is not None:
        # Banded post-analysis: df_mbar is legitimately sparse, so the cycle is solved as a chain of
        # dense per-block MBARs.
        _mbar_result = block_mbar.chained_mbar_result(
            df_subsampled,
            order,
            block_size,
            block_mbar.band_boundaries(order),
            log=log,
            anchors=block_anchors,
        )
    else:
        _mbar_result = pdmbar.mbar(df_subsampled)
    log(
        f"[TIMING] phase=mbar "
        f"wall_s={time.perf_counter() - _mbar_t0:.2f} cores=1 gpu=0 end={time.time():.0f} "
        f"| mbar system={system} block_size={block_size}"
    )
    return _mbar_result, df_mbar


def run_compute_mbar(
    job,
    system_runner,
    config: Config,
    system_type: str,
    memory="2G",
    cores=1,
    disk="3G",
    accelerators=None,
):
    """
    Run the compute_mbar function on one leg's post-analysis output.
    """
    job.log(f"Running MBAR for {system_type}")
    cycle_steps = cycle_steps_from_config(config)
    anchors = config.intermediate_args.block_anchor_states or {}
    return compute_mbar(
        simulation_data=system_runner.post_output,
        temperature=config.intermediate_args.temperature,
        matrix_order=cycle_steps,
        system=system_type,
        block_size=getattr(config.intermediate_args, "post_analysis_block_size", None),
        block_anchors=anchors.get(system_type),
        log=job.log,
    )


def run_exponential_averaging(
    job,
    system_runner,
    temperature: float,
):
    """Execute exponential averaging

    Parameters
    ----------
    system_runner: IntermidateRunner
        A runner class that handles system specific simulations.
    temperature: float
        Specified simulation temperature

    Returns:
    --------
    pdmbar.mbar(df_subsampled): tuple[DataFrame, DataFrame, MBAR]
        DataFrames for the free energies differences (Deltaf_ij), error estimates in free energy difference (dDeltaf_ij), and the pyMBAR object.
    df_mbar: pd.DataFrame
        An formated and chronological arrange DataFrame before any MBAR analysis was performed. (Which can be used to create pdfs of MBAR matrix).
    """
    job.fileStore.logToMaster(f"Running exponential averaging")
    job.fileStore.logToMaster(f"Running a total of {len(system_runner.post_output)} simulations")
    job.fileStore.logToMaster(f"Post output: {system_runner.post_output}")
    return compute_mbar(
        simulation_data=system_runner.post_output,
        temperature=temperature,
        matrix_order=None,
        system="free_flat_bottom",
        log=job.log,
    )
