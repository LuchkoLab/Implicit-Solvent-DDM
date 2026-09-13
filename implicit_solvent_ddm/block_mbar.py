"""Block-chained MBAR.

Solve a thermodynamic cycle as a chain of small dense MBAR sub-problems instead of one
dense N x N solve, so post-analysis only needs a band of the matrix. Block size 2 is the
adjacent-window BAR chain (3N-2 evaluations); larger blocks cost more evaluations and pool
more states per solve.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# pymbar's DEFAULT_SOLVER_PROTOCOL (hybr, adaptive) stalls on this data and returns
# non-converged free energies without raising. See research/STATUS.md and branch
# `solver_protocol`.
BLOCK_SOLVER_PROTOCOL = (
    dict(method="adaptive", options=dict(min_sc_iter=0, maxiter=1000)),
    dict(method="L-BFGS-B", options=dict(maxiter=1000)),
)

GRAD_TOL = 1e-4


# --------------------------------------------------------------------------------------------------
# State identity
# --------------------------------------------------------------------------------------------------
def _fmt(value) -> str:
    """Format one state-tuple field the way ``CycleSteps`` does.

    Parameters
    ----------
    value : str, int or float
        Field value, numeric or already formatted.

    Returns
    -------
    str
        ``f"{float(value)}"`` when numeric, otherwise ``str(value)``.
    """
    if isinstance(value, str):
        try:
            return f"{float(value)}"
        except ValueError:
            return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value)}"
    return str(value)


def state_key(label, extdiel, charge, conformational, orientational=None) -> tuple:
    """Build the canonical state tuple used by ``CycleSteps`` and the MBAR columns.

    Parameters
    ----------
    label : str
        State label, e.g. ``"lambda_window"``, ``"gb_dielectric"``, ``"endstate"``.
    extdiel : str or float
        External dielectric.
    charge : str or float
        Ligand charge scaling.
    conformational : str or float
        Conformational restraint force exponent.
    orientational : str or float, optional
        Orientational restraint force exponent. When given, the restraint field is
        ``"con_orient"`` (halo/complex); when ``None`` it is ``"con"`` alone (apo).

    Returns
    -------
    tuple of str
        ``(state, extdiel, charge, restraints)``.
    """
    restraints = (
        _fmt(conformational)
        if orientational is None
        else f"{_fmt(conformational)}_{_fmt(orientational)}"
    )
    return (str(label), _fmt(extdiel), _fmt(charge), restraints)


def state_key_from_dirargs(directory_args: dict) -> tuple:
    """Return the state tuple for a ``Simulation`` from its ``directory_args``.

    Parameters
    ----------
    directory_args : dict
        Simulation directory arguments, carrying ``state_label``, ``extdiel``, ``charge``,
        ``conformational_restraint`` and optionally ``orientational_restraints``.

    Returns
    -------
    tuple of str
        Canonical state tuple.
    """
    return state_key(
        directory_args["state_label"],
        directory_args["extdiel"],
        directory_args["charge"],
        directory_args["conformational_restraint"],
        directory_args.get("orientational_restraints"),
    )


def canonical_state_key(directory_args: dict, known_states) -> Optional[tuple]:
    """Return the cycle-order spelling of a state, or None if it is genuinely absent.

    ``state_key_from_dirargs`` appends the orientational force whenever
    ``orientational_restraints`` is present. Apo legs carry that key even though they have
    no orientational restraint: ``setup_apply_restraint_windows`` builds its args with
    ``copy(self.no_gb_args)``, which always sets it, and the ``exponent_orientational is
    None`` branch never removes it (``apo_endstate_dirstruct`` hardcodes it too). But
    ``CycleSteps.apply_restraints`` / ``ligand_charges`` / ``no_gb`` spell apo states with
    the conformational force ALONE, and so does the MBAR dataframe. So an apo trajectory
    keys as ``('lambda_window','78.5','1.0','-2.0_8.0')`` while its cycle-order twin is
    ``('lambda_window','78.5','1.0','-2.0')``.

    Prefer the exact key; fall back to the conformational-only spelling. The complex leg is
    genuinely halo (``remove_restraints`` / ``complex_charges`` are compound) and matches on
    the first try, so it never reaches the fallback.

    Parameters
    ----------
    directory_args : dict
        Simulation directory arguments.
    known_states : container of tuple
        The cycle order's state tuples, as returned by ``_post_analysis_pairs``.

    Returns
    -------
    tuple of str or None
        The spelling present in ``known_states``, or None when neither form is.
    """
    key = state_key_from_dirargs(directory_args)
    if key in known_states:
        return key
    label, extdiel, charge, restraints = key
    if "_" in restraints:
        short = (label, extdiel, charge, restraints.split("_", 1)[0])
        if short in known_states:
            return short
    return None


def parm_key(run_args: dict) -> tuple:
    """Return the state tuple a post-analysis mdout was evaluated at.

    Parameters
    ----------
    run_args : dict
        Parsed post-analysis directory arguments.

    Returns
    -------
    tuple of str
        Canonical state tuple for the Hamiltonian side.
    """
    return state_key(
        run_args["state_label"],
        run_args["extdiel"],
        run_args["charge"],
        run_args["conformational_restraint"],
        run_args.get("orientational_restraints"),
    )


def traj_key(run_args: dict) -> tuple:
    """Return the state tuple a post-analysis mdout's trajectory was sampled at.

    Parameters
    ----------
    run_args : dict
        Parsed post-analysis directory arguments.

    Returns
    -------
    tuple of str
        Canonical state tuple for the trajectory side.
    """
    return state_key(
        run_args["traj_state_label"],
        run_args["traj_extdiel"],
        run_args["traj_charge"],
        run_args["trajectory_restraint_conrest"],
        run_args.get("trajectory_restraint_orenrest"),
    )


# --------------------------------------------------------------------------------------------------
# Partitioning
# --------------------------------------------------------------------------------------------------
def band_boundaries(order: Sequence[tuple]) -> list[int]:
    """Find transitions that cross a sub-leg boundary.

    Parameters
    ----------
    order : sequence of tuple
        Cycle-ordered state tuples.

    Returns
    -------
    list of int
        Transition indices ``t`` where ``order[t]`` and ``order[t+1]`` have different state
        labels.
    """
    return [t for t in range(len(order) - 1) if order[t][0] != order[t + 1][0]]


def partition_blocks(
    n_states: int, block_size: int, boundaries: Iterable[int] = ()
) -> list[list[int]]:
    """Partition a cycle into consecutive blocks that share endpoints.

    Parameters
    ----------
    n_states : int
        Number of states in the cycle.
    block_size : int
        Maximum states per block; must be >= 2. A value of 2 gives the BAR chain.
    boundaries : iterable of int, optional
        Transition indices to isolate into their own 2-state block.

    Returns
    -------
    list of list of int
        Blocks of state indices. Consecutive blocks overlap in exactly one state, and the
        last block of a run may be smaller than ``block_size``.

    Raises
    ------
    ValueError
        If ``block_size < 2`` or ``n_states < 2``.
    """
    if block_size < 2:
        raise ValueError(f"block_size must be >= 2 (2 == BAR chain), got {block_size}")
    if n_states < 2:
        raise ValueError(f"need at least 2 states to chain, got {n_states}")

    isolated = {t for t in boundaries if 0 <= t < n_states - 1}
    blocks: list[list[int]] = []
    run: list[int] = []

    def flush(transitions: list[int]) -> None:
        for i in range(0, len(transitions), block_size - 1):
            chunk = transitions[i : i + block_size - 1]
            blocks.append(list(range(chunk[0], chunk[-1] + 2)))

    for t in range(n_states - 1):
        if t in isolated:
            flush(run)
            run = []
            blocks.append([t, t + 1])
        else:
            run.append(t)
    flush(run)
    return blocks


def required_pairs(
    order: Sequence[tuple], block_size: int, boundaries: Iterable[int] = ()
) -> set[tuple]:
    """List the (trajectory, Hamiltonian) pairs a block chain needs evaluated.

    Parameters
    ----------
    order : sequence of tuple
        Cycle-ordered state tuples.
    block_size : int
        Maximum states per block.
    boundaries : iterable of int, optional
        Transition indices to isolate.

    Returns
    -------
    set of tuple
        ``(trajectory_state, hamiltonian_state)`` pairs. The size is the number of
        ``sander imin=5`` jobs the leg requires.
    """
    pairs: set[tuple] = set()
    for block in partition_blocks(len(order), block_size, boundaries):
        for i in block:
            for j in block:
                pairs.add((order[i], order[j]))
    return pairs


def evaluation_count(n_states: int, block_size: int, boundaries: Iterable[int] = ()) -> int:
    """Count the evaluations a block chain needs, without state labels.

    Parameters
    ----------
    n_states : int
        Number of states in the cycle.
    block_size : int
        Maximum states per block.
    boundaries : iterable of int, optional
        Transition indices to isolate.

    Returns
    -------
    int
        Number of distinct (trajectory, Hamiltonian) cells.
    """
    cells: set[tuple[int, int]] = set()
    for block in partition_blocks(n_states, block_size, boundaries):
        for i in block:
            for j in block:
                cells.add((i, j))
    return len(cells)


# --------------------------------------------------------------------------------------------------
# Solving
# --------------------------------------------------------------------------------------------------
def _N_k(df: pd.DataFrame) -> np.ndarray:
    """Count samples per state, aligned to the dataframe columns.

    Parameters
    ----------
    df : pandas.DataFrame
        MBAR matrix, states as columns, sampled state as index.

    Returns
    -------
    numpy.ndarray
        Sample counts in column order, zero for states with no samples.
    """
    counts = df.groupby(df.index.names).count().iloc[:, [0]]
    counts.index = counts.index.tolist()
    counts = pd.merge(
        pd.DataFrame(index=df.columns), counts, left_index=True, right_index=True, how="outer"
    )
    return counts.fillna(0).reindex(df.columns).values.flatten()


def solve_block(df: pd.DataFrame, protocol=BLOCK_SOLVER_PROTOCOL, initial_f_k=None):
    """Run a dense MBAR solve on one block.

    Parameters
    ----------
    df : pandas.DataFrame
        Dense block matrix in reduced units.
    protocol : tuple of dict, optional
        pymbar solver protocol.
    initial_f_k : numpy.ndarray, optional
        Warm-start free energies in kT.

    Returns
    -------
    Delta_f : numpy.ndarray
        Free energy differences in kT.
    dDelta_f : numpy.ndarray
        Statistical errors in kT.
    grad : float
        Maximum absolute gradient of the MBAR objective, or NaN if unavailable.
    """
    import pymbar
    from pymbar import mbar_solvers

    u_kn = df.values.T
    N_k = _N_k(df)
    mbar = pymbar.MBAR(u_kn, N_k, initial_f_k=initial_f_k, solver_protocol=protocol)
    results = mbar.compute_free_energy_differences()
    try:
        grad = float(np.max(np.abs(mbar_solvers.mbar_gradient(u_kn, N_k, mbar.f_k))))
    except Exception:
        grad = float("nan")
    return results["Delta_f"], results["dDelta_f"], grad


def slice_block(df: pd.DataFrame, states: Sequence[tuple]) -> pd.DataFrame:
    """Extract one block's sub-matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Full MBAR matrix, which may contain NaN outside the blocks.
    states : sequence of tuple
        State tuples belonging to the block.

    Returns
    -------
    pandas.DataFrame
        The block's columns and the rows sampled at those states.
    """
    # Rows must be filtered as well as columns, or pymbar's sum(N_k) == n_samples fails.
    block = df[list(states)]
    return block.loc[block.index.isin(list(states))]


def chained_mbar(
    df: pd.DataFrame,
    order: Sequence[tuple],
    block_size: int,
    boundaries: Iterable[int] = (),
    protocol=BLOCK_SOLVER_PROTOCOL,
    log=print,
):
    """Solve a cycle as a chain of per-block MBARs and sum the result.

    Parameters
    ----------
    df : pandas.DataFrame
        Subsampled, cycle-ordered matrix in reduced units. May contain NaN outside the
        blocks; only the dense per-block slices reach pymbar.
    order : sequence of tuple
        Cycle-ordered state tuples.
    block_size : int
        Maximum states per block.
    boundaries : iterable of int, optional
        Transition indices to isolate.
    protocol : tuple of dict, optional
        pymbar solver protocol.
    log : callable, optional
        Logger for blocks that fail to solve cleanly.

    Returns
    -------
    dG : float
        Total free energy difference in kT, summed over blocks.
    err : float
        Errors combined in quadrature. Adjacent blocks share an endpoint state, so this is
        mildly understated.
    per_block : pandas.DataFrame
        One row per block, with its states, dG, error and gradient norm.

    Raises
    ------
    ValueError
        If a block contains unevaluated cells, meaning the data does not match
        ``block_size``.
    """
    blocks = partition_blocks(len(order), block_size, boundaries)
    rows = []
    for b, block in enumerate(blocks):
        states = [order[i] for i in block]
        sub = slice_block(df, states)
        if sub.isna().any().any():
            # Fail rather than dropna(), which would silently solve on fewer samples.
            raise ValueError(
                f"block_mbar: block {b} ({states[0]} -> {states[-1]}) has unevaluated cells; "
                f"post-analysis data does not match block_size={block_size}"
            )
        fe, err, grad = solve_block(sub, protocol)
        dG_b, err_b = float(fe[0, -1]), float(err[0, -1])
        if grad > GRAD_TOL or not math.isfinite(dG_b):
            log(
                f"[block_mbar] WARNING block {b} states {states[0]} -> {states[-1]}: "
                f"dG={dG_b} grad={grad:.3e} -- block did not solve cleanly"
            )
        rows.append(
            {
                "block": b,
                "i_first": block[0],
                "i_last": block[-1],
                "n_states": len(block),
                "state_first": "|".join(map(str, states[0])),
                "state_last": "|".join(map(str, states[-1])),
                "dG_block": dG_b,
                "err_block": err_b,
                "grad": grad,
            }
        )
    per_block = pd.DataFrame(rows)
    dG = float(per_block["dG_block"].sum())
    err = float(np.sqrt((per_block["err_block"] ** 2).sum()))
    return dG, err, per_block


class ChainedMBAR:
    """Stand-in for a pymbar MBAR object built from a chain of block solves.

    Exposes the subset of the MBAR interface the workflow consumes, so a banded run can flow
    through ``postTreatment`` unchanged.

    Parameters
    ----------
    n_states : int
        Number of states in the cycle.
    blocks : list of list of int
        Block state indices.
    overlaps : list of numpy.ndarray
        Per-block overlap matrices, in block order.
    f_k : numpy.ndarray
        Global free energies in kT, one per state.
    """

    def __init__(self, n_states, blocks, overlaps, f_k):
        self.n_states = n_states
        self.blocks = blocks
        self.f_k = f_k
        matrix = np.zeros((n_states, n_states))
        for block, overlap in zip(blocks, overlaps):
            for a, i in enumerate(block):
                for b, j in enumerate(block):
                    matrix[i, j] = overlap[a, b]
        self._overlap = matrix

    def compute_overlap(self) -> dict:
        """Return the block-diagonal overlap matrix.

        Returns
        -------
        dict
            ``{"matrix": ndarray}``. Cells outside any block are zero because they were
            never evaluated, not because the states fail to overlap.
        """
        return {"matrix": self._overlap}


def chained_mbar_result(
    df: pd.DataFrame,
    order: Sequence[tuple],
    block_size: int,
    boundaries: Iterable[int] = (),
    protocol=BLOCK_SOLVER_PROTOCOL,
    log=print,
):
    """Solve a block chain and return it in ``pandasmbar.mbar`` shape.

    Parameters
    ----------
    df : pandas.DataFrame
        Subsampled, cycle-ordered matrix in reduced units.
    order : sequence of tuple
        Cycle-ordered state tuples.
    block_size : int
        Maximum states per block.
    boundaries : iterable of int, optional
        Transition indices to isolate.
    protocol : tuple of dict, optional
        pymbar solver protocol.
    log : callable, optional
        Logger for blocks that fail to solve cleanly.

    Returns
    -------
    free_energies : pandas.DataFrame
        ``Delta_f`` for every state pair, built from the chained global free energies.
    errors : pandas.DataFrame
        ``dDelta_f``, exact along the chain and quadrature-combined across blocks.
    mbar : ChainedMBAR
        Result object exposing ``compute_overlap``.
    """
    blocks = partition_blocks(len(order), block_size, boundaries)
    n = len(order)
    f_k = np.zeros(n)
    var_k = np.zeros(n)
    overlaps = []
    offset = 0.0
    offset_var = 0.0

    for b, block in enumerate(blocks):
        states = [order[i] for i in block]
        sub = slice_block(df, states)
        if sub.isna().any().any():
            # Name the cells, not just the block endpoints: an unevaluated cell means the
            # scoring side and this solve disagree about which (trajectory, Hamiltonian)
            # pairs are required, and the specific row/column is what identifies the
            # disagreement (e.g. a state spelled two different ways -- see
            # canonical_state_key).
            rows, cols = np.where(sub.isna().values)
            missing = sorted(
                {(str(sub.index[i]), str(sub.columns[j])) for i, j in zip(rows, cols)}
            )
            shown = "; ".join(f"traj={r} scored under {c}" for r, c in missing[:5])
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            raise ValueError(
                f"block_mbar: block {b} ({states[0]} -> {states[-1]}) has "
                f"{len(missing)} unevaluated cells; post-analysis data does not match "
                f"block_size={block_size}. Missing: {shown}{more}"
            )
        import pymbar
        from pymbar import mbar_solvers

        u_kn, N_k = sub.values.T, _N_k(sub)
        mbar = pymbar.MBAR(u_kn, N_k, solver_protocol=protocol)
        results = mbar.compute_free_energy_differences()
        try:
            grad = float(np.max(np.abs(mbar_solvers.mbar_gradient(u_kn, N_k, mbar.f_k))))
        except Exception:
            grad = float("nan")
        if grad > GRAD_TOL:
            log(
                f"[block_mbar] WARNING block {b} states {states[0]} -> {states[-1]}: "
                f"grad={grad:.3e} -- block did not solve cleanly"
            )
        overlaps.append(np.asarray(mbar.compute_overlap()["matrix"]))
        for a, i in enumerate(block):
            f_k[i] = offset + float(results["Delta_f"][0, a])
            var_k[i] = offset_var + float(results["dDelta_f"][0, a]) ** 2
        offset = f_k[block[-1]]
        offset_var = var_k[block[-1]]

    delta_f = f_k[None, :] - f_k[:, None]
    d_delta_f = np.sqrt(np.abs(var_k[None, :] - var_k[:, None]))

    index = pd.MultiIndex.from_tuples(order, names=df.index.names)
    free_energies = pd.DataFrame(delta_f, columns=list(order), index=index)
    errors = pd.DataFrame(d_delta_f, columns=list(order), index=index)
    return free_energies, errors, ChainedMBAR(n, blocks, overlaps, f_k)


def sweep_block_sizes(
    df: pd.DataFrame,
    order: Sequence[tuple],
    sizes: Sequence[int],
    align_bands: bool = True,
    protocol=BLOCK_SOLVER_PROTOCOL,
    log=print,
) -> pd.DataFrame:
    """Compare full MBAR against block chains of several block sizes.

    Parameters
    ----------
    df : pandas.DataFrame
        Dense, subsampled, cycle-ordered matrix in reduced units.
    order : sequence of tuple
        Cycle-ordered state tuples.
    sizes : sequence of int
        Block sizes to evaluate.
    align_bands : bool, optional
        Isolate sub-leg transitions into their own blocks. Default True.
    protocol : tuple of dict, optional
        pymbar solver protocol.
    log : callable, optional
        Logger passed to ``chained_mbar``.

    Returns
    -------
    pandas.DataFrame
        One row for the dense reference (``block_size`` 0) and one per block size, with
        dG, error, evaluation count, speedup and delta against the reference.
    """
    n = len(order)
    bounds = band_boundaries(order) if align_bands else []

    dense = df[list(order)]
    dense = dense.loc[dense.index.isin(list(order))]
    fe, err, grad = solve_block(dense, protocol)
    ref_dG, ref_err = float(fe[0, -1]), float(err[0, -1])
    rows = [
        {
            "block_size": 0,
            "label": "full_MBAR",
            "n_states": n,
            "n_blocks": 1,
            "n_evaluations": n * n,
            "speedup": 1.0,
            "dG": ref_dG,
            "err": ref_err,
            "max_grad": grad,
            "delta_vs_full": 0.0,
        }
    ]
    for K in sizes:
        blocks = partition_blocks(n, K, bounds)
        n_eval = evaluation_count(n, K, bounds)
        dG, e, per_block = chained_mbar(df, order, K, bounds, protocol, log=log)
        rows.append(
            {
                "block_size": K,
                "label": "BAR_chain" if K == 2 else f"K{K}_chain",
                "n_states": n,
                "n_blocks": len(blocks),
                "n_evaluations": n_eval,
                "speedup": (n * n) / n_eval,
                "dG": dG,
                "err": e,
                "max_grad": float(per_block["grad"].max()),
                "delta_vs_full": dG - ref_dG,
            }
        )
    return pd.DataFrame(rows)
