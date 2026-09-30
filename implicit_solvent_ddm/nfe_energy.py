"""Analytic energy of AMBER's NFE ``&pmd`` flat-bottom umbrella.

Why this exists
---------------
The restraint band's MBAR matrix does not need ``sander`` at all. Within the band every window shares
the same unbiased potential ``U0(x)`` and differs only by the restraint term, and MBAR free energies
are invariant to a *per-sample* additive constant -- adding ``c_n`` to ``u_k(x_n)`` for every state
``k`` multiplies the numerator and denominator of every weight by ``exp(-c_n)``. So

    u_ij = beta * [ U0(x_i) + sum_g f(R_g(x_i); r3_j, k) ]

and the whole band follows from one per-frame RMSD series per trajectory per group, with ``U0``
coming from an ordinary NFE-free ``imin=5`` pass.

This is exact rather than approximate: ``MULTI_RMSD`` was measured (cb7 and MCL-1, 2026-08-30) to
equal cpptraj ``rms ... mass`` to 3e-5 Angstrom and to be invariant to a rigid transform of the
reference to 3e-8 Angstrom. It also keeps the design independent of whether ``imin=5`` evaluates NFE
biases at all, which the AMBER manual never states.
"""

import numpy as np


def flat_bottom_energy(rmsd, r1=-10.0, r2=0.0, r3=0.0, r4=1000.0, k1=10.0, k2=None):
    """Evaluate the NFE ``&pmd`` anchor potential.

    Transcribed from the AMBER 25 manual Sec. 26.4.4. With the conventional ``r1 = -10``, ``r2 = 0``
    and ``r4 = 1000`` only the flat interior and the upper harmonic wall are reachable, since an RMSD
    is non-negative -- so ``r3`` acts as a leash length in Angstrom, ``r3 = 0`` degenerates to a pure
    harmonic ``0.5*k*R**2``, and ``r3`` beyond the largest sampled RMSD makes the term identically
    zero. That last property is what lets the loosest window *be* the unrestrained endstate.

    Parameters
    ----------
    rmsd : array_like
        Collective-variable values, in Angstrom.
    r1, r2, r3, r4 : float, optional
        Anchor positions, ascending. ``r2``/``r3`` bound the flat interior.
    k1 : float, optional
        Lower-wall force constant, kcal/mol/Angstrom**2.
    k2 : float, optional
        Upper-wall force constant. Defaults to ``k1``, which is how the workflow writes
        ``anchor_strength``.

    Returns
    -------
    numpy.ndarray
        Restraint energy per input value, kcal/mol.
    """
    k2 = k1 if k2 is None else k2
    values = np.asarray(rmsd, dtype=float)
    return np.select(
        [
            values <= r1,
            values <= r2,
            values <= r3,
            values <= r4,
        ],
        [
            k1 * (r1 - r2) * values,
            0.5 * k1 * (values - r2) ** 2,
            np.zeros_like(values),
            0.5 * k2 * (values - r3) ** 2,
        ],
        default=k2 * (r4 - r3) * values,
    )


def band_restraint_energy(rmsd_series, tolerance, force_constant, r1=-10.0, r2=0.0, r4=1000.0,
                          scales=None):
    """Total restraint energy per frame for one window of the RMSD band.

    Sums the independent ``&colvar`` groups, which is what separate blocks in ``cv.in`` give --
    ``U = sum_g U_g`` -- as opposed to the multi-group ``MULTI_RMSD`` form, which would combine the
    groups as ``sqrt(sum_g M_g/M R_g**2)`` instead.

    Parameters
    ----------
    rmsd_series : dict of str to array_like
        Per-frame CV values keyed by group name, e.g. ``{"receptor": ..., "ligand": ...}``. Every
        entry must be the same length and in the same frame order.
    tolerance : float
        This window's ``r3``, in Angstrom, for the primary group.
    force_constant : float
        ``k`` shared by every group and every window.
    r1, r2, r4 : float, optional
        Remaining anchor positions, shared by every group.
    scales : dict of str to float, optional
        Per-group multiplier on ``tolerance``, so a ligand or binding-site group can release at its
        own rate. Missing groups default to 1.0.

    Returns
    -------
    numpy.ndarray
        Total restraint energy per frame, kcal/mol.

    Raises
    ------
    ValueError
        If the groups disagree on frame count, which would silently corrupt every window.
    """
    scales = scales or {}
    lengths = {name: np.asarray(values).size for name, values in rmsd_series.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"colvar groups disagree on frame count: {lengths}")
    total = None
    for name, values in rmsd_series.items():
        term = flat_bottom_energy(
            values, r1=r1, r2=r2, r3=tolerance * scales.get(name, 1.0), r4=r4, k1=force_constant
        )
        total = term if total is None else total + term
    if total is None:
        raise ValueError("rmsd_series is empty; nothing to score")
    return total
