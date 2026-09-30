#!/usr/bin/env python
"""BAR and one-sided EXP estimates of the complex LJ step from cached energies."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from pymbar import other_estimators, timeseries
from scipy.special import logsumexp

KT = 0.0019872041 * 298.0


def lj_work(run, system):
    """Reduced energy differences of the LJ step from the cached matrix.

    Parameters
    ----------
    run : str
        Run directory holding ``.cache/``.
    system : str
        Complex system name.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, float)
        ``w_F = b(U_on - U_off)`` on LJ-off frames, ``w_R = b(U_off - U_on)`` on LJ-on frames,
        and the cached MBAR dG(off -> on) in kcal/mol.
    """
    cache = os.path.join(run, ".cache", system)
    f = pd.read_hdf(os.path.join(cache, f"{system}_formatted.h5"))
    off_col = [c for c in f.columns if c[0] == "no_interactions"][0]
    on_col = [c for c in f.columns if c[0] == "interactions"][0]
    sampled = f.index.get_level_values(0)
    off, on = f[sampled == "no_interactions"], f[sampled == "interactions"]
    w_f = ((off[on_col] - off[off_col]) / KT).dropna().to_numpy()
    w_r = ((on[off_col] - on[on_col]) / KT).dropna().to_numpy()
    fe = pd.read_hdf(os.path.join(cache, f"{system}_fe.h5"))
    states = [tuple(c)[0] for c in fe.columns]
    mbar = fe.values[0][states.index("interactions")] - fe.values[0][states.index("no_interactions")]
    return w_f, w_r, float(mbar)


def decorrelate(w):
    """Keep statistically independent samples.

    Parameters
    ----------
    w : numpy.ndarray
        Time-ordered reduced work values.

    Returns
    -------
    numpy.ndarray
        Subsampled values.
    """
    finite = np.clip(w, None, 1e6)  # clashing frames reach 1e18; clip only for the correlation time
    return w[timeseries.subsample_correlated_data(finite)]


def exp_forward(w_f):
    """dG(off -> on) in kT from insertion: -ln <exp(-w_F)>."""
    return -(logsumexp(-w_f) - np.log(w_f.size))


def exp_reverse(w_r):
    """dG(off -> on) in kT from deletion: +ln <exp(-w_R)>."""
    return logsumexp(-w_r) - np.log(w_r.size)


def bar(w_f, w_r):
    """dG(off -> on) in kT from BAR."""
    return float(other_estimators.bar(w_f, w_r, compute_uncertainty=False)["Delta_f"])


def ess(w):
    """Effective number of frames carrying ``<exp(-w)>``.

    Parameters
    ----------
    w : numpy.ndarray
        Reduced work values.

    Returns
    -------
    float
        ``(sum x)^2 / sum x^2`` with ``x = exp(-w)``, computed in log space.
    """
    return float(np.exp(2 * logsumexp(-w) - logsumexp(-2 * w)))


def bootstrap(fn, samples, n, rng):
    """Bootstrap SD of an estimator, resampling each input independently.

    Parameters
    ----------
    fn : callable
        Estimator taking the sample arrays.
    samples : list of numpy.ndarray
        Inputs to resample.
    n : int
        Bootstrap draws.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    float
        SD in kT.
    """
    draws = [fn(*[s[rng.integers(0, s.size, s.size)] for s in samples]) for _ in range(n)]
    return float(np.std(draws))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run directories holding .cache/, one per replicate")
    parser.add_argument("--system", required=True)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1.0, help="agreement required, kcal/mol")
    parser.add_argument("-o", "--out", help="CSV output path")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    rows = []
    for run in args.runs:
        label = os.path.basename(os.path.normpath(run))
        try:
            w_f_all, w_r_all, mbar = lj_work(run, args.system)
        except (FileNotFoundError, IndexError) as err:
            print(f"{label}: skipped ({err})", file=sys.stderr)
            continue
        w_f, w_r = decorrelate(w_f_all), decorrelate(w_r_all)
        half_f, half_r = w_f_all.size // 2, w_r_all.size // 2
        rows.append(dict(
            run=label,
            MBAR_cached=mbar,
            BAR=bar(w_f, w_r) * KT, BAR_sd=bootstrap(bar, [w_f, w_r], args.bootstrap, rng) * KT,
            EXP_fwd=exp_forward(w_f) * KT, EXP_fwd_sd=bootstrap(exp_forward, [w_f], args.bootstrap, rng) * KT,
            EXP_rev=exp_reverse(w_r) * KT, EXP_rev_sd=bootstrap(exp_reverse, [w_r], args.bootstrap, rng) * KT,
            BAR_1st_half=bar(decorrelate(w_f_all[:half_f]), decorrelate(w_r_all[:half_r])) * KT,
            BAR_2nd_half=bar(decorrelate(w_f_all[half_f:]), decorrelate(w_r_all[half_r:])) * KT,
            n_off=w_f.size, n_on=w_r.size, ESS_fwd=ess(w_f), ESS_rev=ess(w_r),
            frac_off_like_on=float(np.mean(w_f_all < np.percentile(-w_r_all, 95))),
        ))

    if not rows:
        sys.exit("no runs could be read")
    table = pd.DataFrame(rows).set_index("run")
    pd.set_option("display.width", 220)
    print(f"{args.system}: dG(LJ off -> on), kcal/mol (SD = bootstrap on decorrelated samples)")
    print(table[["MBAR_cached", "BAR", "BAR_sd", "EXP_fwd", "EXP_fwd_sd", "EXP_rev", "EXP_rev_sd",
                 "BAR_1st_half", "BAR_2nd_half"]].round(2).to_string())
    print("\nsamples behind each estimate (decorrelated n; ESS = frames effectively carrying the exponential average)")
    print(table[["n_off", "n_on", "ESS_fwd", "ESS_rev", "frac_off_like_on"]].round(3).to_string())

    for label, r in table.iterrows():
        estimates = [r.MBAR_cached, r.BAR, r.EXP_fwd, r.EXP_rev]
        spread = max(estimates) - min(estimates)
        halves = abs(r.BAR_1st_half - r.BAR_2nd_half)
        ok = spread <= args.tolerance and halves <= args.tolerance and min(r.ESS_fwd, r.ESS_rev) > 3
        print(f"{label}: estimators span {spread:.2f}, halves differ {halves:.2f} kcal/mol, "
              f"min ESS {min(r.ESS_fwd, r.ESS_rev):.1f} -> "
              + ("consistent: the thin overlap is not biasing this step" if ok
                 else "NOT consistent: the step needs intermediate LJ windows"))
    if args.out:
        table.to_csv(args.out)


if __name__ == "__main__":
    main()
