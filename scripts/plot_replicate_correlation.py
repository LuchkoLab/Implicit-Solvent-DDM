#!/usr/bin/env python
"""Predicted vs experimental binding free energy from replicate runs, with DBFE-paper metrics."""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
POINT = "#2a78d6"
BAND1, BAND2 = "#d9d8d3", "#ecebe7"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="replicate output directories (each holding .cache/); DIR=N files DIR as replicate N")
    p.add_argument("--expt", required=True, help="CSV with Protein, Ligand, dG_expt (e.g. dbfe/data/alibay_abfe.csv)")
    p.add_argument("--protein", default="MCL-1", help="Protein value to select in --expt")
    p.add_argument("--bounds", nargs="*", type=int, default=[],
                   help="ligands whose experimental value is a detection limit: drawn hollow, left out of the metrics")
    p.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for 95%% CIs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--title", default=None)
    p.add_argument("-o", "--out", required=True, help="figure path; a per-ligand CSV is written next to it")
    return p.parse_args()


def load_replicates(runs):
    """Read every ``deltaG_*.h5`` under each run's ``.cache``.

    Parameters
    ----------
    runs : list of str
        Replicate output directories, optionally ``DIR=N`` to set the replicate number.

    Returns
    -------
    pandas.DataFrame
        One row per (ligand, replicate) with column ``dG``.
    """
    rows = []
    for i, run in enumerate(runs, start=1):
        run, _, rep = run.partition("=")
        rep = int(rep) if rep else i
        for f in sorted(glob.glob(os.path.join(run, ".cache", "*", "deltaG_*.h5"))):
            m = re.search(r"ligand-(\d+)", os.path.basename(f))
            if m is None:
                continue
            df = pd.read_hdf(f).iloc[0]
            # sum the legs: runs from before the flat-bottom fix carry it in deltaG
            dG = df.drop(["deltaG", "flat_bottom_contribution"], errors="ignore").sum()
            rows.append(dict(ligand=int(m.group(1)), rep=rep, dG=float(dG)))
    return pd.DataFrame(rows)


def metrics(pred, expt):
    """RMSE, MUE, Pearson r, Spearman rho, Kendall tau and R^2."""
    err = pred - expt
    r = stats.pearsonr(pred, expt)[0]
    return dict(
        RMSE=np.sqrt(np.mean(err**2)),
        MUE=np.mean(np.abs(err)),
        r=r,
        R2=r**2,
        rho=stats.spearmanr(pred, expt)[0],
        tau=stats.kendalltau(pred, expt)[0],
    )


def bootstrap_ci(pred, expt, n_boot, seed):
    """Percentile 95% CIs of every metric over resampled ligands."""
    rng = np.random.default_rng(seed)
    n = len(pred)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.ptp(pred[idx]) == 0 or np.ptp(expt[idx]) == 0:
            continue
        samples.append(metrics(pred[idx], expt[idx]))
    df = pd.DataFrame(samples)
    return {k: (df[k].quantile(0.025), df[k].quantile(0.975)) for k in df}


def main():
    args = parse_args()
    reps = load_replicates(args.runs)
    if reps.empty:
        raise SystemExit("no deltaG_*.h5 files found")
    ex = pd.read_csv(args.expt)
    ex = ex[ex["Protein"] == args.protein]
    expt_map = dict(zip(ex["Ligand"].astype(int), ex["dG_expt"]))

    g = reps.groupby("ligand")["dG"]
    tab = pd.DataFrame({"n": g.count(), "mean": g.mean(), "SD": g.std(ddof=1)})
    tab["SEM"] = tab["SD"] / np.sqrt(tab["n"])
    tab = tab.join(reps.pivot(index="ligand", columns="rep", values="dG").add_prefix("rep"))
    tab["expt"] = tab.index.map(expt_map)
    tab = tab.dropna(subset=["expt"])

    pred, expt = tab["mean"].to_numpy(), tab["expt"].to_numpy()
    measured = ~tab.index.isin(args.bounds)
    m = metrics(pred[measured], expt[measured])
    ci = bootstrap_ci(pred[measured], expt[measured], args.n_boot, args.seed)
    multi = tab["n"] > 1
    pooled_sd = np.sqrt(((tab.loc[multi, "n"] - 1) * tab.loc[multi, "SD"] ** 2).sum() / (tab.loc[multi, "n"] - 1).sum())

    fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=200)
    sem = tab["SEM"].fillna(0).to_numpy()
    lo = np.floor(min((pred - sem).min(), expt.min()) - 0.5)
    hi = np.ceil(max((pred + sem).max(), expt.max()) + 0.5)
    x = np.array([lo, hi])
    ax.fill_between(x, x - 2, x + 2, color=BAND2, lw=0, label="±2 kcal/mol", zorder=0)
    ax.fill_between(x, x - 1, x + 1, color=BAND1, lw=0, label="±1 kcal/mol", zorder=1)
    ax.plot(x, x, color=MUTED, lw=1, zorder=2)
    ax.errorbar(expt[measured], pred[measured], yerr=sem[measured], fmt="o", ms=6, color=POINT, mec="white", mew=1,
                ecolor=POINT, elinewidth=1.2, capsize=2.5, zorder=3, label="mean ± SEM (replicates)")
    if not measured.all():
        ax.errorbar(expt[~measured], pred[~measured], yerr=sem[~measured], fmt="o", ms=6, mfc="white", mec=POINT,
                    mew=1.2, ecolor=POINT, elinewidth=1.2, capsize=2.5, zorder=3, label="expt is a detection limit")
    labels = [ax.text(xe, yp, str(lig), fontsize=7, color=INK2, zorder=5) for lig, xe, yp in zip(tab.index, expt, pred)]
    try:
        from adjustText import adjust_text
        adjust_text(labels, x=expt, y=pred, ax=ax, expand=(1.3, 1.5),
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    except ImportError:
        for t in labels:
            t.set_position((t.get_position()[0] + 0.12, t.get_position()[1] + 0.12))

    def fmt(k, label, d=2):
        a, b = ci[k]
        return f"{label} = {m[k]:.{d}f} [{a:.{d}f}, {b:.{d}f}]"

    text = "\n".join([
        fmt("RMSE", "RMSE"), fmt("MUE", "MUE"), fmt("r", "Pearson r"), fmt("R2", "R²"),
        fmt("rho", "Spearman ρ"), fmt("tau", "Kendall τ"),
        f"pooled replicate SD = {pooled_sd:.2f}",
    ])
    counts = (f"N = {int(measured.sum())} ligands" + (f" (+{int((~measured).sum())} bounds)" if not measured.all() else "")
              + f", {int(tab['n'].sum())} runs")
    ax.text(0.97, 0.03, text, transform=ax.transAxes, va="bottom", ha="right", fontsize=7.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=BAND1, lw=0.8), zorder=4)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True, edgecolor=BAND1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Experimental ΔG (kcal/mol)", color=INK)
    ax.set_ylabel("Predicted ΔG (kcal/mol)", color=INK)
    ax.set_title(args.title or f"{args.protein}: {len(args.runs)} replicates", fontsize=10, color=INK)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_color(MUTED)
    fig.text(0.5, 0.005, f"Brackets: 95% percentile bootstrap CI over ligands ({args.n_boot} resamples). Energies in kcal/mol.",
             ha="center", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(args.out)

    csv = os.path.splitext(args.out)[0] + ".csv"
    tab.round(3).to_csv(csv)
    print(tab.round(2).to_string())
    print(text)
    print(counts)
    print(f"[done] {args.out}\n[done] {csv}")


if __name__ == "__main__":
    main()
