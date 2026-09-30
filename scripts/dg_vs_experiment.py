#!/usr/bin/env python
"""Correlate computed binding free energies against experiment.

Reads the per-ligand ``deltaG_*.h5`` written by the workflow for one or more
replicate run directories, averages them, and plots the result against the
experimental values in ``data/alibay_abfe.csv`` alongside that paper's own ABFE
numbers for the same ligands.

Usage::

    python scripts/dg_vs_experiment.py --runs mcl1_7remd_2ns_rep1 mcl1_7remd_2ns_rep2 \
        --expt data/alibay_abfe.csv --protein MCL-1 -o mcl1_correlation.png
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"
OURS = "#2a78d6"
REFERENCE = "#eb6834"


def restraint_protocol(run_dir, protein):
    """Label a run by the flat-bottom receptor group its own restraint.RST carries.

    Directory names do not record which protocol a run used, so read it back from the
    file the run actually wrote: a pocket-anchored group is a few dozen atoms, the old
    whole-receptor group is every C-alpha.

    Parameters
    ----------
    run_dir : str
        Run directory.
    protein : str
        Protein name prefixing the ligand directories.

    Returns
    -------
    str
        ``"pocket"``, ``"whole-receptor"`` or ``"unknown"``.
    """
    out = {}
    for path in glob.glob(os.path.join(run_dir, "*", "remd", f"{protein}_ligand-*",
                                       "restraint.RST")):
        lig = re.search(rf"{re.escape(protein)}_ligand-(\d+)", path)
        if not lig:
            continue
        text = open(path).read()
        n = len(re.findall(r"IGR1\(\d+\)", text))
        if not n:
            found = re.search(r"igr1=([\d,]+)", text)
            n = len([v for v in found.group(1).split(",") if v]) if found else 0
        out[int(lig.group(1))] = "pocket" if 0 < n <= 60 else "whole-receptor"
    return out


def collect_deltaG(run_dirs, protein):
    """Per-ligand deltaG from every replicate run.

    Parameters
    ----------
    run_dirs : sequence of str
        Run directories, each holding a ``.cache/<system>/deltaG_<system>.h5``.
    protein : str
        Protein name prefixing the ligand directories.

    Returns
    -------
    pandas.DataFrame
        Indexed by ligand number, one column per replicate.
    """
    rows = []
    for rep, run in enumerate(run_dirs, start=1):
        for path in sorted(glob.glob(os.path.join(run, ".cache", "*", "deltaG_*.h5"))):
            found = re.search(rf"{re.escape(protein)}_ligand-(\d+)", os.path.basename(path))
            if not found:
                continue
            try:
                value = float(pd.read_hdf(path)["deltaG"].iloc[0])
            except (KeyError, IndexError, OSError):
                continue
            rows.append(dict(ligand=int(found.group(1)), rep=f"rep{rep}", dG=value))
    if not rows:
        raise SystemExit(f"no deltaG_*.h5 found for {protein} under {run_dirs}")
    return pd.DataFrame(rows).pivot(index="ligand", columns="rep", values="dG")


def merge_experiment(computed, expt_csv, protein):
    """Join replicate means onto the experimental table."""
    expt = pd.read_csv(expt_csv)
    expt = expt[expt["Protein"] == protein].copy()
    expt["ligand"] = expt["Ligand"].astype(int)          # the csv column is text
    keep = ["dG_expt", "dG_ABFE"] + (["bounds_ABFE"] if "bounds_ABFE" in expt else [])
    reps = [c for c in computed.columns if c.startswith("rep")]
    out = computed.copy()
    out["n"] = out[reps].notna().sum(axis=1)
    out["mean"] = out[reps].mean(axis=1)
    out["sd"] = out[reps].std(axis=1)
    out["range"] = out[reps].max(axis=1) - out[reps].min(axis=1)
    return out.join(expt.set_index("ligand")[keep]), reps


def metrics(pred, obs):
    """Pearson R (with p), Spearman rho, MUE and RMSE."""
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    r, p = stats.pearsonr(pred, obs)
    return dict(R=r, p=p, rho=stats.spearmanr(pred, obs)[0],
                mue=np.abs(pred - obs).mean(),
                rmse=float(np.sqrt(((pred - obs) ** 2).mean())))


def _panel(ax, x, y, yerr, color, title, limits, label_points=None):
    """One predicted-vs-experiment panel with y=x and a +/-1 kcal/mol band."""
    lo, hi = limits
    ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1],
                    color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color=INK_SOFT, lw=1, zorder=1)
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.5,
                    capsize=0, alpha=0.55, zorder=2)
    ax.scatter(x, y, s=64, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    for lig, xi, yi in (label_points or []):
        ax.annotate(f"{lig}", (xi, yi), textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color=INK_SOFT)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)


def plot(table, reps, protein, out_path, protocols):
    """Reproducibility first, then correlation with experiment, then the reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paired = table.dropna(subset=["dG_expt"]).sort_values("dG_expt")
    palette = [OURS, REFERENCE, "#1baf7a"]

    # runs that share a protocol are the only ones comparable to each other
    same = {}
    for r in reps:
        same.setdefault(protocols.get(r, "unknown"), []).append(r)
    pair = next((v[:2] for v in same.values() if len(v) >= 2), None)

    values = np.concatenate([paired[reps].values.ravel(), paired["dG_expt"].values,
                             paired["dG_ABFE"].values])
    values = values[np.isfinite(values)]
    limits = (values.min() - 1.5, values.max() + 1.5)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    if pair:
        a, b = pair
        both = paired[[a, b]].dropna()
        r = stats.pearsonr(both[a], both[b])[0]
        _panel(axes[0], both[a].values, both[b].values, None, OURS,
               f"Reproducibility — {a} vs {b}\n(same protocol: {protocols[a]} restraint)", limits)
        axes[0].set_xlabel(f"{a} ΔG (kcal/mol)", color=INK_SOFT, fontsize=10)
        axes[0].set_ylabel(f"{b} ΔG (kcal/mol)", color=INK_SOFT, fontsize=10)
        axes[0].text(0.04, 0.96, f"R = {r:+.2f}\nmean |diff| = {(both[a]-both[b]).abs().mean():.2f}\n"
                     f"max |diff| = {(both[a]-both[b]).abs().max():.2f}",
                     transform=axes[0].transAxes, va="top", fontsize=9, color=INK_SOFT)

    ax = axes[1]
    _panel(ax, paired["dG_expt"].values, paired[reps[0]].values, None, palette[0],
           "Each run vs experiment", limits)
    ax.collections[-1].remove()
    for run, color in zip(reps, palette):
        ok = paired[run].notna()
        ax.scatter(paired.loc[ok, "dG_expt"], paired.loc[ok, run], s=58, color=color,
                   edgecolor=SURFACE, linewidth=1.2, zorder=3,
                   label=f"{run} — {protocols.get(run, '?')}")
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK_SOFT)
    ax.set_xlabel("experimental ΔG (kcal/mol)", color=INK_SOFT, fontsize=10)
    ax.set_ylabel("computed ΔG (kcal/mol)", color=INK_SOFT, fontsize=10)

    ref = metrics(paired["dG_ABFE"], paired["dG_expt"])
    _panel(axes[2], paired["dG_expt"].values, paired["dG_ABFE"].values,
           paired["bounds_ABFE"].values if "bounds_ABFE" in paired else None,
           REFERENCE, "Alibay et al. ABFE (published reference)", limits)
    axes[2].set_xlabel("experimental ΔG (kcal/mol)", color=INK_SOFT, fontsize=10)
    axes[2].text(0.04, 0.96, f"R = {ref['R']:+.2f}  (p = {ref['p']:.3f})\n"
                 f"MUE = {ref['mue']:.2f}\nRMSE = {ref['rmse']:.2f}",
                 transform=axes[2].transAxes, va="top", fontsize=9, color=INK_SOFT)

    fig.suptitle(f"{protein}: the calculation does not yet reproduce itself",
                 color=INK, fontsize=13, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    # equal-aspect axes shrink vertically and push the x labels outside the rect
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out_path}")
    return paired


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True, help="replicate run directories")
    parser.add_argument("--expt", default="data/alibay_abfe.csv")
    parser.add_argument("--protein", default="MCL-1")
    parser.add_argument("-o", "--output", default="dg_vs_experiment.png")
    args = parser.parse_args()

    computed = collect_deltaG(args.runs, args.protein)
    table, reps = merge_experiment(computed, args.expt, args.protein)
    per_run = {f"rep{i}": restraint_protocol(run, args.protein)
               for i, run in enumerate(args.runs, start=1)}
    # a ligand whose protocol differs between runs is not a replicate of itself
    mixed = sorted({lig for lig in table.index
                    if len({p[lig] for p in per_run.values() if lig in p}) > 1})
    if mixed:
        print(f"excluding mixed-protocol ligands: {mixed}")
        table = table.drop(index=mixed)
    protocols = {r: (sorted({v for k, v in p.items() if k not in mixed})[0]
                     if p else "unknown") for r, p in per_run.items()}
    paired = plot(table, reps, args.protein, args.output, protocols)

    csv_path = os.path.splitext(args.output)[0] + ".csv"
    table.round(3).to_csv(csv_path)
    print(f"wrote {csv_path}")
    print(f"\n{len(paired)} ligands paired with experiment")
    for run in reps:
        ok = paired[run].notna() & paired["dG_expt"].notna()
        if ok.sum() >= 3:
            st = metrics(paired.loc[ok, run], paired.loc[ok, "dG_expt"])
            print(f"  {run} ({protocols.get(run,'?'):>14}) n={ok.sum():2d}  R={st['R']:+.3f} "
                  f"(p={st['p']:.3f})  MUE={st['mue']:.2f}  RMSE={st['rmse']:.2f}")
    st = metrics(paired["dG_ABFE"], paired["dG_expt"])
    print(f"  {'Alibay ABFE':<22} n={len(paired):2d}  R={st['R']:+.3f} (p={st['p']:.3f})  "
          f"MUE={st['mue']:.2f}  RMSE={st['rmse']:.2f}")


if __name__ == "__main__":
    main()
