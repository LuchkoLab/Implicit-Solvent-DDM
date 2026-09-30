#!/usr/bin/env python
"""plot_bar_vs_mbar.py -- figures for the adjacent-window-BAR vs full-MBAR comparison.

Reads the two CSVs written by `bar_vs_mbar.py` and produces:

  1. <prefix>_correlation.png  MBAR vs BAR, one panel per leg + the 3-leg total. Points are
                               (system, replica); the dashed line is y = x, so vertical distance
                               from it IS the estimator error. Annotated with R^2, RMSE, MAE.
  2. <prefix>_heatmap.png      system x leg, mean signed (BAR - MBAR) over the replicas. Diverging
                               about 0 -- blue = BAR under-estimates, red = over-estimates.
  3. <prefix>_overlap.png      per adjacent pair, |BAR - MBAR| vs that pair's MBAR overlap. This is
                               the mechanistic plot: it shows the error is concentrated at the
                               low-overlap steps rather than spread along the cycle.

Usage
-----
  python scripts/plot_bar_vs_mbar.py                       # reads ./bar_vs_mbar_{summary,pairs}.csv
  python scripts/plot_bar_vs_mbar.py --prefix bar_vs_mbar --outdir figures
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# --- palette (validated: 3 categorical slots pass all-pairs CVD + normal-vision in light mode) ---
LEG_COLOR = {"complex": "#2a78d6", "ligand": "#eb6834", "receptor": "#1baf7a", "TOTAL_3LEG": "#4a3aa7"}
SURFACE, INK, INK2, MUTED, GRID, BASELINE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)
# diverging: two poles + NEUTRAL GRAY midpoint (never a hue at the middle), equal arms
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#256abf", "#f0efec", "#d03b3b"])
LEGS = ["complex", "ligand", "receptor", "TOTAL_3LEG"]
LABEL = {"complex": "Complex", "ligand": "Ligand", "receptor": "Receptor",
         "TOTAL_3LEG": "Total (3 legs)"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=1.0)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _short(name):
    """cb7_adn_ligand_1_Hmass -> adn"""
    parts = name.split("_")
    return parts[1] if len(parts) > 1 else name


def correlation(summary, out):
    """MBAR vs BAR per leg. Small multiples: each panel is directly titled, so the leg identity
    never rests on colour alone (the aqua slot is sub-3:1 on this surface)."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.1), facecolor=SURFACE)
    for ax, leg in zip(axes, LEGS):
        g = summary[summary.leg == leg]
        _style(ax)
        if not len(g):
            ax.set_title(LABEL[leg], fontsize=11, color=INK)
            continue
        x, y = g.dG_mbar.to_numpy(), g.dG_bar.to_numpy()
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = 0.06 * (hi - lo or 1.0)
        lims = (lo - pad, hi + pad)

        ax.plot(lims, lims, ls="--", lw=1.2, color=BASELINE, zorder=1)
        # 2px surface ring on the markers so overlapping points stay separable
        ax.scatter(x, y, s=46, color=LEG_COLOR[leg], edgecolor=SURFACE, linewidth=1.6, zorder=3)

        resid = y - x
        rmse = float(np.sqrt((resid ** 2).mean()))
        mae = float(np.abs(resid).mean())
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if len(x) > 2 else float("nan")
        ax.set_xlim(lims), ax.set_ylim(lims)
        ax.set_title(f"{LABEL[leg]}   (n={len(g)})", fontsize=11, color=INK, pad=8)
        ax.set_xlabel("MBAR  $\\Delta G$  (kcal/mol)", fontsize=9, color=INK2)
        if ax is axes[0]:
            ax.set_ylabel("adjacent-window BAR  $\\Delta G$  (kcal/mol)", fontsize=9, color=INK2)
        ax.text(0.04, 0.96,
                f"$R^2$ = {r2:.5f}\nRMSE = {rmse:.3f}\nMAE  = {mae:.3f}\nmax = {np.abs(resid).max():.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=INK2,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=GRID, linewidth=0.8))

    fig.suptitle("Adjacent-window BAR reproduces full MBAR   ·   cb7 host–guest, 3 replicas   "
                 "·   dashed line = perfect agreement",
                 fontsize=12, color=INK, y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


def heatmap(summary, out):
    """system x leg, mean signed error over replicas. Every cell is labelled with its value, so the
    diverging colour is reinforcement rather than the sole channel."""
    piv = (summary.groupby(["system", "leg"])["diff"].mean().unstack("leg").reindex(columns=LEGS))
    piv.index = [_short(s) for s in piv.index]
    piv = piv.sort_values("TOTAL_3LEG", na_position="last")
    n_rep = summary.groupby(["system", "leg"]).size().max()

    vmax = float(np.nanmax(np.abs(piv.to_numpy())))
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(piv) + 2.2), facecolor=SURFACE)
    im = ax.imshow(piv.to_numpy(), cmap=DIVERGING, norm=TwoSlopeNorm(0.0, -vmax, vmax), aspect="auto")

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.to_numpy()[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5, color=MUTED)
                continue
            # ink flips to white only where the fill is dark enough to need it
            shade = "#ffffff" if abs(v) > 0.62 * vmax else INK
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8, color=shade)

    ax.set_xticks(range(len(LEGS)), [LABEL[l].replace(" (3 legs)", "") for l in LEGS], fontsize=9,
                  color=INK2)
    ax.set_yticks(range(len(piv)), piv.index, fontsize=9, color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(LEGS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(piv), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)  # 2px surface gap between cells
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("mean (BAR − MBAR), kcal/mol", fontsize=9, color=INK2)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_visible(False)

    ax.set_title(f"Signed estimator error by system and leg\nmean of {n_rep} replicas · "
                 f"blue = BAR lower, red = BAR higher",
                 fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


def overlap(pairs, out):
    """The mechanistic plot: per-pair error against that pair's MBAR overlap."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor=SURFACE)
    _style(ax)
    FLOOR = 1e-5  # exact-zero overlaps are clipped here so they are visible on a log axis
    legs = [l for l in ["complex", "ligand", "receptor"] if (pairs.leg == l).any()]
    for leg in legs:
        g = pairs[pairs.leg == leg]
        ax.scatter(g.overlap_min.clip(lower=FLOOR), g.abs_diff.clip(lower=1e-7), s=15, alpha=0.45,
                   color=LEG_COLOR[leg], edgecolor="none", label=f"{LABEL[leg]} ({len(g)} pairs)",
                   zorder=3)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.axvline(0.04, color=BASELINE, ls="--", lw=1.2, zorder=2)

    # annotate in axes coords so nothing can collide with the legend
    ax.text(0.04, -0.085, "ALS threshold\n0.04", transform=ax.get_xaxis_transform(),
            fontsize=8, color=INK2, ha="center", va="top", linespacing=1.3)
    n_zero = int((pairs.overlap_min <= 0).sum())
    if n_zero:
        ax.text(FLOOR, -0.085, f"overlap = 0\n({n_zero} pairs, clipped)",
                transform=ax.get_xaxis_transform(), fontsize=8, color=MUTED, ha="center",
                va="top", linespacing=1.3)

    ax.set_xlabel("MBAR overlap of the adjacent pair (min direction)", fontsize=9.5, color=INK2,
                  labelpad=26)
    ax.set_ylabel("| BAR − MBAR |  for that step  (kcal/mol)", fontsize=9.5, color=INK2)
    ax.set_title("Where the BAR chain loses accuracy: the low-overlap steps",
                 fontsize=11.5, color=INK, pad=10)
    leg_box = ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    leg_box.set_zorder(5)
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


def overlap_cycle(pairs, out):
    """Adjacent-pair overlap along the cycle, system x step, one panel per leg.

    This is the superdiagonal of each leg's overlap matrix -- the only part the BAR chain actually
    depends on -- for every system at once. Sequential (magnitude), one hue, light->dark, on a log
    scale because the values span 1e-5 to 0.5. Cells at or below the ALS threshold are outlined.
    """
    legs = [l for l in ["complex", "ligand", "receptor"] if (pairs.leg == l).any()]
    fig, axes = plt.subplots(1, len(legs), figsize=(5.6 * len(legs), 6.2), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    FLOOR = 1e-5

    for ax, leg in zip(axes, legs):
        g = pairs[pairs.leg == leg]
        piv = g.groupby(["system", "i"])["overlap_min"].mean().unstack("i")
        piv.index = [_short(s) for s in piv.index]
        piv = piv.sort_index()
        vals = np.clip(piv.to_numpy(), FLOOR, None)

        # range chosen so the 0.003-0.4 band where nearly all the data sits is actually resolved;
        # anything worse than 0.003 saturates at the darkest step (and is red-outlined anyway)
        im = ax.imshow(np.log10(vals), cmap="Blues_r", aspect="auto",
                       vmin=np.log10(3e-3), vmax=np.log10(0.4))
        # outline every cell that fails the ALS overlap threshold
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if vals[i, j] < 0.04:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="#d03b3b", linewidth=1.4, zorder=4))
        ax.set_xlabel("adjacent-window step along the cycle", fontsize=9, color=INK2)
        ax.set_title(f"{LABEL[leg]}   ({piv.shape[1]} steps)", fontsize=11, color=INK, pad=8)
        ax.set_yticks(range(len(piv)), piv.index, fontsize=8.5, color=INK2)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        if ax is not axes[0]:
            ax.set_yticklabels([])

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("adjacent-pair overlap  (log$_{10}$)", fontsize=9, color=INK2)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    fig.suptitle("Overlap between neighbouring windows, every system   ·   "
                 "red outline = below the 0.04 threshold   ·   dark = poor overlap",
                 fontsize=12, color=INK, y=0.98)
    fig.savefig(out, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


def overlap_matrices(ovdir, out):
    """The full N x N MBAR overlap matrices written by `bar_vs_mbar.py --save-overlap`."""
    import glob as _glob

    files = sorted(_glob.glob(os.path.join(ovdir, "*.overlap.npz")))
    if not files:
        print(f"no *.overlap.npz in {ovdir} — run bar_vs_mbar.py --save-overlap first")
        return
    fig, axes = plt.subplots(1, len(files), figsize=(5.4 * len(files), 5.0), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, files):
        d = np.load(f, allow_pickle=True)
        m = np.clip(np.asarray(d["overlap"], dtype=float), 1e-6, None)
        system, rep, leg = os.path.basename(f).replace(".overlap.npz", "").split("__")
        im = ax.imshow(np.log10(m), cmap="Blues_r", vmin=-5, vmax=np.log10(0.5))
        ax.set_title(f"{LABEL.get(leg, leg)}  ·  {_short(system)}/{rep}  ({len(m)} states)",
                     fontsize=10.5, color=INK, pad=8)
        ax.set_xlabel("state index (cycle order)", fontsize=9, color=INK2)
        if ax is axes[0]:
            ax.set_ylabel("state index (cycle order)", fontsize=9, color=INK2)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)
        for s in ax.spines.values():
            s.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("MBAR overlap  (log$_{10}$)", fontsize=9, color=INK2)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    fig.suptitle("Full MBAR overlap matrices   ·   BAR uses only the first off-diagonal; "
                 "MBAR uses every cell   ·   darker = less overlap",
                 fontsize=12, color=INK, y=0.99)
    fig.savefig(out, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


def table(summary):
    """Printed table view -- the relief obligation for the sub-3:1 slot, and handy for Slack."""
    print("\n=== BAR - MBAR by leg (kcal/mol) ===")
    print(f"{'leg':<16}{'n':>4}{'mean':>9}{'RMSE':>8}{'MAE':>8}{'max|d|':>9}{'R^2':>10}")
    for leg in LEGS:
        g = summary[summary.leg == leg]
        if not len(g):
            continue
        r = g.dG_bar.to_numpy() - g.dG_mbar.to_numpy()
        r2 = float(np.corrcoef(g.dG_mbar, g.dG_bar)[0, 1] ** 2)
        print(f"{LABEL[leg]:<16}{len(g):>4}{r.mean():>+9.3f}{np.sqrt((r**2).mean()):>8.3f}"
              f"{np.abs(r).mean():>8.3f}{np.abs(r).max():>9.3f}{r2:>10.6f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="bar_vs_mbar", help="CSV prefix from bar_vs_mbar.py")
    ap.add_argument("--outdir", default=".", help="where to write the PNGs")
    ap.add_argument("--overlap-dir", default=None,
                    help="directory of *.overlap.npz from `bar_vs_mbar.py --save-overlap`; "
                         "adds the full N x N overlap-matrix figure")
    args = ap.parse_args()

    summary = pd.read_csv(f"{args.prefix}_summary.csv")
    pairs = pd.read_csv(f"{args.prefix}_pairs.csv")
    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, os.path.basename(args.prefix))

    correlation(summary, f"{base}_correlation.png")
    heatmap(summary, f"{base}_heatmap.png")
    overlap(pairs, f"{base}_overlap.png")
    overlap_cycle(pairs, f"{base}_overlap_cycle.png")
    if args.overlap_dir:
        overlap_matrices(args.overlap_dir, f"{base}_overlap_matrices.png")
    table(summary)


if __name__ == "__main__":
    main()
