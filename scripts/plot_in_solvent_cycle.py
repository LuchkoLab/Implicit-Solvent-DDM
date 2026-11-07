#!/usr/bin/env python
"""Draw the in-solvent decoupling cycle as a three-leg state diagram."""
import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

COLORS = {
    "lj": "#4C78A8",
    "screen": "#72B7B2",
    "charge": "#F58518",
    "restraint": "#9D9D9D",
    "endstate": "#54A24B",
}

# (title, rows, stage): rows are (ligand q, ligand SCREEN, ligand-receptor LJ, restraints)
COMPLEX = [
    ("no_interactions", ("q 0", "SCREEN 0", "LJ 0", "restr max"), "lj"),
    ("lj_0.2", ("q 0", "SCREEN 0", "LJ 0.2", "restr max"), "lj"),
    ("interactions", ("q 0", "SCREEN 0", "LJ 1", "restr max"), "lj"),
    ("screening_0.5", ("q 0", "SCREEN 0.5", "LJ 1", "restr max"), "screen"),
    ("electrostatics", ("q 0", "SCREEN 1", "LJ 1", "restr max"), "screen"),
    ("electrostatics", ("q 0.5", "SCREEN 1", "LJ 1", "restr max"), "charge"),
    ("electrostatics", ("q 1", "SCREEN 1", "LJ 1", "restr max"), "charge"),
    ("lambda_window", ("q 1", "SCREEN 1", "LJ 1", "restr ladder"), "restraint"),
    ("endstate", ("q 1", "SCREEN 1", "LJ 1", "no restr"), "endstate"),
]
LIGAND = [
    ("endstate", ("q 1", "no restr"), "endstate"),
    ("lambda_window", ("q 1", "restr ladder"), "restraint"),
    ("anchor", ("q 1", "restr max"), "restraint"),
    ("electrostatics", ("q 0.5", "restr max"), "charge"),
    ("electrostatics", ("q 0", "restr max"), "charge"),
]
RECEPTOR = [
    ("endstate", ("no ligand", "no restr"), "endstate"),
    ("lambda_window", ("no ligand", "restr ladder"), "restraint"),
    ("anchor", ("no ligand", "restr max"), "restraint"),
]


def draw_leg(ax, y, label, states, width=1.55, gap=0.45, height=1.35):
    """Draw one leg as a row of state boxes joined by arrows.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    y : float
        Bottom of the row.
    label : str
        Leg name.
    states : list of tuple
        ``(title, rows, stage)`` in cycle order.
    width, gap, height : float
        Box geometry.
    """
    ax.text(-0.25, y + height / 2, label, ha="right", va="center", fontsize=12, fontweight="bold")
    for i, (title, rows, stage) in enumerate(states):
        x = i * (width + gap)
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=COLORS[stage], edgecolor="black", linewidth=0.8, alpha=0.9))
        ax.text(x + width / 2, y + height - 0.17, title, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white")
        for j, row in enumerate(rows):
            ax.text(x + width / 2, y + height - 0.45 - 0.24 * j, row, ha="center", va="center",
                    fontsize=8, color="white")
        if i:
            ax.add_patch(FancyArrowPatch((x - gap + 0.04, y + height / 2), (x - 0.04, y + height / 2),
                                         arrowstyle="-|>", mutation_scale=12, linewidth=1.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/in_solvent_cycle.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(19, 7.2))
    draw_leg(ax, 4.2, "Complex", COMPLEX)
    draw_leg(ax, 2.2, "Ligand", LIGAND)
    draw_leg(ax, 0.2, "Receptor", RECEPTOR)
    ax.text(0, 6.05, "Every state: GB water (igb_solvent, extdiel 78.5, user saltcon), GPU for complex/receptor",
            fontsize=11, style="italic")
    ax.text(6.3, 0.9, r"$\Delta G_{bind} = \Delta G_{complex} + \Delta G_{ligand} + \Delta G_{receptor}"
            r" + \Delta G_{Boresch} + \Delta G_{flat\ bottom}$", fontsize=13, va="center")
    ax.legend(handles=[
        Patch(color=COLORS["lj"], label="ligand-receptor LJ on (q 0, SCREEN 0)"),
        Patch(color=COLORS["screen"], label="ligand GB screening on (q 0)"),
        Patch(color=COLORS["charge"], label="ligand charges"),
        Patch(color=COLORS["restraint"], label="restraints"),
        Patch(color=COLORS["endstate"], label="endstate"),
    ], loc="lower right", fontsize=9, frameon=False, bbox_to_anchor=(1.0, 0.0))
    ax.set_xlim(-1.6, 9 * 2.0)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    fig.savefig(args.out, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
