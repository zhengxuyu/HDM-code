"""Draw the single-column excess-correctness routing figure.

Run with the repository's plotting Python:

    /Users/yuzhengxu/miniforge3/bin/python3 scripts/fig_curves.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper-assets/figures/fig_curves.pdf"

# Deliberately not Matplotlib's default palette. HDM is the only heavy, solid,
# saturated curve; the baselines remain distinguishable in greyscale by dashing.
STYLES = {
    "HDM (ours)": dict(color="#17365D", lw=2.15, ls="-", zorder=6),
    "Fast confidence": dict(color="#C44E38", lw=1.15, ls=(0, (4.0, 1.6)), zorder=4),
    "ThinkSwitcher*": dict(color="#16817A", lw=1.15, ls=(0, (2.0, 1.3)), zorder=5),
    "KNN-Router*": dict(color="#76528B", lw=1.05, ls=(0, (1.0, 1.25)), zorder=3),
    "RouteLLM*": dict(color="#C58A16", lw=1.15, ls=(0, (5.0, 1.3, 1.0, 1.3)), zorder=4),
    "Random": dict(color="#777777", lw=0.85, ls="-", zorder=1),
}


def distinct_points(tokens, accuracy):
    """Return actual swept operating points in increasing token order."""
    points = np.column_stack((tokens, accuracy)).astype(float)
    points = points[np.argsort(points[:, 0], kind="stable")]
    # Threshold grids repeat identical operating points. Remove only exact repeats;
    # do not hull, interpolate, smooth, or otherwise change the measured sweep.
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.any(points[1:] != points[:-1], axis=1)
    return points[keep, 0], points[keep, 1]


def load_curves():
    with (ROOT / "paper-assets/data/routing-curves.json").open() as stream:
        cell = json.load(stream)["JevBench,Qwen3.8-27B"]
    with (ROOT / "results/ts-27b-jev-matched.json").open() as stream:
        switch = json.load(stream)["curve"]
    with (ROOT / "results/routers-27b-jev-matched.json").open() as stream:
        routers = json.load(stream)["curves"]

    curves = {
        "HDM (ours)": tuple(cell["htm"]),
        "Fast confidence": tuple(cell["confidence"]),
        "ThinkSwitcher*": ([row["tokens"] for row in switch],
                            [row["accuracy"] for row in switch]),
        "KNN-Router*": ([row["tokens"] for row in routers["KNN-Router"]],
                        [row["accuracy"] for row in routers["KNN-Router"]]),
        "RouteLLM*": ([row["tokens"] for row in routers["RouteLLM"]],
                      [row["accuracy"] for row in routers["RouteLLM"]]),
        "Random": ([0.0, cell["slow_tokens"]], [cell["fast"], cell["slow"]]),
    }
    return cell, curves


def main():
    cell, curves = load_curves()
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 6.0,
        "axes.labelsize": 6.0,
        "axes.linewidth": 0.5,
        "xtick.labelsize": 5.6,
        "ytick.labelsize": 5.6,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "legend.fontsize": 5.35,
        "pdf.fonttype": 42,
    })

    # Exactly one 3.3-inch ICML column by 2 inches = 237.6 x 144 points.
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    handles = []
    for label in STYLES:  # dict insertion order is the required legend order
        tokens, accuracy = distinct_points(*curves[label])
        random_accuracy = (cell["fast"]
                           + tokens / cell["slow_tokens"]
                           * (cell["slow"] - cell["fast"]))
        excess = (accuracy - random_accuracy) * cell["n"]
        line, = ax.plot(tokens, excess, label=label, **STYLES[label])
        handles.append(line)

        # The markers disclose that this apparent segment joins only six choices.
        if label == "ThinkSwitcher*":
            ax.plot(tokens, excess, linestyle="none", marker="o", markersize=2.8,
                    markerfacecolor="white", markeredgecolor=STYLES[label]["color"],
                    markeredgewidth=0.75, zorder=7)

    ax.set_xlim(0, 591.1645)
    ax.set_xticks([0, 100, 200, 300, 400, 500, 591])
    ax.set_ylim(-2.85, 10.0)
    ax.set_yticks([-2, 0, 2, 4, 6, 8, 10])
    ax.set_xlabel("thinking tokens a question", labelpad=2.0)
    ax.set_ylabel("items correct beyond random routing\nat the same token spend (of 231)",
                  labelpad=2.4)
    ax.grid(axis="y", color="#D8D8D8", lw=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # A dedicated strip above the axes costs no data range and cannot obscure data.
    legend = fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
                        bbox_to_anchor=(0.56, 0.985), borderaxespad=0,
                        handlelength=2.2, handletextpad=0.38, columnspacing=0.95,
                        labelspacing=0.32,
                        title="* reproduced here; no public code")
    legend.get_title().set_fontsize(5.2)

    # Manual margins keep the PDF canvas exact (bbox_inches='tight' would change it).
    fig.subplots_adjust(left=0.205, right=0.982, bottom=0.205, top=0.695)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    plt.close(fig)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
