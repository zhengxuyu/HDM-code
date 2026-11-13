"""Every router's accuracy against tokens, and the areas under those curves.

**`scripts/fig_curves.py` draws the published figure**, not this. This remains the
analysis tool -- it reports the area under each rule's curve, and its `--style`
switches between views while deciding what to draw -- and its `--output` no longer
defaults to the published path, so the two cannot silently disagree.


One panel per backbone, both on JevBench's held-out items, the venue and setting
of Table~\\ref{tab:baselines}. Each rule sweeps its own knob over every cut, so a
line is the whole family that rule reaches rather than one operating point, and
the panels share an axis because the modes are held fixed and only the rule varies.

Three of the lines are our implementations of published methods, and each is
labelled `(reproduced)` in the legend for the same reason: none of the three ships
code we could run. ThinkSwitcher's appendix gives Ray Tune *search ranges* rather
than the values it chose, so `thinkswitcher.py` reruns that search; KNN-Router and
RouteLLM are the versions Pan et al. specify for exactly this binary routing
comparison, and `routers.py` implements them. The label is not a hedge -- it says
which numbers depend on how well we ran someone else's method.

Reads the curve data this repository records rather than recomputing it, so the
figure and the table cannot drift apart:

    paper-assets/data/routing-curves.json  ours and the confidence threshold
    results/ts-*-tuned.json               the switch, after its own search
    results/routers-*.json                KNN-Router and RouteLLM

    python3 scripts/plot_curves.py --venue JevBench --output paper-assets/figures/fig_curves.pdf
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

# One colour and dash per rule, shared across panels. Ours is the only solid
# heavy line; the reproduced baselines are dashed, random is dotted.
STYLE = {
    "\\textsc{Htm} (ours)":      dict(color="#1b3a6b", lw=2.0, ls="-",  zorder=5),
    "Fast confidence":           dict(color="#c2541c", lw=1.4, ls=(0, (2.4, 1.2))),
    "ThinkSwitcher (reproduced)": dict(color="#2e7d4f", lw=1.2, ls=(0, (1.6, 1.4))),
    "KNN-Router (reproduced)":   dict(color="#7b4fa0", lw=1.2, ls=(0, (3.4, 1.2, 0.8, 1.2))),
    # Magenta, not the olive it had: olive against our navy is two dark colours at
    # 1.3pt, and these two curves differ by 1.4 of 231 items on average, so a reader
    # who cannot tell the lines apart cannot see that they nearly coincide either.
    "RouteLLM (reproduced)":     dict(color="#b5377f", lw=1.4, ls=(0, (5, 1.4))),
    "Random":                    dict(color="#777777", lw=1.1, ls=":"),
}

# In the legend a reproduced method is starred and the legend's title says what the
# star means, because spelling "(reproduced)" out six times over makes the legend
# wider than the empty corner it has to fit in.
SHORT = {"\\textsc{Htm} (ours)": "HTM (ours)", "Fast confidence": "Fast confidence",
         "ThinkSwitcher (reproduced)": "ThinkSwitcher*",
         "KNN-Router (reproduced)": "KNN-Router*",
         "RouteLLM (reproduced)": "RouteLLM*", "Random": "Random"}

# A marker per rule for the scatter. Ours is the filled one; every baseline is open,
# so where they land on top of us the fill still reads.
MARKER = {"\\textsc{Htm} (ours)": ("o", True, 11),
          "Fast confidence": ("s", False, 9),
          "ThinkSwitcher (reproduced)": ("v", False, 9),
          "KNN-Router (reproduced)": ("^", False, 9),
          "RouteLLM (reproduced)": ("D", False, 7),
          # `x` draws no face, so an unfilled `x` draws nothing at all; it is marked
          # as filled here, which for a faceless marker just means "use the colour".
          "Random": ("x", True, 8)}

# `Random` is the zero line in the excess view and saying so twice is noise, so the
# legend names it for what it is there: the reference every other marker is measured
# against.
EXCESS_SHORT = dict(Random="Random (the reference)")

# Below this many distinct operating points, a line between them is a claim about
# cuts the rule does not offer, so its points are drawn on top of the line.
FEW = 12


def operating_points(tokens, accuracy, every):
    """The rule's own achievable operating points, thinned to a readable number.

    A sweep offers one point per cut -- up to one per item -- which is a cloud
    rather than a figure. Thinning by rank keeps points the rule actually reaches
    and spaces them along its own frontier, where thinning by token budget would
    put several markers inside one flat step and none across a jump.
    """
    order = np.argsort(tokens, kind="stable")
    x, y = np.asarray(tokens)[order], np.asarray(accuracy)[order]
    # Thin the *distinct* points, not the knob's grid. A threshold sweep spends most
    # of its grid at the two ends, where every nearby value escalates all or none, so
    # thinning the grid leaves a rule looking sparse in the middle where it in fact
    # offers the most choices.
    _, first = np.unique(np.stack([x, y]), axis=1, return_index=True)
    distinct = np.sort(first)
    keep = distinct[:: max(1, len(distinct) // every)]
    if keep[-1] != distinct[-1]:
        keep = np.append(keep, distinct[-1])
    return x[keep], y[keep]


def monotone_hull(tokens, accuracy):
    """The frontier of a sweep: the best accuracy reachable at each cost or less.

    A sweep's raw curve is not monotone -- one more escalation can lose an item --
    and a reader comparing two wiggling lines at a budget is reading noise. The
    running maximum is what a deployment could actually choose, since it may always
    pick a cheaper cut of its own knob.
    """
    order = np.argsort(tokens, kind="stable")
    return np.asarray(tokens)[order], np.maximum.accumulate(np.asarray(accuracy)[order])


def read_at(tokens, accuracy, budgets):
    tokens, accuracy = monotone_hull(tokens, accuracy)
    return np.interp(budgets, tokens, accuracy, left=accuracy[0], right=accuracy[-1])


def lines_for(venue, model, ours, extra):
    """Every rule's (tokens, accuracy) for one panel, as a dict of label to pair."""
    cell = ours[f"{venue},{model}"]
    out = {"\\textsc{Htm} (ours)": tuple(cell["htm"]),
           "Fast confidence": tuple(cell["confidence"])}
    for label, path in extra.items():
        if not path or not pathlib.Path(path).exists():
            continue
        loaded = json.load(open(path))
        if "curves" in loaded:                      # routers.py writes several
            for name, rows in loaded["curves"].items():
                out[f"{name} (reproduced)"] = ([r["tokens"] for r in rows],
                                               [r["accuracy"] for r in rows])
        else:                                       # thinkswitcher.py writes one
            out[label] = ([r["tokens"] for r in loaded["curve"]],
                          [r["accuracy"] for r in loaded["curve"]])
    # Random escalates a uniform share, so it is the straight line between the
    # endpoints and needs no sweep.
    share = np.linspace(0.0, 1.0, 51)
    out["Random"] = (share * cell["slow_tokens"],
                     cell["fast"] + share * (cell["slow"] - cell["fast"]))
    return out, cell


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--curves", default="paper-assets/data/routing-curves.json")
    parser.add_argument("--venue", default="JevBench")
    parser.add_argument("--models", nargs="+", default=["Qwen3.8-27B"])
    parser.add_argument("--switch", nargs="+",
                        default=["results/ts-27b-jev-matched.json"])
    parser.add_argument("--routers", nargs="+",
                        default=["results/routers-27b-jev-matched.json"])
    parser.add_argument("--style", choices=("excess", "scatter", "line"),
                        default="excess",
                        help="excess plots each operating point against what a coin "
                             "buys at the same spend, which is the comparison the "
                             "section makes and the only one with no interpolation "
                             "in it; scatter plots raw accuracy, where every rule's "
                             "shared rise with budget uses up the axis; line joins "
                             "the points, drawing operating points no rule offers")
    parser.add_argument("--points", type=int, default=26,
                        help="operating points drawn per rule")
    parser.add_argument("--output", default="figures-draft/fig_curves.pdf",
                        help="not the published figure; fig_curves.py draws that")
    arguments = parser.parse_args()

    ours = json.load(open(arguments.curves))
    # A single panel has the whole column and can be set at a readable size; two
    # panels share it and everything shrinks.
    one = len(arguments.models) == 1
    plt.rcParams.update({"font.size": 6.0 if one else 5.6, "font.family": "serif",
                         "text.usetex": False, "axes.linewidth": 0.5,
                         "xtick.labelsize": 5.6 if one else 4.8,
                         "ytick.labelsize": 5.6 if one else 4.8,
                         "legend.fontsize": 5.4 if one else 4.5,
                         "mathtext.fontset": "dejavuserif"})
    # One ICML column is 3.3in. A `figure*` spanning both would be wider but costs
    # its height from both columns, which is three column-inches the body does not
    # have, so this stays single-column and the legend goes under the panels.
    figure, panels = plt.subplots(1, len(arguments.models),
                                  figsize=(3.3, 2.0 if one else 1.85), squeeze=False)
    panels = panels[0]

    areas = {}
    for panel, model, switch, routers in zip(panels, arguments.models,
                                             arguments.switch, arguments.routers):
        lines, cell = lines_for(arguments.venue, model, ours,
                                {"ThinkSwitcher (reproduced)": switch, "routers": routers})
        top = min(max(x) for x, _ in lines.values())
        budgets = np.linspace(0.0, top, 201)
        # The coin's accuracy at a spend, which is exact: escalating a random share
        # gains that share of the fast-to-slow difference at that share of the cost.
        coin = lambda t: (cell["fast"]
                          + (t / cell["slow_tokens"]) * (cell["slow"] - cell["fast"]))
        for label, (x, y) in lines.items():
            style = STYLE.get(label, dict(lw=1.0))
            if arguments.style == "excess":
                name = SHORT.get(label, label)
                px, py = operating_points(x, y, 10 ** 6)   # every distinct point
                ex = (py - coin(px)) * cell["n"]
                panel.plot(px, ex, label=EXCESS_SHORT.get(name, name), **style)
                # A rule with few cuts gets its points marked, because a line through
                # six of them draws a frontier it does not offer. A rule with 50 or
                # more needs no marks and markers would only crowd the panel.
                if len(px) <= FEW:
                    marker = MARKER.get(label, ("o", False, 8))[0]
                    panel.plot(px, ex, linestyle="none", marker=marker,
                               markersize=2.2, markerfacecolor="white",
                               markeredgecolor=style["color"], markeredgewidth=0.7,
                               zorder=style.get("zorder", 3) + 1)
            elif arguments.style == "scatter":
                marker, filled, size = MARKER.get(label, ("o", False, 8))
                px, py = operating_points(x, y, arguments.points)
                panel.scatter(px, py, label=SHORT.get(label, label), marker=marker,
                              s=size, linewidths=0.7,
                              c=style["color"] if filled else "none",
                              # a faceless marker ignores edgecolors and warns
                              **({} if marker in ("x", "+")
                                 else {"edgecolors": style["color"]}),
                              zorder=style.get("zorder", 3))
            else:
                hx, hy = monotone_hull(x, y)
                panel.plot(hx, hy, label=SHORT.get(label, label), **style)
            areas.setdefault(model, {})[label] = float(
                np.trapezoid(read_at(x, y, budgets), budgets) / top)
        if arguments.style == "excess":
            panel.axhline(0.0, color="#777777", lw=0.7, ls="-", zorder=1)
            panel.set_ylabel(f"items correct beyond random routing,\nat the same "
                             f"cost (of {cell['n']})",
                             fontsize=6.0 if one else 5.0)
        else:
            panel.axhline(cell["fast"], color="#bbbbbb", lw=0.5, ls="-", zorder=0)
            panel.axhline(cell["slow"], color="#bbbbbb", lw=0.5, ls="-", zorder=0)
        # Anchored right: the legend takes the upper left, which is the one region no
        # curve can enter -- accuracy cannot be high where the spend is zero.
        if arguments.style != "excess":
            panel.annotate("Fast only", (cell["slow_tokens"], cell["fast"]),
                           textcoords="offset points", xytext=(-2, 2.5),
                           fontsize=5.0 if one else 4.2, color="#888888", ha="right")
            panel.annotate("Slow only", (cell["slow_tokens"], cell["slow"]),
                           textcoords="offset points", xytext=(-2, -6.5),
                           fontsize=5.0 if one else 4.2, color="#888888", ha="right")
        else:
            panel.annotate("escalating at random", (cell["slow_tokens"] * 0.55, 0.0),
                           textcoords="offset points", xytext=(0, 3),
                           fontsize=5.0 if one else 4.2, color="#777777", ha="center")
        panel.set_title(f"{model} ({cell['n']} items)" if one
                        else f"{model} ({cell['n']})",
                        fontsize=6.4 if one else 5.6, pad=2.5)
        panel.set_xlabel("thinking tokens a question", fontsize=6.0 if one else 5.0)
        panel.set_xlim(0, cell["slow_tokens"] * 1.02)
        panel.grid(alpha=0.18, lw=0.4)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
    if arguments.style != "excess":
        panels[0].set_ylabel(f"accuracy on {arguments.venue}",
                             fontsize=6.0 if one else 5.0)
    # The legend sits in the lower right, which every curve leaves empty: they run
    # from (0, fast) up to (slow tokens, slow), so accuracy is high wherever the
    # spend is. Under the panels it would cost 31pt of height the body cannot spare.
    # Under the coin line is empty in the excess view, since only a rule that is
    # worse than chance goes there; raw accuracy leaves the upper left empty instead.
    if arguments.style == "excess":
        # Every rule returns to zero at both ends, so the two lower corners are empty
        # whatever the data does in between; a little headroom below is enough for a
        # legend laid out across, and costs far less than the band the marker version
        # needed.
        for panel in panels:
            low, high = panel.get_ylim()
            panel.set_ylim(low - 0.30 * (high - low), high + 0.04 * (high - low))
    legend = panels[-1].legend(loc="lower left" if arguments.style == "excess"
                               else "upper left", frameon=False,
                               ncol=3 if arguments.style == "excess" else 1,
                               columnspacing=1.1,
                               handlelength=2.0 if one else 1.7,
                               labelspacing=0.3 if one else 0.22,
                               handletextpad=0.35, borderaxespad=0.2,
                               title="* reproduced here; no public code",
                               alignment="left")
    legend.get_title().set_fontsize(5.0 if one else 4.2)

    figure.tight_layout(pad=0.3)
    pathlib.Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output)
    print(f"wrote {arguments.output}")

    print("\nArea under each curve, normalised by the shared budget range")
    for model, row in areas.items():
        best = max(row.values())
        print(f"  {model}")
        for label, area in sorted(row.items(), key=lambda kv: -kv[1]):
            name = label.replace("\\textsc{Htm}", "HTM")
            print(f"    {name:30s} {area:.4f}  ({area - best:+.4f})")


if __name__ == "__main__":
    main()
