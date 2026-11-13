"""What each rule converts thinking tokens into, and what it must spend to match us.

Area under the accuracy-token curve answers "which rule is better overall" and
hides both numbers a deployment picks between. This reports them separately,
against \\method{}'s own operating point at `lambda = 0`:

  at matched tokens    what accuracy each rule reaches for what we spend
  at matched accuracy  what each rule must spend to reach what we reach
  conversion           accuracy points bought per 100 thinking tokens

The conversion rate has a reference that makes it readable. A coin's accuracy gain
is proportional to its spend -- escalating a random share `e` gains `e` of the
full fast-to-slow difference at `e` of the full cost -- so **random's conversion is
the same at every budget, and it is the no-information rate**. A router's own rate
divided by it is what the router's information is worth, in tokens. A rule at
`1.0x` is buying accuracy no faster than a coin.

Two conversions are reported because they answer different questions. The average
rate, from zero to a point, is what the whole spend bought. The marginal rate is
what the *next* hundred tokens buy there, and it falls as a rule exhausts the items
worth escalating; it is the number that says whether to spend more.

Area under the accuracy-token curve answers "which rule is better overall" and
hides both numbers a deployment actually picks between. This reports the two
directions separately, against \\method{}'s own operating point at `lambda = 0`:

  at matched tokens    what accuracy each rule reaches for what we spend
  at matched accuracy  what each rule must spend to reach what we reach

The second is the one that can be unanswerable, and saying so is part of the
result: a rule whose curve never reaches our accuracy has no matched-accuracy
cost, and reporting its cheapest point instead would credit it for a target it
misses. Those cells print as `never`, with the rule's own ceiling beside them.

Both readings are taken off each rule's monotone frontier -- the best accuracy
reachable at a cost or less -- because a raw sweep is not monotone and a reader
comparing two wiggling lines at one budget is reading noise. Within a block of
tied scores the frontier's chord is achievable, since escalating a random subset
of a tie gives exactly that chord.

    python3 scripts/matched_cost.py --venue JevBench
"""
import argparse
import json

import numpy as np

from plot_curves import lines_for, monotone_hull


def conversion(accuracy, fast, tokens):
    """Accuracy points bought per 100 thinking tokens a question, from zero."""
    return 100.0 * 100.0 * (accuracy - fast) / tokens if tokens > 0 else float("nan")


def marginal(tokens_curve, accuracy_curve, budget, window):
    """Accuracy points the next 100 tokens buy at this budget, by a local slope.

    Read over a window rather than between adjacent grid points, because the
    frontier is a step function: adjacent points give either zero or a slope set
    by one item's token count, neither of which is the local rate.
    """
    lo = max(0.0, budget - window / 2)
    hi = budget + window / 2
    x, y = monotone_hull(tokens_curve, accuracy_curve)
    span = hi - lo
    if span <= 0:
        return float("nan")
    rise = (np.interp(hi, x, y, left=y[0], right=y[-1])
            - np.interp(lo, x, y, left=y[0], right=y[-1]))
    return 100.0 * 100.0 * rise / span


def accuracy_at(tokens, accuracy, budget):
    x, y = monotone_hull(tokens, accuracy)
    return float(np.interp(budget, x, y, left=y[0], right=y[-1]))


def tokens_for(tokens, accuracy, target):
    """The cheapest spend on this frontier that reaches `target`, or None.

    `None` rather than the frontier's last point: a rule that never reaches the
    target has no cost for it, and substituting its ceiling would report a rule
    as cheap at an accuracy it does not achieve.
    """
    x, y = monotone_hull(tokens, accuracy)
    reached = np.nonzero(y >= target - 1e-12)[0]
    if len(reached) == 0:
        return None
    first = reached[0]
    if first == 0:
        return float(x[0])
    # y is non-decreasing, so the target is crossed between first-1 and first.
    lo, hi = y[first - 1], y[first]
    if hi == lo:
        return float(x[first])
    return float(x[first - 1] + (x[first] - x[first - 1]) * (target - lo) / (hi - lo))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--curves", default="paper-assets/data/routing-curves.json")
    parser.add_argument("--venue", default="JevBench")
    parser.add_argument("--models", nargs=2, default=["Qwen3.8-27B", "Qwen3.5-4B"])
    parser.add_argument("--switch", nargs=2, required=True)
    parser.add_argument("--routers", nargs=2, required=True)
    parser.add_argument("--point", nargs=2, type=float, required=True,
                        metavar=("TOK", "TOK"),
                        help="our own operating point's token spend, per backbone")
    arguments = parser.parse_args()

    ours = json.load(open(arguments.curves))
    for model, switch, routers, spend in zip(arguments.models, arguments.switch,
                                             arguments.routers, arguments.point):
        lines, cell = lines_for(arguments.venue, model, ours,
                                {"ThinkSwitcher (reproduced)": switch, "routers": routers})
        mine = lines["\\textsc{Hdm} (ours)"]
        reach = accuracy_at(*mine, spend)
        print(f"\n=== {arguments.venue}, {model}: {cell['n']} items, "
              f"fast {cell['fast']:.4f}, slow {cell['slow']:.4f} at "
              f"{cell['slow_tokens']:.0f} tok")
        print(f"    our operating point: {spend:.0f} tok, {reach:.4f}")
        # The coin's rate, which is the same at every budget and is the reference.
        coin = conversion(cell["slow"], cell["fast"], cell["slow_tokens"])
        window = cell["slow_tokens"] / 8.0
        print(f"    random converts {coin:.3f} points per 100 tokens at every budget, "
              f"which is the no-information rate")
        print(f"\n    {'Rule':28s} {'acc':>7s} {'diff':>8s} {'conv':>6s} {'x coin':>7s} "
              f"{'marg':>6s} {'tok @ our acc':>13s} {'ratio':>7s} {'ceiling':>8s}")
        rows = []
        for label, (x, y) in lines.items():
            got = accuracy_at(x, y, spend)
            rows.append((label, got, tokens_for(x, y, reach),
                         float(monotone_hull(x, y)[1][-1]),
                         conversion(got, cell["fast"], spend),
                         marginal(x, y, spend, window)))
        for label, got, cost, ceiling, conv, marg in sorted(rows, key=lambda r: -r[4]):
            name = label.replace("\\textsc{Hdm}", "HDM")
            if cost is None:
                cheap, ratio = f"{'never':>13s}", f"{'--':>7s}"
            else:
                cheap, ratio = f"{cost:13.0f}", f"{cost / spend:7.2f}"
            print(f"    {name:28s} {got:7.4f} {got - reach:+8.4f} {conv:6.3f} "
                  f"{conv / coin:7.2f} {marg:6.3f} {cheap} {ratio} {ceiling:8.4f}")

        print(f"\n    Conversion across the sweep, points per 100 tokens "
              f"(marginal in brackets)")
        grid = np.linspace(cell["slow_tokens"] / 8, cell["slow_tokens"], 8)
        order = [l for l, *_ in sorted(rows, key=lambda r: -r[4])]
        print(f"    {'tok':>6s} " + " ".join(
            f"{l.replace(chr(92)+chr(92)+'textsc{Hdm}','HDM')[:13]:>15s}" for l in order))
        for budget in grid:
            cells = []
            for label in order:
                x, y = lines[label]
                cells.append(f"{conversion(accuracy_at(x, y, budget), cell['fast'], budget):6.2f}"
                             f" [{marginal(x, y, budget, window):5.2f}]")
            print(f"    {budget:6.0f} " + " ".join(f"{c:>15s}" for c in cells))


if __name__ == "__main__":
    main()
