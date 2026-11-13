"""Accuracy against thinking tokens, as a frontier, for every router.

Matching the escalation rate does not match the cost: the head escalates the
long questions, so at one rate it and a confidence threshold spend different
numbers of tokens, and the paper reports cost in tokens. This compares whole
curves instead, so no router is handed an operating point.

Each router sweeps its own score: for every cut `k` the top `k` questions by
that score are escalated, which costs the tokens their slow passes took and
reaches the accuracy that escalation set reaches. That is one point; the sweep
over every `k` is the router's frontier. Accuracy is then read off each
frontier at a shared grid of token budgets by linear interpolation, so the
comparison is at equal cost rather than at equal coverage.

Random needs no sweep: escalating a uniformly random subset of size `k` has
expected accuracy `e*slow + (1-e)*fast` at `e = k/n` and expected cost `e` of
the slow-only token count, which is the straight line between the endpoints.

    uv run python scripts/frontiers.py results/records-sglang-29734.npz \
        --head results/head-27b-sglang-29734.safetensors --output results/frontiers-27b.json
"""
import argparse
import json

import numpy as np

from litjev.core.block import HeadMetadata, escalation_gain
from litjev.core.objective import outcome_labels, train_head
from litjev.data.collect import open_records
from litjev.serving import load_head
from litjev.train.training import features_for, holdout_mask

# The order `distribution_stats` writes them in, with the sign that makes a
# higher score mean "escalate".
STATS = {"max probability": (0, -1), "normalised entropy": (1, +1),
         "top-two margin": (2, -1), "concentration": (3, -1), "log K": (4, +1)}


def frontier(score, fast, slow, tokens):
    """(tokens, accuracy) at every cut of this score, cheapest first.

    One descending sort and two cumulative sums give every `k` at once. Ties are
    broken by a stable sort rather than by the implementation's order, so a block
    of equal scores does not make the curve depend on the machine.
    """
    order = np.argsort(-score, kind="stable")
    cost = np.concatenate([[0.0], np.cumsum(tokens[order])]) / len(score)
    gained = np.concatenate([[0.0], np.cumsum(slow[order].astype(float)
                                              - fast[order].astype(float))])
    return cost, fast.mean() + gained / len(score)


def read_at(cost, accuracy, budgets):
    """Accuracy at each token budget, interpolated along one frontier.

    Beyond the frontier's own end the router cannot spend more, so it holds at
    its last point rather than being extrapolated.
    """
    return np.interp(budgets, cost, accuracy, left=accuracy[0], right=accuracy[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records")
    parser.add_argument("--head", help="a trained head to serve as-is; omit to refit")
    parser.add_argument("--evaluate",
                        help="a second record set to score on, for a benchmark the head "
                             "only transfers to. Fitting then uses `records`' training "
                             "categories and every item here is evaluated.")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--seeds", type=int, default=5,
                        help="refits, used only when --head is omitted")
    parser.add_argument("--grid", type=int, default=41,
                        help="token budgets between 0 and the slow-only cost")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    records, metadata = open_records(arguments.records)
    position = list(metadata["feature_layers"]).index(arguments.layer)
    fast = np.asarray(records["fast_correct"], dtype=bool)
    slow = np.asarray(records["slow_correct"], dtype=bool)
    tokens = np.asarray(records["thinking_tokens"], dtype=np.float64)
    labels = outcome_labels(fast, slow)
    stats = np.asarray(records["stats"], dtype=np.float64)
    features = features_for(records, (position,))
    stats_block = features_for(records, ())
    held, level = holdout_mask(records["category"], 0.25, 0)
    test = np.asarray(held, dtype=bool)
    train = ~test
    if arguments.evaluate:
        # A transfer benchmark: the head is fitted on the same training
        # categories and neither side has seen these items.
        target, target_meta = open_records(arguments.evaluate)
        target_position = list(target_meta["feature_layers"]).index(arguments.layer)
        if target_meta.get("readout", "coded") != metadata.get("readout", "coded"):
            raise SystemExit("the two record sets were collected under different readouts")
        ft = np.asarray(target["fast_correct"], dtype=bool)
        st = np.asarray(target["slow_correct"], dtype=bool)
        tt = np.asarray(target["thinking_tokens"], dtype=np.float64)
        eval_features = features_for(target, (target_position,))
        eval_stats = np.asarray(target["stats"], dtype=np.float64)
        eval_block = {"features": eval_features,
                      "stats_block": features_for(target, ())}
    else:
        ft, st, tt = fast[test], slow[test], tokens[test]
        eval_features, eval_stats = features[test], stats[test]
        eval_block = {"features": features[test], "stats_block": stats_block[test]}
    n = len(ft)

    print(f"{arguments.records}: {metadata['model_id'].split('/')[-1]}, "
          f"budget {metadata.get('budget')}, layer {arguments.layer}, split by {level}")
    print(f"{n} held-out: fast {ft.mean():.4f}, slow {st.mean():.4f}, "
          f"slow tokens {tt.mean():.0f}/q\n")

    curves = {}
    if arguments.head:
        head = load_head(arguments.head)
        gains = np.array([escalation_gain(row) for row in head.predict(eval_features)])
        curves["HTM head (served)"] = [frontier(gains, ft, st, tt)]
        print(f"served head: {arguments.head.split('/')[-1]}, "
              f"layer {head.metadata.feature_layers}, pca {head.metadata.pca_dim}")
    else:
        shapes = {"HTM head (layer -1, raw)": dict(width=256, pca=0, layers=(arguments.layer,),
                                                   block=features, key="features"),
                  "HTM head (layer -1, PCA-64)": dict(width=64, pca=64, layers=(arguments.layer,),
                                                      block=features, key="features"),
                  "stats-only head": dict(width=32, pca=0, layers=(), block=stats_block,
                                          key="stats_block")}
        for name, shape in shapes.items():
            runs = []
            for seed in range(arguments.seeds):
                block, _ = train_head(
                    HeadMetadata(model_id=metadata["model_id"], revision=metadata["revision"],
                                 hidden_size=int(metadata["hidden_size"]),
                                 feature_layers=shape["layers"], hidden_width=shape["width"],
                                 pca_dim=shape["pca"]),
                    shape["block"][train], labels[train], epochs=30, seed=seed)
                gains = np.array([escalation_gain(row) for row in
                                  block.predict(eval_block[shape["key"]])])
                runs.append(frontier(gains, ft, st, tt))
            curves[name] = runs

    for name, (index, sign) in STATS.items():
        curves[name] = [frontier(sign * eval_stats[:, index], ft, st, tt)]

    budgets = np.linspace(0.0, float(tt.mean()), arguments.grid)
    # Random spends a share of the slow-only cost and gains that share of the
    # endpoint difference, which is the straight line and needs no sweep.
    share = budgets / tt.mean()
    random = ft.mean() + share * (st.mean() - ft.mean())

    read = {name: np.mean([read_at(cost, accuracy, budgets) for cost, accuracy in runs], axis=0)
            for name, runs in curves.items()}
    read["random"] = random

    order = sorted(read, key=lambda name: -float(np.trapezoid(read[name], budgets)))
    print("Accuracy at a shared token budget, read off each router's own frontier")
    print(f"{'tokens/q':>9s} {'share':>6s} " + " ".join(f"{name[:13]:>13s}" for name in order))
    for index in range(0, len(budgets), max(1, len(budgets) // 10)):
        print(f"{budgets[index]:9.0f} {share[index]:6.0%} "
              + " ".join(f"{read[name][index]:13.3f}" for name in order))

    print("\nArea under the accuracy-token curve, normalised by the budget range")
    span = budgets[-1] - budgets[0]
    areas = {name: float(np.trapezoid(read[name], budgets)) / span for name in order}
    best = max(areas.values())
    for name in order:
        print(f"  {name:28s} {areas[name]:.4f}  ({areas[name] - best:+.4f})")

    print("\nBudgets at which each router leads, of "
          f"{len(budgets)} on the grid")
    stacked = np.vstack([read[name] for name in order])
    winners = np.asarray(order)[stacked.argmax(axis=0)]
    for name in order:
        print(f"  {name:28s} {int((winners == name).sum()):3d}")

    if arguments.output:
        from litjev.run.streaming import write_json_atomic
        write_json_atomic(arguments.output, {
            "records": arguments.records, "head": arguments.head,
            "model_id": metadata["model_id"], "budget": metadata.get("budget"),
            "held_out": n, "fast": float(ft.mean()), "slow": float(st.mean()),
            "slow_tokens": float(tt.mean()), "budgets": budgets.tolist(),
            "accuracy": {name: read[name].tolist() for name in read},
            "area": areas})
        print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
