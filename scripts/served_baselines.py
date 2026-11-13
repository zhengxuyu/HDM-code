"""Parameter-free routers against the head, at the condition the paper serves.

`scripts/free_baselines.py` answered this on the 4B at B=1024 reading layer 24,
which is not a condition any published row runs at. This script answers it at the
served one -- Qwen3.8-27B, B=4096, layer -1 -- and changes two things about the
comparison besides the condition.

**All five statistics, not one.** The earlier sweep at this budget compared the
head against max probability alone, while on the 4B table the other three
(normalised entropy, top-two margin, concentration) scored at or above it. `log K`
is the control: it is a surface property of the question rather than of the
distribution, and it should score at chance.

**The threshold has to find its own operating point.** Every comparison we have
run handed it one: the sweep matched escalation rates, and `experiments/EXP.md`'s
margin arms chose tau from five values on the test set. That is the half of the
decision the head does for free -- it emits a gain on the same scale as the price,
so a deployment sets lambda and gets a rate -- and a threshold emits an ordering
with no price in it. So for each lambda both sides are priced:

    head        escalate question i when gain_i > lambda
    threshold   choose tau on the TRAINING categories to maximise the same
                expected reward, then apply that tau to the held-out ones

The reward is the paper's, `u * exp(-lambda * T)` with `T = 1` for one escalation
(`objective.cdpo_loss`), so escalating question i pays `slow_i * exp(-lambda)` and
keeping the fast answer pays `fast_i`. Both sides maximise one scalar, and what
separates them is whether choosing the operating point is free.

The rate-matched column is kept beside it, because it is the generous reading and
the one the deleted appendix reported: hand the threshold the head's own held-out
escalation count and let it pick those questions by its score.

    uv run python scripts/served_baselines.py results/records-sglang-29734.npz \
        --layer -1 --output results/served-baselines.json
"""
import argparse
import json

import numpy as np

from hde.core.block import HeadMetadata, escalation_gain
from hde.core.block_utils import auroc
from hde.core.objective import outcome_labels, train_head
from hde.data.collect import open_records
from hde.run.streaming import write_json_atomic
from hde.train.training import features_for, holdout_mask

# The order `distribution_stats` writes them in.
STATS = {"max probability": 0, "normalised entropy": 1, "top-two margin": 2,
         "concentration": 3, "log K": 4}
# Higher must mean "escalate", so confidence-like statistics are negated.
SIGN = {"max probability": -1, "normalised entropy": +1, "top-two margin": -1,
        "concentration": -1, "log K": +1}

# The prices Table 1 reports, inside the finer sweep Figure 2 draws.
PAPER_PRICES = (-0.1, -0.05, 0.0, 0.05, 0.1)


def reward(escalate, fast, slow, lam):
    """Mean of the paper's reward over a set of questions.

    Escalating pays `slow * exp(-lambda)` and keeping the fast answer pays `fast`,
    which is `cdpo_loss`'s reward with T = 1 for one escalation. This is the
    quantity both sides of the comparison maximise, so it is written once.
    """
    return float(np.where(escalate, slow * np.exp(-lam), fast).mean())


def priced_threshold(score, fast, slow, lam):
    """The tau a deployment would pick for this price, chosen on training data only.

    Only the cuts between distinct scores are legal thresholds, and the set a cut
    escalates is the top k by score, so one descending sort and two cumulative sums
    give the objective at every candidate at once. Returns a tau for which
    `score > tau` is that best set: the next score down, or `-inf` for all of them.
    """
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    # The reward of escalating the top k: the slow outcome, discounted, on those,
    # and the fast outcome on the rest. k = 0 keeps every fast answer.
    slow_gain = np.concatenate([[0.0], np.cumsum(slow[order])]) * np.exp(-lam)
    fast_kept = fast.sum() - np.concatenate([[0.0], np.cumsum(fast[order])])
    objective = (slow_gain + fast_kept) / len(score)
    # A cut inside a block of equal scores is not reachable by any tau, so only
    # the boundaries between distinct values are candidates.
    legal = np.concatenate([[True], sorted_score[:-1] > sorted_score[1:], [True]])
    objective = np.where(legal, objective, -np.inf)
    best = int(np.argmax(objective))
    return -np.inf if best == len(score) else float(sorted_score[best])


def arm(escalate, fast, slow, tokens, lam):
    """What one router at one price costs and buys, on the held-out questions."""
    return {"escalation": float(escalate.mean()),
            "accuracy": float(np.where(escalate, slow, fast).mean()),
            "tokens": float(np.where(escalate, tokens, 0.0).mean()),
            "reward": reward(escalate, fast, slow, lam)}


def top_k(score, k):
    """The k highest scores, ties broken by a fixed order rather than the sort's.

    `np.argsort` is not stable by default and inside a block of equal scores its
    order is the implementation's, which is how `log K`'s accuracy row came out
    differently on two machines.
    """
    chosen = np.zeros(len(score), dtype=bool)
    chosen[np.argsort(-score, kind="stable")[:k]] = True
    return chosen


def load(path, layer):
    """One record set, as the arrays this comparison needs."""
    records, metadata = open_records(path)
    position = list(metadata["feature_layers"]).index(layer)
    fast = np.asarray(records["fast_correct"], dtype=bool)
    slow = np.asarray(records["slow_correct"], dtype=bool)
    return {
        "path": path, "metadata": metadata,
        "fast": fast, "slow": slow,
        "tokens": np.asarray(records["thinking_tokens"], dtype=np.float64),
        "labels": outcome_labels(fast, slow),
        "stats": np.asarray(records["stats"], dtype=np.float64),
        "features": features_for(records, (position,)),
        "stats_block": features_for(records, ()),
        "category": np.asarray(records["category"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", help="the set the head and the taus are fitted on")
    parser.add_argument("--evaluate",
                        help="a second record set to evaluate on, for a benchmark the "
                             "head only transfers to. Fitting then uses `records`' "
                             "training categories and evaluation uses every item here.")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--seeds", type=int, default=20,
                        help="refits of anything trained; the statistics have nothing to refit")
    parser.add_argument("--prices", default="-0.1:1.0:0.01",
                        help="start:stop:step, inclusive of stop")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    source = load(arguments.records, arguments.layer)
    metadata = source["metadata"]
    held, level = holdout_mask(source["category"], 0.25, 0)
    fit = ~np.asarray(held, dtype=bool)

    if arguments.evaluate:
        # A transfer benchmark: the head and every tau are fitted on the same
        # training categories as before and neither side sees these items. The
        # statistics are read off this set's own distributions, as they would be
        # at serving time.
        target = load(arguments.evaluate, arguments.layer)
        keep = np.ones(len(target["fast"]), dtype=bool)
    else:
        target = source
        keep = np.asarray(held, dtype=bool)

    blocks = {
        "HDM head (layer -1, raw)": dict(width=256, pca=0, layers=(arguments.layer,),
                                         key="features"),
        "HDM head (layer -1, PCA-64)": dict(width=64, pca=64, layers=(arguments.layer,),
                                            key="features"),
        "stats-only head": dict(width=32, pca=0, layers=(), key="stats_block"),
    }

    start, stop, step = (float(value) for value in arguments.prices.split(":"))
    prices = np.round(np.arange(start, stop + step / 2, step), 10)

    ff, fs = source["fast"][fit], source["slow"][fit]
    ef = target["fast"][keep]
    es = target["slow"][keep]
    et = target["tokens"][keep]
    helps = ~ef & es

    print(f"fit on {arguments.records}: {metadata['model_id'].split('/')[-1]}, "
          f"budget {metadata.get('budget')}, layer {arguments.layer}, "
          f"{int(fit.sum())} questions outside the held-out categories (split by {level})")
    print(f"evaluate on {target['path']}: "
          f"{target['metadata']['model_id'].split('/')[-1]}, "
          f"budget {target['metadata'].get('budget')}, {int(keep.sum())} items")
    print(f"  fast {ef.mean():.3f}, slow {es.mean():.3f}, thinking helps "
          f"{helps.mean():.3f}, both wrong {(~ef & ~es).mean():.3f}, "
          f"slow tokens {et.mean():.0f}/q")
    print(f"{len(prices)} prices from {prices[0]} to {prices[-1]}, "
          f"{arguments.seeds} seeds for anything refit\n")

    # The statistics are read straight off the records, so they carry no seed.
    fit_scores = {name: SIGN[name] * source["stats"][fit, index]
                  for name, index in STATS.items()}
    eval_scores = {name: SIGN[name] * target["stats"][keep, index]
                   for name, index in STATS.items()}

    results = {}
    for name, score in eval_scores.items():
        results[name] = [{"auroc": float(auroc(score, helps)), "priced": [],
                          "fitted": [], "matched": []}]
    for name in list(blocks) + ["random"]:
        results[name] = []

    for seed in range(arguments.seeds):
        gains, gains_fit = {}, {}
        for name, shape in blocks.items():
            head_metadata = HeadMetadata(
                model_id=metadata["model_id"], revision=metadata["revision"],
                hidden_size=int(metadata["hidden_size"]),
                feature_layers=shape["layers"],
                hidden_width=shape["width"], pca_dim=shape["pca"])
            head, _ = train_head(head_metadata, source[shape["key"]][fit],
                                 source["labels"][fit], epochs=30, seed=seed)
            gains[name] = np.array([escalation_gain(row) for row in
                                    head.predict(target[shape["key"]][keep])])
            # The head's own score on the fitting set, so a tau can be chosen for
            # it the way the statistics get one: the two sides then differ in what
            # they read and not in how much fitting they are allowed.
            gains_fit[name] = np.array([escalation_gain(row) for row in
                                        head.predict(source[shape["key"]][fit])])
        rng = np.random.default_rng(seed)
        gains["random"] = rng.normal(size=int(keep.sum()))
        gains_fit["random"] = rng.normal(size=int(fit.sum()))

        for name, gain in gains.items():
            trial = {"auroc": float(auroc(gain, helps)), "priced": [], "fitted": [],
                     "matched": []}
            for lam in prices:
                # A gain is on the price's scale, so this rule needs no fitting.
                trial["priced"].append(arm(gain > lam, ef, es, et, lam))
                tau = priced_threshold(gains_fit[name], ff, fs, lam)
                trial["fitted"].append(arm(gain > tau, ef, es, et, lam))
                trial["matched"].append(None)
            results[name].append(trial)
        print(f"  seed {seed}", end="\r", flush=True)

    served = results["HDM head (layer -1, raw)"]
    reference = [float(np.mean([trial["priced"][index]["escalation"] for trial in served]))
                 for index in range(len(prices))]
    for name in eval_scores:
        trial = results[name][0]
        for index, lam in enumerate(prices):
            tau = priced_threshold(fit_scores[name], ff, fs, lam)
            trial["priced"].append(arm(eval_scores[name] > tau, ef, es, et, lam))
            trial["priced"][-1]["tau"] = tau
            # A statistic has no unfitted rule, so its fitted arm is its priced one.
            trial["fitted"].append(trial["priced"][-1])
            count = int(round(reference[index] * int(keep.sum())))
            trial["matched"].append(arm(top_k(eval_scores[name], count), ef, es, et, lam))

    def summarise(name, key, index, field):
        values = [trial[key][index][field] for trial in results[name]
                  if trial[key][index] is not None]
        array = np.asarray(values, dtype=float)
        return array.mean(), (array.std(ddof=1) if len(array) > 1 else 0.0)

    print("Routing AUROC on the evaluated items")
    order = sorted(results, key=lambda name: -np.mean([t["auroc"] for t in results[name]]))
    for name in order:
        values = np.asarray([trial["auroc"] for trial in results[name]])
        tail = (f"{values.mean():.4f}" if len(values) == 1
                else f"{values.mean():.4f} +- {values.std(ddof=1):.4f}")
        print(f"  {name:30s} {tail}")

    print(f"\nPriced: each router at the price, every tau chosen on the "
          f"{int(fit.sum())} fitting questions")
    for lam in PAPER_PRICES:
        index = int(np.argmin(np.abs(prices - lam)))
        print(f"\n  lambda = {prices[index]:+.2f}")
        print(f"    {'router':30s} {'esc':>7s} {'acc':>7s} {'tok/q':>8s} {'reward':>8s}")
        for name in order:
            esc, _ = summarise(name, "priced", index, "escalation")
            accuracy, spread = summarise(name, "priced", index, "accuracy")
            tok, _ = summarise(name, "priced", index, "tokens")
            value, _ = summarise(name, "priced", index, "reward")
            print(f"    {name:30s} {esc:6.1%} {accuracy:7.3f} {tok:8.1f} {value:8.4f}"
                  + (f"  (sd {spread:.3f})" if spread else ""))

    print("\nRate-matched: the statistics handed the served head's own escalation rate")
    for lam in PAPER_PRICES:
        index = int(np.argmin(np.abs(prices - lam)))
        print(f"\n  lambda = {prices[index]:+.2f}, rate {reference[index]:.1%}")
        print(f"    {'router':30s} {'esc':>7s} {'acc':>7s} {'tok/q':>8s} {'reward':>8s}")
        for name in list(eval_scores) + ["HDM head (layer -1, raw)"]:
            key = "matched" if name in eval_scores else "priced"
            esc, _ = summarise(name, key, index, "escalation")
            accuracy, _ = summarise(name, key, index, "accuracy")
            tok, _ = summarise(name, key, index, "tokens")
            value, _ = summarise(name, key, index, "reward")
            print(f"    {name:30s} {esc:6.1%} {accuracy:7.3f} {tok:8.1f} {value:8.4f}")

    print(f"\nOver all {len(prices)} prices, how often each statistic reaches the "
          "head's reward")
    for key, label in (("priced", "head priced at lambda, no fitting"),
                       ("fitted", "head thresholded on the fitting set, like the statistics")):
        print(f"  against the {label}")
        head_reward = np.asarray([[cell["reward"] for cell in trial[key]]
                                  for trial in results["HDM head (layer -1, raw)"]]
                                 ).mean(axis=0)
        for name in eval_scores:
            their = np.asarray([cell["reward"] for cell in results[name][0][key]])
            print(f"    {name:28s} ahead at {(their > head_reward).sum():3d}, "
                  f"behind at {(their < head_reward).sum():3d}, "
                  f"tied at {(their == head_reward).sum():3d} of {len(prices)}")

    if arguments.output:
        write_json_atomic(arguments.output, {
            "fit_records": arguments.records, "evaluate_records": target["path"],
            "layer": arguments.layer, "model_id": metadata["model_id"],
            "budget": metadata.get("budget"),
            "evaluate_model_id": target["metadata"]["model_id"],
            "evaluate_budget": target["metadata"].get("budget"),
            "evaluated": int(keep.sum()), "fitted_on": int(fit.sum()),
            "split_level": level, "seeds": arguments.seeds,
            "prices": prices.tolist(), "routers": results})
        print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
