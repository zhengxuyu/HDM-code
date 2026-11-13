#!/usr/bin/env python
"""What the discount does to the off-policy head update, on the recorded episodes.

The MiniWoB loop is collect, train, evaluate, and only the middle step is a
function of the reward. That step is CPU-seconds and the episodes it reads are on
disk, so the reward change can be measured here before any GPU time is spent --
and the answer decides what price range the cluster run should sweep, which is
the one thing the old range cannot tell us (see below).

**What this cannot show.** Whether routing got better. A recorded episode says
the mode it took was right or wrong; it does not say what the other mode would
have done, so no per-decision label for "thinking helped" exists here. That
comparison needs rollouts. What this does show is whether the update is informed
at all -- effective sample size, how much of the corpus carries any price signal,
how often the clip binds -- and where on the lambda axis the learned head
actually moves, which is a property of the trained head and needs no environment.

**Why the old price range is not reusable.** lambda is compared against the gain,
and the gain stopped being a probability difference and became a log ratio. The
difference is bounded by how much the fast mode gets wrong; the log ratio is not
bounded at all. So (0.0, 0.25) is not the same interval it was, and the three
recorded eval prices 0 / 0.1 / 0.2 are not the same three operating points. The
sweep below is what replaces guessing.

Usage:
    python scripts/reward_offline.py \
        --episodes experiments/browser-gate/results/browser-pmpo-click-c0783e7/episodes-round0.jsonl \
        --head demo/assets/head-27b-lastlayer.safetensors \
        --out experiments/browser-gate/results/reward-offline.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hde.core.block import DecisionHead, escalation_gain_batch
from slime_hdm import head_trainer
from slime_hdm.head_trainer import (
    HeadRLConfig,
    flatten_episodes,
    priced_value,
    train_head_on_episodes,
    train_head_on_smeared_returns,
)


# The pricing this replaced, kept here and nowhere else: it exists to be compared
# against, not to be callable from the library.
def linear_value(rewards, spent, prices):
    return rewards - prices * spent


def read_episodes(path):
    episodes, signature = [], None
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "signature" in record:
            signature = record["signature"]
        elif "terminal_reward" in record and record.get("head_features"):
            episodes.append(record)
    return episodes, signature


def price_response(head, features, prices):
    """The share of decisions the trained head escalates, at each price.

    This is the deployment-facing curve and the reason the script exists: it says
    which lambdas are operating points and which are degenerate.
    """
    with torch.no_grad():
        outcome = torch.softmax(head(torch.as_tensor(features)), dim=-1)
        gain = escalation_gain_batch(outcome).numpy()
    return gain, {str(price): float((gain > price).mean()) for price in prices}


def carries_price_signal(value_function, rewards, spent, prices):
    """The share of decisions whose value moves when the price does.

    Under the discount a failed episode is worth zero at every price, so its
    advantage is constant in lambda and the price term learns nothing from it.
    Under the subtraction every decision with an escalation moved. This is the
    cost of removing the inversion, measured rather than described.

    `value_function` is passed rather than looked up: the first version of this
    called the `priced_value` this module imported, which is the discount whatever
    the comparison had patched, and so reported one number for both arms.
    """
    low, high = min(prices), max(prices)
    moved = np.abs(value_function(rewards, spent, high)
                   - value_function(rewards, spent, low)) > 1e-12
    return float(moved.mean())


def run(episodes, metadata, config, seeds, prices):
    """Three arms. The first two differ only in how the price enters the reward.

    The third is the cross-entropy baseline, trained on the same episodes by the
    same file, and it is here to separate two explanations for a near-binary gain:
    a policy-gradient loss with nothing holding its four-way distribution to a
    proper scoring rule, or a corpus that genuinely says each decision was
    hopeless or certain. The baseline has the anchor and the same data, so if its
    gain is graded and the policy arms' are not, the loss is the cause.
    """
    out = {}
    for name, value_function in (("discount", priced_value), ("subtraction", linear_value)):
        original = head_trainer.priced_value
        head_trainer.priced_value = value_function
        try:
            rows = []
            for seed in seeds:
                head, report = train_head_on_episodes(metadata, episodes, config, seed=seed)
                features, _, _, rewards, spent, _ = flatten_episodes(episodes, config)
                gain, response = price_response(head, features, prices)
                rows.append({
                    "seed": seed,
                    "effective_fraction": report["effective_fraction"],
                    "clipped_fraction_mean": report["clipped_fraction_mean"],
                    "zero_gradient_fraction_mean": report["zero_gradient_fraction_mean"],
                    "advantage_spread": report["advantage_spread"],
                    "gain_mean": float(gain.mean()),
                    "gain_spread": float(gain.std()),
                    # A saturated head answers "obviously yes" or "obviously no"
                    # and lambda has no purchase on it. The quantiles say whether
                    # the price grid is in the wrong place or the head is in a
                    # corner, which the escalation rates alone cannot separate.
                    "gain_quantiles": {
                        str(q): float(np.quantile(gain, q))
                        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
                    },
                    "gain_within_the_price_grid": float(
                        ((gain > min(prices)) & (gain < max(prices))).mean()),
                    # The two places a log ratio piles up: the GAIN_EPS floor,
                    # where P(fast right) underflowed, and exactly zero, where the
                    # head says the two modes do equally well.
                    "at_the_eps_floor": float((gain > 13.8).mean()),
                    "exactly_zero": float((np.abs(gain) < 1e-9).mean()),
                    "escalation_rate_by_price": response,
                })
            features, _, _, rewards, spent, _ = flatten_episodes(episodes, config)
            out[name] = {
                "seeds": rows,
                "decisions_with_price_signal": carries_price_signal(
                    value_function, rewards, spent, prices),
            }
        finally:
            head_trainer.priced_value = original

    rows = []
    for seed in seeds:
        head, report = train_head_on_smeared_returns(metadata, episodes, config, seed=seed)
        features, _, _, _, _, _ = flatten_episodes(episodes, config)
        gain, response = price_response(head, features, prices)
        rows.append({
            "seed": seed,
            "gain_mean": float(gain.mean()),
            "gain_spread": float(gain.std()),
            "gain_quantiles": {str(q): float(np.quantile(gain, q))
                               for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)},
            "gain_within_the_price_grid": float(
                ((gain > min(prices)) & (gain < max(prices))).mean()),
            "at_the_eps_floor": float((gain > 13.8).mean()),
            "exactly_zero": float((np.abs(gain) < 1e-9).mean()),
            "cloned_escalation_rate": report["cloned_escalation_rate"],
            "escalation_rate_by_price": response,
        })
    out["cross_entropy_baseline"] = {"seeds": rows}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--head", required=True, help="read for its metadata only")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    # Spanning the gain's whole reachable span, which for a log ratio floored at
    # GAIN_EPS = 1e-6 is (-13.8, 13.8). A grid inside (0, 0.25) cannot see where
    # this head's gain actually sits; that was the first mistake made here.
    parser.add_argument("--prices", type=float, nargs="+",
                        default=[-1.0, 0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 13.0, 14.0])
    parser.add_argument("--lambda-range", type=float, nargs=2, default=[0.0, 0.25])
    args = parser.parse_args()

    episodes, signatures = [], []
    for path in args.episodes:
        found, signature = read_episodes(path)
        episodes += found
        signatures.append({"path": path, "episodes": len(found), "signature": signature})

    metadata = DecisionHead.load(args.head).metadata
    config = HeadRLConfig(lambda_range=tuple(args.lambda_range))
    result = run(episodes, metadata, config, args.seeds, args.prices)

    _, _, _, rewards, spent, episode_index = flatten_episodes(episodes, config)
    payload = {
        "schema": "litjev.reward-offline.v1",
        "sources": signatures,
        "episodes": int(len(episodes)),
        "decisions": int(len(rewards)),
        "returns": sorted({float(r) for r in rewards}),
        "episodes_succeeded": float(np.mean(
            [episode["terminal_reward"] for episode in episodes])),
        "escalations_per_episode": {
            "mean": float(np.mean([spent[episode_index == i][0]
                                   for i in range(len(episodes))])),
            "max": float(max(spent)),
        },
        "config": vars(config),
        "prices": args.prices,
        "results": result,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{len(episodes)} episodes, {len(rewards)} decisions, "
          f"{payload['episodes_succeeded']:.3f} succeeded")
    for name, block in result.items():
        if "decisions_with_price_signal" in block:
            print(f"\n{name}: {block['decisions_with_price_signal']:.1%} "
                  "of decisions carry a price signal")
        else:
            print(f"\n{name}:")
        rows = block["seeds"]
        for key in ("effective_fraction", "clipped_fraction_mean",
                    "zero_gradient_fraction_mean", "gain_spread",
                    "gain_within_the_price_grid", "at_the_eps_floor",
                    "exactly_zero"):
            values = [row.get(key) for row in rows]
            if any(value is None for value in values):
                continue
            print(f"  {key:30s} {np.mean(values):8.4f} +/- {np.std(values):.4f}")
        print("  gain quantiles, mean over seeds")
        for q in ("0.0", "0.1", "0.25", "0.5", "0.75", "0.9", "1.0"):
            values = [row["gain_quantiles"][q] for row in rows]
            print(f"    q{q:<5s} {np.mean(values):8.3f} +/- {np.std(values):.3f}")
        print("  escalation rate by price")
        for price in args.prices:
            values = [row["escalation_rate_by_price"][str(price)] for row in rows]
            print(f"    lambda={price:<6g} {np.mean(values):6.3f} +/- {np.std(values):.3f}"
                  f"   seeds {[round(v, 3) for v in values]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
