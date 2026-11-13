"""Where the value is lost: in the estimate, or in the decision?

Three nulls in a row asked for a reason rather than another experiment. This
splits the distance between a routed agent and a perfect one into two parts.

  oracle            what a router that knew both outcomes would reach
  offline optimal   the head's probabilities, allocated with full lookahead
  online DP         the same probabilities, rationed step by step
  online myopic     the same probabilities, spent as soon as they look good

offline optimal to oracle is the estimation gap: what the head does not know.
online DP to offline optimal is the optimisation gap: what not seeing the rest
of the episode costs. A loss trained for the decision can only attack the
second. If the first is much larger, no loss will show a difference, which is
what Section 5 reports.
"""
import os
import sys

import numpy as np

from hde._legacy.heads import CDPOConfig, HeadMetadata, outcome_labels, train_head
from hde.data.collect import open_records
from hde.episodes import (
    form_episodes,
    greedy_progress_allocation,
    mode_probabilities,
    oracle_allocation,
    realised_progress,
)
from hde.online import online_thresholds, run_online, solve_values
from hde.run.streaming import write_json_atomic
from hde.train.training import features_for, holdout_mask

records, meta = open_records(sys.argv[1])
position = list(meta["feature_layers"]).index(24)
LENGTH, seeds = int(os.environ.get("LENGTH", "5")), int(os.environ.get("SEEDS", "20"))

fast = np.asarray(records["fast_correct"], dtype=bool)
slow = np.asarray(records["slow_correct"], dtype=bool)
test, _ = holdout_mask(records["category"], 0.25, seed=0)
features = features_for(records, (position,))
head_meta = HeadMetadata(model_id=meta["model_id"], revision=meta["revision"],
                         hidden_size=int(meta["hidden_size"]), feature_layers=(24,),
                         hidden_width=64, pca_dim=64)
PLAIN_CE = CDPOConfig(policy_weight=0.0, calibration_weight=0.0,
                      anchor_weight=1.0, weighted_anchor=False)
fast_t, slow_t = fast[test], slow[test]

rows = {}
for seed in range(seeds):
    head, _ = train_head(head_meta, features[~test], outcome_labels(fast, slow)[~test],
                         epochs=30, seed=seed, objective="cdpo", cdpo=PLAIN_CE)
    # The online rule is fitted on the training split: it may not see the episodes it runs on.
    train_fast, train_slow = mode_probabilities(head.predict(features[~test]))
    p_fast, p_slow = mode_probabilities(head.predict(features[test]))
    episodes = form_episodes(records["category"][test], length=LENGTH, seed=seed)
    for budget in range(1, LENGTH):
        thresholds = online_thresholds(solve_values(train_fast, train_slow, LENGTH, budget))
        for name, value in (
            ("online myopic", run_online(episodes, p_fast, p_slow, fast_t, slow_t, budget)),
            ("online DP", run_online(episodes, p_fast, p_slow, fast_t, slow_t, budget, thresholds)),
            ("offline optimal", realised_progress(
                episodes, greedy_progress_allocation(episodes, p_fast, p_slow, budget),
                fast_t, slow_t)),
            ("oracle", realised_progress(
                episodes, oracle_allocation(episodes, fast_t, slow_t, budget), fast_t, slow_t)),
        ):
            rows.setdefault((budget, name), []).append(value)

names = ("online myopic", "online DP", "offline optimal", "oracle")
print(f"episode length {LENGTH}, {seeds} seeds; mean steps completed before the first error\n")
print(f"{'budget':>7}" + "".join(f"{n:>21}" for n in names) +
      f"{'optimisation gap':>20}{'estimation gap':>18}{'ratio':>8}")
for budget in range(1, LENGTH):
    cells = [f"{np.mean(rows[(budget, n)]):.4f}±{np.std(rows[(budget, n)], ddof=1):.4f}"
             for n in names]
    optimisation = np.array(rows[(budget, "offline optimal")]) - np.array(rows[(budget, "online DP")])
    estimation = np.array(rows[(budget, "oracle")]) - np.array(rows[(budget, "offline optimal")])
    # The optimisation gap can be zero when the budget is loose enough that
    # lookahead buys nothing; a ratio is then meaningless rather than large.
    ratio = (estimation.mean() / optimisation.mean()
             if optimisation.mean() > 1e-6 else float("nan"))
    print(f"{budget:>7}" + "".join(f"{c:>21}" for c in cells) +
          f"{optimisation.mean():>+20.4f}{estimation.mean():>+18.4f}{ratio:>8.1f}")

write_json_atomic(sys.argv[2] if len(sys.argv) > 2 else "results/online-gap.json",
                  {f"{k[0]}|{k[1]}": v for k, v in rows.items()})
