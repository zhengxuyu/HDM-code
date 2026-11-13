"""Does the coupling in an agentic episode change what a router should do?

Before asking whether a decision-focused loss helps, check that the decision
problem is actually different. Same head, same budget, same predictions: only
the score the allocation ranks by changes. If the multiplicative score does not
beat the additive one on an all-steps-correct reward, there is no coupling for
any loss to exploit and the rest of the experiment is pointless.
"""
import os
import sys

import numpy as np

from hde._legacy.heads import CDPOConfig, HeadMetadata, outcome_labels, train_head
from hde.data.collect import open_records
from hde.episodes import (
    additive_score,
    allocate,
    episode_success,
    form_episodes,
    multiplicative_score,
    oracle_allocation,
    step_accuracy,
)
from hde.run.streaming import write_json_atomic
from hde.train.training import features_for, holdout_mask

records, meta = open_records(sys.argv[1])
LAYER, PCA, EPOCHS = 24, 64, 30
position = list(meta["feature_layers"]).index(LAYER)

fast = np.asarray(records["fast_correct"], dtype=bool)
slow = np.asarray(records["slow_correct"], dtype=bool)
labels = outcome_labels(fast, slow)
test, _ = holdout_mask(records["category"], 0.25, seed=0)
features = features_for(records, (position,))
head_meta = HeadMetadata(model_id=meta["model_id"], revision=meta["revision"],
                         hidden_size=int(meta["hidden_size"]), feature_layers=(LAYER,),
                         hidden_width=64, pca_dim=PCA)
# The single-turn winner: full cross-entropy on the four-way outcome, unweighted.
PLAIN_CE = CDPOConfig(policy_weight=0.0, calibration_weight=0.0,
                      anchor_weight=1.0, weighted_anchor=False)

LENGTHS = (2, 3, 4)
seeds = int(os.environ.get("SEEDS", "10"))
rows = {}
for seed in range(seeds):
    head, _ = train_head(head_meta, features[~test], labels[~test], epochs=EPOCHS,
                         seed=seed, objective="cdpo", cdpo=PLAIN_CE)
    outcomes = head.predict(features[test])
    fast_t, slow_t = fast[test], slow[test]
    scores = {"additive": additive_score(outcomes), "multiplicative": multiplicative_score(outcomes)}
    for length in LENGTHS:
        episodes = form_episodes(records["category"][test], length=length, seed=seed)
        rng = np.random.default_rng(1000 + seed)
        scores["random"] = rng.random(len(fast_t))
        for budget in range(length + 1):
            for name, score in scores.items():
                escalate = allocate(episodes, score, budget)
                key = (length, budget, name)
                rows.setdefault(key, []).append(
                    (episode_success(episodes, escalate, fast_t, slow_t),
                     step_accuracy(escalate, fast_t, slow_t)))
            oracle = oracle_allocation(episodes, fast_t, slow_t, budget)
            rows.setdefault((length, budget, "oracle"), []).append(
                (episode_success(episodes, oracle, fast_t, slow_t),
                 step_accuracy(oracle, fast_t, slow_t)))

print(f"{seeds} seeds; episode succeeds only if every step is right\n")
for length in LENGTHS:
    n_ep = len(form_episodes(records["category"][test], length=length, seed=0))
    print(f"--- episode length {length}  ({n_ep} episodes) ---")
    print(f"{'budget':>7} " + "".join(f"{n:>26}" for n in
                                      ("additive", "multiplicative", "random", "oracle")))
    for budget in range(length + 1):
        cells = []
        for name in ("additive", "multiplicative", "random", "oracle"):
            a = np.array(rows[(length, budget, name)])
            cells.append(f"{a[:,0].mean():.4f}±{a[:,0].std(ddof=1):.4f}"
                         if len(a) > 1 else f"{a[0,0]:.4f}")
        print(f"{budget:>7} " + "".join(f"{c:>26}" for c in cells))
    print()

write_json_atomic(sys.argv[2] if len(sys.argv) > 2 else "results/agentic.json",
                  {(f"{k[0]}|{k[1]}|{k[2]}"): v for k, v in rows.items()})
