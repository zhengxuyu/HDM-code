"""Does training for the episode reward beat training for likelihood?

The baseline gets everything the proposed method gets except the loss: the same
features, the same bottleneck, the same early stopping, and the same
position-aware allocation rule at evaluation. The single-turn comparison failed
because the baseline was handicapped by a class weighting; nothing here is
allowed to handicap it.
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
from hde.run.streaming import write_json_atomic
from hde.train.episode_training import EpisodeCDPOConfig, train_episode_head
from hde.train.training import features_for, holdout_mask

records, meta = open_records(sys.argv[1])
LAYER, EPOCHS = 24, 30
position = list(meta["feature_layers"]).index(LAYER)
LENGTH = int(os.environ.get("LENGTH", "5"))
seeds = int(os.environ.get("SEEDS", "10"))
WEIGHTS = tuple(float(w) for w in os.environ.get("WEIGHTS", "0,1,5,20").split(","))

fast = np.asarray(records["fast_correct"], dtype=bool)
slow = np.asarray(records["slow_correct"], dtype=bool)
labels = outcome_labels(fast, slow)
test, _ = holdout_mask(records["category"], 0.25, seed=0)
train = ~test
features = features_for(records, (position,))
head_meta = HeadMetadata(model_id=meta["model_id"], revision=meta["revision"],
                         hidden_size=int(meta["hidden_size"]), feature_layers=(LAYER,),
                         hidden_width=64, pca_dim=64)
PLAIN_CE = CDPOConfig(policy_weight=0.0, calibration_weight=0.0,
                      anchor_weight=1.0, weighted_anchor=False)

train_index = np.flatnonzero(train)
rows = {}
for seed in range(seeds):
    # Training episodes are formed among training questions only; the indices
    # below are into the full array, so the head never sees a test question.
    local = form_episodes(records["category"][train], length=LENGTH, seed=seed)
    train_episodes = [train_index[e] for e in local]

    heads = {"cross-entropy": train_head(head_meta, features[train], labels[train], epochs=EPOCHS,
                                         seed=seed, objective="pmpo", pmpo=PLAIN_CE)[0]}
    # Sweep how hard the episode reward pulls against the anchor. Weight 0 is the
    # same anchor as the baseline and should land on it, which is the control that
    # says the episode trainer itself introduces no difference.
    for weight in WEIGHTS:
        heads[f"episode PMPO w={weight}"] = train_episode_head(
            head_meta, features, fast, slow, train_episodes,
            config=EpisodeCDPOConfig(progress_weight=weight), epochs=EPOCHS, seed=seed)[0]
    test_episodes = form_episodes(records["category"][test], length=LENGTH, seed=seed)
    fast_t, slow_t = fast[test], slow[test]
    for name, head in heads.items():
        p_fast, p_slow = mode_probabilities(head.predict(features[test]))
        for budget in range(LENGTH + 1):
            escalate = greedy_progress_allocation(test_episodes, p_fast, p_slow, budget)
            rows.setdefault((budget, name), []).append(
                realised_progress(test_episodes, escalate, fast_t, slow_t))
    for budget in range(LENGTH + 1):
        rows.setdefault((budget, "oracle"), []).append(
            realised_progress(test_episodes,
                              oracle_allocation(test_episodes, fast_t, slow_t, budget),
                              fast_t, slow_t))

names = ("cross-entropy",) + tuple(f"episode PMPO w={w}" for w in WEIGHTS) + ("oracle",)
print(f"episode length {LENGTH}, {seeds} seeds; value = mean steps completed before the first error")
print("both heads allocate with the same position-aware rule\n")
def paired(a, b):
    d = np.array(rows[a]) - np.array(rows[b])
    if d.std(ddof=1) == 0:
        return "  0"
    t_stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return f"{d.mean():+.4f} (t={t_stat:+.2f})"

print(f"{'budget':>7}" + "".join(f"{n:>22}" for n in names))
for budget in range(LENGTH + 1):
    cells = []
    for n in names:
        a = np.array(rows[(budget, n)])
        cells.append(f"{a.mean():.4f}\u00b1{a.std(ddof=1):.4f}")
    print(f"{budget:>7}" + "".join(f"{c:>22}" for c in cells))

print("\npaired against cross-entropy, same seeds, same allocation rule")
print(f"{'budget':>7}" + "".join(f"{f'w={w}':>22}" for w in WEIGHTS))
for budget in range(1, LENGTH):
    print(f"{budget:>7}" + "".join(
        f"{paired((budget, f'episode PMPO w={w}'), (budget, 'cross-entropy')):>22}"
        for w in WEIGHTS))

write_json_atomic(sys.argv[2] if len(sys.argv) > 2 else "results/agentic-train.json",
                  {f"{k[0]}|{k[1]}": v for k, v in rows.items()})
