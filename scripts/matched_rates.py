"""Head against random and against confidence, all at the same escalation rates,
all on one measurement: the collection's own 346 held-out questions.

Table 8's head column comes from the benchmark run and its random column is a
closed form, so neither needs records. A confidence column does, and the records
are the collection, whose 27B slow endpoint is not the benchmark's. Everything
here is therefore read off the collection alone, so the comparison is internal.
"""
import numpy as np

from hde.core.block import HeadMetadata, escalation_gain
from hde.core.objective import outcome_labels, train_head
from hde.data.collect import open_records
from hde.train.training import features_for, holdout_mask

import argparse

PATH = None  # set from the command line
STATS = {"max prob": (0, -1), "entropy": (1, +1), "margin": (2, -1), "concentr": (3, -1)}

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("records")
parser.add_argument("--rates", default="0.99,0.63,0.48,0.47,0.38")
parser.add_argument("--seeds", type=int, default=5)
arguments = parser.parse_args()
RATES = [float(v) for v in arguments.rates.split(",")]
SEEDS = arguments.seeds

records, metadata = open_records(arguments.records)
position = list(metadata["feature_layers"]).index(-1)
fast = np.asarray(records["fast_correct"], bool)
slow = np.asarray(records["slow_correct"], bool)
tokens = np.asarray(records["thinking_tokens"], float)
labels = outcome_labels(fast, slow)
stats = np.asarray(records["stats"], float)
features = features_for(records, (position,))
stats_block = features_for(records, ())
held, _ = holdout_mask(records["category"], 0.25, 0)
test = np.asarray(held, bool)
train = ~test
ft, st, tt = fast[test], slow[test], tokens[test]
n = int(test.sum())

shapes = {
    "head raw":  dict(width=256, pca=0, layers=(-1,), block=features),
    "head PCA":  dict(width=64, pca=64, layers=(-1,), block=features),
    "stats-only": dict(width=32, pca=0, layers=(), block=stats_block),
}
gains = {name: [] for name in shapes}
for seed in range(SEEDS):
    for name, shape in shapes.items():
        head, _ = train_head(
            HeadMetadata(model_id=metadata["model_id"], revision=metadata["revision"],
                         hidden_size=int(metadata["hidden_size"]),
                         feature_layers=shape["layers"], hidden_width=shape["width"],
                         pca_dim=shape["pca"]),
            shape["block"][train], labels[train], epochs=30, seed=seed)
        gains[name].append(np.array([escalation_gain(row)
                                     for row in head.predict(shape["block"][test])]))

def at_rate(score, rate):
    k = int(round(rate * n))
    pick = np.zeros(n, bool)
    pick[np.argsort(-score, kind="stable")[:k]] = True
    return float(np.where(pick, st, ft).mean()), float(np.where(pick, tt, 0.0).mean())

print(f"{n} held-out, fast {ft.mean():.4f}, slow {st.mean():.4f}, "
      f"slow tokens {tt.mean():.0f}/q, {SEEDS} seeds for the heads\n")
names = list(shapes) + list(STATS)
print(f"{'rate':>5s} {'random':>7s} " + " ".join(f"{m:>9s}" for m in names))
for rate in RATES:
    random = ft.mean() + rate * (st.mean() - ft.mean())
    cells = []
    for name in shapes:
        values = [at_rate(g, rate)[0] for g in gains[name]]
        cells.append(f"{np.mean(values):.3f}" + (f"±{np.std(values, ddof=1):.3f}"
                                                 if SEEDS > 1 else ""))
    for name, (index, sign) in STATS.items():
        cells.append(f"{at_rate(sign * stats[test, index], rate)[0]:.3f}      ")
    print(f"{rate:5.0%} {random:7.3f} " + " ".join(f"{c:>9s}" for c in cells))
