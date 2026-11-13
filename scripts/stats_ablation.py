"""What the eight statistics add on top of the hidden state, at layer 24 behind PCA-64.

Table~\\ref{tab:features} varies what the head reads only by *removing* the hidden
state: its `Dim` column is $2560+8$ or $64+8$, so every row with a hidden state also
carries the statistics, and the stats-only row carries them alone. Nothing in it
isolates the hidden state, so the marginal value of the eight statistics given it is
untested.

This adds the missing cell. The statistics are **zeroed rather than dropped**:
`STATS_DIM` is baked into `HeadMetadata.input_dim` and the forward pass splits its
input at `hidden_dim`, so removing the columns would mean changing a library other
people's runs go through. A constant column is given unit scale by
`constant_safe_scale`, not amplified, so zeroing leaves the architecture, the
parameter count and the code path identical and takes only the information away.

All three configurations are fitted here in one pass rather than read from the
published table, because `experiments/EXP.md` records that the selection sweep no
longer reproduces that table's numbers under the current code -- the chosen candidate
flips and the PCA rows moved when SVD signs were pinned. A new number set beside an
old one would be two code paths, not one ablation.

    uv run python scripts/stats_ablation.py results/records-4b-27063.npz
"""
import argparse

import numpy as np

from litjev.core.block import HeadMetadata
from litjev.core.objective import outcome_labels, train_head
from litjev.train.training import (DEFAULT_LAMBDAS, evaluate_head, features_for,
                                   group_folds, holdout_mask)
from litjev.data.collect import open_records


def probe(meta, features, labels, fast, slow, train_index, folds, test, seed, epochs):
    """One candidate's cross-validated and held-out AUROCs, as the sweep computes them."""
    fold_gain, fold_fast = [], []
    for fold in range(int(folds.max()) + 1):
        validation_rows = train_index[folds == fold]
        fit_rows = train_index[folds != fold]
        if len(validation_rows) == 0 or len(fit_rows) < 2:
            continue
        head, _ = train_head(meta, features[fit_rows], labels[fit_rows],
                             epochs=epochs, seed=seed)
        scored = evaluate_head(head, features[validation_rows], fast[validation_rows],
                               slow[validation_rows], DEFAULT_LAMBDAS)
        fold_gain.append(scored["auroc_gain_vs_helps"])
        fold_fast.append(scored["auroc_fast_correct"])
    head, _ = train_head(meta, features[train_index], labels[train_index],
                         epochs=epochs, seed=seed)
    held = evaluate_head(head, features[test], fast[test], slow[test], DEFAULT_LAMBDAS)
    return (float(np.nanmean(fold_gain)), float(np.nanmean(fold_fast)),
            float(held["auroc_gain_vs_helps"]), float(held["auroc_fast_correct"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--pca", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10, help="probe_epochs of the sweep")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    records, metadata = open_records(arguments.records)
    layers = tuple(metadata["feature_layers"])
    if arguments.layer not in layers:
        raise SystemExit(f"{arguments.records} carries layers {layers}, not {arguments.layer}")
    position = layers.index(arguments.layer)

    fast = np.asarray(records["fast_correct"], bool)
    slow = np.asarray(records["slow_correct"], bool)
    labels = outcome_labels(fast, slow)
    test, level = holdout_mask(records["category"], 0.25, arguments.seed)
    train_index = np.flatnonzero(~test)
    folds = group_folds(np.asarray(records["category"])[~test], 4, arguments.seed + 1)

    base = {"model_id": metadata["model_id"], "revision": metadata["revision"],
            "hidden_size": int(metadata["hidden_size"]),
            "readout": metadata.get("readout", "coded")}
    hidden_and_stats = features_for(records, (position,))
    stats_width = hidden_and_stats.shape[1] - int(metadata["hidden_size"])
    hidden_only = hidden_and_stats.copy()
    hidden_only[:, -stats_width:] = 0.0                 # the information, not the width
    stats_block = features_for(records, ())

    print(f"{arguments.records}: {metadata['model_id'].split('/')[-1]}, "
          f"layers {layers}, split by {level}, {len(train_index)} train, "
          f"{int(test.sum())} test, {stats_width} statistics\n")

    # Every configuration the table reports, fitted in one pass. Mixing a new row
    # measured today with rows measured on the code that produced the published
    # table would be two measurements in one column; refitting all of them is what
    # makes the new row comparable with the rest.
    runs = [("stats only", stats_block,
             HeadMetadata(**base, feature_layers=(), hidden_width=32, pca_dim=0))]
    for layer in layers:
        block = features_for(records, (layers.index(layer),))
        runs.append((f"layer {layer}, raw", block,
                     HeadMetadata(**base, feature_layers=(layer,), hidden_width=256,
                                  pca_dim=0)))
        runs.append((f"layer {layer}, PCA-{arguments.pca}", block,
                     HeadMetadata(**base, feature_layers=(layer,), hidden_width=64,
                                  pca_dim=arguments.pca)))
    # The ablation is run at both bottlenecks so it pairs with whichever row the
    # sweep selects, rather than only with the one that happened to win before.
    runs.append((f"layer {arguments.layer}, raw, no stats", hidden_only,
                 HeadMetadata(**base, feature_layers=(arguments.layer,), hidden_width=256,
                              pca_dim=0)))
    runs.append((f"layer {arguments.layer}, PCA-{arguments.pca}, no stats", hidden_only,
                 HeadMetadata(**base, feature_layers=(arguments.layer,), hidden_width=64,
                              pca_dim=arguments.pca)))
    print(f"{'configuration':38s} {'CV rescue':>10s} {'CV right':>9s} "
          f"{'test rescue':>12s} {'test right':>11s}")
    out = {}
    for name, features, meta in runs:
        cv_gain, cv_fast, t_gain, t_fast = probe(
            meta, features, labels, fast, slow, train_index, folds, test,
            arguments.seed, arguments.epochs)
        out[name] = (cv_gain, t_gain)
        print(f"{name:38s} {cv_gain:10.4f} {cv_fast:9.4f} {t_gain:12.4f} {t_fast:11.4f}")

    full = out[f"layer {arguments.layer}, raw"]
    bare = out[f"layer {arguments.layer}, raw, no stats"]
    only = out["stats only"]
    print()
    print(f"\nwhat the {stats_width} statistics add on top of the hidden state:"
          f" CV {full[0] - bare[0]:+.4f}, test {full[1] - bare[1]:+.4f}")
    print(f"what the hidden state adds on top of the statistics:"
          f" CV {full[0] - only[0]:+.4f}, test {full[1] - only[1]:+.4f}")


if __name__ == "__main__":
    main()
