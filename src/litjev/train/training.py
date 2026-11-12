"""Train and evaluate the decision head from collected records.

Split by category so the reported numbers measure transfer to unseen subjects,
never memorization of the training questions.
"""

from __future__ import annotations

import hashlib
import math
from itertools import pairwise

import numpy as np

from litjev.core.block import (
    GAIN_EPS,
    HeadMetadata,
    escalation_gain,
    fast_confidence,
)
from litjev.core.block_utils import auroc, coverage_accuracy_curve
from litjev.core.objective import OBJECTIVES, normalise_objective, outcome_labels, train_head

DEFAULT_LAMBDAS = (-0.2, -0.1, -0.05, 0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def _hash_mask(keys, fraction, seed):
    scores = np.array(
        [
            int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            for key in keys
        ]
    )
    return scores < fraction


def holdout_mask(categories, fraction, seed=0):
    """Deterministic category-level holdout.

    Returns (mask, level). Level is "category" when whole categories are held out,
    or "example" when too few categories exist for that to leave both sides populated;
    the per-example split then measures within-subject generalization only.
    """
    categories = np.asarray(categories)
    if categories.size and any(categories):
        mask = _hash_mask(categories, fraction, seed)
        if mask.any() and not mask.all():
            return mask, "category"
    mask = _hash_mask(np.arange(len(categories)).astype(str), fraction, seed)
    if mask.all() or not mask.any():
        raise ValueError("Holdout split left one side empty; adjust --holdout-fraction or seed")
    return mask, "example"


def group_folds(categories, k, seed=0):
    """Fold id per example: whole categories share a fold; per-example when categories are few."""
    categories = np.asarray(categories)
    unique = sorted({str(c) for c in categories if str(c)})
    if len(unique) >= k:
        fold_of = {
            name: int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:8], 16) % k
            for name in unique
        }
        folds = np.array([fold_of[str(c)] for c in categories])
        if len(set(folds.tolist())) == k:
            return folds
    return np.arange(len(categories)) % k


def features_for(records, layer_positions):
    hidden = records["hidden"][:, list(layer_positions), :].astype(np.float32)
    return np.concatenate([hidden.reshape(len(hidden), -1), records["stats"]], axis=1)


def expected_calibration_error(confidence, correct, bins=10):
    """ECE with equal-width bins; empty bins contribute nothing."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in pairwise(edges):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            error += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(error)


def gain_calibration(gain, fast_correct, slow_correct, bins=10):
    """Is the predicted gain on the scale lambda is priced in?

    ECE measures whether the reported confidence matches how often the fast mode
    is right. This measures the analogous thing for the gain. A router whose gain
    is only a ranking score passes AUROC and fails here, and lambda then has no
    meaning it can be set by.

    **The realised side is a log ratio, because the predicted side is.** The gain
    became `log P(slow right) - log P(fast right)` when the reward became a
    discount; this compared it against the accuracy *difference*
    `mean(slow) - mean(fast)` for one commit, which is a different quantity in
    different units, so both the error and the slope were uninterpretable. Within
    a bin the realised gain is now `log mean(slow correct) - log mean(fast
    correct)`, floored by `GAIN_EPS` exactly as the prediction is. Values recorded
    before this are on the old difference scale and do not compare with values
    recorded after it.

    A log ratio is not defined per question -- one question gives a mean of 0 or 1
    -- so unlike ECE this is a per-bin statistic throughout: the slope is the
    count-weighted regression of realised on predicted across bins, 1 when the
    scale is right and below 1 when the spread is exaggerated.

    Bins hold equal counts, because the gain concentrates near zero and
    equal-width bins would leave the tails empty.
    """
    gain = np.asarray(gain, dtype=np.float64)
    fast = np.asarray(fast_correct, dtype=np.float64)
    slow = np.asarray(slow_correct, dtype=np.float64)
    # Stable, because the gain ties often -- a saturated head returns the same
    # value for many questions -- and an unstable sort then splits a tie group
    # between bins however the C library happens to partition it. The bin
    # contents of a reported statistic should not depend on that.
    order = np.argsort(gain, kind="stable")
    total = len(gain)
    predicted, realised, weights = [], [], []
    for chunk in np.array_split(order, min(bins, total)):
        if not len(chunk):
            continue
        predicted.append(gain[chunk].mean())
        realised.append(math.log(max(slow[chunk].mean(), GAIN_EPS))
                        - math.log(max(fast[chunk].mean(), GAIN_EPS)))
        weights.append(len(chunk) / total)
    predicted = np.array(predicted)
    realised = np.array(realised)
    weights = np.array(weights)

    error = float((weights * np.abs(predicted - realised)).sum())
    centre = (weights * predicted).sum()
    spread = (weights * (predicted - centre) ** 2).sum()
    if spread <= 0:
        return error, float("nan")
    covariance = (weights * (predicted - centre)
                  * (realised - (weights * realised).sum())).sum()
    return error, float(covariance / spread)


def evaluate_head(head, features, fast_correct, slow_correct, lambdas=DEFAULT_LAMBDAS):
    outcome = head.predict(features)
    confidence = np.array([fast_confidence(row) for row in outcome])
    gain = np.array([escalation_gain(row) for row in outcome])
    fast_correct = np.asarray(fast_correct, dtype=bool)
    gain_error, gain_slope = gain_calibration(gain, fast_correct, slow_correct)
    return {
        "auroc_fast_correct": auroc(confidence, fast_correct),
        "auroc_gain_vs_helps": auroc(gain, ~fast_correct & slow_correct),
        "brier_fast_correct": float(np.mean((confidence - fast_correct) ** 2)),
        "ece_fast_correct": expected_calibration_error(confidence, fast_correct),
        "gain_calibration_error": gain_error,
        "gain_calibration_slope": gain_slope,
        "curve": coverage_accuracy_curve(gain, fast_correct, slow_correct, lambdas),
    }


def _selection_score(result):
    """Routing objective first; fall back to fast-correct AUROC when no escalation helped."""
    gain = result["auroc_gain_vs_helps"]
    return float(gain) if np.isfinite(gain) else float(np.nan_to_num(result["auroc_fast_correct"]))


def run_training(
    records,
    metadata,
    holdout_fraction=0.25,
    select_layers=1,
    epochs=30,
    probe_epochs=10,
    seed=0,
    lambdas=DEFAULT_LAMBDAS,
    selection_folds=4,
    pca_dim=64,
    objective="supervised",
    cdpo=None,
    train_seed=None,
    force_candidate=None,
    pmpo=None,
):
    """Select a feature set by grouped cross-validation over training categories,
    retrain it on all training data, report on held-out categories. Returns (head, report).

    Candidates are the stats-only head (no hidden states, the floor), one head per
    collected layer with raw hidden states, and one per layer through a whitened PCA
    bottleneck. Selection never sees the test split.
    """
    layers = tuple(metadata["feature_layers"])
    train_seed = seed if train_seed is None else train_seed  # splits stay fixed across seeds
    fast = np.asarray(records["fast_correct"], dtype=bool)
    slow = np.asarray(records["slow_correct"], dtype=bool)
    labels = outcome_labels(fast, slow)
    test, split_level = holdout_mask(records["category"], holdout_fraction, seed)
    train = ~test
    # Grouped folds over the training side only, for model selection.
    train_index = np.flatnonzero(train)
    folds = group_folds(np.asarray(records["category"])[train], selection_folds, seed + 1)
    base = {
        "model_id": metadata["model_id"],
        "revision": metadata["revision"],
        "hidden_size": int(metadata["hidden_size"]),
        # Carried from the records rather than left at the default. All three
        # HeadMetadata constructions below take **base, so this is what stops a
        # head fitted on content features being stamped "coded" and then served
        # under the other readout with nothing refusing it, which is the silent
        # mismatch `HeadMetadata.readout` exists to prevent.
        "readout": metadata.get("readout", "coded"),
    }
    candidates = [{"name": "stats_only", "layers": (), "positions": (), "width": 32, "pca": 0}]
    for position, layer in enumerate(layers):
        candidates.append(
            {
                "name": f"layer_{layer}",
                "layers": (layer,),
                "positions": (position,),
                "width": 256,
                "pca": 0,
            }
        )
        candidates.append(
            {
                "name": f"layer_{layer}_pca{pca_dim}",
                "layers": (layer,),
                "positions": (position,),
                "width": 64,
                "pca": pca_dim,
            }
        )
        # The two cells that make the pair above a 2x2 rather than a diagonal.
        # `layer_X` is width 256 with no projection and `layer_X_pca64` is width
        # 64 with one, so a difference between them is PCA *and* a fourfold
        # narrowing, and the paper once read such a difference as PCA alone.
        # These isolate the two: same width, projection on or off; same
        # projection, width 256 or 64.
        candidates.append(
            {
                "name": f"layer_{layer}_w64",
                "layers": (layer,),
                "positions": (position,),
                "width": 64,
                "pca": 0,
            }
        )
        candidates.append(
            {
                "name": f"layer_{layer}_pca{pca_dim}_w256",
                "layers": (layer,),
                "positions": (position,),
                "width": 256,
                "pca": pca_dim,
            }
        )
    if select_layers > 1 and len(layers) > 1:
        candidates.append(
            {
                "name": "all_layers",
                "layers": layers,
                "positions": tuple(range(len(layers))),
                "width": 256,
                "pca": 0,
            }
        )
    sweep = []
    for candidate in candidates:
        meta = HeadMetadata(
            **base,
            feature_layers=candidate["layers"],
            hidden_width=candidate["width"],
            pca_dim=candidate["pca"],
        )
        features = features_for(records, candidate["positions"])
        fold_scores, fold_fast, fold_gain = [], [], []
        for fold in range(selection_folds):
            val_index = train_index[folds == fold]
            fit_index = train_index[folds != fold]
            if len(val_index) == 0 or len(fit_index) < 2:
                continue
            probe, _ = train_head(
                meta, features[fit_index], labels[fit_index], epochs=probe_epochs, seed=train_seed
            )
            validation = evaluate_head(
                probe, features[val_index], fast[val_index], slow[val_index], lambdas
            )
            fold_scores.append(_selection_score(validation))
            fold_fast.append(validation["auroc_fast_correct"])
            fold_gain.append(validation["auroc_gain_vs_helps"])
        probe, _ = train_head(
            meta, features[train_index], labels[train_index], epochs=probe_epochs, seed=train_seed
        )
        held_out = evaluate_head(probe, features[test], fast[test], slow[test], lambdas)
        sweep.append(
            {
                "name": candidate["name"],
                "layers": list(candidate["layers"]),
                "pca_dim": candidate["pca"],
                "validation": {
                    "folds": len(fold_scores),
                    "auroc_fast_correct": float(np.nanmean(fold_fast)) if fold_fast else None,
                    "auroc_gain_vs_helps": float(np.nanmean(fold_gain)) if fold_gain else None,
                },
                "test": held_out,
                "selection_score": float(np.mean(fold_scores)) if fold_scores else -1.0,
            }
        )
    best = max(sweep, key=lambda row: row["selection_score"])
    if force_candidate is not None:
        best = next((row for row in sweep if row["name"] == force_candidate), None)
        if best is None:
            raise ValueError(f"Unknown candidate {force_candidate!r}")
    chosen = next(c for c in candidates if c["name"] == best["name"])
    chosen_layers = tuple(chosen["layers"])
    meta = HeadMetadata(
        **base, feature_layers=chosen_layers, hidden_width=chosen["width"], pca_dim=chosen["pca"]
    )
    features = features_for(records, chosen["positions"])
    # Train the chosen feature set under both objectives; return the requested one.
    heads, histories, finals = {}, {}, {}
    objective = normalise_objective(objective)
    # `pmpo=` is the name this argument carried for one revision, kept for the
    # scripts written under it; passing both is a mistake, not a precedence
    # question.
    if pmpo is not None and cdpo is not None:
        raise ValueError("pass cdpo=, not both cdpo= and its former name pmpo=")
    cdpo = cdpo if cdpo is not None else pmpo
    for name in OBJECTIVES:
        heads[name], histories[name] = train_head(
            meta,
            features[train],
            labels[train],
            epochs=epochs,
            seed=train_seed,
            objective=name,
            cdpo=cdpo,
        )
        finals[name] = evaluate_head(heads[name], features[test], fast[test], slow[test], lambdas)
    head, history, final = heads[objective], histories[objective], finals[objective]
    stats_only = next(row for row in sweep if row["name"] == "stats_only")["test"]
    max_probability = records["fast_probability"][test]
    concentration = records["stats"][test, 3]
    report = {
        "train_count": int(train.sum()),
        "test_count": int(test.sum()),
        "holdout_fraction": holdout_fraction,
        "split_level": split_level,
        "selection_folds": selection_folds,
        "categories_seen": sorted({str(c) for c in np.asarray(records["category"])}),
        "held_out_categories": sorted({str(c) for c in np.asarray(records["category"])[test]}),
        "fast_accuracy": float(fast[test].mean()),
        "slow_accuracy": float(slow[test].mean()),
        "thinking_helps_rate": float((~fast[test] & slow[test]).mean()),
        "thinking_hurts_rate": float((fast[test] & ~slow[test]).mean()),
        "candidates": sweep,
        "chosen": best["name"],
        "forced": force_candidate is not None,
        "chosen_layers": list(chosen_layers),
        "objective": objective,
        "train_seed": train_seed,
        "by_objective": finals,
        "baseline_auroc_max_probability": auroc(max_probability, fast[test]),
        "baseline_auroc_concentration": auroc(concentration, fast[test]),
        "baseline_stats_probe": stats_only,
        "head": final,
        "loss_history": history,
    }
    head.metadata = HeadMetadata(
        **base,
        feature_layers=chosen_layers,
        hidden_width=chosen["width"],
        pca_dim=chosen["pca"],
        training={
            key: report[key]
            for key in (
                "train_count",
                "test_count",
                "held_out_categories",
                "fast_accuracy",
                "slow_accuracy",
                "chosen_layers",
            )
        }
        # How the head was trained belongs in the file. A head trained under one
        # objective is not interchangeable with one trained under another, and a
        # checkpoint that does not say which it was invites benchmarking the wrong
        # one months later.
        | {
            "auroc_fast_correct": final["auroc_fast_correct"],
            "objective": objective,
            "class_weighted": False,
            "epochs": epochs,
            "train_seed": train_seed,
        },
    )
    return head, report
