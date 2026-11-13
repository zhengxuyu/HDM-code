"""Is the routing signal about the question, or about the subject it belongs to?

The reported routing AUROC ranks every held-out question against every other.
A head that had learned only which subjects reward thinking would score well on
that ranking and be useless inside a subject, which is where a deployment lives:
questions arrive one at a time and are not competing with a different field.

So this trains one head per category, leaving that category out, and scores each
category's questions against one another only. Two numbers come back: the pooled
AUROC over all out-of-fold predictions, which is the usual quantity, and the mean
of the per-category AUROCs, which is the part that survives when subject is held
constant. The gap between them is what the subject alone was worth.

    uv run python scripts/within_category.py results/records-4b.npz --layer 24
"""
import argparse

import numpy as np

from litjev._legacy.heads import HeadMetadata, auroc, outcome_labels
from litjev._legacy.routing import escalation_gain
from litjev.data.collect import open_records
from litjev.run.streaming import write_json_atomic
from litjev.train.training import features_for, train_head


def gains_for(head, features):
    outcome = head.predict(features)
    return np.array([escalation_gain(row) for row in outcome])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records")
    parser.add_argument("--layer", type=int, default=24, help="the layer the paper selected")
    parser.add_argument("--pca", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-questions", type=int, default=40,
                        help="a category smaller than this cannot support an AUROC")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    records, metadata = open_records(arguments.records)
    layers = list(metadata["feature_layers"])
    if arguments.layer not in layers:
        raise SystemExit(f"layer {arguments.layer} was not collected; have {layers}")
    position = layers.index(arguments.layer)

    fast = np.asarray(records["fast_correct"], dtype=bool)
    slow = np.asarray(records["slow_correct"], dtype=bool)
    labels = outcome_labels(fast, slow)
    helps = labels == 2
    categories = np.asarray([str(c) for c in records["category"]])
    features = features_for(records, (position,))

    head_metadata = HeadMetadata(
        model_id=metadata["model_id"], revision=metadata["revision"],
        hidden_size=int(metadata["hidden_size"]), feature_layers=(arguments.layer,),
        hidden_width=arguments.width, pca_dim=arguments.pca)

    # Leave one category out, so no question is ranked by a head that has seen its
    # subject. This is stricter than the paper's split, which holds out three.
    gains = np.zeros(len(labels), dtype=np.float64)
    for name in sorted(set(categories)):
        held = categories == name
        head, _ = train_head(head_metadata, features[~held], labels[~held],
                             epochs=arguments.epochs, seed=arguments.seed)
        gains[held] = gains_for(head, features[held])
        print(f"  {name:22s} {held.sum():4d} questions, {helps[held].sum():3d} helped",
              flush=True)

    pooled = auroc(gains, helps)
    rows = []
    for name in sorted(set(categories)):
        held = categories == name
        positives = int(helps[held].sum())
        negatives = int(held.sum() - positives)
        if held.sum() < arguments.min_questions or positives == 0 or negatives == 0:
            rows.append({"category": name, "questions": int(held.sum()),
                         "positives": positives, "auroc": None})
            continue
        rows.append({"category": name, "questions": int(held.sum()),
                     "positives": positives,
                     "auroc": float(auroc(gains[held], helps[held])),
                     "pairs": positives * negatives})

    scored = [row for row in rows if row["auroc"] is not None]
    weight = sum(row["pairs"] for row in scored)
    weighted = sum(row["auroc"] * row["pairs"] for row in scored) / weight
    plain = sum(row["auroc"] for row in scored) / len(scored)

    print(f"\npooled AUROC over all out-of-fold questions   {pooled:.4f}")
    print(f"mean within-category AUROC (pair-weighted)    {weighted:.4f}")
    print(f"mean within-category AUROC (unweighted)       {plain:.4f}")
    print(f"categories scored: {len(scored)} of {len(rows)}\n")
    for row in sorted(scored, key=lambda r: r["auroc"]):
        print(f"  {row['category']:22s} {row['auroc']:.4f}  "
              f"({row['positives']}/{row['questions']} helped)")

    if arguments.output:
        write_json_atomic(arguments.output,
                          {"records": arguments.records, "layer": arguments.layer,
                           "pooled_auroc": float(pooled),
                           "within_category_weighted": float(weighted),
                           "within_category_unweighted": float(plain),
                           "per_category": rows})


if __name__ == "__main__":
    main()
