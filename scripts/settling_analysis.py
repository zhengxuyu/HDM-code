"""Does thinking help more on questions the fast pass settles late?

Two questions, and the order matters. The first is about the recorded outcomes
and holds whatever our head learned: among questions of the same subject, does
the slow mode rescue the late-settling ones more often? That tests the paper's
explanation. The second is whether the head's predicted value of thinking tracks
the settling layer, which tests whether it found the signal.

Settling late correlates with being unsure, and the paper's whole point is that
being unsure is not the same as thinking helping. So every comparison is also
reported with the fast mode's confidence partialled out; a correlation that does
not survive that is the confidence signal wearing a different name.

    uv run python scripts/settling_analysis.py settling-4b.npz records-4b.npz
"""
import argparse
import json

import numpy as np

from hde._legacy.heads import auroc, outcome_labels
from hde.run.streaming import write_json_atomic


def settling_layer(logits, count):
    """The last layer at which the argmax over candidates changes."""
    choices = np.argmax(logits[:, :count], axis=1)
    changes = np.flatnonzero(choices[1:] != choices[:-1])
    return int(changes[-1] + 1) if len(changes) else 0


def spearman(a, b):
    order_a = np.argsort(np.argsort(a))
    order_b = np.argsort(np.argsort(b))
    return float(np.corrcoef(order_a, order_b)[0, 1])


def partial(a, b, control):
    """Correlation of a and b with `control` regressed out of both, on ranks."""
    def rank(v):
        return np.argsort(np.argsort(v)).astype(float)

    x, y, z = rank(a), rank(b), rank(control)
    z = np.stack([z, np.ones_like(z)], axis=1)

    def residual(v):
        return v - z @ np.linalg.lstsq(z, v, rcond=None)[0]

    return float(np.corrcoef(residual(x), residual(y))[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("settling")
    parser.add_argument("records")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    lens = np.load(arguments.settling, allow_pickle=True)
    records = np.load(arguments.records, allow_pickle=True)

    depth = json.loads(str(lens["metadata"]))["depth"]
    settling = {
        str(identifier): settling_layer(logits, count)
        for identifier, logits, count in zip(
            lens["question_id"], lens["layer_logits"], lens["candidate_count"])
    }

    identifiers = [str(q) for q in records["question_id"]]
    keep = np.array([q in settling for q in identifiers])
    if keep.sum() < len(identifiers):
        print(f"joined {keep.sum()} of {len(identifiers)} recorded questions")
    layer = np.array([settling[q] for q in identifiers if q in settling], dtype=float)

    fast = np.asarray(records["fast_correct"], dtype=bool)[keep]
    slow = np.asarray(records["slow_correct"], dtype=bool)[keep]
    confidence = np.asarray(records["fast_probability"], dtype=float)[keep]
    categories = np.asarray([str(c) for c in records["category"]])[keep]
    helps = outcome_labels(fast, slow) == 2

    print(f"depth {depth}, settling layer: median {np.median(layer):.0f}, "
          f"mean {layer.mean():.1f}, {100 * (layer == 0).mean():.0f}% settle at the embedding\n")

    print("Q1  does thinking actually help more on late-settling questions?")
    print(f"  AUROC of settling layer for 'thinking rescues it'   {auroc(layer, helps):.4f}")
    print(f"  AUROC of (1 - confidence) for the same              "
          f"{auroc(-confidence, helps):.4f}")
    print(f"  Spearman(settling, helps)                           {spearman(layer, helps):.4f}")
    print(f"  ... with confidence partialled out                  "
          f"{partial(layer, helps.astype(float), confidence):.4f}")

    rows = []
    for name in sorted(set(categories)):
        mask = categories == name
        positives = int(helps[mask].sum())
        if mask.sum() < 40 or positives == 0 or positives == mask.sum():
            continue
        rows.append((name, float(auroc(layer[mask], helps[mask])), positives, int(mask.sum())))
    if rows:
        within = sum(r[1] for r in rows) / len(rows)
        print(f"\n  within-category AUROC of the settling layer, mean {within:.4f}")
        for name, value, positives, total in sorted(rows, key=lambda r: -r[1]):
            print(f"    {name:22s} {value:.4f}  ({positives}/{total} helped)")

    print("\nQ2  has the head found it?")
    print("  (run with a head's gains to fill this in)")

    if arguments.output:
        write_json_atomic(arguments.output, {
            "depth": depth,
            "settling_auroc_helps": float(auroc(layer, helps)),
            "confidence_auroc_helps": float(auroc(-confidence, helps)),
            "spearman_settling_helps": spearman(layer, helps),
            "partial_out_confidence": partial(layer, helps.astype(float), confidence),
            "within_category": [
                {"category": n, "auroc": v, "positives": p, "questions": t}
                for n, v, p, t in rows],
        })


if __name__ == "__main__":
    main()
