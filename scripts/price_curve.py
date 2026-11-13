"""The price-of-thinking curve, as `figures/fig_lambda.tex` reads it.

Four columns, one row per price: `lambda rate cost acc`, where `rate` is the
share of questions escalated, `cost` is thinking tokens as a fraction of the
slow-only row's, and `acc` is the accuracy the routed policy reaches. That is
the shape of `paper-assets/curve27b.dat`, which nothing in this tree wrote.

Computed from recorded passes rather than re-served, which is what the benchmark
itself validates: at the prices that were served, the escalation set, the token
count and the accuracy reproduce it. Serving 111 prices instead would cost 111
runs of the slow mode to learn nothing the records do not already hold.

    uv run python scripts/price_curve.py results/records-sglang-29734.npz \
        --head results/head-27b-sglang-29734.safetensors > paper-assets/curve27b.dat
"""
import argparse
import sys

import numpy as np

from hde.core.block import escalation_gain
from hde.data.collect import open_records
from hde.serving import load_head
from hde.train.training import features_for, holdout_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records")
    parser.add_argument("--head", required=True)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--prices", default="-0.1:1.0:0.01",
                        help="start:stop:step, inclusive of stop")
    parser.add_argument("--all-questions", action="store_true",
                        help="score every record rather than the held-out categories")
    arguments = parser.parse_args()

    head = load_head(arguments.head)
    records, metadata = open_records(arguments.records)
    # A head fitted under one readout reads a different quantity under the other,
    # and nothing in its output would say so, so the mismatch is refused here.
    collected = metadata.get("readout", "coded")
    if head.metadata.readout != collected:
        raise SystemExit(
            f"this head was fitted on the {head.metadata.readout!r} readout and these "
            f"records were collected under {collected!r}; its statistics are a "
            f"different quantity on a different scale")
    position = list(metadata["feature_layers"]).index(arguments.layer)
    fast = np.asarray(records["fast_correct"], dtype=bool)
    slow = np.asarray(records["slow_correct"], dtype=bool)
    tokens = np.asarray(records["thinking_tokens"], dtype=np.float64)
    features = features_for(records, (position,))
    if arguments.all_questions:
        keep = np.ones(len(fast), dtype=bool)
    else:
        keep = np.asarray(holdout_mask(records["category"], 0.25, 0)[0], dtype=bool)
    ft, st, tt = fast[keep], slow[keep], tokens[keep]
    gains = np.array([escalation_gain(row) for row in head.predict(features[keep])])

    start, stop, step = (float(value) for value in arguments.prices.split(":"))
    prices = np.round(np.arange(start, stop + step / 2, step), 10)
    slow_tokens = tt.mean()

    print(f"{arguments.records}: {metadata['model_id'].split('/')[-1]}, readout "
          f"{collected}, {int(keep.sum())} questions, slow-only {st.mean():.4f} at "
          f"{slow_tokens:.0f} tokens", file=sys.stderr)
    print("lambda rate cost acc")
    for price in prices:
        escalate = gains > price
        rate = float(escalate.mean())
        cost = float(np.where(escalate, tt, 0.0).mean() / slow_tokens) if slow_tokens else 0.0
        accuracy = float(np.where(escalate, st, ft).mean())
        print(f"{price:.3f} {rate:.5f} {cost:.5f} {accuracy:.5f}")


if __name__ == "__main__":
    main()
