"""How much of a multiple-choice column is the question, and how much is the position.

The constructed candidate sets showed a readout that takes the second option
99--100% of the time. The native multiple-choice benchmarks load their options in
source order, so if the readout carries the same prior there, part of the fast
column in the paper's main tables is a position and not a judgement.

The accuracy bounds it loosely already: a policy that always picks one position
scores the frequency of gold at that position, which on ten-way questions is
about a tenth, and the fast mode scores 0.538 and 0.665. So the prior cannot be
doing the work. It can still be worth a few points, and that is what this
measures rather than argues:

    chosen        where the readout put its answer
    gold          where the answer actually was
    accuracy by gold position, which is what a prior distorts

Reads the record files `litjev-collect` writes. Files written before the position
columns existed report their absence rather than a histogram of nothing.

    uv run python scripts/position_bias.py results/records-4b-27063.npz
"""
import argparse

import numpy as np


def histogram(values, width):
    counts = np.bincount(values[values >= 0], minlength=width)[:width]
    total = max(counts.sum(), 1)
    return counts, counts / total


def bar(share, scale=40):
    return "#" * int(round(share * scale))


def main():
    from litjev.data.collect import load_records
    from litjev.run.streaming import write_json_atomic

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", nargs="+")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    # The pass this reads is normally a fast-only one: the slow path costs the
    # thinking and answers none of this question.
    merged, metadata = load_records(arguments.records, allow_fast_only=True)
    chosen = np.asarray(merged["chosen_index"], dtype=np.int64)
    gold = np.asarray(merged["gold_index"], dtype=np.int64)
    if (chosen < 0).all():
        raise SystemExit(
            f"these records carry no position columns: they were written in "
            f"{metadata.get('record_format')}, before the columns existed. The question "
            f"needs a re-collection, not a re-reading -- which option a readout chose is "
            f"not recoverable from whether it was right")

    counted = np.asarray(merged["option_count"], dtype=np.int64)
    width = int(counted.max()) if (counted > 0).any() else int(max(chosen.max(), gold.max()) + 1)
    correct = np.asarray(merged["fast_correct"], dtype=bool)
    usable = (chosen >= 0) & (gold >= 0)
    print(f"{usable.sum()} of {len(chosen)} questions carry both positions, "
          f"{width} options at most, fast accuracy {correct[usable].mean():.3f}\n")

    chosen_counts, chosen_share = histogram(chosen[usable], width)
    gold_counts, gold_share = histogram(gold[usable], width)
    print(f"{'pos':>4}  {'chosen':>7} {'gold':>7}  {'acc|gold':>8}  chosen")
    by_gold = {}
    for index in range(width):
        here = usable & (gold == index)
        accuracy = float(correct[here].mean()) if here.any() else float("nan")
        by_gold[index] = {"questions": int(here.sum()), "accuracy": accuracy}
        print(f"{index:>4}  {chosen_share[index]:>7.3f} {gold_share[index]:>7.3f}  "
              f"{accuracy:>8.3f}  {bar(chosen_share[index])}")

    # How far the readout's positions sit from where the answers are. A readout
    # reading the question tracks the gold histogram; one with a prior does not.
    # Reported as total variation, which is the share of answers that would have
    # to move position for the two to agree.
    drift = 0.5 * np.abs(chosen_share - gold_share).sum()
    flat = np.full(width, 1 / width)
    print(f"\nchosen vs gold, total variation   {drift:.3f}"
          f"   (0 = the readout's positions are distributed like the answers')")
    print(f"gold vs uniform                   {0.5 * np.abs(gold_share - flat).sum():.3f}"
          f"   (how unevenly this benchmark places its answers)")
    spread = [value["accuracy"] for value in by_gold.values()
              if value["questions"] and not np.isnan(value["accuracy"])]
    if spread:
        print(f"accuracy by gold position         {min(spread):.3f} to {max(spread):.3f}"
              f"   (a prior shows up here, as easy and hard positions)")

    if arguments.output:
        write_json_atomic(arguments.output, {
            "records": arguments.records, "record_format": metadata.get("record_format"),
            "model": metadata.get("model_id"), "dataset": metadata.get("dataset"),
            "questions": int(usable.sum()), "options": width,
            "fast_accuracy": float(correct[usable].mean()),
            "chosen_share": chosen_share.tolist(), "gold_share": gold_share.tolist(),
            "chosen_counts": chosen_counts.tolist(), "gold_counts": gold_counts.tolist(),
            "by_gold_position": by_gold,
            "chosen_vs_gold_total_variation": float(drift),
        })
        print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
