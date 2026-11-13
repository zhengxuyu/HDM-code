"""Two records files over the same questions, column by column.

Written for the readout change: 29978 collected its features through the causal
LM's full-sequence logits, and a replay collects them through the gathered
readout. If the change is sound the fast-leg columns are the same numbers, and
the fast leg is the whole of what the decision head reads.

**The selection is checked before the columns are.** A column diff between two
files over different question sets is a number with no meaning, and this project
has already published one comparison whose two sides were not the same
population. So the fields that decide *which* questions and *what* was asked
must agree exactly, and the run stops if they do not.

`slow_correct` and `thinking_tokens` are not compared when either side is
fast-only. They depend on generated text, and batched greedy generation is not
bit-reproducible across batch compositions, so a difference there would be the
known nondeterminism rather than this change.
"""
import argparse
import json
import sys

import numpy as np

from litjev.data.collect import open_records

# Must match, or the two files are not over the same work.
SELECTION = ("model_id", "revision", "dataset_name", "dataset_split", "dataset_revision",
             "dataset_digest_sha256", "offset", "stride", "limit", "categories", "phase",
             "feature_layers", "budget", "unified_prompt", "code_offset", "slot_format",
             "record_format", "selected", "available", "batching")

# Depend on generated text, so not comparable across a fast-only replay.
GENERATED = ("slow_correct", "thinking_tokens", "has_slow")

# Must be exactly equal, not equal to a tolerance: these are the decision head's
# inputs, and a change in them is a change in every routing decision downstream.
GATED = ("hidden", "stats", "fast_probabilities", "fast_probability")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("replay")
    parser.add_argument("--allow-differing", default="",
                        help="comma-separated metadata keys permitted to differ, with a "
                             "reason given on the command line that produced the run")
    args = parser.parse_args()

    permitted = {k.strip() for k in args.allow_differing.split(",") if k.strip()}
    left, left_meta = open_records(args.reference)
    right, right_meta = open_records(args.replay)

    disagreements = []
    for key in SELECTION:
        a, b = left_meta.get(key, "<absent>"), right_meta.get(key, "<absent>")
        if a != b and key not in permitted:
            disagreements.append(f"  {key}: {a!r} != {b!r}")
    if disagreements:
        print("the two files are not over the same work:", file=sys.stderr)
        print("\n".join(disagreements), file=sys.stderr)
        return 2

    fast_only = bool(left_meta.get("fast_only")) or bool(right_meta.get("fast_only"))
    columns, verdict = {}, True
    for name in sorted(set(left.files) & set(right.files) - {"metadata"}):
        if fast_only and name in GENERATED:
            columns[name] = {"skipped": "depends on generated text"}
            continue
        a, b = left[name], right[name]
        if a.shape != b.shape:
            columns[name] = {"shape": [list(a.shape), list(b.shape)]}
            verdict = False
            continue
        if a.dtype.kind in "fc":
            # **NaN-aware, because a padded column is otherwise unreadable.**
            # `fast_probabilities` is padded to the widest candidate set, so it
            # carries NaN where a question has fewer: on the 500-record file,
            # 3,152 of 8,000 entries. `np.array_equal` is false whenever a NaN is
            # present and `max` is NaN, so the first version of this reported the
            # one column that holds the full candidate distribution as "not
            # identical, delta NaN, 0 rows differing" -- three readings that
            # cannot all be true, and the column the head's features come from.
            finite = ~(np.isnan(a) | np.isnan(b))
            delta = np.abs(a.astype(np.float64)[finite] - b.astype(np.float64)[finite])
            identical = bool(np.array_equal(a, b, equal_nan=True))
            moved = np.abs(np.nan_to_num(a) - np.nan_to_num(b)) > 0
            columns[name] = {
                "bit_identical": identical,
                # Whether the two sides pad in the same places. A column can
                # agree everywhere it is defined and still be a different
                # question set, which this separates from a value difference.
                "nan_positions_agree": bool(np.array_equal(np.isnan(a), np.isnan(b))),
                "max_abs_delta": float(delta.max()) if delta.size else 0.0,
                "rows_differing": int(moved.any(axis=tuple(range(1, a.ndim))).sum())
                                  if a.ndim > 1 else int(moved.sum())}
            # The three inputs the decision head reads. All pass through this
            # change untouched, so all three are gated on exact equality rather
            # than on a tolerance: `hidden` and `stats` are its input vector and
            # `fast_probabilities` is what `stats` is computed from, so a
            # difference in any of them is a difference in the features.
            if name in GATED and not identical:
                verdict = False
        else:
            mismatches = int((a != b).sum())
            columns[name] = {"identical": mismatches == 0, "mismatches": mismatches}
            if mismatches:
                verdict = False

    report = {"reference": args.reference, "replay": args.replay,
              "rows": int(len(left["question_id"])), "fast_only": fast_only,
              "selection_agrees": True, "columns": columns, "pass": verdict}
    print(json.dumps(report, indent=2))
    print("PASS" if verdict else "FAIL")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
