"""Re-read the slow column from what the model wrote, without running it again.

A three-policy run records the whole response for every question, so a fault in
reading the answer out of it is a scoring fault and not a generation fault. This
reads the saved responses, extracts again with whatever `litjev.construct` now
does, grades against the same gold answers, and writes a new record beside the
old one.

It exists because 27866 ran from a commit whose `ANSWER_LINE` took the last
match on a line, so `Answer: 12. This answer uses the formula` scored as the
sentence. Nothing about the generation was wrong, and rerunning would spend the
slow mode's tokens again to get the same text.

What it cannot repair is a response that was never saved, or one saved as a
tail: a record written before the whole response was kept has nothing to read.
Those rows are reported and left out rather than carried over at their old
score, because a column mixing two scorings is worse than a column missing rows.

    uv run python scripts/rescore_slow.py results/open-three-27866.json \\
        --output results/open-three-27866-rescored.json
"""
import argparse
import json

import numpy as np


def main():
    from litjev.candidates.construct import balanced, correct, extract_answer
    from litjev.run.streaming import write_json_atomic

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record")
    parser.add_argument("--output", required=True)
    parser.add_argument("--answer-form", choices=["expression", "integer"],
                        help="only for a record written before the form was saved")
    arguments = parser.parse_args()

    with open(arguments.record) as handle:
        original = json.load(handle)
    rows = original["rows"]
    # From the record, not from the dataset's name: a record that does not say
    # what form its answers take cannot be re-scored without guessing, and a
    # wrong guess changes every verdict in the column.
    form = arguments.answer_form or original.get("answer_form")
    if form is None:
        raise KeyError(
            f"{arguments.record} does not record answer_form, so whether its answers "
            f"are expressions or integers can only be guessed. Re-run the producer at "
            f"a commit that records it, or pass --answer-form.")
    symbolic = form == "expression"

    rescored, unreadable, changed, unbalanced = [], [], 0, []
    for row in rows:
        written = row.get("slow_text")
        if written is None:
            unreadable.append(row["id"])
            continue
        answer, closed = extract_answer(written)
        was = bool(row.get("slow_correct"))
        now = bool(answer and correct(answer, str(row["gold"]), symbolic=symbolic))
        changed += was != now
        if not balanced(answer):
            unbalanced.append(row["id"])
        rescored.append({**row, "slow_answer": answer, "slow_correct": now,
                         "slow_left_its_thinking": bool(closed),
                         "slow_answer_before": row.get("slow_answer"),
                         "slow_correct_before": was})

    if unreadable:
        print(f"{len(unreadable)} rows carry no saved response and are left out: "
              f"{unreadable[:5]}{' ...' if len(unreadable) > 5 else ''}")
    slow_before = np.mean([row["slow_correct_before"] for row in rescored])
    slow_after = np.mean([row["slow_correct"] for row in rescored])
    truncated = sum(not row["slow_left_its_thinking"] for row in rescored)
    print(f"{len(rescored)} of {len(rows)} rows re-scored, {changed} changed verdict")
    print(f"  slow column   {slow_before:.3f} -> {slow_after:.3f}")
    print(f"  never left their thinking   {truncated}/{len(rescored)}")
    if unbalanced:
        print(f"  {len(unbalanced)} extracted answers have unbalanced braces, which means "
              f"a cut landed inside one: {unbalanced[:5]}"
              f"{' ...' if len(unbalanced) > 5 else ''}")
    fast = np.mean([row["fast_correct"] for row in rescored])
    # The fast column reads no free text, so re-scoring cannot move it. If it
    # moves, the re-scoring is wrong and the slow column should not be believed.
    if len(rescored) == len(rows) and "fast" in original:
        if abs(fast - float(original["fast"])) > 1e-9:
            raise AssertionError(
                f"re-scoring moved the fast column, {original['fast']:.4f} -> {fast:.4f}, "
                f"which it cannot do: it reads no free text")
    print(f"  fast column   {fast:.3f} (unchanged, as it must be)")
    oracle = np.mean([row["fast_correct"] or row["slow_correct"] for row in rescored])
    print(f"  oracle        {oracle:.3f} (better of the two per question)")

    write_json_atomic(arguments.output,
                      {**original, "rows": rescored,
                       "rescored_from": arguments.record,
                       "rescored_rows": len(rescored),
                       "rows_without_saved_response": unreadable,
                       "slow_before": float(slow_before), "slow": float(slow_after),
                       "fast": float(fast), "oracle": float(oracle),
                       "answer_form": form,
                       "unbalanced_extractions": unbalanced})
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
