"""Which code the readout picks, conditioned on how many codes there were.

The readout takes the first listed candidate once in 478 questions. What it takes
instead was first read off the mode alone, which supports two stories equally:

    position     it prefers a slot -- the second, or the last, or some offset
                 from the end -- and the code it lands on is whatever sits there.
    code         it prefers particular code tokens, and the slot it lands on is
                 wherever those codes happen to be.

Nothing here separates them, and an earlier version of this file claimed it did.
`candidate_codes` hands out `A, B, C…` in slot order, so a distribution over
letters *is* a distribution over slots, at every width and over any number of
questions. Mass on `B` surviving where `G` leads looked like something no slot
rule could produce; it is what "slot 6 with probability two thirds, else slot 1"
produces, and "one slot per question" constrains one question rather than the
marginal across them.

What the distribution does say is narrower and still worth having. The preference
is not one fixed slot: it sits on index 1 up to width 6, index 6 at widths 7 to
9, index 9 at width 10. It is not the last slot either, since at widths 8 and 9
the last slot takes almost nothing while index 6 takes most of it. And it is
concentrated on a few indices rather than spread. Every simple slot rule of the
form "always the k-th" or "always the last" is out; a width-dependent one, or a
mixture, is not, and neither is a preference for the letters themselves.

Reads the per-question rows from a `propose_gate` sweep. Rotating the codes
against the slots is the only thing that separates letter from slot -- the P3 row
in experiments/EXP.md.

    uv run python scripts/readout_codes.py experiments/math-500/results/*.csv
"""
import argparse
import csv
import string
from collections import Counter, defaultdict


def rows_of(paths, rank, temperature=None):
    found = []
    for path in paths:
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row["rank"] != rank or row["degenerate"] != "0":
                    continue
                if temperature is not None and row["temperature"] != temperature:
                    continue
                found.append(row)
    return found


def table(rows, letters):
    """share of each code, by how many candidates the question offered."""
    by_width = defaultdict(Counter)
    for row in rows:
        by_width[int(row["option_count"])][row["readout_code"]] += 1
    print("width    n  " + "".join(f"{code:>6}" for code in letters))
    shares = {}
    for width in sorted(by_width):
        counted = by_width[width]
        total = sum(counted.values())
        shares[width] = {code: counted[code] / total for code in letters if counted[code]}
        print(f"{width:>5} {total:>4}  " + "".join(
            f"{counted[code] / total:>6.2f}" if counted[code] else f"{'-':>6}"
            for code in letters))
    return shares


def surviving(shares):
    """Codes that keep mass at widths where a later code takes the mode.

    Not a discriminating quantity, though it was described as one here. A slot
    mixture produces the same marginal. It says the preference is over a few
    indices rather than one, which rules out "always the k-th" and "always the
    last" and nothing finer.
    """
    found = {}
    for width, share in shares.items():
        if not share:
            continue
        mode = max(share, key=share.get)
        behind = {code: value for code, value in share.items()
                  if code < mode and value >= 0.05}
        if behind:
            found[width] = {"mode": mode, "mode_share": share[mode], "still_held": behind}
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rows", nargs="+", help="per-question CSVs from a propose_gate sweep")
    parser.add_argument("--rank", default="likelihood")
    parser.add_argument("--temperature", help="one temperature; the default pools them")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    rows = rows_of(arguments.rows, arguments.rank, arguments.temperature)
    if not rows:
        raise SystemExit(f"no non-degenerate rows at rank {arguments.rank!r}"
                         + (f", temperature {arguments.temperature}"
                            if arguments.temperature else ""))
    letters = list(string.ascii_uppercase[: max(int(row["option_count"]) for row in rows)])
    first = sum(int(row["readout_index"]) == 0 for row in rows)
    print(f"{len(rows)} questions with a real choice"
          + (f" at t={arguments.temperature}" if arguments.temperature else
             f" pooled over t={sorted({r['temperature'] for r in rows}, key=float)}"))
    print(f"the first listed candidate is taken {first} times ({first / len(rows):.1%})\n")

    shares = table(rows, letters)
    held = surviving(shares)
    print("\nCodes still holding 5% or more at a width where a later code leads:")
    if not held:
        print("  none -- the preference is on one index at each width")
    for width, found in sorted(held.items()):
        behind = ", ".join(f"{code} {value:.2f}" for code, value in sorted(found["still_held"].items()))
        print(f"  width {width:>2}: {found['mode']} leads at {found['mode_share']:.2f}, "
              f"and {behind} keeps mass behind it")
    print("\nRead this as: the preference is spread over a few indices, not one, so")
    print("\"always the k-th\" and \"always the last\" are out. It does NOT separate a")
    print("letter preference from a slot one -- codes are bound to slots, so a mixture of")
    print("slot preferences gives the same marginal. Only rotating the codes against the")
    print("slots separates them (P3 in experiments/EXP.md).")

    if arguments.output:
        from litjev.run.streaming import write_json_atomic

        write_json_atomic(arguments.output, {
            "rows": arguments.rows, "rank": arguments.rank,
            "temperature": arguments.temperature, "questions": len(rows),
            "first_listed_taken": first,
            "share_by_width": {str(width): share for width, share in shares.items()},
            "held_behind_the_mode": {str(width): found for width, found in held.items()}})
        print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
