"""Does the second forward pass earn its place, and if not, which thing is wrong.

`propose_gate` reports `first_accuracy` and `readout_accuracy` over the questions
that produced a real choice. Across a temperature sweep the readout sits below
candidate one at high temperatures and above it at low ones, and there are three
readings of that, which the same numbers cannot tell apart:

    construction   at a low temperature the proposals collapse, so the second
                   candidate is often a near-duplicate of the first and taking it
                   costs nothing; at a high one they genuinely differ and taking
                   the second is a real error. Then the gap tracks how many
                   distinct candidates there are, not the temperature.
    denominator    a degenerate question is excluded, and 58% are degenerate at
                   0.1 against 4% at 1.0, so the two temperatures are scored on
                   different question sets. Then the gap is about which questions
                   got in, and disappears on a set held fixed.
    readout        the readout is simply worse at one temperature than the other.

So this cuts both ways at once: by candidate count, and on the questions that are
non-degenerate at every temperature given. Reads the `--output` files
`propose_gate` already writes; nothing here needs a new run.

    uv run python scripts/readout_vs_first.py results/gate-t0.1.json results/gate-t1.0.json
"""
import argparse
import json
from collections import defaultdict


def entries_of(report, rank):
    """One (question id, entry) per row, for the rank asked for."""
    found = {}
    for row in report["rows"]:
        entry = row["by_rank"].get(rank)
        if entry is not None:
            found[row["id"]] = entry
    return found


def accuracy(entries, ids):
    """(first, readout, n) over the ids that had a real choice."""
    usable = [entries[key] for key in ids
              if key in entries and entries[key]["reason"] is None]
    if not usable:
        return float("nan"), float("nan"), 0
    return (sum(entry["first_right"] for entry in usable) / len(usable),
            sum(entry["readout_right"] for entry in usable) / len(usable),
            len(usable))


def report_line(label, first, readout, count):
    gap = readout - first
    verdict = "readout ahead" if gap > 0 else "readout behind" if gap < 0 else "level"
    print(f"  {label:<28} n={count:<5} first {first:.3f}  readout {readout:.3f}  "
          f"{gap:+.3f}  {verdict}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", help="propose_gate --output files")
    parser.add_argument("--rank", default="likelihood")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    loaded = []
    for path in arguments.reports:
        with open(path) as handle:
            report = json.load(handle)
        entries = entries_of(report, arguments.rank)
        if not entries:
            raise SystemExit(f"{path} has no rows at rank {arguments.rank!r}; it has "
                             f"{sorted({r for row in report['rows'] for r in row['by_rank']})}")
        loaded.append((report.get("temperature"), path, entries))
    loaded.sort(key=lambda item: (item[0] is None, item[0]))

    # The questions every temperature could make a choice on. Comparing anything
    # else across temperatures compares question sets as much as readouts.
    shared = None
    for _, _, entries in loaded:
        usable = {key for key, entry in entries.items() if entry["reason"] is None}
        shared = usable if shared is None else (shared & usable)
    print(f"{len(shared)} questions are non-degenerate at every temperature given "
          f"({', '.join(str(t) for t, _, _ in loaded)})\n")

    findings = {}
    for temperature, path, entries in loaded:
        name = f"t={temperature}" if temperature is not None else path
        print(name)
        everything = accuracy(entries, entries)
        report_line("as the gate reports it", *everything)
        fixed = accuracy(entries, shared)
        report_line("on the shared question set", *fixed)

        by_count = defaultdict(list)
        for key, entry in entries.items():
            if entry["reason"] is None and key in shared:
                by_count[entry["candidates"]].append(entry)
        counted = {}
        for size in sorted(by_count):
            rows = by_count[size]
            first = sum(entry["first_right"] for entry in rows) / len(rows)
            readout = sum(entry["readout_right"] for entry in rows) / len(rows)
            counted[size] = {"questions": len(rows), "first": first, "readout": readout}
            report_line(f"shared, {size} candidates", first, readout, len(rows))

        # Where the readout puts its answer, among questions that offered the
        # position at all, which is the conditioning the raw histogram lacks.
        chosen = defaultdict(int)
        for entry in entries.values():
            if entry["reason"] is None:
                chosen[entry["readout_index"]] += 1
        total = max(sum(chosen.values()), 1)
        print("    readout index      " + "  ".join(
            f"{index}:{chosen.get(index, 0) / total:.2f}" for index in sorted(chosen)))
        findings[name] = {"reported": dict(zip(("first", "readout", "n"), everything)),
                          "shared": dict(zip(("first", "readout", "n"), fixed)),
                          "by_candidate_count": counted,
                          "readout_index_share": {str(k): v / total for k, v in chosen.items()}}
        print()

    print("Reading it: if the gap moves with the candidate count inside the shared set, the")
    print("construction explains it. If the gap is there as reported and gone on the shared")
    print("set, the denominator explains it. If it survives both cuts, the readout does.")

    if arguments.output:
        from litjev.run.streaming import write_json_atomic

        write_json_atomic(arguments.output,
                          {"reports": arguments.reports, "rank": arguments.rank,
                           "shared_questions": len(shared), "by_temperature": findings})
        print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
