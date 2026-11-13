"""How much of WebArena the enumerated text actions could reach, measured.

Option (b) of the action-space decision offers `fill`/`select_option` over spans
of the goal, so a task is reachable only if the value it is scored on appears in
its goal. That is a claim about 694 tasks, and the first version of it in the
README was "most of the 411 program_html ones" -- an estimate, which code-reviewer
replaced with a count. This is that count, as a script, so it is rerun whenever the
span rules change rather than remembered.

Per task, every value its validators require is compared against `goal_spans` of
its intent:

    exact     every required value is a span of the goal
    partial   some are
    none      none are

`none` is an upper bound on what is unreachable, not a count of it: a task
needing no typing at all lands there too, and clicking may be enough for it. The
honest reading is that `exact` is reachable, `none` is not reachable *by typing*,
and `partial` needs a look.

    python scripts/webarena_span_coverage.py [-o out.json]
"""

import argparse
import json
import re
from pathlib import Path


def required_values(config):
    """Every string a task's validators require, with where it has to appear.

    The channel matters, which the first version of this missed. A value under
    `reference_answers` is the **answer the agent sends**; one under
    `program_html` is text that has to end up **on the page**. Typing can satisfy
    those two in different ways, so they cannot share one coverage rule.
    """
    values = []
    evaluation = config.get("eval") or {}
    for check in evaluation.get("program_html") or []:
        contents = check.get("required_contents") or {}
        for key in ("must_include", "exact_match", "must_exclude"):
            value = contents.get(key)
            for item in ([value] if isinstance(value, str) else list(value or [])):
                values.append((str(item), key, "page"))
    references = evaluation.get("reference_answers") or {}
    for key in ("must_include", "exact_match"):
        value = references.get(key)
        for item in ([value] if isinstance(value, str) else list(value or [])):
            values.append((str(item), key, "answer"))
    # A `url_match` task has no required *content*, which the first version read as
    # "needs no typing". 16 of the 66 are graded on a search-results URL --
    # `Search for "usb wifi"` wants `catalogsearch/result/?q=usb+wifi` -- and a
    # query parameter is something typed into a box. code-reviewer found this;
    # treating those as click-only overstated the click-only ceiling by 16.
    for value in query_values(evaluation.get("reference_url") or ""):
        values.append((value, "query", "url"))
    return [row for row in values if row[0].strip()]


def query_values(url):
    """The query terms a reference URL carries, which have to be typed somewhere."""
    from urllib.parse import parse_qs, unquote_plus, urlsplit

    found = []
    for key, items in parse_qs(urlsplit(url).query).items():
        if key.lower() in ("q", "query", "search", "s", "keyword"):
            found.extend(unquote_plus(item) for item in items)
    return [value for value in found if value.strip()]


def covered_by(value, key, channel, spans):
    """Whether typing one of `spans` satisfies this requirement, conservatively.

    Three rules rather than one, because what counts differs by channel, and the
    first version used `value in span or span in value` both ways -- which scored
    the span `books` as covering a full reddit URL, the span
    `a11yproject on 3/5/2023` as covering the required answer `1`, and three
    university names as covering "The order is Princeton University, Yale
    University, Harvard University". None of those is produced by typing the span.
    code-reviewer found all three.

    - **answer channel: equality.** `must_include` on an answer is satisfiable by
      sending any superstring, so `"1"` would be "covered" by sending a date that
      happens to contain a 1. That scores a validator loophole as a capability.
      The claim is that the enumerated set can *answer*, so the span has to be the
      answer.
    - **url channel: equality.** A query parameter is typed, and a longer phrase
      types a different query.
    - **page, `exact_match`: equality.** The field must end up equal to it.
    - **page, `must_include`: a whole-word occurrence.** Typing a superstring does
      put the value on the page, so containment is legitimate here -- but on word
      boundaries, or a one-character requirement is met by any span containing
      that character.

    `span in value` is never coverage: typing `books` does not put a whole URL
    anywhere.
    """
    value = value.strip().lower()
    for span in spans:
        span = span.strip().lower()
        if not span:
            continue
        if span == value:
            return True
        if channel == "url":
            # The query has to be typed exactly: a longer phrase types a different
            # query. `Search for "batteries for iphone 13"` is graded on
            # `q=iphone+13`, so the quoted span does not satisfy it, and counting
            # that as reachable is the generous reading this script exists to stop
            # making.
            continue
        if channel == "page" and key == "must_include" and value:
            if re.search(rf"(?<!\w){re.escape(value)}(?!\w)", span):
                return True
    return False


def classify(config, spans):
    """Whether typing could satisfy every requirement, some of them, or none."""
    values = required_values(config)
    if not values:
        return "no_required_values"
    covered = [covered_by(value, key, channel, spans) for value, key, channel in values]
    if all(covered):
        return "exact"
    return "partial" if any(covered) else "none"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output",
                        default="experiments/browser-gate/webarena-span-coverage.json")
    parser.add_argument("--spans", type=int, default=4)
    args = parser.parse_args()

    from collections import Counter

    from slime_hdm.envs.browser import goal_spans
    from slime_hdm.envs.gate import _webarena_configs, judged_by_a_model

    configs = [c for c in _webarena_configs() if not judged_by_a_model(c)]
    rows, counts = {}, Counter()
    by_validator = {}
    for config in configs:
        spans = goal_spans(config.get("intent", ""), limit=args.spans)
        verdict = classify(config, spans)
        kinds = tuple(sorted((config.get("eval") or {}).get("eval_types") or []))
        rows[str(config["task_id"])] = {
            "verdict": verdict, "spans": spans,
            "required": [{"value": value, "rule": key, "channel": channel}
                         for value, key, channel in required_values(config)],
            "eval_types": list(kinds),
        }
        counts[verdict] += 1
        by_validator.setdefault("+".join(kinds), Counter())[verdict] += 1

    record = {"tasks": len(configs), "spans_per_goal": args.spans,
              "counts": dict(counts),
              "by_validator": {k: dict(v) for k, v in by_validator.items()},
              "per_task": rows}
    Path(args.output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"{len(configs)} tasks, {args.spans} spans per goal")
    for verdict, count in counts.most_common():
        print(f"  {count:4d}  {verdict}")
    print("\nby validator:")
    for kinds, inner in sorted(by_validator.items()):
        print(f"  {kinds:28} " + "  ".join(f"{v}={n}" for v, n in sorted(inner.items())))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
