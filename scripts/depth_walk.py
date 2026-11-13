"""Run the sixteen-arrow walk at both ends of the price, and record the trace.

The paper's appendix asserted that the 4B backbone answers k=1 at 43 per cent and
k=2 at chance, and k=14 at 99 per cent with thinking switched on. Those figures
were removed from the paper because nothing outside it recorded them. This writes
the record: every question, both modes' answers, the gold answer, the tokens and
seconds each mode spent, what the slow mode wrote, and the value the head put on
thinking before either mode ran.

    python scripts/depth_walk.py --base http://127.0.0.1:8800 --out results/walk.json

Two requests per depth, not one per question: the fast pass batches a request's
questions behind one shared prefill, which is the serving path the numbers should
come from. lambda = 1 is never-think and lambda = -1 is always-think, the two ends
of the slider and the two rows a routed result is read against.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litjev.data.depth_walk import SYMBOLS, as_request, sweep  # noqa: E402

NEVER, ALWAYS = 1.0, -1.0


def post(base, body, timeout):
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/systemtwo/debug",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return json.loads(reply.read())
    except urllib.error.HTTPError as error:
        detail = json.loads(error.read() or b"{}").get("detail", str(error))
        raise SystemExit(f"the server refused the request: {detail}") from error


def run(base, steps, instances, budget, seed, timeout):
    arrows, by_depth = sweep(steps, instances=instances, seed=seed)
    rows = []
    for depth in steps:
        state, schema, gold = by_depth[depth]
        sides = {}
        for label, price in (("fast", NEVER), ("slow", ALWAYS)):
            started = time.perf_counter()
            payload = post(base, as_request(state, schema, lambda_=price, budget=budget), timeout)
            sides[label] = (payload, time.perf_counter() - started)
            print(f"  k={depth:<3} {label:4} "
                  f"{payload['diagnostics']['timing']['decision_seconds']:7.2f} s  "
                  f"{payload['result']['usage']['output_tokens']:6} tokens", flush=True)

        for name in gold:
            entry = {"depth": depth, "question": name, "gold": gold[name],
                     "instructions": schema[name].instructions}
            for label, (payload, _wall) in sides.items():
                answer = payload["result"]["answers"][name]
                field = payload["diagnostics"]["fields"][name]
                entry[label] = {
                    "answer": answer.get(answer["type"]),
                    "correct": str(answer.get(answer["type"])) == gold[name],
                    "probabilities": answer.get("probabilities"),
                    "tokens": field.get("generated_tokens", 0),
                    "mode": field["system"],
                    "reasoning": field.get("reasoning"),
                }
            # The head speaks before either mode runs, so it is read from the
            # never-think pass: the same v it would price any lambda against.
            fast_field = sides["fast"][0]["diagnostics"]["fields"][name]
            entry["value_of_thinking"] = fast_field.get("escalation_gain")
            entry["confidence"] = fast_field.get("concentration")
            rows.append(entry)

        for label, (payload, wall) in sides.items():
            rows[-1].setdefault("timing", {})[label] = {
                "round_trip_seconds": wall, **payload["diagnostics"]["timing"]}
    return arrows, rows


def summarise(rows):
    """Accuracy per depth per mode, and what the head said, as the demo plots it."""
    depths = sorted({row["depth"] for row in rows})
    out = []
    for depth in depths:
        here = [row for row in rows if row["depth"] == depth]
        values = [row["value_of_thinking"] for row in here
                  if row["value_of_thinking"] is not None]
        out.append({
            "depth": depth,
            "n": len(here),
            "fast_accuracy": sum(row["fast"]["correct"] for row in here) / len(here),
            "slow_accuracy": sum(row["slow"]["correct"] for row in here) / len(here),
            "rescued": sum(row["slow"]["correct"] and not row["fast"]["correct"]
                           for row in here) / len(here),
            "spoiled": sum(row["fast"]["correct"] and not row["slow"]["correct"]
                           for row in here) / len(here),
            "slow_tokens": sum(row["slow"]["tokens"] for row in here) / len(here),
            "mean_value_of_thinking": (sum(values) / len(values)) if values else None,
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8800")
    parser.add_argument("--max-depth", type=int, default=14)
    parser.add_argument("--instances", type=int, default=SYMBOLS)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--out", default="results/depth-walk.json")
    args = parser.parse_args()

    steps = list(range(1, args.max_depth + 1))
    print(f"{len(steps)} depths x {args.instances} walks x 2 modes, chance = 1/{SYMBOLS}",
          flush=True)
    arrows, rows = run(args.base, steps, args.instances, args.budget, args.seed, args.timeout)
    trace = {
        "schema_version": "litjev.compare.v1",
        "task": "sixteen-arrow walk",
        "chance": 1 / SYMBOLS,
        "arrows": {str(k): v for k, v in sorted(arrows.items())},
        "settings": {"instances": args.instances, "budget": args.budget,
                     "seed": args.seed, "never_think": NEVER, "always_think": ALWAYS},
        "summary": summarise(rows),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=1, allow_nan=False))
    print(f"\nwrote {out}")
    print(f"{'k':>3} {'fast':>7} {'slow':>7} {'rescued':>8} {'spoiled':>8} "
          f"{'slow tok':>9} {'mean v':>8}")
    for line in trace["summary"]:
        value = line["mean_value_of_thinking"]
        print(f"{line['depth']:>3} {line['fast_accuracy']:>7.3f} {line['slow_accuracy']:>7.3f} "
              f"{line['rescued']:>8.3f} {line['spoiled']:>8.3f} {line['slow_tokens']:>9.0f} "
              f"{'-' if value is None else f'{value:>8.4f}'}")


if __name__ == "__main__":
    main()
