"""Routers at fixed rules, each reported at the cost it actually spends.

Sweeping a threshold over the test set and reading the best point off the
resulting curve is not something a deployment can do: it picks the operating
point with the answers in hand. So every router here is one rule, stated in
advance, and the table reports what that rule spends as well as what it reaches.

    head          escalate when the estimated gain exceeds the published price
    confidence    escalate when the fast mode's top probability falls below c,
                  for the round values a practitioner would reach for
    random        escalate each question independently with probability one half

Because the rules are fixed and not matched to one another, the routers do not
spend the same number of tokens, so accuracy alone does not order them: read
both columns, and a router is only ahead where it wins on each.

    uv run python scripts/fixed_rules.py results/records-sglang-30813.npz \
        --head results/head-27b-content.safetensors
"""
import argparse
import json
from math import comb

import numpy as np

from hde.core.block import escalation_gain
from hde.data.collect import open_records
from hde.serving import load_head
from hde.train.training import features_for, holdout_mask

CONFIDENCES = (0.9, 0.8, 0.5, 0.3)
PRICES = (0.0, 0.05, 0.1, 0.5)   # the prices Table 1 publishes


def mcnemar(left, right):
    """Two-sided exact binomial over the items exactly one of two rules gets right.

    The accuracy difference between two rules is carried entirely by those items,
    and on a benchmark of a few hundred there are often fewer than ten of them,
    so the difference is reported with the count it rests on rather than alone.
    """
    only_left = int((left & ~right).sum())
    only_right = int((~left & right).sum())
    total = only_left + only_right
    if not total:
        return only_left, only_right, float("nan")
    near = min(only_left, only_right)
    mass = sum(comb(total, i) for i in range(near + 1)) / 2 ** total
    return only_left, only_right, min(1.0, 2 * mass)


def outcome(escalate, fast, slow, tokens):
    """What one rule costs and reaches, and what each of its escalations bought.

    Accuracy and total tokens say what a rule reached and what it spent, but not
    how it spent it. A rule that escalates twice as often for the same accuracy
    paid for escalations that bought nothing, and the aggregate token column
    absorbs that without naming it. So the escalation set is split four ways:

        useful   escalated, fast wrong and slow right -- bought an answer
        wasted   escalated, both modes agree -- tokens spent, nothing changed
        harmful  escalated, fast right and slow wrong -- tokens spent, answer lost
        missed   not escalated, though thinking would have rescued it

    `yield` is useful over escalated: of the escalations the rule made, the share
    that bought something. It is counted over every escalation a rule makes --
    eighty to a hundred of them -- rather than over the handful of items two
    rules' accuracies differ by, so it separates a discriminating router from a
    spending one where the accuracy column cannot.
    """
    escalated = int(escalate.sum())
    useful = int((escalate & ~fast & slow).sum())
    harmful = int((escalate & fast & ~slow).sum())
    wasted = escalated - useful - harmful
    return {"rate": float(escalate.mean()),
            "tokens": float(np.where(escalate, tokens, 0.0).mean()),
            "accuracy": float(np.where(escalate, slow, fast).mean()),
            "escalated": escalated, "useful": useful, "wasted": wasted,
            "harmful": harmful, "missed": int((~escalate & ~fast & slow).sum()),
            "yield": (useful / escalated) if escalated else float("nan")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records")
    parser.add_argument("--head", required=True)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--held-out", action="store_true",
                        help="score only the categories this head was not fitted on, which "
                             "is what a collection the head was trained from can answer; a "
                             "benchmark it only transfers to is held out in full already")
    parser.add_argument("--seeds", type=int, default=20,
                        help="independent Bernoulli draws, for random's spread")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    records, metadata = open_records(arguments.records)
    head = load_head(arguments.head)
    # A head fitted under one readout reads a different quantity under the other
    # and nothing in its output would say so, so the mismatch is refused here.
    collected = metadata.get("readout", "coded")
    if head.metadata.readout != collected:
        raise SystemExit(
            f"this head was fitted on the {head.metadata.readout!r} readout and these "
            f"records were collected under {collected!r}")

    position = list(metadata["feature_layers"]).index(arguments.layer)
    keep = (np.asarray(holdout_mask(records["category"], 0.25, 0)[0], dtype=bool)
            if arguments.held_out else np.ones(len(records["fast_correct"]), dtype=bool))
    fast = np.asarray(records["fast_correct"], dtype=bool)[keep]
    slow = np.asarray(records["slow_correct"], dtype=bool)[keep]
    tokens = np.asarray(records["thinking_tokens"], dtype=np.float64)[keep]
    top = np.asarray(records["stats"], dtype=np.float64)[keep, 0]
    features = features_for(records, (position,))[keep]
    gains = np.array([escalation_gain(row) for row in head.predict(features)])
    n = len(fast)

    print(f"{arguments.records}: {metadata['model_id'].split('/')[-1]}, readout "
          f"{collected}, budget {metadata.get('budget')}, {n} items"
          + (" (held-out categories)" if arguments.held_out else ""))
    # What any router can win: the items the fast mode gets wrong and the slow
    # mode gets right. Nothing a router does to the rest changes the accuracy,
    # so this count, not the item count, is what separates two rules.
    rescuable = int((~fast & slow).sum())
    spoiled = int((fast & ~slow).sum())
    print(f"fast {fast.mean():.4f} at 0 tokens, slow {slow.mean():.4f} at "
          f"{tokens.mean():.0f} tokens/q")
    print(f"{rescuable} items thinking rescues, {spoiled} it spoils: the whole "
          f"margin any router can win or lose\n")

    rows = {}
    for price in PRICES:
        rows[f"head, lambda={price}"] = outcome(gains > price, fast, slow, tokens)
    for cut in CONFIDENCES:
        rows[f"confidence < {cut}"] = outcome(top < cut, fast, slow, tokens)

    # Each question is drawn on its own, which is the rule as stated; the spread
    # over seeds is what a single draw would have been worth reporting without.
    draws = [outcome(np.random.default_rng(seed).random(n) < 0.5, fast, slow, tokens)
             for seed in range(arguments.seeds)]
    keys = ("rate", "tokens", "accuracy", "escalated", "useful", "wasted",
            "harmful", "missed", "yield")
    rows["random, p=0.5"] = {key: float(np.mean([draw[key] for draw in draws]))
                             for key in keys}
    spread = {key: float(np.std([draw[key] for draw in draws], ddof=1))
              for key in keys}

    print(f"{'rule':24s} {'rate':>7s} {'tokens/q':>9s} {'accuracy':>9s}")
    for name, row in rows.items():
        line = (f"{name:24s} {row['rate']:7.1%} {row['tokens']:9.0f} "
                f"{row['accuracy']:9.4f}")
        if name.startswith("random"):
            line += f"   +- {spread['accuracy']:.4f} over {arguments.seeds} draws"
        print(line)

    print(f"\nWhat each rule's escalations bought, of the {rescuable} thinking rescues")
    print(f"{'rule':24s} {'escalated':>9s} {'useful':>7s} {'wasted':>7s} "
          f"{'harmful':>8s} {'missed':>7s} {'yield':>7s}")
    for name, row in rows.items():
        print(f"{name:24s} {row['escalated']:9.0f} {row['useful']:7.0f} "
              f"{row['wasted']:7.0f} {row['harmful']:8.0f} {row['missed']:7.0f} "
              f"{row['yield']:7.1%}")

    print("\nEach head rule against each confidence rule. A pair is decided only "
          "where one\nside is both cheaper and more accurate; p is over the items "
          "exactly one gets right.")
    print(f"{'pair':26s} {'d tokens':>9s} {'d acc':>8s} {'win/lose':>10s} {'p':>6s}  verdict")
    for price in PRICES:
        he = gains > price
        for cut in CONFIDENCES:
            ce = top < cut
            ha, ca = np.where(he, slow, fast), np.where(ce, slow, fast)
            delta_tokens = (np.where(he, tokens, 0.0).mean()
                            - np.where(ce, tokens, 0.0).mean())
            delta_accuracy = ha.mean() - ca.mean()
            win, lose, p = mcnemar(ha, ca)
            if delta_tokens <= 0 and delta_accuracy >= 0:
                verdict = "head dominates" if p < 0.05 else "head cheaper, accuracy tied"
            elif delta_tokens >= 0 and delta_accuracy <= 0:
                verdict = "confidence dominates" if p < 0.05 else "confidence cheaper, accuracy tied"
            else:
                verdict = "incomparable, one axis each"
            name = f"head l={price} vs conf<{cut}"
            print(f"{name:26s} {delta_tokens:+9.0f} {delta_accuracy:+8.4f} "
                  f"{win:4d}/{lose:4d} {p:6.3f}  {verdict}")

    if arguments.output:
        from hde.run.streaming import write_json_atomic
        write_json_atomic(arguments.output, {
            "records": arguments.records, "head": arguments.head,
            "model_id": metadata["model_id"], "readout": collected,
            "budget": metadata.get("budget"), "items": n,
            "held_out": bool(arguments.held_out),
            "fast": float(fast.mean()), "slow": float(slow.mean()),
            "slow_tokens": float(tokens.mean()),
            "rescuable": rescuable, "spoiled": spoiled,
            "rules": rows, "random_sd": spread, "seeds": arguments.seeds})
        print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
