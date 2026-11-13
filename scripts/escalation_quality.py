"""Where two rules disagree about escalating, whose choice was right.

Yield over a rule's whole escalation set is not comparable between rules that
escalate different amounts: any rule looks precise when it escalates seldom. So
this takes the items exactly one of the two escalates and asks what share of each
side's unilateral escalations were rescues.

**The share, not the count.** Counting them and testing the split at one half
assumes the two sides escalated unilaterally the same number of times, and when
one rule's escalation set contains the other's that number is zero on one side:
the rule that escalates more picks up more rescues for no reason but escalating
more. The rates are compared with Fisher's exact test on the two-by-two of
rescue against which side escalated.
"""
import argparse
from math import comb, lgamma

import numpy as np

from litjev.core.block import escalation_gain
from litjev.data.collect import open_records
from litjev.serving import load_head
from litjev.train.training import features_for, holdout_mask

def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]], summing tables no likelier than this.

    No scipy in the serving environment, and a hypergeometric sum over margins
    this small is a few dozen terms.
    """
    def log_choose(n, k):
        if k < 0 or k > n:
            return float("-inf")
        return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

    row1, row2, col1 = a + b, c + d, a + c
    total = row1 + row2
    if not (row1 and row2 and col1 and (b + d)):
        return float("nan")
    import math
    probs = [math.exp(log_choose(row1, i) + log_choose(row2, col1 - i) - log_choose(total, col1))
             for i in range(max(0, col1 - row2), min(row1, col1) + 1)]
    observed = math.exp(log_choose(row1, a) + log_choose(row2, c) - log_choose(total, col1))
    return min(1.0, sum(p for p in probs if p <= observed * (1 + 1e-9)))


parser = argparse.ArgumentParser()
parser.add_argument("records"); parser.add_argument("head")
parser.add_argument("--held-out", action="store_true")
args = parser.parse_args()

records, metadata = open_records(args.records)
head = load_head(args.head)
pos = list(metadata["feature_layers"]).index(-1)
keep = (np.asarray(holdout_mask(records["category"], 0.25, 0)[0], bool)
        if args.held_out else np.ones(len(records["fast_correct"]), bool))
fast = np.asarray(records["fast_correct"], bool)[keep]
slow = np.asarray(records["slow_correct"], bool)[keep]
tok = np.asarray(records["thinking_tokens"], float)[keep]
top = np.asarray(records["stats"], float)[keep, 0]
gain = np.array([escalation_gain(r) for r in head.predict(features_for(records, (pos,))[keep])])
rescuable = ~fast & slow

rules = {f"head l={p}": gain > p for p in (0.0, 0.5)}
rules.update({f"conf<{c}": top < c for c in (0.9, 0.8, 0.5, 0.3)})
rules["random"] = np.random.default_rng(0).random(len(fast)) < 0.5

print(f"{args.records.split('/')[-1]}  {metadata['model_id'].split('/')[-1]}  "
      f"{len(fast)} items{', held-out' if args.held_out else ''}, "
      f"{int(rescuable.sum())} rescuable")
print("\nOf each side's unilateral escalations, the share that were rescues")
print(f"{'pair':24s} {'head alone':>16s} {'other alone':>16s} {'p':>7s}  reading")
for a in ("head l=0.0", "head l=0.5"):
    for b in ("conf<0.9", "conf<0.8", "conf<0.5", "conf<0.3", "random"):
        only_a, only_b = rules[a] & ~rules[b], rules[b] & ~rules[a]
        na, nb = int(only_a.sum()), int(only_b.sum())
        ra, rb = int((only_a & rescuable).sum()), int((only_b & rescuable).sum())
        p = fisher_exact(ra, na - ra, rb, nb - rb)
        if not na or not nb:
            reading = f"no disagreement on one side ({a if not na else b} escalates a subset)"
        else:
            share_a, share_b = ra / na, rb / nb
            reading = ("head better" if share_a > share_b else
                       "other better" if share_b > share_a else "tied")
            if not (p == p) or p >= 0.05:
                reading += ", not significant"
        pa = f"{ra:3d}/{na:3d} {ra/na:5.1%}" if na else f"{ra:3d}/{na:3d}   --  "
        pb = f"{rb:3d}/{nb:3d} {rb/nb:5.1%}" if nb else f"{rb:3d}/{nb:3d}   --  "
        print(f"{a+' vs '+b:24s} {pa:>16s} {pb:>16s} {p:7.3f}  {reading}")
