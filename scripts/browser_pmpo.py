"""HDM + CDPO against its baselines on BrowserGym, which is the paper's missing result.

Every head that produces a number in the paper is `objective: supervised`, and
CDPO's only evidence is a 4B ablation in the one regime its own appendix says the
method is **not** for: a single question, both arms recorded, the expected reward
exact, cross-entropy available. This is the other regime -- an episode whose
reward arrives once at the end, where no decision has an outcome of its own -- and
it is the claim the method was written for.

Four policies, on **the same evaluation tasks**, so the comparison is a comparison:

    never       the fast mode at every step, token-free
    always      reason and write the action at every step
    smeared CE  the same head and features, trained by cross-entropy on the
                episode's success smeared onto its steps -- baseline (A)
    CDPO        the same head and features, trained on episode returns with the
                per-decision importance ratio and the clipped surrogate

What the two trained heads share is everything except the loss: one
`flatten_episodes` call produces the rows for both, the trunk is the same, and at
evaluation both are read through the log-ratio gain against the same lambda.
The repository has made the opposite mistake before -- the single-question
comparison failed because the baseline carried a class weighting the method did
not -- so nothing here is allowed to handicap it.

Baseline (B), cross-entropy on one-step counterfactual labels, is **not** here: it
needs a fork at every step the two modes disagree on, run to termination, which is
a separate and much larger collection. It is the strongest baseline and it is the
honest next one; this run cannot claim to have beaten it.

**Baseline (C) is measured by this script and does not come from the gate.** I had
told exps that the gate's paired `never`/`always` episodes were (C) on the same
tasks; exps pointed out that this only holds if the tasks, the seed, `--max-steps`,
`--budget` and the action set all match, which they do not -- the gate runs all 54
click tasks and this evaluates on the held-out quarter of them. Aligning the two
would mean evaluating on tasks the head trained on, which is the thing
`split_tasks` exists to prevent. So the constant policies are run here, on the
same held-out tasks, at the same seed, through the same rollout, and the gate is
the **precondition** -- is there anywhere declining is safe -- and a cost
reference, not a baseline column.

The venue is MiniWoB through `slime_htm.envs.browser_environment`, which is
BrowserGym: no containers, no ServiceNow, nothing external.

    # beside a running scripts/serve_decision.sbatch, whose node it prints
    python scripts/browser_pmpo.py --decision-url http://$NODE:30001 \
      --head results/head-27b.safetensors \
      --venue miniwob --group click --output results/browser-pmpo

`--head` is read for its metadata only: which layer and what width. No weights are
reused, because each round trains a head from scratch on that round's returns.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from litjev._legacy.heads import DecisionHead
from litjev.run.provenance import commit_now
from litjev.run.streaming import write_json_atomic
from slime_htm import ClientScorer, HeadRLConfig, LoopConfig, run
from slime_htm.head_trainer import train_head_on_smeared_returns
from slime_htm.loop import BLIND_EPISODES, ENVIRONMENT_BROKEN
from slime_htm.resume import (
    append_row,
    attempts_by_index,
    checked_rows,
    episode_rows,
    read_rows,
    resume_plan,
)
from slime_htm.rollout import rollout_episodes

# What only a finished evaluation episode carries, for `slime_htm.resume`.
EVAL_KEY = "episode_steps"

# How much of the page the policy is shown. Named once, because the run's
# signature has to carry the same number the environment was built with.
STATE_CHARS = 6000


@dataclass
class Played:
    """One evaluated episode, kept as small as its two readers need.

    Not an `EpisodeResult`: that holds a hidden-state vector per step, about
    27 KB of it, and the evaluation has to write every episode to disk as it
    finishes. These are the fields `summarise` and `paired_test` touch, and the
    per-step pair behind them is what makes the escalation count auditable rather
    than asserted.
    """

    episode: int
    reward: float
    finished: bool = False
    ended: str | None = None
    steps: list = field(default_factory=list)
    # **Wall time, which no run before 2026-10-06 recorded.** Every time figure
    # in the gate README and in the slow-only cost table was derived from
    # 28831's 2.29s fast / 10.95s slow -- a different run over a different task
    # set -- and the paper's metric is score net of cost. It cannot be recovered
    # afterwards, so the cell schema has to carry it from the run that measures
    # it. exps raised this; the user approved adding it on 2026-10-06.
    # `None` means "this run did not record time", which is not the same as zero
    # and must not become zero on the way to a summary.
    seconds: float | None = None

    @property
    def timed(self):
        """Did the run that wrote this record times at all?

        **`None`, not `0.0`, for a cell written before 05e58bc** --
        code-reviewer's condition, and it is the pattern `5ca7ed3` had just
        finished fixing on `used_outside_action`: a `.get(key, 0.0)` turns "was
        never recorded" into "took no time", and a summary then prints `0.0s`
        for 28853's cells as though they had been measured. Absence has to stay
        distinguishable from zero all the way to the printed line.
        """
        return all(step.get("seconds") is not None for step in self.steps)

    @property
    def slow_seconds(self):
        """Time on the steps that escalated, which is what a price buys. `None`
        when this record predates timing."""
        if not self.timed:
            return None
        return sum(float(step["seconds"]) for step in self.steps
                   if step["escalated"])

    @property
    def fast_seconds(self):
        if not self.timed:
            return None
        return sum(float(step["seconds"]) for step in self.steps
                   if not step["escalated"])

    @property
    def escalations(self):
        return sum(bool(step["escalated"]) for step in self.steps)

    @property
    def generated_tokens(self):
        return sum(int(step["generated_tokens"]) for step in self.steps)


def digest(value):
    """A short, stable hash of anything JSON can hold. For signatures only."""
    import hashlib

    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str)
                        .encode()).hexdigest()[:12]


THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE")


def thread_settings():
    """The environment variables the head refit's result depends on.

    Belongs in `evaluation_signature`, not the collection's: threads change the
    fitted heads, and at `--rounds 1` the collection trains nothing -- it plays
    the explore coin. Guarding the corpus with them would discard a valid 81
    episode collection because a resubmission landed on a differently sized
    node. code-reviewer caught that before the run started; I had put them in
    the wrong one.
    """
    import os

    return {name: os.environ.get(name) for name in THREAD_VARIABLES}


def marginal_value(episodes):
    """What one escalation bought, in score, seconds and tokens.

    **The number 28853 should have been priced from and was not.** Its
    `lambda = 0.1` was the midpoint of the interval the head trains over, chosen
    without asking what an escalation is worth; afterwards it came out at about
    0.003 of a point, so 0.1 put every routing policy at a loss and the
    pre-registered test compared two losers. That was a design error, and this is
    the function that prevents the next one.

    The estimator is the one the pre-registration names, fixed before any number
    exists: a least squares slope of the episode's terminal reward on its
    escalation **count**, over the episodes of one seed. Per seed, never pooled:
    two single-seed numbers have already been overturned on this venue in one
    day. Seconds and tokens are the same slope against the same count, which is
    what makes a price per second expressible rather than only a price per
    escalation.

    **Do not price a run from this. 29074 showed the confounding flips its
    sign.** The caveat below was already here -- episodes differ in difficulty
    and the behaviour policy escalated at random, so the slope mixes "thinking
    helped" with "hard episodes got more thinking" -- but I wrote it as a
    limitation and named this function in the pre-registration as the thing a
    price should come from. On 29074's corpus it reports **-0.0739 reward per
    escalation**, while every direct between-policy contrast on the same run is
    non-negative:

        this estimator (within-corpus regression)   -0.0739
        always against never                        +0.0191
        each of the nine routed cells vs never      +0.0000 .. +0.1250

    **Seven positive, two exactly zero, none negative** -- `priced_ce` at seeds
    0 and 2 move the reward not at all. "Nine of nine positive" was my phrasing
    and code-reviewer corrected it. The direction still disagrees with the
    slope, and that is the point: the confounding does not shade the magnitude,
    it reverses the sign, so a price chosen from the slope would be chosen from
    the wrong one.

    **One limit on the comparison, also code-reviewer's.** The slope is fitted
    over the *training* tasks the corpus was collected on and the contrasts are
    over the *held-out* tasks the cells evaluate, so the two are not computed on
    the same questions. That is a reason to trust neither as a point estimate
    and no reason to trust the slope: a sign that flips between training and
    held-out tasks is not a quantity to price with either way. `between_policy_value` is
    what a price should come from; this stays because the number is free and
    worth recording, and because the gap between the two is itself the
    demonstration that an observational slope cannot price a decision.
    """
    import numpy as np

    by_seed = {}
    for episode in episodes:
        steps = episode.get("head_features") or []
        if not steps:
            continue
        seed = episode.get("seed", 0)
        by_seed.setdefault(seed, []).append((
            sum(1 for step in steps if step.get("escalated")),
            float(episode.get("terminal_reward", 0.0)),
            # `None` when the run that wrote this did not record time. Kept as
            # `None` rather than coerced to 0.0, because a slope fitted against
            # zeros would print `+0.00s per escalation` and read as a
            # measurement that thinking is instantaneous.
            episode.get("seconds"),
            sum(int(step.get("generated_tokens", 0)) for step in steps)))

    report, lines = {}, []
    for seed, rows in sorted(by_seed.items()):
        counts = np.array([row[0] for row in rows], dtype=float)
        entry = {"episodes": len(rows), "mean_escalations": float(counts.mean())}
        for index, name in ((1, "reward"), (2, "seconds"), (3, "tokens")):
            raw = [row[index] for row in rows]
            # Two different reasons to have no answer, kept apart: the quantity
            # was never recorded, or it was recorded and did not vary.
            if any(value is None for value in raw):
                entry[f"per_escalation_{name}"] = None
                entry[f"{name}_recorded"] = False
                continue
            entry[f"{name}_recorded"] = True
            if counts.std() < 1e-9:
                entry[f"per_escalation_{name}"] = None
                continue
            slope, _ = np.polyfit(counts, np.array(raw, dtype=float), 1)
            entry[f"per_escalation_{name}"] = float(slope)
        report[str(seed)] = entry
        reward = entry["per_escalation_reward"]
        if reward is None:
            tail = ("not recorded" if not entry.get("reward_recorded", True)
                    else "no variation in escalation count, slope undefined")
        else:
            seconds = entry["per_escalation_seconds"]
            tail = (f"{reward:+.4f} reward, "
                    + ("time not recorded" if seconds is None
                       else f"{seconds:+.2f}s")
                    + f", {entry['per_escalation_tokens']:+.0f} tokens per escalation")
        lines.append(f"seed {seed}: {len(rows)} episodes, "
                     f"{counts.mean():.2f} escalations/ep, {tail}")
    return {"per_seed": report, "lines": lines,
            "estimator": "least squares slope of terminal reward on escalation "
                         "count, per seed, not pooled",
            "caveat": "a correlation under a random behaviour policy, not a causal "
                      "effect: harder episodes also drew more escalations"}


def between_policy_value(cells, shared=None):
    """One escalation's worth, from the contrast between two policies.

    **This is the estimator a price should come from**, and
    `marginal_value`'s within-corpus slope is not: on 29074 the two disagree in
    sign, because the behaviour policy escalated at random but harder episodes
    still drew more escalations, and that confound is larger than the effect.

    `(reward_b - reward_a) / (escalations_b - escalations_a)` over the task
    indices both cells contain. **Not "both finished"**, which the first version
    of this docstring claimed and the code never did -- it intersects indices and
    does not look at `finished`. Said accurately rather than made true, because
    the pairing the cells already guarantee is by index. It is a contrast between two *policies* on the same tasks,
    so the difficulty mix is identical on both sides and cannot drive it. It is
    still not causal for a single decision -- the two policies differ in which
    steps they escalate as well as how many -- so it bounds an order of
    magnitude, which is all a price needs.

    Returns `None` when the escalation counts do not differ, rather than
    dividing by something near zero.
    """
    pairs = {}
    names = sorted(cells)
    if shared is None:
        shared = sorted(set.intersection(*(set(cells[n]) for n in names)))
    if not shared:
        return pairs

    def means(name):
        rows = [cells[name][index] for index in shared]
        return (sum(row.reward for row in rows) / len(rows),
                sum(row.escalations for row in rows) / len(rows))

    base_reward, base_escalations = means("never") if "never" in cells else (None, None)
    if base_reward is None:
        return pairs
    for name in names:
        if name == "never":
            continue
        reward, escalations = means(name)
        gap = escalations - base_escalations
        pairs[name] = None if abs(gap) < 0.05 else (reward - base_reward) / gap
    return pairs


def played_from(row):
    """A stored row back into the shape the readers take."""
    # `.get` for `seconds`: cells written before 2026-10-06 do not have it, and
    # a read of an old cell should give 0.0 rather than raise. The signature
    # already sets those files aside -- the commit changed -- so this is for
    # reading an archive, not for mixing one into a run.
    return Played(episode=int(row["index"]), reward=float(row["terminal_reward"]),
                  finished=bool(row.get("finished")), ended=row.get("ended"),
                  steps=row[EVAL_KEY],
                  seconds=(None if row.get("seconds") is None
                           else float(row["seconds"])))


def play(path, make_environment, scorer, head, *, price, episodes, budget, max_steps,
         seed, max_attempts, escalate_prob=None, signature=None):
    """One policy on the evaluation tasks, resumably. Returns {index: Played}.

    The evaluation is the larger half of this run -- five policies, three prices
    and three seeds come to about 460 episodes against 81 collected -- and it had
    no resume at all: a hang anywhere in it sent the next process back to the
    first evaluation episode. At the hang rate exps measured, roughly 1% an
    episode, a 460-episode phase that restarts from zero completes about once in
    a hundred attempts, which is not a run. So each cell streams its episodes and
    a restart reads them back.

    Resuming does not change what the policies do: a head decides deterministically
    from the step's own features, and the constant policies compare `rng.random()`
    against 0 or 1, so no draw of it can change the decision. The one thing a
    restart must not do is re-pair the tasks, and it cannot: every row carries the
    episode index it was played at.
    """
    rows = checked_rows(path, signature)
    attempts = attempts_by_index(rows, EVAL_KEY)
    pending, given_up = resume_plan(rows, episodes, max_attempts, EVAL_KEY)
    if given_up:
        print(f"    {len(given_up)} episode(s) given up after {max_attempts} "
              f"attempt(s): {given_up}", flush=True)
    if pending:
        if len(pending) < episodes:
            print(f"    resuming: {episodes - len(pending) - len(given_up)} already "
                  f"played, {len(pending)} to go", flush=True)
        environment = make_environment()

        def start(index):
            append_row(path, {"index": index, "attempt": attempts.get(index, 0),
                              "ended": "started"})

        dropped = []

        def keep(result):
            attempt = attempts.get(result.episode, 0)
            if not result.steps or not getattr(environment, "complete", True):
                ended = ("no_steps" if not result.steps
                         else getattr(environment, "ended", "incomplete"))
                print(f"    dropping episode {result.episode}, which ended {ended}",
                      flush=True)
                append_row(path, {"index": result.episode, "attempt": attempt,
                                  "ended": ended})
                dropped.append(ended)
                # **The collection's blind-run guard, on this path too.**
                # code-reviewer verified that a dead environment aborts the
                # collection at three dropped episodes; the evaluation had only
                # the per-index attempt cap, so a browser that never came back
                # would have burned 27 tasks x `--max-attempts` in **every** cell
                # -- about 1200 browser launches across fifteen of them -- before
                # the empty intersection stopped the run. Terminating, but a waste
                # of hours, and the asymmetry was mine: I told exps where the
                # bound was on each path and only then noticed one was far away.
                if (len(dropped) >= BLIND_EPISODES
                        and not episode_rows(read_rows(path), EVAL_KEY)):
                    print(f"    {len(dropped)} episode(s) of this cell dropped "
                          f"({sorted(set(dropped))}) with nothing kept, so it has "
                          f"collected nothing. Exiting {ENVIRONMENT_BROKEN}: do "
                          f"not resume on this code -- check one episode by hand",
                          flush=True)
                    raise SystemExit(ENVIRONMENT_BROKEN)
                return
            append_row(path, {
                "index": result.episode, "attempt": attempt,
                "terminal_reward": float(result.reward),
                "finished": bool(result.finished),
                "ended": result.ended,
                "seconds": float(getattr(result, "seconds", 0.0)),
                EVAL_KEY: [{"escalated": bool(step.escalated),
                            "generated_tokens": int(step.generated_tokens),
                            "seconds": float(getattr(step, "seconds", 0.0))}
                           for step in result.steps]})

        try:
            # Until nothing is pending, because a dropped evaluation episode is
            # not a hole in one policy: the paired read drops that task from
            # **every** policy, so one drop costs the comparison a task rather
            # than costing this cell an episode. The cap is what terminates it.
            while pending:
                rollout_episodes(
                    environment, scorer, head, lambda_=price, episodes=episodes,
                    budget=budget, max_steps=max_steps, escalate_prob=escalate_prob,
                    rng=np.random.default_rng(seed), on_episode=keep, on_start=start,
                    indices=pending)
                rows = read_rows(path)
                attempts = attempts_by_index(rows, EVAL_KEY)
                pending, given_up = resume_plan(rows, episodes, max_attempts, EVAL_KEY)
                if pending:
                    print(f"    retrying {len(pending)} dropped episode(s) {pending}",
                          flush=True)
                elif given_up:
                    print(f"    {len(given_up)} episode(s) given up after "
                          f"{max_attempts} attempt(s): {given_up}", flush=True)
        finally:
            environment.close()
    return {int(row["index"]): played_from(row)
            for row in episode_rows(read_rows(path), EVAL_KEY)}


def constant_policy(path, make_environment, scorer, *, escalate, **played):
    """`never` at escalate=0 and `always` at escalate=1, through the same rollout.

    Baseline (C). The rollout takes `escalate_prob` when there is no head, so the
    two constant policies need no separate implementation and cannot drift from
    the routed one. Their price is 0 because they have no decision to price: the
    value of the same episodes at each price is arithmetic in `summarise`.

    Returns the episodes, like `play`. It used to return `summarise(results)`,
    and the evaluation then called `summarise` on that dict and handed it to
    `paired_test` beside real episodes -- which would have died on the first
    constant policy with `AttributeError: 'str' object has no attribute 'reward'`.
    No test reached it because it needs a browser.
    """
    return play(path, make_environment, scorer, None, price=0.0,
                escalate_prob=float(escalate), **played)


def paired_test(mine, theirs, price):
    """Whether my value beats theirs **task by task**, with a p-value.

    Episodes are paired: the same 27 held-out tasks at the same seed under each
    policy, so the comparison is within task and not between means. The test is
    the gate's own -- an exact one-sided sign test on the discordant pairs, the
    tasks where the two policies' values differ -- because a mean difference over
    27 tasks with no test is a number and not a conclusion, and this repository
    refuses a verdict without one elsewhere.
    """
    from math import comb

    ahead = behind = 0
    for one, other in zip(mine, theirs, strict=True):
        a = one.reward - price * one.escalations
        b = other.reward - price * other.escalations
        if a > b:
            ahead += 1
        elif b > a:
            behind += 1
    discordant = ahead + behind
    # Both tails are reported because this function fed a table whose question
    # had no direction, and the one-sided value was read there as a test of "no
    # difference". It is not one: a one-sided p near 0.75 is equally consistent
    # with a large deficit the other way. `sign_test_p` stays one-sided for the
    # directional reads that cite it ("is this row behind slow-only"), and
    # `sign_test_p_two_sided` is the one to quote when no direction was fixed in
    # advance -- which is the default case.
    one_sided = (1.0 if discordant == 0
                 else sum(comb(discordant, k) for k in range(ahead, discordant + 1))
                 / 2 ** discordant)
    two_sided = (1.0 if discordant == 0 else min(
        1.0, 2 * sum(comb(discordant, k) for k in range(min(ahead, behind) + 1))
        / 2 ** discordant))
    return {"ahead": ahead, "behind": behind, "tied": len(mine) - discordant,
            "sign_test_p": one_sided, "sign_test_p_two_sided": two_sided}


def read_cells(output):
    """Every evaluation cell on disk, as `{(policy, price, seed): [Played]}`.

    For reading a run **back**. 28853 is running from `c0783e7`, which computes
    no `primary_test.seed`, no `reverse` and no `matched_seeds` -- so the outcome
    table in `experiments/browser-gate/README.md` names keys its `policies.json`
    will not have. code-reviewer pointed that out: it is my own "from `<sha>` on"
    practice applied to the reading side, which I had written for the producing
    side only.

    Rather than a second script with a second copy of the definitions -- the
    mistake that made the resume arithmetic wrong twice -- `--verdict` recomputes
    the report through the same `primary_verdict` and `matched_seeds` that a live
    run uses. The cell files are the state; the episodes are not replayed.
    """
    import re

    found = {}
    for path in sorted(Path(output).glob("eval-*.jsonl")):
        name = path.stem[len("eval-"):]
        played = [played_from(row) for row in episode_rows(read_rows(path), EVAL_KEY)]
        routed = re.fullmatch(r"(.+)-price([0-9.eE+-]+)-seed(\d+)", name)
        if routed:
            found[(routed.group(1), float(routed.group(2)),
                   int(routed.group(3)))] = played
        else:
            found[(name,)] = played
    return found


def shared_tasks(cells):
    """The episode indices every cell finished, which is the only pairable set."""
    if not cells:
        return []
    return sorted(set.intersection(*({p.episode for p in v} for v in cells.values())))


def matched_seeds(routed, seeds, price, mine="pmpo", against="priced_ce"):
    """Per-seed differences at one price, by **matched** seed.

    `mine` fitted at seed *s* against `against` fitted at seed *s* -- never
    against a mean across seeds, which would compare one initialisation with an
    average of three and call the difference a direction. code-reviewer asked for
    the definition before the numbers, because the README's outcome table decides
    which sentence the paper prints from `pmpo_ahead_at` and `reversed_at`.

    A pure function because the source-string test for the inline version could
    not fail: code-reviewer flipped `difference` to `theirs - mine` in a clean
    copy and the suite still passed 53/53. Third time today I asserted that a
    computation existed rather than that it was right, so this one is tested on
    hand-built episodes whose answer is known.
    """
    rows = []
    for seed in seeds:
        ours = summarise(routed[(mine, price, seed)], price)["value"]
        theirs = summarise(routed[(against, price, seed)], price)["value"]
        rows.append({"seed": seed, mine: ours, against: theirs,
                     "difference": ours - theirs})
    return {
        "price": price,
        "definition": f"{mine} at seed s minus {against} at seed s, by mean value "
                      f"at the middle price. Matched seeds, never against a mean",
        "rows": rows,
        "pmpo_ahead_at": [r["seed"] for r in rows if r["difference"] > 0],
        "reversed_at": [r["seed"] for r in rows if r["difference"] < 0],
    }


# **The price the primary test is registered at.** 28853 ran at 0.1 and that is
# what the pre-registration fixes, so a run whose `--lambdas` does not include it
# has no pre-registered primary -- it has a descriptive comparison at whatever
# price it did run. code-reviewer raised this when exps proposed `--lambdas 0.0`,
# where `middle = prices[0]` makes the single price look like an interval's
# midpoint and `<- PRIMARY` prints over a test nobody registered.
#
# Changing this constant is changing the pre-registration, which is a thing to do
# deliberately and before data, not by passing a different `--lambdas`.
PRIMARY_PRICE = 0.1


def primary_verdict(routed, price, seed, mine="pmpo", against="priced_ce"):
    """The pre-registered test, and the reverse one that gives "lost" a threshold.

    Both directions, because a deficit of one pair is not a defeat: overstating a
    negative result is the same error as overstating a positive one. The forward
    test is `mine` against `against`; `reverse` swaps them, so the README's table
    can read a loss off a threshold rather than off a sign.
    """
    forward = paired_test(routed[(mine, price)], routed[(against, price)], price)
    registered = abs(price - PRIMARY_PRICE) < 1e-9
    return {
        "mine": mine, "against": against, "price": price,
        # "registered" or "descriptive", so a reader of the JSON does not have to
        # know what the pre-registered price is to know whether they are looking
        # at the pre-registered test.
        "status": "registered" if registered else "descriptive",
        "registered_price": PRIMARY_PRICE,
        "reason": ("isolates the off-policy treatment at the price the heads were "
                   "trained around" if registered else
                   f"**descriptive, not the pre-registered test**: the primary is "
                   f"registered at price {PRIMARY_PRICE:g} and this run evaluated "
                   f"at {price:g}. The numbers below are a comparison at the price "
                   f"that was run, and no pre-registered sentence is licensed by "
                   f"them"),
        # Which seed the p-value is from: `routed[(name, price)]` aliases the
        # first seed and `paired_test` reads that, so this is 27 paired tasks at
        # one seed. A reader seeing one p would otherwise take it for the run.
        "seed": seed,
        "seed_note": "the paired test runs on the first training seed's episodes. "
                     "The other seeds appear in `matched_seeds`, not in this "
                     "p-value",
        **forward,
        "reverse": paired_test(routed[(against, price)], routed[(mine, price)],
                               price),
    }


def primary_heading(primary):
    """How to label a stored `primary_test`: (status, heading, note).

    A function so the defaulting is testable. **`"unknown"`, not
    `"registered"`** -- an absent key taking the favourable label is the shape
    three of today's defects had: `used_outside_action` missing read as "did not
    act outside" and manufactured a reasoning effect, `seconds` missing read as
    0.0, and the bid count missing read as "no bids to check". A JSON written
    before this field existed cannot say whether its price was the registered
    one, so it must not claim it was.
    """
    status = primary.get("status", "unknown")
    if status == "registered":
        return status, "primary", ""
    price = primary.get("registered_price", PRIMARY_PRICE)
    heading = f"{status.upper()} (registered price is {price:g})"
    note = "no pre-registered sentence is licensed by the line above"
    if status != "descriptive":
        note += ("; this file predates the status field, so whether its price "
                 "was the registered one is unrecorded")
    return status, heading, note


def summarise(results, price=None):
    """What a policy earned, and what it earned **net of what thinking cost**.

    `value = reward - price x escalations` is the headline, not accuracy.

    **The reason given here used to be the withdrawn one, and this is the third
    place it was found.** It read "28831 measured hurts = 0 on this group, so
    always-think *is* the oracle and no router can beat it on accuracy". That is
    **seed 0 only**: seed 1 has `hurts = 0.074`, and combined `oracle - always =
    +0.037`, so a router that declines where thinking hurts beats always-think on
    accuracy -- at any price, including zero. `docs/datasets.md` has carried the
    withdrawal since 2026-10-05 under my own name, and the argument still
    reappeared here and in the gate README. Withdrawing an argument is not the
    same as remembering that it was withdrawn.

    The real reason for the priced headline is the user's, 2026-10-05: the metric
    should be score net of cost. The quantity on which a win is possible is the one
    the
    objective already optimises: the routed reward net of the price. At a given
    price `always` pays for every step and a router that declines where declining
    is safe pays less for the same accuracy, so the combined score separates them
    where accuracy cannot.

    This is the user's correction of 2026-10-05, and it is the right one: the
    metric was wrong rather than the venue. Reporting accuracy and escalation side
    by side leaves a reader to combine them; the objective already says how.
    """
    if not results:
        return {"episodes": 0}
    rewards = np.array([r.reward for r in results], dtype=float)
    escalations = np.array([r.escalations for r in results], dtype=float)
    summary = {
        "episodes": len(results),
        "reward": float(rewards.mean()),
        "finished": float(np.mean([r.finished for r in results])),
        "escalations": float(escalations.mean()),
        "escalation_rate": float(
            escalations.sum() / max(sum(len(r.steps) for r in results), 1)),
        "tokens": float(np.mean([r.generated_tokens for r in results])),
        "steps": float(np.mean([len(r.steps) for r in results])),
        # Why the episodes stopped, counted here rather than left to a reader of
        # the logs. exps audited the two episode loops and found that the same
        # event -- the slow mode writing something unexecutable -- was `ended` on
        # the gate's path and a `metadata` key on this one, so a single count of
        # bad actions across both runs would have missed this path's silently.
        # The field is shared now and this is where the count lands.
        "ended": {reason: sum(1 for r in results if r.ended == reason)
                  for reason in sorted({r.ended for r in results if r.ended})},
    }
    if price is not None:
        summary["price"] = float(price)
        summary["value"] = float((rewards - price * escalations).mean())
    return summary


def recorded_lambdas(output):
    """The `--lambdas` **order** the run recorded, or None if it never wrote one.

    The order matters and sorting destroys it: the primary test's price is the
    middle of the list as given, so a run launched `0.2,0.0,0.1` pre-registered
    0.0. `policies.json` carries `lambdas` verbatim from `args.lambdas`, which
    makes it the authority over anything inferred from file names.
    """
    path = Path(output) / "policies.json"
    if not path.exists():
        return None
    try:
        found = json.loads(path.read_text()).get("lambdas")
    except (json.JSONDecodeError, OSError):
        return None
    return [float(value) for value in found] if found else None


def diagnostic_report(constants, routed, shared, prices, seeds, why):
    """每个 cell 自己的数字,**不做任何配对**。

    `--eval-only` 之后的分析原本假设五个策略都在:`cells = {("never",): ...}` 会
    KeyError,而且是在两个 cell 都播完(约 5 小时)之后才炸,exit 1、`stacks.log` 不增长、
    **不写任何报告**。code-reviewer 复现了 live 和 read-back 两条路。

    我的 `--eval-only` 帮助文本自己写的是「这张表是诊断而不是对比」,但代码照样去跑
    配对检验。这个函数就是那句话的实现:缺了策略,任何对比都不可比,所以只报每个 cell
    自己的数字,不报 primary、不报 seed spread、不报配对 p 值。
    """
    cells = {}
    for name, played in constants.items():
        cells[name] = summarise(played)
    for key, played in routed.items():
        if len(key) != 3:
            continue
        name, price, seed = key
        cells[f"{name}-price{price:g}-seed{seed}"] = summarise(played, price)
    return {
        "kind": "diagnostic",
        "why": why,
        "incomplete": True,
        "paired_tasks": len(shared),
        "lambdas": prices,
        "training_seeds": seeds,
        "cells": cells,
        "note": "没有配对检验、没有 primary、没有 seed spread:缺少策略时任何对比都"
                "不可比。要对比就把五个策略在同一个 --eval-max-steps 下全部跑完",
    }


def verdict(output):
    """Recompute the report's deciding fields from the cells already on disk.

    Through the **same** `primary_verdict` and `matched_seeds` a live run calls,
    so a run from a base that lacked them -- 28853 -- is read with `86bd61f`'s
    definitions rather than a second implementation of them.
    """
    cells = read_cells(output)
    constants = {name[0]: v for name, v in cells.items() if len(name) == 1}
    routed = {key: v for key, v in cells.items() if len(key) == 3}
    if not routed:
        raise SystemExit(f"no routed cells under {output}; nothing to read")
    prices = sorted({key[1] for key in routed})
    seeds = sorted({key[2] for key in routed})
    objectives = sorted({key[0] for key in routed})

    # **The pre-registered middle comes from the run's own `--lambdas` order, not
    # from sorting.** A live run takes `prices[len(prices)//2]` over the order it
    # was given; sorting the cell prices agrees only if that order was sorted, so
    # a run launched with `--lambdas 0.2,0.0,0.1` pre-registered 0.0 and a sorted
    # read-back would test 0.1 instead -- the primary test at a price the run did
    # not name. code-reviewer found it. 28853's log reads
    # `evaluated at [0.0, 0.1, 0.2]`, so the two agree there; the guard is for
    # the next run, and for anyone reading this one without that log.
    recorded = recorded_lambdas(output)
    if recorded is None:
        middle = prices[len(prices) // 2]
        print(f"no policies.json under {output}, so the middle price is taken "
              f"from the sorted cell prices: {middle:g}. If the run used an "
              f"unsorted --lambdas this is not the price it pre-registered",
              flush=True)
    elif set(recorded) != set(prices):
        # Two causes, and the message has to name both, because it will be read
        # by whoever hit it: the files really are from different runs, **or** a
        # price did not survive the `:g` in the cell name. `--lambdas 1/3` is
        # written `price0.333333` and parses back to a different float, so the
        # sets differ for a formatting reason rather than a provenance one.
        # code-reviewer found that this fails safe; a message pointing at the
        # wrong cause is still a message that sends someone the wrong way.
        raise SystemExit(
            f"policies.json records --lambdas {recorded} but the cells hold "
            f"{prices}, so guessing which is right would put a price in the "
            f"verdict that no run pre-registered. Either the files and the "
            f"report are from different runs, or a price did not survive the "
            f"`:g` formatting in the cell names -- which happens for a value "
            f"like 1/3, needs more digits than `:g` gives, and is a naming "
            f"limitation rather than a mismatched run")
    else:
        middle = recorded[len(recorded) // 2]
        if middle != prices[len(prices) // 2]:
            print(f"--lambdas was {recorded}, so the pre-registered middle is "
                  f"{middle:g} and NOT the sorted {prices[len(prices) // 2]:g}. "
                  f"Using the run's own order", flush=True)

    shared = shared_tasks(cells)
    if not shared:
        raise SystemExit("no task finished under every cell, so nothing pairs")

    def ordered(played):
        by_index = {p.episode: p for p in played}
        return [by_index[index] for index in shared]

    constants = {name: ordered(v) for name, v in constants.items()}
    routed = {key: ordered(v) for key, v in routed.items()}
    for key in [k for k in routed if len(k) == 3 and k[2] == seeds[0]]:
        routed[(key[0], key[1])] = routed[key]

    missing = [name for name in ("pmpo", "priced_ce")
               if (name, middle) not in routed]
    if missing or not constants:
        found = diagnostic_report(
            constants, routed, shared, prices, seeds,
            f"缺少 {missing or 'constant policies'},所以不做配对")
        found["read_back_from"] = str(output)
        return found

    report = {
        "read_back_from": str(output),
        "note": "recomputed from the cell files, not from a live run: the base "
                "that produced them did not carry these fields",
        "paired_tasks": len(shared), "lambdas": prices,
        "training_seeds": seeds,
        # Per seed and labelled, with the pooled figures beside it. The routed
        # entries read `routed[(name, price)]`, which is aliased to `seeds[0]`
        # above, so they are the first seed's episodes -- and `policies` said
        # nothing about that, which is how 29773's seed-0 reward of 0.4444 was
        # read as the run's against a pooled 0.4691. A cost column must come from
        # `policies_pooled`.
        "policies": {
            f"price_{price:g}": {
                **{name: summarise(v, price) for name, v in constants.items()},
                **{name: dict(summarise(routed[(name, price)], price),
                              seed=seeds[0],
                              seed_note="the first training seed's episodes, not "
                                        "a mean over seeds; `policies_pooled` "
                                        "holds them together")
                   for name in objectives if (name, price) in routed},
            }
            for price in prices
        },
        "policies_pooled": {
            f"price_{price:g}": {
                **{name: summarise(v, price) for name, v in constants.items()},
                **{name: dict(summarise([one for seed in seeds
                                         for one in routed.get((name, price, seed), [])],
                                        price), seeds=list(seeds))
                   for name in objectives
                   if any((name, price, seed) in routed for seed in seeds)},
            }
            for price in prices
        },
        "primary_test": primary_verdict(routed, middle, seeds[0]),
        "matched_seeds": matched_seeds(routed, seeds, middle),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--decision-url",
                        help="a running scripts/serve_decision.sbatch. The engine "
                             "**cannot** be in this process: Playwright's sync API "
                             "runs an event loop on the calling thread and SGLang's "
                             "in-process Engine drives its own, so the first "
                             "scoring call dies with 'Cannot run the event loop "
                             "while another loop is running'. The gate never met "
                             "this because it has always gone over HTTP")
    parser.add_argument("--head", help="for its metadata only. Required unless "
                                      "--verdict, which reads files")
    parser.add_argument("--venue", default="miniwob")
    parser.add_argument("--group", default="click")
    parser.add_argument("--rounds", type=int, default=1,
                        help="**1 by default.** Round 0 explores at --explore-prob, "
                             "so its corpus has balanced escalate/decline "
                             "decisions -- which is what both trainers need. Later "
                             "rounds collect under the *learned* head, and with "
                             "hurts = 0 on this group that head escalates almost "
                             "everywhere, so coverage would collapse exactly where "
                             "it has to learn to decline. CDPO is off-policy by "
                             "construction, so one collection is the honest minimum")
    parser.add_argument("--round-eval-episodes", type=int, default=0,
                        help="per-round evaluation, which is diagnostic: the result "
                             "is the four-policy read at the end, on the same tasks. "
                             "0 skips it and saves a third of the run")
    parser.add_argument("--episodes", type=int, default=0,
                        help="collected per round. 0 is **three** passes over the "
                             "training tasks: the count has to come from the split, "
                             "and a fixed default past its end crashed round 0 "
                             "before any head was trained. Three rather than two "
                             "because 27 training tasks is the thin side of this "
                             "run, and a pass is a new instance of each task")
    parser.add_argument("--eval-episodes", type=int, default=0,
                        help="0 is exactly one pass over the held-out tasks, which "
                             "is the only count that weights each of them once")
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--collect-only", action="store_true",
                        help="collect, fit the round's own head, report what one "
                             "escalation is worth, and stop before evaluating. "
                             "**Not the baseline heads**: those are fitted after "
                             "this returns, so a second invocation refits all "
                             "nine at a thread-sensitive cost and is not free -- "
                             "the plural here used to read 'the heads' and "
                             "understated that. The price has to be chosen from "
                             "the corpus rather than from the training interval's "
                             "midpoint, which is what 28853 did")
    parser.add_argument("--eval-only",
                        help="comma-separated policies to evaluate, e.g. "
                             "`always,pmpo`. The rest are skipped and the report "
                             "records them as absent. **The resulting table is a "
                             "diagnostic, not a comparison**: a policy read at a "
                             "different --eval-max-steps than another is not "
                             "comparable with it, and `verdict` refuses to pair "
                             "cells whose signatures differ. Use it to ask "
                             "whether a longer horizon moves one policy, then "
                             "rerun everything if it does")
    parser.add_argument("--eval-max-steps", type=int, default=0,
                        help="step cap for the **evaluation** only; 0 means use "
                             "--max-steps. Separate from the collection's cap "
                             "because they are separate experiments: 28853 left "
                             "8 of 27 tasks unfinished under `always` at 25 "
                             "steps, every one of them against the cap, and "
                             "asking whether more steps finishes them must not "
                             "also throw away the 81 collected episodes -- which "
                             "it would, since `max_steps` is in the collection's "
                             "signature and changing it sets the corpus aside. "
                             "The head is then the same head, read at a longer "
                             "horizon, which is the question rather than a "
                             "confound")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.0)
    parser.add_argument("--explore-prob", type=float, default=0.5)
    parser.add_argument("--price-range", default="0.0,0.25",
                        help="the interval the training price is drawn from, per "
                             "decision. It has to bracket --lambdas, since a gain "
                             "above every price drawn escalates on every draw and "
                             "the objective cannot tell those apart. This is what "
                             "makes one head serve every price")
    parser.add_argument("--cost-weight", type=float, default=0.05,
                        help="what one escalation costs in units of reward, "
                             "subtracted from the return. **Not 0.** At zero the "
                             "return is pure reward, thinking is free, and the "
                             "reward-maximising policy is to think everywhere -- so "
                             "CDPO would be trained to do the opposite of what it "
                             "is for. 0.05 puts it in the same range as the lambda "
                             "grid, where a five-step episode escalating "
                             "throughout costs a quarter of a success")
    parser.add_argument("--lambdas", default="0.0,0.1,0.2",
                        help="the prices to evaluate at. **A curve, not a point.** "
                             "28831 measured hurts = 0, so always-think *is* the "
                             "oracle and no router can beat it on accuracy: the "
                             "axis on which a win is possible is cost, and two "
                             "heads with one ceiling are distinguished by their "
                             "accuracy-cost curve. Table 2 reports five lambdas "
                             "for the same reason")
    parser.add_argument("--probe-prob", type=float, default=0.0,
                        help="how often a step that did not escalate also runs the "
                             "slow path, recording what it would have chosen "
                             "without changing what was taken. Escalated steps are "
                             "probed free regardless, since their fast "
                             "distribution is already computed. The gain is not "
                             "identifiable from a corpus collected at 0 -- see "
                             "experiments/browser-gate/REWARD-OFFLINE.md. Enters "
                             "the collection signature, so a probed corpus cannot "
                             "be resumed onto an unprobed one")
    parser.add_argument("--eval-fraction", type=float, default=0.5,
                        help="0.5 gives 27 train / 27 eval on the 54 click tasks. "
                             "**Not 0.25.** That gave 14 held-out tasks, under the "
                             "MIN_PAIRS of 20 this repository refuses a verdict "
                             "below, so the headline comparison could not have "
                             "reached significance in its own terms. 27 is the most "
                             "`split_tasks` can give, since its stride is at least 2")
    parser.add_argument("--min-eval-tasks", type=int, default=20,
                        help="refuse to run if the held-out set is smaller, which is "
                             "the gate's MIN_PAIRS applied to this comparison")
    parser.add_argument("--step-timeout", type=float, default=120.0,
                        help="seconds one browser call may take. The second guard "
                             "layer exits the process at 1.5x this with a stack, "
                             "since 28832's hang was a Playwright greenlet a signal "
                             "cannot unwind. A launcher loop then resumes: every "
                             "phase streams its episodes as they finish and comes "
                             "back by **index**, so a kill costs the episode it "
                             "took and nothing else")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="how many times one episode index may be started "
                             "before it is given up on. A hang takes the process, "
                             "so the launcher loops this script; the attempt is "
                             "recorded **before** the episode runs, which is the "
                             "only way a failure that leaves nothing behind can be "
                             "counted. Without the cap a task that hangs every time "
                             "consumes the whole job. This is the gate's "
                             "--max-attempts on the path this script actually uses")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-seeds", type=int, default=3,
                        help="how many training seeds to fit each head at, on the "
                             "**same** corpus. Training is CPU-seconds; what costs "
                             "episodes is reading them. The full curve is read at "
                             "seed 0 and the other seeds at the middle price only, "
                             "which is what says a difference is the objective "
                             "rather than training noise -- a one-seed difference "
                             "over 27 tasks cannot tell those apart")
    parser.add_argument("--output", required=True)
    parser.add_argument("--verdict", action="store_true",
                        help="recompute the deciding fields from the evaluation "
                             "cells already under --output and write "
                             "verdict.json, running no episode and needing no "
                             "service. For reading back a run whose base did not "
                             "compute them -- 28853 has no primary_test.seed, no "
                             "reverse and no matched_seeds -- through the same "
                             "functions a live run uses rather than a second "
                             "implementation of the definitions")
    args = parser.parse_args()

    # `--decision-url` and `--head` are required for a run and meaningless for a
    # read-back, so they are checked here rather than by argparse. The docstring
    # said `--verdict` "needs no service" while argparse rejected the command
    # without one -- a property claimed and not implemented, found by running it.
    if not args.verdict:
        missing = [name for name, value in (("--decision-url", args.decision_url),
                                            ("--head", args.head)) if not value]
        if missing:
            parser.error(f"{' and '.join(missing)} required unless --verdict")

    if args.verdict:
        # Before anything that needs a model, a browser or a service: this reads
        # files.
        found = verdict(args.output)
        write_json_atomic(str(Path(args.output) / "verdict.json"), found)
        primary = found["primary_test"]
        # The same three-way label as the live path: a read-back that prints
        # "primary" over a descriptive comparison is the more dangerous of the
        # two, because the JSON it came from outlives the log.
        status, heading, note = primary_heading(primary)
        print(f"paired on {found['paired_tasks']} task(s); "
              f"{heading} {primary['mine']} vs {primary['against']} at price "
              f"{primary['price']:g}, seed {primary['seed']}: "
              f"ahead {primary['ahead']} behind {primary['behind']} "
              f"p = {primary['sign_test_p']:.4g} one-sided / "
              f"{primary['sign_test_p_two_sided']:.4g} two-sided, "
              f"reverse p = {primary['reverse']['sign_test_p']:.4g} one-sided / "
              f"{primary['reverse']['sign_test_p_two_sided']:.4g} two-sided")
        if note:
            print(f"  {note}")
        print(f"matched seeds: CDPO ahead at {found['matched_seeds']['pmpo_ahead_at']}, "
              f"reversed at {found['matched_seeds']['reversed_at']}")
        print(f"wrote {Path(args.output) / 'verdict.json'}")
        return

    from inference import DecisionClient
    from slime_htm.envs.browser_environment import BrowserEnvironment, split_tasks
    from slime_htm.envs.gate import VENUES, group_tasks, missing_environment

    absent = missing_environment(args.venue)
    if absent:
        parser.error(f"{args.venue} needs {', '.join(absent)} in the environment")

    from importlib import import_module

    import_module(VENUES[args.venue]["registers"])
    found = group_tasks(args.venue, args.group)
    training, evaluating = split_tasks(found, args.eval_fraction)
    print(f"{args.venue}/{args.group}: {len(found)} tasks -> {len(training)} train, "
          f"{len(evaluating)} eval, disjoint", flush=True)

    metadata = DecisionHead.load(args.head).metadata
    print(f"head reads layers {metadata.feature_layers}, width {metadata.input_dim}",
          flush=True)
    client = DecisionClient(args.decision_url)
    health = client.health()
    print(f"decision service: {health}", flush=True)
    scorer = ClientScorer(client)
    # Fail here rather than at episode one: a service without the hidden state
    # cannot collect for CDPO, which trains on it, and the only way to know is to
    # ask. 28840 died at the first scoring call of the first episode; a probe
    # costs one request.
    probe = client.fast("probe", ["click('0')", "click('1')"], want_hidden=True)
    if not probe.get("hidden_state"):
        parser.error(
            "the decision service returned no hidden state for /fast. CDPO trains "
            "on the readout state at every step, so this service cannot collect "
            "for it -- it is running a build without want_hidden")
    # And it has to be the width this head reads. A service holding another model
    # or another head would otherwise collect a whole round -- hours -- and fail
    # only when `DecisionHead(metadata)` met wrong-width features. Both numbers go
    # in the message, because "a shape mismatch" without them tells a launcher
    # nothing about which side to change.
    width = len(probe["hidden_state"])
    if width != metadata.hidden_dim:
        parser.error(
            f"the service returns a {width}-float hidden state and this head reads "
            f"{metadata.hidden_dim} ({metadata.hidden_size} x "
            f"{len(metadata.feature_layers)} layer(s) {list(metadata.feature_layers)}). "
            f"The service is holding a different model or a different head; one of "
            f"the two has to change before anything is collected")
    print(f"hidden state: {width} floats per step, matching the head's "
          f"{metadata.hidden_size} x {len(metadata.feature_layers)}", flush=True)

    # Collection cycles -- a MiniWoB task is parameterised by its seed, so a second
    # pass at seed+1 is new data -- and evaluation does not, because a second pass
    # would weight some tasks twice in the mean.
    if len(evaluating) < args.min_eval_tasks:
        parser.error(
            f"{len(evaluating)} held-out task(s) is under --min-eval-tasks "
            f"{args.min_eval_tasks}. One task is "
            f"{100 / max(len(evaluating), 1):.0f} percentage points, and this "
            f"repository refuses a verdict below 20 pairs elsewhere; a headline "
            f"result should not be held to a weaker standard than a gate. Raise "
            f"--eval-fraction, or pick a group with more tasks")
    price_range = tuple(float(v) for v in args.price_range.split(","))
    if len(price_range) != 2 or not price_range[1] > price_range[0]:
        parser.error(f"--price-range must be low,high with high > low, got {args.price_range}")
    evaluated_prices = [float(v) for v in args.lambdas.split(",") if v.strip()]
    outside = [p for p in evaluated_prices if not price_range[0] <= p <= price_range[1]]
    if outside:
        parser.error(
            f"--lambdas {outside} fall outside --price-range {price_range}. The "
            f"training price is drawn from that interval, so a head asked at a "
            f"price it was never trained near is being read off the end of what it "
            f"learned -- `draw_prices` says the range must bracket the prices a "
            f"deployment will use")
    print(f"training price drawn per decision from {price_range}, "
          f"evaluated at {evaluated_prices}", flush=True)

    # Three passes over the training tasks, cycling: a MiniWoB task is
    # parameterised by its seed, so passes are instances and not replays. The
    # head trains on decisions rather than tasks -- 81 episodes is roughly 400
    # decisions a round -- and that is the thin side of this run, which the
    # effective sample size in each round's report is what measures.
    collected = args.episodes or 3 * len(training)
    evaluated = args.eval_episodes or len(evaluating)
    if evaluated > len(evaluating):
        parser.error(
            f"--eval-episodes {evaluated} is more than the {len(evaluating)} held-out "
            f"tasks, and evaluation does not cycle: every task is read once")
    print(f"collecting {collected} episode(s) per round over {len(training)} tasks "
          f"({collected / len(training):.1f} passes), evaluating {evaluated} over "
          f"{len(evaluating)}", flush=True)

    stacks = str(Path(args.output) / "stacks.log")

    def environment(tasks, seed, cycle=False):
        return BrowserEnvironment(tasks, seed=seed, limit=52,
                                  state_chars=STATE_CHARS,
                                  cycle=cycle, step_timeout=args.step_timeout,
                                  stack_file=stacks)

    def make_environment(index, evaluation):
        """Train and evaluate on disjoint task sets, as `split_tasks` guarantees.

        A different seed per round on the training side, so repeated rounds draw
        different instances of the same tasks rather than replaying one.
        """
        if evaluation:
            return environment(evaluating, args.seed)
        return environment(training, args.seed + index * 1000, cycle=True)

    # What the episodes on disk were collected under. A second invocation with
    # one flag changed -- a different split, group, seed or budget -- would
    # otherwise resume onto them and the run would answer two questions at once.
    collection_signature = {
        "venue": args.venue, "group": args.group, "seed": args.seed,
        "budget": args.budget, "max_steps": args.max_steps,
        "explore_prob": args.explore_prob, "lambda": args.lambda_,
        # Without this a probed corpus and an unprobed one digest identically and
        # `--resume` merges them, which is the pooling-across-configurations error
        # in the one place nobody would look for it.
        "probe_prob": args.probe_prob,
        "price_range": list(price_range), "episodes": collected,
        "training_tasks": digest(training), "evaluating_tasks": digest(evaluating),
        # **What the policy may choose from, and what state it sees.** Without
        # these the signature was blind to the fast mode: making `content_free`
        # the default in c0783e7 changed the policy and left this dict identical,
        # so a resume would have mixed episodes from the sealed fast mode into a
        # corpus the fixed one was extending and said nothing. exps caught it
        # before the relaunch; the guard was mine and it would have waved it
        # through.
        "offered_actions": environment(training, args.seed).offered_actions,
        "state_chars": STATE_CHARS,
        "step_timeout": args.step_timeout,
        # **Threads are NOT here**, and that was a real mistake -- see
        # `thread_settings()` and `evaluation_signature` below. They change the
        # fitted heads, not the collected episodes: at `--rounds 1` the
        # collection plays the explore coin and trains nothing, so guarding the
        # corpus with them would throw away a valid 81-episode collection
        # (about 1.2h) because a resubmission landed on a node with a different
        # `SLURM_CPUS_ON_NODE` -- while the heads would have been refitted
        # anyway. code-reviewer caught it before the run started.
        #
        # They *do* belong here once `--rounds > 1`, because a mid-collection
        # head steers the behaviour policy, and that is what the conditional in
        # `thread_settings` is for.
        # **The general guard, which is code-reviewer's point and the right one.**
        # Enumerating the settings that matter is how this was missed once
        # already: the next change to the fast mode would be a field nobody
        # thought to add. The commit changes whenever the code does, so a
        # behaviour change sets stale files aside whether or not it has a field.
        # `commit_now` reads git, and on the cluster -- where there is no `.git`
        # -- the `REVISION` file that a sync replaces, which is what makes a
        # swapped tree visible. It can return "unknown", and then two different
        # unknown trees would match: it warns on stderr when it does, and that
        # warning is the thing to act on.
        "commit": commit_now(),
        # Only when a round can train a head that then steers the behaviour
        # policy. At `--rounds 1` this is `{}` and a thread change costs nothing.
        **({"threads": thread_settings()} if args.rounds > 1 else {})}

    # The rows the CDPO trainer sees, kept so the baseline is fitted to identical
    # data. A second collection would make the comparison about the sample.
    corpus = {}
    head, rounds = run(
        metadata, make_environment, scorer,
        LoopConfig(rounds=args.rounds, episodes_per_round=collected,
                   eval_episodes=args.round_eval_episodes, budget=args.budget,
                   max_steps=args.max_steps, lambda_=args.lambda_,
                   explore_prob=args.explore_prob, max_attempts=args.max_attempts,
                   probe_prob=args.probe_prob,
                   head=HeadRLConfig(lambda_=args.lambda_,
                                     lambda_range=price_range,
                                     cost_weight=args.cost_weight,
                                     behaviour_prob=args.explore_prob),
                   output=args.output),
        seed=args.seed, signature=collection_signature,
        on_corpus=lambda index, episodes: corpus.__setitem__(index, episodes))

    last = max(corpus) if corpus else None
    if last is None:
        parser.error("no episodes were collected, so nothing can be compared")
    print(f"\nCDPO head trained on {len(corpus[last])} episodes from round {last}",
          flush=True)

    # **Stop here when the price has not been chosen yet.** 28853's `lambda` was
    # the training interval's midpoint, picked without estimating what one
    # escalation is worth -- about 0.003 of a point, so 0.1 put every routing
    # policy at a loss and the pre-registered test compared two losers. The
    # marginal value is a property of the collected corpus, so it can only be
    # read after collection and must be read before the price is fixed.
    #
    # Evaluation is the expensive half (2.8h of a 4h run against 1.2h), so
    # splitting there costs nothing: the corpus and the heads are on disk and a
    # second invocation with `--lambdas` resumes against the same signature.
    # **Always reported, not only under `--collect-only`.** It costs one pass
    # over a corpus already in memory, and a run that evaluates at a price
    # should still record what an escalation was worth at the time -- otherwise
    # the next person reading the cells has to re-derive it, which is how
    # 28853's price came to be the training interval's midpoint.
    report = marginal_value(corpus[last])
    print("\nmarginal value of one escalation, per seed, no pooling:", flush=True)
    for line in report["lines"]:
        print(f"  {line}", flush=True)
    value_path = Path(args.output) / "marginal-value.json"
    value_path.write_text(json.dumps(report, indent=1, default=str))
    print(f"  written to {value_path}", flush=True)

    if args.collect_only:
        # **"The heads are kept" was wrong** -- code-reviewer's condition. The
        # corpus is on disk and resumes; the heads are *refitted* on the second
        # invocation, and exps measured that refit moving with thread count on
        # the cluster. So the two invocations are only comparable if the CPU
        # allocation is identical, which is a launcher property and recorded in
        # the signature below so a mismatch is visible rather than silent.
        print("\n--collect-only: stopping before evaluation. **The corpus is kept "
              "and resumes; the heads are refitted** on the next invocation, and "
              "that refit is sensitive to thread count -- run the second "
              "invocation with the same CPU allocation and OMP_NUM_THREADS, which "
              "the signature records.", flush=True)
        return

    # Baseline (A), on the same rows and with the same trunk. Trained from the
    # final round's corpus, which is the one the CDPO head was last fitted to.
    head_config = HeadRLConfig(lambda_=args.lambda_, lambda_range=price_range,
                               cost_weight=args.cost_weight,
                               behaviour_prob=args.explore_prob)
    seeds = list(range(args.seed, args.seed + max(args.train_seeds, 1)))
    baselines, baseline_reports = {}, {}
    # Two cross-entropy baselines, not one. **Unpriced** is what a practitioner
    # with no counterfactual would do, and a win over it says the price in the
    # objective mattered. **Priced** optimises the same quantity CDPO does, fitted
    # by cross-entropy instead of a clipped policy gradient -- so a win over *it*
    # is attributable to what CDPO actually adds: the per-decision importance
    # ratio and the clip. Without the second, the sharpest objection to a CDPO win
    # is that CDPO was handed the price and the baseline was not. code-reviewer
    # reached the same conclusion independently and proposed it as the next
    # baseline; it costs 108 episodes, so it is in this run instead.
    # "drawn", not a fixed midpoint: the priced baseline must marginalise over the
    # price exactly as CDPO does, or the two differ in the loss **and** in price
    # marginalisation and a CDPO win cannot be credited to the importance ratio
    # and the clip. code-reviewer caught that the first version passed one fixed
    # price while this file claimed the per-decision draw.
    for name, price in (("smeared_ce", None), ("priced_ce", "drawn")):
        baselines[name], baseline_reports[name] = {}, {}
        for seed in seeds:
            fitted, fit_report = train_head_on_smeared_returns(
                metadata, corpus[last], head_config, seed=seed, price=price)
            baselines[name][seed] = fitted
            baseline_reports[name][seed] = fit_report
            fitted.save(Path(args.output) / f"head-{name}-seed{seed}.safetensors")
        print(f"  {name}: {json.dumps(baseline_reports[name][seeds[0]])}", flush=True)

    # CDPO at the same seeds, on the same corpus. Training is CPU-seconds; what
    # costs episodes is reading a head, so every head is fitted at every seed and
    # only the reads are rationed -- the full curve at the first seed, the others
    # at the middle price. A one-seed difference over 27 tasks cannot tell the
    # objective from training noise.
    from slime_htm.head_trainer import train_head_on_episodes

    pmpo_heads, pmpo_reports = {seeds[0]: head}, {seeds[0]: rounds[-1]}
    for seed in seeds[1:]:
        fitted, fit_report = train_head_on_episodes(
            metadata, corpus[last], head_config, seed=seed)
        pmpo_heads[seed] = fitted
        pmpo_reports[seed] = fit_report
        fitted.save(Path(args.output) / f"head-pmpo-seed{seed}.safetensors")
    baselines["pmpo"] = pmpo_heads
    report_seeds = {"seeds": seeds, "pmpo": {str(s): pmpo_reports[s] for s in seeds}}

    report = {
        "venue": args.venue, "group": args.group,
        "train_tasks": training, "eval_tasks": evaluating,
        "rounds": rounds, "baseline_reports": baseline_reports,
        "training_seeds": report_seeds,
        "episodes_trained_on": len(corpus[last]),
        # What this run can and cannot show, in the file rather than only in a
        # README, because the file is what a reader will have.
        #
        # **Two seeds, not one.** The single-seed version of this block said
        # `oracle == always` and therefore that accuracy headroom above
        # always-think was zero and a tie at the top was the best available
        # outcome. 28844 (seed 1) refutes it: `hurts` is 0.074 there, so its
        # oracle beats always by +0.074 and the two-seed combination by +0.037.
        # exps built the no-headroom argument and withdrew it; I had repeated it.
        # The headroom is small -- about one task of the 27 -- but it is not zero,
        # so a router that matches always-think on accuracy has not hit a ceiling
        # that excuses it. The priced value stays the headline for the reason the
        # user gave, not because accuracy is unmeasurable.
        "ceiling": {
            "source": "28831 + 28844, click group, seeds 0 and 1 combined "
                      "(108 pairs, exact sign test p = 9.65e-06)",
            "combined": {"never": 0.296, "always": 0.519, "oracle": 0.556,
                         "both_right": 0.259, "helps": 0.259, "hurts": 0.037,
                         "helps_by_reach": 0.241, "helps_by_reasoning": 0.019},
            "per_seed": {"seed0": {"hurts": 0.0, "always": 0.481, "oracle": 0.481,
                                   "helps_by_reasoning": 0.0},
                         "seed1": {"hurts": 0.074, "always": 0.556, "oracle": 0.630,
                                   "helps_by_reasoning": 0.037}},
            "note": "oracle - always = +0.037 combined, so there IS accuracy "
                    "headroom; `oracle == always` was seed 0 only and must not "
                    "be cited. Reach still dominates reasoning about 13:1 "
                    "(0.241 vs 0.019), so the licensed claim about escalation is "
                    "mostly 'it reaches steps the cheap mode cannot act on'",
            "seed_spread_warning": "`never` moves 15 points between the two seeds "
                                   "on the same 54 tasks, so any single-seed "
                                   "number from this venue is labelled with its "
                                   "seed or not quoted",
            "both_right_in_eval_half": 4,
            "room_note": "4 of the 12 both_right tasks fall in the evaluation "
                         "half at seed 0, so a router's operating room there is "
                         "about 4 tasks and one task is 3.7 points -- exps "
                         "computed this from the rows before the run, and it is "
                         "itself a seed-0 figure",
        },
        "policies": {}}

    print("\nreading every policy on the same evaluation tasks", flush=True)
    # Every cell writes its episodes to its own file under --output, so a
    # launcher loop re-reads them instead of replaying them. The name is the
    # identity of the cell: the file for `pmpo` at price 0.1 from training seed 2
    # is not the file for any other.
    # The signature carries a digest of **the corpus the heads were fitted to**,
    # not only the flags. Collection finishes before this point -- every pending
    # episode is retried or given up on inside `run` -- so the corpus cannot grow
    # between two invocations, and if that ever stopped holding, the cells played
    # by the earlier head would be set aside instead of being read beside the
    # later one's under a single column.
    eval_max_steps = args.eval_max_steps or args.max_steps
    evaluation_signature = dict(
        collection_signature, evaluated=evaluated,
        # **What the head refit is sensitive to.** `--collect-only` keeps the
        # corpus and the heads are refitted on the next invocation, and exps
        # measured that refit moving with thread count on this cluster. So two
        # invocations under different CPU allocations train different heads from
        # identical data -- a difference that would read as the objective. This
        # is the signature that covers the heads; the collection's does not and
        # must not, or a thread change costs a recollection.
        threads=thread_settings(),
        # The evaluation's own cap, so a longer-horizon read sets the eval cells
        # aside without touching the corpus. The collection's `max_steps` stays
        # in the inherited part.
        eval_max_steps=eval_max_steps,
        corpus=digest([[episode.get("index"), episode["terminal_reward"]]
                       for episode in corpus[last]]),
        train_seeds=sorted(seeds), lambdas=sorted(
            float(v) for v in args.lambdas.split(",") if v.strip()))
    played = dict(episodes=evaluated, budget=args.budget, max_steps=eval_max_steps,
                  seed=args.seed, max_attempts=args.max_attempts,
                  signature=evaluation_signature)

    def cell(name):
        return Path(args.output) / f"eval-{name}.jsonl"

    # The constant policies run **once**. They do not depend on the price, so
    # their value at each price is arithmetic on the same episodes rather than
    # another 54 episodes per price.
    wanted = ({name.strip() for name in args.eval_only.split(",") if name.strip()}
              if args.eval_only else None)
    report["evaluated_policies"] = sorted(wanted) if wanted else "all"
    report["incomplete"] = wanted is not None

    constants = {}
    for name, escalate in (("never", 0.0), ("always", 1.0)):
        if wanted is not None and name not in wanted:
            continue
        print(f"  {name}", flush=True)
        constants[name] = constant_policy(
            cell(name), lambda: environment(evaluating, args.seed), scorer,
            escalate=escalate, **played)

    prices = [float(value) for value in args.lambdas.split(",") if value.strip()]
    report["lambdas"] = prices
    routed = {}
    print(f"\nthe routed policies, at prices {prices}", flush=True)
    middle = prices[len(prices) // 2]
    for name in ("smeared_ce", "priced_ce", "pmpo"):
        if wanted is not None and name not in wanted:
            continue
        for seed in seeds:
            # The full curve at the first seed; the others at the middle price,
            # which is what separates the objective from training noise.
            for price in (prices if seed == seeds[0] else [middle]):
                print(f"  {name}@{price:g} seed {seed}", flush=True)
                routed[(name, price, seed)] = play(
                    cell(f"{name}-price{price:g}-seed{seed}"),
                    lambda: environment(evaluating, args.seed), scorer,
                    baselines[name][seed], price=price, **played)
                if seed == seeds[0]:
                    routed[(name, price)] = routed[(name, price, seed)]

    # **The same tasks, or no comparison.** Every policy is read on the indices
    # that *all* of them finished. A dropped episode in one cell would otherwise
    # shorten that cell alone, and `paired_test` pairs by position: it would be
    # comparing one policy's task 7 against another's task 8, which is worse than
    # having no number. `strict=True` would catch the length mismatch; nothing
    # would catch the misalignment if the lengths happened to match.
    cells = {**{(name,): played for name, played in constants.items()},
             **{key: value for key, value in routed.items() if len(key) == 3}}
    shared = sorted(set.intersection(*(set(value) for value in cells.values())))

    def ordered(cell_episodes):
        return [cell_episodes[index] for index in shared]

    # **`--eval-only` stops here.** Everything below pairs policies: the five-way
    # table, the seed spread, `primary_verdict` against `priced_ce`, and the
    # paired tests against `never`/`always`/`smeared_ce`. With a policy skipped
    # those all index a cell that was never played -- and the failure lands
    # **after** every episode has run, so it would exit 1 with no stacks.log
    # growth and write no report at all, losing the whole evaluation.
    # code-reviewer reproduced it on both paths. My own `--eval-only` help text
    # says the table "is a diagnostic, not a comparison"; this is that sentence
    # implemented rather than asserted.
    if wanted is not None:
        found = diagnostic_report(
            {name: ordered(played) for name, played in constants.items()},
            {key: ordered(value) for key, value in routed.items()
             if len(key) == 3},
            shared, prices, seeds,
            f"--eval-only {sorted(wanted)},所以跳过所有配对分析")
        found["evaluated_policies"] = sorted(wanted)
        write_json_atomic(str(Path(args.output) / "policies.json"), found)
        print(f"\n诊断报告(无配对):{len(found['cells'])} 个 cell,"
              f"{len(shared)} 个共同任务", flush=True)
        for name, row in sorted(found["cells"].items()):
            print(f"  {name:28} 分数 {row['reward']:.3f}  升级 "
                  f"{row['escalations']:.2f}  步 {row['steps']:.2f}  "
                  f"token {row['tokens']:.0f}", flush=True)
        print(f"\nwrote {Path(args.output) / 'policies.json'}", flush=True)
        return
    lost = sorted(set(range(evaluated)) - set(shared))
    report["paired_tasks"] = len(shared)
    report["unpaired_tasks"] = lost
    if lost:
        print(f"\n{len(lost)} of {evaluated} evaluation task(s) did not finish under "
              f"every policy and are dropped from all of them: {lost}", flush=True)
    if len(shared) < args.min_eval_tasks:
        # Not an exit: the collection and the heads are on disk and the table is
        # still worth having. But the verdict is not, and the report says so in
        # the file rather than leaving a reader to count.
        report["underpowered"] = {
            "paired_tasks": len(shared), "min_eval_tasks": args.min_eval_tasks,
            "note": "under the MIN_PAIRS this repository refuses a verdict below, "
                    "so the sign test here is descriptive and not a conclusion"}
        print(f"\n*** {len(shared)} paired task(s) is under --min-eval-tasks "
              f"{args.min_eval_tasks}: the table below is descriptive, not a "
              f"verdict ***", flush=True)
    if not shared:
        parser.error("no evaluation task finished under every policy, so there is "
                     "nothing to pair")

    constants = {name: ordered(value) for name, value in constants.items()}
    routed = {key: ordered(value) for key, value in routed.items()}
    for name, value in constants.items():
        print(f"  {name:12} {json.dumps(summarise(value))}", flush=True)
    for name in ("smeared_ce", "priced_ce", "pmpo"):
        for seed in seeds:
            for price in (prices if seed == seeds[0] else [middle]):
                print(f"  {name}@{price:g} seed {seed}: "
                      f"{json.dumps(summarise(routed[(name, price, seed)], price))}",
                      flush=True)

    # What the seeds say at the middle price: a difference that survives them is
    # the objective, and one that does not is noise.
    print(f"\nacross {len(seeds)} training seeds at price {middle:g}", flush=True)
    report["seed_spread"] = {}
    for name in ("smeared_ce", "priced_ce", "pmpo"):
        values = [summarise(routed[(name, middle, seed)], middle)["value"]
                  for seed in seeds]
        report["seed_spread"][name] = {
            "values": values, "mean": float(np.mean(values)),
            "spread": float(np.std(values)), "min": min(values), "max": max(values)}
        print(f"  {name:12} {np.mean(values):+.3f} +- {np.std(values):.3f}  "
              f"{[round(v, 3) for v in values]}", flush=True)

    # The table the claim is read off: one row per price, four policies, scored on
    # `reward - price x escalations`. The constants are re-scored rather than
    # re-run, since their behaviour does not depend on the price.
    # Paired, task by task, against every other policy at every price. This is
    # what turns the table into a claim: "CDPO's value exceeded X's on k of 27
    # tasks and fell short on m, one-sided p = ...".
    print("\nCDPO against each baseline, paired by task", flush=True)
    # **One primary test, pre-registered here before the run.** Twelve sign tests
    # over three prices and four opponents, read uncorrected, would let one
    # p < 0.05 in twelve be reported as the result. The primary is CDPO against
    # the priced cross-entropy head at the middle price -- the comparison that
    # isolates what CDPO adds, at the price the head was trained around -- with
    # the seed spread as its robustness check. Everything else is secondary and
    # labelled so in the output.
    report["primary_test"] = primary_verdict(routed, middle, seeds[0])
    report["matched_seeds"] = matched_seeds(routed, seeds, middle)
    print(f"\nmatched seeds at price {middle:g}: CDPO ahead at "
          f"{report['matched_seeds']['pmpo_ahead_at']}, reversed at "
          f"{report['matched_seeds']['reversed_at']}", flush=True)
    report["paired"] = {}
    for price in prices:
        against = {}
        for name in ("never", "always"):
            against[name] = paired_test(routed[("pmpo", price)], constants[name], price)
        for name in ("smeared_ce", "priced_ce"):
            against[name] = paired_test(routed[("pmpo", price)],
                                        routed[(name, price)], price)
        report["paired"][f"price_{price:g}"] = against
        for name, row in against.items():
            matched = (name == report["primary_test"]["against"]
                       and price == report["primary_test"]["price"])
            # Three labels, not two. `<- PRIMARY` over a price the
            # pre-registration does not name is the thing code-reviewer caught:
            # it reads as the registered test to anyone skimming the log, and
            # `middle = prices[0]` makes a lone `--lambdas 0.0` look like an
            # interval's midpoint.
            if matched and report["primary_test"]["status"] == "registered":
                label = "   <- PRIMARY"
            elif matched:
                label = "   <- descriptive (not the registered price)"
            else:
                label = "   (secondary)"
            print(f"  {price:>5.3g} vs {name:12} ahead {row['ahead']:>3} "
                  f"behind {row['behind']:>3} tied {row['tied']:>3} "
                  f"p = {row['sign_test_p']:.4g} one-sided / "
                  f"{row['sign_test_p_two_sided']:.4g} two-sided{label}", flush=True)

    columns = ("never", "always", "smeared_ce", "priced_ce", "pmpo")
    # **The header named only the price and every cell was `value`.** Read off
    # this table 29773 looks like a negative result -- `never` winning, `pmpo`
    # behind both cross-entropy heads, reversed on all three seeds -- and it was
    # reported that way and nearly written up that way. On `reward` the same run
    # has the routed policy between the endpoints, losing no task to fast-only
    # across 81 paired evaluations. Both quantities are printed now: a summary
    # whose quantity has to be looked up elsewhere is one that will be quoted
    # wrong, and `value` charges per escalation, so it is blind to how long an
    # escalation ran and cannot speak to token cost at all.
    print("\n" + f"{'price':>7}  {'quantity':>8}"
          + "".join(f"{name:>12}" for name in columns)
          + f"{'winner':>12}", flush=True)
    for price in prices:
        # Stored per seed, and now saying so. `routed[(name, price)]` is aliased
        # to `seeds[0]` further up, so a two-tuple read is the first seed's
        # episodes and nothing in the block it produced mentioned a seed -- while
        # `ceiling.seed_spread_warning` in the same file says any single-seed
        # figure from this venue must be labelled or not quoted.
        row = {name: summarise(constants[name], price) for name in constants}
        for name in ("smeared_ce", "priced_ce", "pmpo"):
            row[name] = summarise(routed[(name, price)], price)
            row[name]["seed"] = seeds[0]
            row[name]["seed_note"] = (
                "the first training seed's episodes, not a mean over seeds; "
                "`policies_pooled` holds them together")
        report["policies"][f"price_{price:g}"] = row
        # Pooled, which is what a table row describing the run has to come from.
        # With equal episode counts per seed this equals a mean of per-seed means
        # for every level quantity -- reward, tokens, seconds -- and differs for
        # the escalation *rate*, which is a ratio: pooling escalations over steps
        # weights an episode once where averaging three rates weights a seed
        # once, and on 29773 that is 0.263 against 0.315. Only the pooled rate
        # reproduces the escalations and steps printed beside it.
        pooled = {name: summarise(constants[name], price) for name in constants}
        for name in ("smeared_ce", "priced_ce", "pmpo"):
            episodes = [one for seed in seeds
                        for one in routed.get((name, price, seed), [])]
            if episodes:
                pooled[name] = summarise(episodes, price)
                pooled[name]["seeds"] = list(seeds)
        report.setdefault("policies_pooled", {})[f"price_{price:g}"] = pooled
        best = max(pooled, key=lambda name: pooled[name]["value"])
        for quantity in ("reward", "value"):
            print(f"{price:>7.3g}  {quantity:>8}"
                  + "".join(f"{pooled[name][quantity]:>12.3f}" for name in columns)
                  + (f"{best:>12}" if quantity == "value" else ""), flush=True)

    write_json_atomic(str(Path(args.output) / "policies.json"), report)
    print(f"\nwrote {Path(args.output) / 'policies.json'}", flush=True)


if __name__ == "__main__":
    main()
