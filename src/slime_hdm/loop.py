"""Collect, update the head, collect again.

slime's own loop drives Megatron, which has nothing to update here, so the outer
loop is this. One round is: roll out with the current head, update the head from
the rewards those rollouts earned, evaluate, keep going.

Each round reports the effective sample size alongside the reward, because a
round that improved the reward while the importance weights collapsed has not
shown an improvement. The head from round `k` is the behaviour policy for round
`k + 1`, so its escalation probability is recorded on every step and the next
round's ratios are against what actually acted, not against a nominal rate.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from litjev.core.block import escalation_gain
from litjev.core.block_utils import build_features
from slime_hdm.head_trainer import HeadRLConfig, train_head_on_episodes
from slime_hdm.resume import (
    append_row,
    attempts_by_index,
    checked_rows,
    episode_rows,
    jsonable,
    read_rows,
    resume_plan,
)
from slime_hdm.rollout import rollout_episodes

# The bookkeeping moved to `slime_hdm.resume`, so the collection here and the
# evaluation in `scripts/browser_pmpo.py` resume through one implementation
# rather than two. Kept under its old name for the callers that have it.
_jsonable = jsonable


@dataclass
class LoopConfig:
    rounds: int = 3
    episodes_per_round: int = 100
    eval_episodes: int = 0
    budget: int = 256
    max_steps: int = 50
    lambda_: float = 0.0
    # How many times one episode index may be started before it is given up on.
    # The Playwright hang exps captured takes the process, so a launcher loops
    # this script; without a cap an index that hangs every time loops forever.
    max_attempts: int = 3
    # The first round has no head to act with, so it explores at this rate.
    explore_prob: float = 0.5
    # How often a step that did not escalate also runs the slow path, to record
    # what it would have chosen. Zero collects nothing of the sort, which is what
    # every corpus before this had. See `StepRecord.probe_probabilities` for why
    # the gain is not identifiable without it. **This belongs in the collection
    # signature**: a probed and an unprobed corpus are otherwise indistinguishable
    # by digest and a resume can silently merge them.
    probe_prob: float = 0.0
    head: HeadRLConfig = field(default_factory=HeadRLConfig)
    output: str = "results/hdm-rl"


def behaviour_probability(head, features, lambda_, temperature):
    """What the acting policy's escalation probability was, per step.

    The next round reweights against this. Recording it beats assuming a rate,
    because after round one the acting policy is a head and its probability
    varies by step.
    """
    if head is None:
        return None
    outcome = head.predict(features)
    gain = np.array([escalation_gain(row) for row in outcome])
    return 1.0 / (1.0 + np.exp(-(gain - lambda_) / temperature))


def load_corpus(path):
    """A round's episodes, as `train_head_on_episodes` wants them.

    Reads either shape: the per-episode JSONL this writes now, or the
    end-of-round `.npz` it used to write. The JSONL is what survives a kill
    **during** collection, which the `.npz` did not: it was written after the
    round completed, so at `--rounds 1` a death during collection left nothing and
    the restart began at episode zero. exps computed the consequence from tonight's
    hang rate -- about 1% an episode over 194 episodes -- as a 57% chance the
    collection dies, and the new process-exit guard makes that the expected
    outcome rather than a tail risk, since exiting is its whole job.
    """
    from pathlib import Path

    path = Path(path)
    if path.suffix == ".jsonl":
        return read_rows(path)
    stored = np.load(path, allow_pickle=True)
    return json.loads(str(stored["episodes"]))


# Which field only a real episode carries, for `slime_hdm.resume`: the markers in
# the same file have an index and an outcome and no features.
EPISODE_KEY = "head_features"

# How many dropped episodes **with nothing kept** mean the environment is broken
# rather than the tasks hard. Not "in a row": the count is every drop so far in
# this process, and the second half of the condition -- that no episode has been
# kept -- is what disarms it. One good episode proves the browser works, so later
# drops are hard tasks. code-reviewer asked for the wording, since "in a row"
# implies an adjacency the code does not check. The same 3 as `gate.BLIND_EPISODES`, defined here rather than
# imported: this loop also drives AlfWorld, and importing the browser gate for a
# constant would put browsergym on that path.
BLIND_EPISODES = 3

# **Exit code for "a human has to look at this", and why it is not an exception.**
# A launcher loops this script to survive the hang guard's process exit, so a
# `RuntimeError` here is just another crash to relaunch: it would drop three more
# episodes, abort again, and spend every index's attempt cap -- turning a guard
# that exists to *stop* into a slow drain. code-reviewer found that. 3 is
# distinct from the hang guard's exit and from argparse's 2, and the launcher
# must not resume on it. `experiments/browser-gate/README.md` carries it in the
# contract table beside the other codes.
ENVIRONMENT_BROKEN = 3


def trainable(rows):
    """The rows a trainer can read, in the order they were collected."""
    return episode_rows(rows, EPISODE_KEY)


def corpus_path(output, index):
    """Where a round's episodes are streamed, one JSON object per line."""
    from pathlib import Path

    return Path(output) / f"episodes-round{index}.jsonl"


def run(metadata, make_environment, scorer, config, seed=0, on_corpus=None,
        signature=None):
    """Run the loop. `make_environment(round, evaluation)` yields a fresh environment.

    Returns (head, rounds), where each round records what it collected, what it
    learned, and how much of the sample the reweighting left usable.

    `signature` is what the episodes on disk are checked against before they are
    resumed onto: the environment lives behind `make_environment`, so only the
    caller knows which tasks and which split these rows were collected over.

    `on_corpus(index, episodes)` is handed each round's episodes as the trainer
    sees them -- the same rows, before any training. It exists so a baseline can
    be fitted to **identical** data rather than to a second collection: the
    comparison CDPO has to win is against another loss, not against another
    sample.
    """
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    head, rounds = None, []
    for index in range(config.rounds):
        rng = np.random.default_rng(seed + index)
        stream = corpus_path(output, index)

        def built(result, head=head):
            """One episode in the shape the trainers read."""
            features = np.stack([
                build_features(step.hidden, step.probabilities[: step.candidates], "choice")
                for step in result.steps])
            acting = behaviour_probability(head, features, config.lambda_,
                                           config.head.temperature)
            # `EpisodeResult.row` and `StepRecord.row` decide what gets written,
            # here and in `rollout.samples_from`, so this path cannot list a
            # shorter set of keys than the other one -- which is how a probed
            # collection came to write a corpus identical to an unprobed one.
            # `behaviour_prob` is the one field that is not a step's own: it is
            # the price-dependent probability the acting head gave this step, and
            # the off-policy ratio is unrecoverable without it.
            #
            # The episode aggregates come with it, and the seconds among them are
            # written **on the collection path too**, not only in the evaluation
            # cells. The collection is where the marginal value of an escalation
            # gets estimated, and that estimate is what the next run's price has
            # to be chosen from -- 28853's lambda = 0.1 was the training
            # interval's midpoint, picked without it, and it landed where every
            # routing policy loses.
            return result.row(head_features=[
                s.row(behaviour_prob=(config.explore_prob if acting is None
                                      else float(acting[position])))
                for position, s in enumerate(result.steps)])

        def append(row, stream=stream):
            return append_row(stream, row)

        # Already collected, if a previous process died partway. The guard that
        # catches the Playwright greenlet hang works by exiting, so this is the
        # expected path and not an edge case.
        def kept_so_far(stream=stream):
            return trainable(read_rows(stream))

        rows = checked_rows(stream, signature)
        attempts = attempts_by_index(rows, EPISODE_KEY)
        pending, given_up = resume_plan(rows, config.episodes_per_round,
                                        config.max_attempts, EPISODE_KEY)
        if rows:
            print(f"round {index}: resuming with {len(trainable(rows))} episode(s) in "
                  f"{stream}; {len(pending)} to run, {len(given_up)} given up after "
                  f"{config.max_attempts} attempt(s)", flush=True)
        noted = {int(row["index"]) for row in rows
                 if row.get("ended") == "given_up" and row.get("index") is not None}
        for episode_index in list(given_up):
            # Recorded once -- hence `noted`, since a launcher loop re-enters this
            # with the same indices still given up -- so the round's own row says
            # how many episodes the environment took rather than leaving the
            # shortfall unexplained.
            if episode_index not in noted:
                noted.add(episode_index)
                append({"index": episode_index,
                        "attempt": attempts.get(episode_index, 0), "ended": "given_up"})

        collecting = make_environment(index, False)

        def start(episode_index):
            """Mark the attempt **before** it runs.

            A hang that takes the process writes nothing, so without this the
            attempt leaves no trace and a launcher looping this script retries
            that index for as long as it is allowed to.
            """
            append({"index": episode_index,
                    "attempt": attempts.get(episode_index, 0), "ended": "started"})

        dropped = []

        def keep(result, environment=collecting):
            """Append each episode **as it finishes**, flushed.

            Per episode and not per round: the `.npz` was written after a round
            completed, so a death during collection left nothing behind and the
            restart began at zero. The gate survived both of tonight's hangs with
            one episode lost each because its rows stream this way.
            """
            attempt = attempts.get(result.episode, 0)
            if not result.steps or not getattr(environment, "complete", True):
                # A step that did not return ends the episode, and the episode is
                # not the policy's failure: keeping it would train on a zero the
                # browser manufactured. Dropped -- by index, so a retry comes back
                # to this task and not to whichever one the count lands on.
                ended = ("no_steps" if not result.steps
                         else getattr(environment, "ended", "incomplete"))
                print(f"  dropping episode {result.episode}, which ended {ended}",
                      flush=True)
                append({"index": result.episode, "attempt": attempt, "ended": ended})
                dropped.append(ended)
                # **The gate's `blind_run` guard, on this path.** The gate scores a
                # `no_candidates` episode and aborts the run when the first few
                # are all that -- a run that saw no actions has measured nothing,
                # and the alternative is a full run reporting "both near zero"
                # about an environment rather than about thinking. This path
                # *drops* such an episode instead, which is right for training --
                # an episode with no decisions has no gradient and is not the
                # policy's failure -- but dropping made the same breakage silent:
                # 28845 lost 13.8% of its episodes to a missing scroll action and
                # it took exps reading the log to notice. Aborting is not a
                # verdict and cannot bias one.
                if len(dropped) >= BLIND_EPISODES and len(kept_so_far()) == 0:
                    print(f"{len(dropped)} episode(s) dropped with nothing kept "
                          f"({sorted(set(dropped))}), so nothing has been "
                          f"collected. The page is not loading, or the candidate "
                          f"rule is reading something this environment does not "
                          f"provide -- check one episode by hand before spending "
                          f"the rest of the run. Exiting "
                          f"{ENVIRONMENT_BROKEN}: **do not resume on this code**, "
                          f"it is for a person and a resume would drop three more "
                          f"and spend the attempt caps", flush=True)
                    raise SystemExit(ENVIRONMENT_BROKEN)
                return
            append({"index": result.episode, "attempt": attempt, **built(result)})

        # Until there is nothing left pending, not once. A dropped episode has to
        # be retried **in this process**, because the corpus is what the heads are
        # trained from: if collection handed on a corpus that a later resume could
        # still add to, the evaluation cells cached by this process would have been
        # played by a head fitted to fewer episodes than the ones cached by the
        # next, and the table would be comparing two different heads under one
        # name. The cap is what makes this terminate.
        while pending:
            rollout_episodes(
                collecting, scorer, head,
                lambda_=config.lambda_, episodes=config.episodes_per_round,
                budget=config.budget, max_steps=config.max_steps,
                escalate_prob=None if head is not None else config.explore_prob,
                rng=rng, on_episode=keep, on_start=start, indices=pending,
                probe_prob=config.probe_prob)
            rows = load_corpus(stream)
            attempts = attempts_by_index(rows, EPISODE_KEY)
            pending, retried = resume_plan(rows, config.episodes_per_round,
                                           config.max_attempts, EPISODE_KEY)
            if pending:
                print(f"round {index}: retrying {len(pending)} dropped episode(s) "
                      f"{pending}", flush=True)
            for episode_index in retried:
                if episode_index not in noted:
                    noted.add(episode_index)
                    append({"index": episode_index,
                            "attempt": attempts.get(episode_index, 0),
                            "ended": "given_up"})

        rows = load_corpus(stream)
        episodes = trainable(rows)
        if not episodes:
            # The same code and for the same reason: a resume reads the rows
            # back, finds every index given up, has nothing pending, and raises
            # again -- a loop that never advances.
            print(f"round {index} collected no usable episodes, so there is "
                  f"nothing to train on. Exiting {ENVIRONMENT_BROKEN}: do not "
                  f"resume on this code", flush=True)
            raise SystemExit(ENVIRONMENT_BROKEN)
        _, abandoned = resume_plan(rows, config.episodes_per_round,
                                   config.max_attempts, EPISODE_KEY)

        if on_corpus is not None:
            on_corpus(index, episodes)
        head, report = train_head_on_episodes(metadata, episodes, config.head, seed=seed + index)
        head.save(output / f"head-round{index}.safetensors")
        # From the corpus and not from `results`, which holds only what **this**
        # process collected: after a resume that is the tail of the round, and a
        # mean over the tail is not the round's mean.
        steps_of = [episode["head_features"] for episode in episodes]
        row = {
            "round": index,
            # **Which games this round played**, not how many it asked for. An
            # environment that cannot say gets None, and that is itself worth
            # recording: ALFWorld's fast and slow endpoints were measured on
            # different games and no record said so, which was only recoverable
            # by comparing task strings afterwards.
            "games": (list(collecting.game_files)
                      if hasattr(collecting, "game_files") else None),
            "episodes": len(episodes),
            "given_up": len(abandoned),
            # Counted here rather than left in the log. exps found 28845's 13.8%
            # by reading it; a number in the round's own row is what makes the
            # next one visible without anybody watching.
            "dropped": len(dropped),
            "dropped_reasons": {reason: dropped.count(reason)
                                for reason in sorted(set(dropped))},
            "collected_reward": float(np.mean([e["terminal_reward"] for e in episodes])),
            "collected_escalations": float(np.mean(
                [sum(bool(s["escalated"]) for s in steps) for steps in steps_of])),
            "collected_tokens": float(np.mean(
                [sum(int(s["generated_tokens"]) for s in steps) for steps in steps_of])),
            "finished": float(np.mean([bool(e.get("finished")) for e in episodes])),
            "behaviour": "explore" if index == 0 else f"head-round{index - 1}",
            **{key: report[key] for key in (
                "decisions", "decisions_per_episode", "effective_sample_size",
                "effective_fraction", "ratio_max", "clipped_fraction",
                "clipped_fraction_mean", "zero_gradient_fraction",
                "zero_gradient_fraction_mean", "advantage_spread",
                "escalation_rate_learned", "escalation_rate_behaviour", "gain_spread")},
        }
        if config.eval_episodes:
            evaluated = rollout_episodes(
                make_environment(index, True), scorer, head, lambda_=config.lambda_,
                episodes=config.eval_episodes, budget=config.budget,
                max_steps=config.max_steps, rng=np.random.default_rng(10_000 + index))
            row["eval_reward"] = float(np.mean([r.reward for r in evaluated]))
            row["eval_escalations"] = float(np.mean([r.escalations for r in evaluated]))
            row["eval_tokens"] = float(np.mean([r.generated_tokens for r in evaluated]))
        rounds.append(row)
        (output / "rounds.json").write_text(json.dumps(
            {"config": asdict(config), "rounds": rounds}, indent=1, default=str) + "\n")
        print(json.dumps(row), flush=True)
    return head, rounds
