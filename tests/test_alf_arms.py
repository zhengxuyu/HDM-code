"""The ALFWorld arms: that each one is a decision rule over the loop's step function.

No engine, no alfworld. What is worth pinning here is not that the script runs but
that the four arms *differ only in the decision rule*, because the alternative --
an arm with its own step loop -- is what this venue already paid for once: the
fast and slow endpoints of 27309 and 27343 were measured by separate code and
played different games, and every paired figure built on them paired two
populations.

So each test drives `rollout_episodes` with the arm's own policy and the same
stubs the rollout's own tests use, and asserts what the arm must do to the
escalations. A test that re-derived the mapping instead of calling `arm_policy`
could not go red when the mapping changed.
"""
import json

import numpy as np
import pytest

from conftest import load_script
from slime_htm.rollout import rollout_episodes
from test_slime_htm_rollout import Corridor, Scorer

alf = load_script("alf_rollout")


def run(policy, head=None, escalate_prob=0.5, lambda_=0.0, length=4, fast_right=None):
    """One episode under `policy`, through the rollout the loop collects with.

    `fast_right` defaults to a corridor the fast mode walks correctly, so every
    arm reaches every step and a count of escalations is a count over the same
    trajectory. With the fast mode wrong the corridor ends at the first mistake,
    which makes the arms' step counts differ for a reason that has nothing to do
    with the mapping under test -- the first version of this file asserted equal
    step counts against exactly that setup.
    """
    environment = Corridor(fast_right=(fast_right if fast_right is not None
                                       else (True,) * length), length=length)
    scorer = Scorer(environment)
    results = rollout_episodes(
        environment, scorer, head,
        lambda_=lambda_, episodes=1, budget=16,
        escalate_prob=alf.arm_policy(policy, escalate_prob),
        rng=np.random.default_rng(0), max_steps=length, question_type="choice")
    (result,) = results
    return result, scorer


def test_fast_arm_never_escalates():
    """And it reaches the steps: zero escalations over zero steps proves nothing."""
    result, scorer = run("fast")
    assert len(result.steps) == 4
    assert [step.escalated for step in result.steps] == [False] * 4
    assert scorer.slow_calls == 0


def test_slow_arm_escalates_every_step():
    result, scorer = run("slow")
    assert len(result.steps) == 4
    assert [step.escalated for step in result.steps] == [True] * 4
    assert scorer.slow_calls == 4


def test_endpoints_are_the_same_code_path():
    """On a trajectory neither arm diverges from, they differ in the escalations alone.

    The property the separate-loop version could not have: same step count, same
    candidate count per step, same recorded fast distribution -- because the fast
    pass runs identically either way and only the decision after it differs. It
    needs a corridor the fast mode walks correctly; where escalating changes the
    action the trajectories legitimately differ, which is the next test.
    """
    fast, _ = run("fast")
    slow, _ = run("slow")
    assert len(fast.steps) == len(slow.steps) == 4
    assert ([step.candidates for step in fast.steps]
            == [step.candidates for step in slow.steps])
    for one, other in zip(fast.steps, slow.steps):
        assert np.allclose(one.probabilities, other.probabilities)


def test_escalating_can_change_the_trajectory():
    """Which is why the step counts above needed a corridor the fast mode walks.

    Here the fast mode is wrong at every step, so the fast arm ends at its first
    mistake and the slow arm walks the corridor out. Recorded because it is the
    thing that makes an arm comparison on this venue a comparison of trajectories
    rather than of per-step decisions.
    """
    fast, _ = run("fast", fast_right=(False,) * 4)
    slow, _ = run("slow", fast_right=(False,) * 4)
    assert len(fast.steps) == 1 and fast.reward == 0.0
    assert len(slow.steps) == 4 and slow.reward == 1.0


def test_random_arm_passes_its_probability_through():
    """At 0 and 1 the random arm is the endpoints, which is what makes it a probability."""
    never, scorer_never = run("random", escalate_prob=0.0)
    always, scorer_always = run("random", escalate_prob=1.0)
    assert not any(step.escalated for step in never.steps)
    assert all(step.escalated for step in always.steps)
    assert scorer_never.slow_calls == 0
    assert scorer_always.slow_calls == len(always.steps)


def test_head_arm_leaves_the_decision_to_the_head():
    """`arm_policy` must return None for the head, or the coin would decide instead.

    This is the one mapping whose failure is silent: any float here would make the
    routed arm a random policy that still reports a `head` summary and a plausible
    success rate.
    """
    assert alf.arm_policy("head", 0.5) is None
    assert alf.arm_policy("head", None) is None


def test_the_rollout_refuses_a_head_arm_with_no_head():
    """Which is why returning None above is safe rather than merely intended."""
    with pytest.raises(ValueError, match="escalate_prob"):
        rollout_episodes(Corridor(), Scorer(Corridor()), None,
                         lambda_=0.0, episodes=1, budget=16,
                         escalate_prob=alf.arm_policy("head", 0.5),
                         rng=np.random.default_rng(0), max_steps=1)


def test_every_declared_arm_has_a_rule():
    """`.get` returns None for an undefined name, and None means "the head decides".

    So a policy that `--policy` accepts and `arm_policy` has no rule for would run
    as the routed arm while reporting its own name -- a plausible success rate for
    a policy nobody ran. The two are kept in agreement by iterating the declared
    set rather than by a comment saying they match.
    """
    assert set(alf.POLICIES) == {"fast", "slow", "random", "confidence", "head"}
    for name in alf.POLICIES:
        # Two rule tables now, since a confidence cut cannot be written as a coin
        # probability. Each arm must be answered by exactly one of them: by none
        # and it runs as the head's, by both and the rollout refuses it.
        coin = alf.arm_policy(name, 0.5)
        cut = alf.confidence_rule(name, 0.8)
        if name == "head":
            assert coin is None and cut is None
        else:
            assert (coin is None) != (cut is None), (
                f"{name} is answered by {'both' if coin is not None else 'neither'} "
                f"rule table")


def test_the_sglang_arm_writes_what_it_measured(tmp_path, monkeypatch):
    """`play_through_the_loop` end to end, with the engine and the games stubbed.

    Both of its heavy dependencies load lazily -- `SGLangScorer.load` imports
    sglang and `AlfworldEnvironment.load` imports alfworld, each inside the
    method -- so the function is reachable here with neither installed. Worth
    reaching: until this existed the only way to find out whether the summary was
    built from the episodes or from something adjacent was a multi-hour 27B job.
    """
    import argparse

    from inference.sglang_backend import SGLangScorer
    from slime_htm.envs.alfworld.environment import AlfworldEnvironment

    class Games(Corridor):
        """A two-episode shard that also answers the shard questions."""

        def __init__(self):
            super().__init__(fast_right=(True, True), length=2)
            self.resets = 0

        @property
        def episodes(self):
            return 2

        @property
        def game_files(self):
            return ["game-a", "game-b"]

        @property
        def task(self):
            return f"task for episode {self.resets - 1}"

        def reset(self, episode):
            self.resets += 1
            return super().reset(episode)

    games = Games()
    monkeypatch.setattr(AlfworldEnvironment, "load",
                        classmethod(lambda cls, **kwargs: games))
    monkeypatch.setattr(SGLangScorer, "load",
                        classmethod(lambda cls, settings: _Provider(settings)))

    output = tmp_path / "arm.json"
    alf.play_through_the_loop(argparse.Namespace(
        model="/models/qwen38-27b", split="valid_unseen", policy="slow",
        escalate_prob=0.5, decision_head=None, lambda_=0.0, offset=0, games=2,
        game_list=None, max_steps=2, budget=16, seed=0, output=str(output)))

    summary = json.loads(output.read_text())
    assert summary["policy"] == "slow"
    assert summary["backend"] == "sglang"
    assert summary["games"] == 2 and len(summary["episodes"]) == 2
    # The shard it played, not the shard it asked for.
    assert summary["game_files"] == ["game-a", "game-b"]
    # Every step escalated, since this is the slow arm, and the summary's mean is
    # over the episodes rather than a constant.
    assert summary["escalations_per_episode"] == 2.0
    assert summary["success_rate"] == 1.0
    # **The per-episode task text, read while that episode was current.** A task
    # recorded after the loop would be the last episode's twice.
    assert [row["task"] for row in summary["episodes"]] == [
        "task for episode 0", "task for episode 1"]
    assert summary["hidden_states_requested"] is True

    # **The fast arm over the same stubs**, because on the slow arm every step
    # escalates and `escalations` and `len(steps)` are the same number there: the
    # assertion above passes just as well against a count of steps. It took a
    # deliberate break to notice, which is the argument for running both.
    games.resets = 0
    fast_output = tmp_path / "fast.json"
    alf.play_through_the_loop(argparse.Namespace(
        model="/models/qwen38-27b", split="valid_unseen", policy="fast",
        escalate_prob=0.5, decision_head=None, lambda_=0.0, offset=0, games=2,
        game_list=None, max_steps=2, budget=16, seed=0, output=str(fast_output)))
    fast = json.loads(fast_output.read_text())
    assert fast["escalations_per_episode"] == 0.0
    assert fast["mean_steps"] == 2.0
    assert fast["generated_per_episode"] == 0.0


class _Provider:
    """A `score`/`think` provider, which is the surface `StepScorer` needs."""

    def __init__(self, settings):
        self.settings = settings

    @property
    def hidden_size(self):
        return 4

    def score(self, context, schema):
        from litjev.core.pipeline_utils import RawFieldScores

        (name, question), = schema.items()
        return (RawFieldScores(
            name=name, logits=np.zeros(len(question.criteria), dtype=np.float32),
            input_tokens=1, hidden=np.zeros((1, 4), dtype=np.float32),
            generated_tokens=0, provenance={}),)

    def think(self, context, schema, names, budget):
        from litjev.core.pipeline_utils import RawFieldScores

        (name, question), = schema.items()
        # The slow mode prefers the first action, as the stub scorer above does.
        logits = np.full(len(question.criteria), -10.0, dtype=np.float32)
        logits[0] = 10.0
        return (RawFieldScores(
            name=name, logits=logits, input_tokens=1, hidden=None,
            generated_tokens=budget, provenance={}),)


def test_a_rounds_head_records_its_own_fit_and_not_the_seed_heads(tmp_path):
    """The fault exps found in `head-round0.safetensors` before 29700.

    The loop reads `--head` for its *shape* and fits a new head each round, and
    `DecisionHead(metadata, ...)` carried that seed head's whole metadata --
    including a `training` block describing the 27B's MMLU-Pro fit: `train_count
    1154`, `test_count 346`, three held-out categories and `auroc_fast_correct
    0.8758`. On a head fitted to 150 ALFWorld episodes, a venue with none of
    those quantities. Nothing raised and the file read as a well-provenanced head.

    Checked through `save`/`load` rather than on the object, because the file is
    what a reader opens months later and the metadata is serialised on the way.
    """
    from dataclasses import replace

    from litjev.core.block import DecisionHead, HeadMetadata
    from slime_htm.head_trainer import HeadRLConfig, train_head_on_episodes
    from episode_corpus import corpus

    # A seed head carrying exactly the provenance that leaked.
    seed_meta = HeadMetadata(
        model_id="m", revision="r", hidden_size=8, feature_layers=(-1,),
        training={"train_count": 1154, "test_count": 346,
                  "held_out_categories": ["chemistry", "economics", "psychology"],
                  "auroc_fast_correct": 0.8758177222717811})
    episodes = corpus()
    head, report = train_head_on_episodes(seed_meta, episodes, HeadRLConfig(), seed=0)
    path = tmp_path / "head-round0.safetensors"
    head.save(path)
    training = DecisionHead.load(path).metadata.training

    # None of the seed head's fit survives.
    for leaked in ("train_count", "test_count", "held_out_categories"):
        assert leaked not in training, f"{leaked} came from the seed head"
    assert training.get("auroc_fast_correct") != 0.8758177222717811

    # And this fit's own numbers are there, including the three that say whether
    # the head may be read as a trained policy at all.
    assert training["venue"] == "episodes"
    assert training["episodes"] == len(episodes)
    assert training["decisions"] == report["decisions"]
    for needed in ("clipped_fraction_mean", "zero_gradient_fraction_mean",
                   "effective_fraction"):
        assert needed in training, f"{needed} is how a saturated fit is spotted"

    # The structural fields are the seed head's and must survive, since serving
    # reads only those.
    loaded = DecisionHead.load(path).metadata
    assert replace(loaded, training={}) == replace(seed_meta, training={})
