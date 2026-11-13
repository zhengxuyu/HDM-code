"""The rollout's logic, with no engine, no environment and no slime installed."""
import math
import time

import numpy as np
import pytest

from slime_hdm.rollout import rollout_episodes, samples_from


class Corridor:
    """Three steps; the reward arrives only at the end, and only if every step was right.

    Action 0 is right at every step. The fast mode prefers it when `fast_right`
    says so for that step, and the slow mode always prefers it, so an episode
    succeeds exactly when every step the fast mode would get wrong was escalated.
    """

    def __init__(self, fast_right=(True, True, True), length=3):
        self.fast_right, self.length = fast_right, length
        self.position, self.broken = 0, False

    @property
    def episodes(self):
        return 1

    def reset(self, episode):
        self.position, self.broken = 0, False
        return f"step {self.position}", ["right", "wrong"]

    def step(self, action):
        if action != "right":
            self.broken = True
        self.position += 1
        done = self.position >= self.length or self.broken
        reward = 1.0 if (done and not self.broken) else 0.0
        actions = [] if done else ["right", "wrong"]
        return f"step {self.position}", actions, done, reward


class Scorer:
    def __init__(self, env, width=4):
        self.env, self.width, self.slow_calls = env, width, 0

    def fast(self, context, actions):
        right = self.env.fast_right[min(self.env.position, len(self.env.fast_right) - 1)]
        probabilities = np.array([0.9, 0.1] if right else [0.1, 0.9])
        return probabilities, np.full(self.width, float(self.env.position), dtype=np.float32)

    def slow(self, context, actions, budget):
        self.slow_calls += 1
        # Three values, as both real scorers return. This double returned two,
        # which is why the rollout's two-value unpack stayed green.
        return np.array([0.99, 0.01]), budget, "reasoning"


class Head:
    """Escalates when the gain the test asked for exceeds lambda."""

    def __init__(self, gain):
        self.gain = gain

    def predict(self, features):
        # outcome order: both right, fast right slow wrong, fast wrong slow right, both wrong.
        # The gain is log P(slow ok) - log P(fast ok), so an outcome with the gain
        # the test asked for puts P(slow ok) = 1 and P(fast ok) = exp(-gain).
        fast_right = math.exp(-max(self.gain, 0.0))
        return np.array([[fast_right, 0.0, 1.0 - fast_right, 0.0]])


def test_a_random_policy_explores_both_actions_and_records_every_step():
    env = Corridor()
    scorer = Scorer(env)
    results = rollout_episodes(env, scorer, head=None, lambda_=0.0, episodes=1, budget=7,
                              escalate_prob=1.0, rng=np.random.default_rng(0))
    (result,) = results
    assert len(result.steps) == 3 and result.finished
    assert result.escalations == 3 and result.generated_tokens == 21
    assert all(step.hidden.shape == (4,) for step in result.steps)


def test_thinking_rescues_the_steps_the_fast_mode_would_lose():
    env = Corridor(fast_right=(True, False, True))
    without = rollout_episodes(env, Scorer(env), head=None, lambda_=0.0, episodes=1, budget=1,
                               escalate_prob=0.0, rng=np.random.default_rng(0))
    assert without[0].reward == 0.0          # step 1 goes wrong and ends the episode
    env = Corridor(fast_right=(True, False, True))
    with_thinking = rollout_episodes(env, Scorer(env), head=None, lambda_=0.0, episodes=1,
                                     budget=1, escalate_prob=1.0, rng=np.random.default_rng(0))
    assert with_thinking[0].reward == 1.0


def test_the_head_escalates_only_above_lambda():
    env = Corridor()
    quiet = rollout_episodes(env, Scorer(env), Head(gain=0.05), lambda_=0.1, episodes=1, budget=1)
    assert quiet[0].escalations == 0
    env = Corridor()
    eager = rollout_episodes(env, Scorer(env), Head(gain=0.2), lambda_=0.1, episodes=1, budget=1)
    assert eager[0].escalations == 3
    assert all(step.gain == pytest.approx(0.2) for step in eager[0].steps)


def test_a_rollout_needs_either_a_head_or_an_exploration_rate():
    env = Corridor()
    with pytest.raises(ValueError, match="escalate_prob"):
        rollout_episodes(env, Scorer(env), head=None, lambda_=0.0, episodes=1, budget=1)


class FakeSample:
    class Status:
        COMPLETED = "completed"
        TRUNCATED = "truncated"


def test_samples_carry_the_terminal_reward_and_a_zero_loss_mask_by_default():
    env = Corridor()
    results = rollout_episodes(env, Scorer(env), head=None, lambda_=0.0, episodes=1, budget=2,
                               escalate_prob=1.0, rng=np.random.default_rng(0))
    (batch,) = samples_from(results, FakeSample, rollout_id=3, trainable=False)
    (sample,) = batch
    assert sample.reward == 1.0
    assert sample.loss_mask == [0, 0, 0], "a frozen backbone must not be trained on these"
    assert sample.tokens == []
    assert sample.rollout_id == 3
    assert sample.status == FakeSample.Status.COMPLETED
    assert len(sample.metadata["head_features"]) == 3
    assert sample.metadata["terminal_reward"] == 1.0


def test_training_the_backbone_has_to_be_asked_for_explicitly():
    env = Corridor()
    results = rollout_episodes(env, Scorer(env), head=None, lambda_=0.0, episodes=1, budget=1,
                               escalate_prob=0.0, rng=np.random.default_rng(0))
    (batch,) = samples_from(results, FakeSample, rollout_id=0, trainable=True)
    assert set(batch[0].loss_mask) == {1}


def test_an_unfinished_episode_is_marked_truncated_not_completed():
    env = Corridor(length=99)
    results = rollout_episodes(env, Scorer(env), head=None, lambda_=0.0, episodes=1, budget=1,
                               escalate_prob=0.0, rng=np.random.default_rng(0), max_steps=2)
    (batch,) = samples_from(results, FakeSample, rollout_id=0, trainable=False)
    assert batch[0].status == FakeSample.Status.TRUNCATED


class Probed:
    """A corridor where the two modes disagree at every step, so a probe is visible."""

    def __init__(self, length=6):
        self.length, self.position = length, 0

    def reset(self, episode):
        self.position = 0
        return f"step {self.position}", ["right", "wrong", "other"]

    def step(self, action):
        self.position += 1
        done = self.position >= self.length
        return (f"step {self.position}", [] if done else ["right", "wrong", "other"],
                done, 1.0 if done else 0.0)


class TwoMinds:
    """The fast mode prefers `right`; the slow mode prefers `other`. Always."""

    #: How long a slow pass takes, so a test can see it in the clock.
    slow_seconds = 0.0

    def fast(self, context, actions):
        return np.array([0.5, 0.3, 0.2]), np.zeros(4, dtype=np.float32)

    def slow(self, context, actions, budget):
        if self.slow_seconds:
            time.sleep(self.slow_seconds)
        return np.array([0.1, 0.2, 0.7]), 9, "reasoning"


def _replay(probe_prob, seed=11, episodes=3):
    return rollout_episodes(Probed(), TwoMinds(), head=None, lambda_=0.0,
                            episodes=episodes, budget=16,
                            escalate_prob=0.5, rng=np.random.default_rng(seed),
                            max_steps=8, probe_prob=probe_prob)


def test_probing_does_not_change_the_trajectory():
    """The probe runs the mode not taken and records what it would have done. If
    it changed which action was taken, or shifted the random stream the escalation
    decision draws from, the collected episodes would no longer be samples from
    the behaviour policy and the whole off-policy update would be invalid.

    So: the same seed, probing off and fully on, must give the same actions, the
    same escalations and the same rewards.
    """
    off, on = _replay(0.0), _replay(1.0)
    assert len(off) == len(on) == 3
    for quiet, probed in zip(off, on, strict=True):
        assert [s.action for s in quiet.steps] == [s.action for s in probed.steps]
        assert [s.escalated for s in quiet.steps] == [s.escalated for s in probed.steps]
        assert quiet.reward == probed.reward
    # `probe_prob` rations only the probe that costs a generation. The free one
    # on an escalated step happens either way, so what must be absent with
    # probing off is the slow probe, not every probe.
    assert not [s for e in off for s in e.steps if s.probe_mode == "slow"]
    assert [s for e in on for s in e.steps if s.probe_mode == "slow"], "vacuous"
    assert sum(e.probe_tokens for e in off) == 0
    assert sum(e.probe_tokens for e in on) > 0


def test_an_escalated_step_is_probed_for_nothing():
    """The fast distribution is computed to make the decision and was then thrown
    away. Recording it is the free half of the probe, so an escalated step is
    probed whatever `probe_prob` says and costs no tokens for it."""
    episodes = _replay(0.0)
    escalated = [s for e in episodes for s in e.steps if s.escalated]
    assert escalated, "the test would be vacuous"
    for step in escalated:
        assert step.probe_mode == "fast"
        assert step.probe_tokens == 0
        assert step.probe_action == "right"          # what the fast mode prefers
        assert step.action == "other"                # what the slow mode took
    # And with probing off, no unescalated step is probed.
    assert all(s.probe_mode is None for e in episodes for s in e.steps if not s.escalated)


def test_a_probe_is_not_charged_to_the_policy():
    """A probe's tokens were spent to learn something, not to answer. Counting
    them as the policy's cost would make collecting labels look like a more
    expensive policy, which is the column the method is judged on."""
    episodes = _replay(1.0)
    for episode in episodes:
        slow_probes = [s for s in episode.steps if s.probe_mode == "slow"]
        assert slow_probes, "the test would be vacuous"
        assert episode.probe_tokens == 9 * len(slow_probes)
        # The policy's own cost is its escalations, and nothing else.
        assert episode.generated_tokens == 9 * episode.escalations


def test_the_probe_records_whether_thinking_would_have_acted_differently():
    """The quantity the probe exists to supply: these two modes disagree at every
    state, so every probed step must be counted as a disagreement."""
    episodes = _replay(1.0)
    for episode in episodes:
        assert episode.probed_steps == len(episode.steps)
        assert episode.modes_disagreed == len(episode.steps)

    # And when the modes agree, nothing is counted, whatever the reward.
    class OneMind(TwoMinds):
        def slow(self, context, actions, budget):
            return np.array([0.5, 0.3, 0.2]), 9, "reasoning"

    agreeing = rollout_episodes(Probed(), OneMind(), head=None, lambda_=0.0,
                                episodes=1, budget=16, escalate_prob=0.5,
                                rng=np.random.default_rng(11), max_steps=8,
                                probe_prob=1.0)
    assert agreeing[0].probed_steps == len(agreeing[0].steps)
    assert agreeing[0].modes_disagreed == 0


def test_the_probe_reaches_the_head_trainer():
    """It is useless if it stops at the rollout."""
    class Sample:
        class Status:
            COMPLETED = "completed"
            TRUNCATED = "truncated"

    episodes = _replay(1.0, episodes=1)
    (batch,) = samples_from(episodes, Sample, rollout_id=0, trainable=False)
    metadata = batch[0].metadata
    assert metadata["probed_steps"] == episodes[0].probed_steps
    assert metadata["modes_disagreed"] == episodes[0].modes_disagreed
    assert metadata["probe_tokens"] == episodes[0].probe_tokens
    assert metadata["probe_tokens"] != metadata["generated_tokens"]
    for row, step in zip(metadata["head_features"], episodes[0].steps, strict=True):
        assert row["probe_mode"] == step.probe_mode
        assert row["probe_action"] == step.probe_action
        assert np.allclose(row["probe_probabilities"], step.probe_probabilities)


class FakeClock:
    """A monotonic clock the test advances itself.

    The scorer's slow pass is already fake, so measuring real time to check a
    bookkeeping property measured nothing but the machine. exps found the version
    that slept and compared two rollouts within 20 ms failing about one suite run
    in sixteen -- the tolerance sat near the 97th percentile of the noise -- and
    a flaky guard gets quarantined, which silently stops checking the property it
    was written for.

    **It advances only when the scorer advances it, never on a read.** A clock
    that moves when it is read cannot test this property at all: the probed path
    makes more clock calls than the unprobed one, so `seconds` would differ
    because of the reads and not because any time was charged, and the test would
    report a defect that is not there. exps hit exactly that writing an
    independent check, and it is the version anyone reaches for first.
    """

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def test_a_probe_is_not_charged_to_the_policys_clock():
    """The same argument as `probe_tokens`, in the other currency.

    `seconds` is the other cost column the method is judged on, and a step the
    policy answered with the fast mode must not be charged for a slow pass run
    only to produce a label. The first version of the probe separated the tokens
    and left the seconds inflated: at MiniWoB's recorded step times a marginal
    slow pass is about 9.3 s, so probing a tenth of the non-escalated steps would
    have added 7.9% to the corpus's wall clock inside the reported column. Nor is
    it subtractable afterwards unless it was recorded, which is why
    `probe_seconds` exists.

    With an injected clock the property is exact: probing must leave `seconds`
    bit-identical, not merely close.
    """
    PROBE = 4.0

    def replay(probe_prob):
        clock = FakeClock()

        class Slow(TwoMinds):
            def slow(self, context, actions, budget):
                clock.advance(PROBE)
                return super().slow(context, actions, budget)

        episodes = rollout_episodes(
            Probed(), Slow(), head=None, lambda_=0.0, episodes=2, budget=16,
            escalate_prob=0.5, rng=np.random.default_rng(11), max_steps=8,
            probe_prob=probe_prob, clock=clock)
        return episodes

    off, on = replay(0.0), replay(1.0)
    probed = [s for e in on for s in e.steps if s.probe_mode == "slow"]
    assert probed, "the test would be vacuous"
    assert all(s.probe_seconds == PROBE for s in probed)

    for quiet, loud in zip(off, on, strict=True):
        for a, b in zip(quiet.steps, loud.steps, strict=True):
            assert b.seconds == a.seconds, (a.seconds, b.seconds)
        assert loud.seconds == quiet.seconds
        assert loud.probe_seconds == PROBE * len(
            [s for s in loud.steps if s.probe_mode == "slow"])
        assert quiet.probe_seconds == 0.0

    # An escalated step probes by reusing what it already computed, so it adds no
    # time either -- the free half is free in both currencies.
    for step in (s for e in on for s in e.steps if s.probe_mode == "fast"):
        assert step.probe_seconds == 0.0
        assert step.probe_tokens == 0


def test_probing_refuses_a_generator_it_cannot_make_independent():
    """A fixed-seed fallback would give every episode the same probe stream and
    quietly correlate the labels instead of raising. exps caught that."""
    with pytest.raises(TypeError, match="spawn"):
        rollout_episodes(Probed(), TwoMinds(), head=None, lambda_=0.0, episodes=1,
                         budget=4, escalate_prob=0.5,
                         rng=np.random.RandomState(0), max_steps=4, probe_prob=0.5)


class WritesFreely:
    """A venue whose escalations write their action, as the browser's do."""

    def __init__(self, length=4):
        self.length, self.position = length, 0

    def reset(self, episode):
        self.position = 0
        return "page 0", ["click('1')", "click('2')"]

    def slow_instructions(self):
        return "write one click"

    def act_freely(self, text):
        self.position += 1
        done = self.position >= self.length
        return ("page", [] if done else ["click('1')", "click('2')"], done,
                1.0 if done else 0.0), None

    def step(self, action):
        self.position += 1
        done = self.position >= self.length
        return ("page", [] if done else ["click('1')", "click('2')"], done,
                1.0 if done else 0.0)


class WritingMinds:
    """Fast prefers click(1); the written slow path always answers click(2)."""

    def fast(self, context, actions):
        return np.array([0.7, 0.3]), np.zeros(4, dtype=np.float32)

    def slow(self, context, actions, budget):
        return np.array([0.2, 0.8]), 9, "reasoning"

    def solve(self, context, instructions, budget):
        return "click('2')", 11, False


def test_the_probe_fires_where_escalation_writes_its_action():
    """The browser venue's escalations take the `written` path, and both probe
    branches used to require `not written` -- so the half documented as
    unconditional and free never fired there at all, and 44% of a browser
    corpus's steps would have carried no probe. The free half must not depend on
    which slow path the venue uses, because the fast distribution is in hand
    either way."""
    escalating = rollout_episodes(
        WritesFreely(), WritingMinds(), head=None, lambda_=0.0, episodes=1,
        budget=16, escalate_prob=1.0, rng=np.random.default_rng(0), max_steps=6,
        probe_prob=0.0)
    steps = escalating[0].steps
    assert steps and all(s.escalated for s in steps)
    for step in steps:
        assert step.probe_mode == "fast", "the free probe must fire when writing"
        assert step.probe_tokens == 0 and step.probe_seconds == 0.0
        assert step.probe_action == "click('1')"    # what the fast mode wanted
        assert step.action == "click('2')"          # what the slow mode wrote
    assert escalating[0].modes_disagreed == len(steps)


def test_a_probe_runs_the_slow_path_this_venue_actually_uses():
    """Where escalating writes its action, a probe that read the slow answer off
    the candidate codes instead would record what a mode the policy never takes
    would have chosen. It has to write too, and parse the text the same way --
    without executing it, which is what keeps the trajectory untouched."""
    quiet = rollout_episodes(
        WritesFreely(), WritingMinds(), head=None, lambda_=0.0, episodes=1,
        budget=16, escalate_prob=0.0, rng=np.random.default_rng(0), max_steps=6,
        probe_prob=1.0)
    steps = quiet[0].steps
    assert steps and not any(s.escalated for s in steps)
    for step in steps:
        assert step.probe_mode == "slow_written", step.probe_mode
        assert step.probe_tokens == 11              # solve's cost, not slow's 9
        assert step.probe_action == "click('2')"    # parsed from the written text
        assert step.action == "click('1')"          # the fast mode still acted

    # And in a venue without `act_freely` the probe reads the candidates, as before.
    plain = rollout_episodes(
        Probed(), TwoMinds(), head=None, lambda_=0.0, episodes=1, budget=16,
        escalate_prob=0.0, rng=np.random.default_rng(0), max_steps=6, probe_prob=1.0)
    assert all(s.probe_mode == "slow" for s in plain[0].steps)


def test_probe_prob_reaches_the_collection_and_the_signature():
    """A probe nobody can switch on is a probe that collects nothing. The flag
    was absent from both the loop config and the launcher when the first
    collection was specified, so the run would have paid full GPU cost for an
    unprobed corpus."""
    import inspect
    import pathlib

    from slime_hdm.loop import LoopConfig, run

    assert "probe_prob" in {f for f in LoopConfig.__dataclass_fields__}
    assert LoopConfig().probe_prob == 0.0, "probing is opt-in"
    # The loop must hand it to the rollout, not merely hold it.
    assert "probe_prob=config.probe_prob" in inspect.getsource(run)

    # **And the launcher must expose it and seal it into the signature.** The
    # previous version of this test checked neither, despite its name: a refactor
    # dropping `probe_prob` from the signature dict would have stayed green, and
    # that dict is the only thing stopping a probed corpus being resumed onto an
    # unprobed one. exps pointed that out.
    launcher = (pathlib.Path(__file__).resolve().parents[1]
                / "scripts" / "browser_pmpo.py").read_text()
    assert '"--probe-prob"' in launcher, "no flag means nothing can switch it on"
    # By balancing braces, not by slicing to the first `}` or to one at a given
    # indent: the dict holds nested dicts and closes inline on its last line, so
    # both of those break on reformatting rather than on the thing being checked.
    # My first two attempts did exactly that; exps caught the first.
    start = launcher.index("collection_signature = {") + len("collection_signature = ")
    depth, end = 0, None
    for offset, character in enumerate(launcher[start:], start):
        depth += (character == "{") - (character == "}")
        if depth == 0:
            end = offset + 1
            break
    assert end is not None, "the collection signature dict does not close"
    signature = launcher[start:end]
    assert '"probe_prob": args.probe_prob' in signature, (
        "probe_prob must be in the collection signature, or a probed corpus and "
        "an unprobed one digest alike and --resume merges them")


class Canonicalising(WritesFreely):
    """A venue that owns its action grammar, as `BrowserEnvironment` does."""

    def canonical_action(self, text):
        line = text.strip().splitlines()[-1].strip().strip("`*")
        if "click(" not in line:
            return None, "no action call was written"
        bid = line.split("click(")[1].split(")")[0].strip("'\"")
        if bid not in {"1", "2"}:
            return None, f"{bid!r} is not an element on this page"
        return f"click({bid!r})", None          # repr, the candidate rendering


def _one_probed_step(writes, probe_prob=1.0, escalate_prob=0.0):
    return rollout_episodes(
        Canonicalising(), writes, head=None, lambda_=0.0, episodes=1, budget=16,
        escalate_prob=escalate_prob, rng=np.random.default_rng(0), max_steps=1,
        probe_prob=probe_prob)[0].steps[0]


def test_a_formatting_difference_is_not_a_disagreement():
    """The written form and the enumerated form of the same action are different
    strings: a model writing `click("2")`, backticking it, or adding a sentence
    means the candidate `click('2')`. Comparing raw can only manufacture
    disagreement and never agreement, so the error has a direction and inflates
    the one number this probe exists to produce."""
    for written in ('click("2")', '`click("2")`', 'I will click.\nclick(\'2\')',
                    '**click("2")**'):
        class Writes(WritingMinds):
            def solve(self, context, instructions, budget):
                return written, 11, False

        step = _one_probed_step(Writes())
        assert step.probe_problem is None, (written, step.probe_problem)
        assert step.probe_action == "click('2')", (written, step.probe_action)

    # The fast mode took click('1'), so this is a real disagreement and counts.
    class Agrees(WritingMinds):
        def solve(self, context, instructions, budget):
            return '`click("1")`', 11, False

    step = _one_probed_step(Agrees())
    assert step.probe_action == step.action == "click('1')"
    assert step.probe_problem is None


def test_a_probe_that_wrote_nothing_executable_refused_rather_than_disagreed():
    """A probe whose text is not an action did not choose differently. Folding it
    into the disagreement count is the substitution `act_freely` declines to make
    for the taken action."""
    class Rambles(WritingMinds):
        def solve(self, context, instructions, budget):
            return "I am not sure what to do here.", 11, False

    episode = rollout_episodes(
        Canonicalising(), Rambles(), head=None, lambda_=0.0, episodes=1,
        budget=16, escalate_prob=0.0, rng=np.random.default_rng(0), max_steps=3,
        probe_prob=1.0)[0]
    assert episode.probed_steps == len(episode.steps)
    assert episode.probe_refusals == len(episode.steps)
    assert episode.modes_disagreed == 0, "a refusal is not a disagreement"
    assert episode.modes_agreed == 0, "nor an agreement"
    for step in episode.steps:
        assert step.probe_problem == "no action call was written"

    # An out-of-range bid is a refusal too, and names itself.
    class Invents(WritingMinds):
        def solve(self, context, instructions, budget):
            return "click('99')", 11, False

    step = _one_probed_step(Invents())
    assert step.probe_problem == "'99' is not an element on this page"
    assert step.probe_action is None


def test_the_three_probe_outcomes_partition_the_probed_steps():
    """agreed + disagreed + refused must be exactly the probed steps, or some
    probe is being counted twice or not at all."""
    class Mixed(WritingMinds):
        def __init__(self): self.n = 0
        def solve(self, context, instructions, budget):
            self.n += 1
            return ["click('1')", "click('2')", "nothing at all"][self.n % 3], 11, False

    episode = rollout_episodes(
        Canonicalising(length=9), Mixed(), head=None, lambda_=0.0, episodes=1,
        budget=16, escalate_prob=0.0, rng=np.random.default_rng(0), max_steps=9,
        probe_prob=1.0)[0]
    assert episode.probed_steps > 0
    assert (episode.modes_agreed + episode.modes_disagreed
            + episode.probe_refusals) == episode.probed_steps


def test_a_written_step_records_what_it_executes_not_its_last_line():
    """`record.action` was the raw last line on every written step, so the corpus
    misdescribed the taken action on 44% of a browser run's steps -- independently
    of the agreement rate."""
    class Decorates(WritingMinds):
        def solve(self, context, instructions, budget):
            return 'Thinking done.\n`click("2")`', 11, False

    step = _one_probed_step(Decorates(), probe_prob=0.0, escalate_prob=1.0)
    assert step.escalated
    assert step.action == "click('2')", step.action


def test_whether_a_choice_was_even_on_offer_is_recorded():
    """The reach/reasoning split. A disagreement where the slow mode picked a
    different element of the same offered set is reasoning; one where it wrote an
    element never offered is reach. The step row keeps `candidates` as a count, so
    the offered list is gone after the run and this boolean is the only way the
    distinction survives."""
    class Reaches(WritingMinds):
        def solve(self, context, instructions, budget):
            return "click('2')", 11, False          # offered

    step = _one_probed_step(Reaches())
    assert step.probe_action == "click('2')"
    assert step.probe_action_offered is True, "click('2') is on offer here"

    class OffPage(WritesFreely):
        """Its parser accepts a bid the fast mode was never offered."""
        def canonical_action(self, text):
            bid = text.strip().split("click(")[1].split(")")[0].strip("'\"")
            return f"click({bid!r})", None

    class Writes(WritingMinds):
        def solve(self, context, instructions, budget):
            return "click('99')", 11, False

    reach = rollout_episodes(OffPage(), Writes(), head=None, lambda_=0.0,
                             episodes=1, budget=16, escalate_prob=0.0,
                             rng=np.random.default_rng(0), max_steps=1,
                             probe_prob=1.0)[0].steps[0]
    assert reach.probe_action == "click('99')"
    assert reach.probe_action_offered is False, "it was never on offer"
    assert reach.probe_problem is None, "it parsed; it just was not offered"

    # And a refusal has no offered-ness to report rather than a false one.
    class Rambles(WritingMinds):
        def solve(self, context, instructions, budget):
            return "no idea", 11, False

    assert _one_probed_step(Rambles()).probe_action_offered is None

    # **The same on the taken side**, which nothing pinned. On an unparsable
    # written step `action` is the raw-line fallback, so `in actions` is false and
    # `False` would assert the slow mode reached outside the offered set -- a
    # reach claim about a text that named no action at all.
    escalated = rollout_episodes(
        Canonicalising(), Rambles(), head=None, lambda_=0.0, episodes=1,
        budget=16, escalate_prob=1.0, rng=np.random.default_rng(0), max_steps=1,
        probe_prob=0.0)[0].steps[0]
    assert escalated.action_problem == "no action call was written"
    assert escalated.action_offered is None, escalated.action_offered

    # Where it did parse and was on offer, it says so.
    class Picks(WritingMinds):
        def solve(self, context, instructions, budget):
            return "click('2')", 11, False

    took = rollout_episodes(
        Canonicalising(), Picks(), head=None, lambda_=0.0, episodes=1, budget=16,
        escalate_prob=1.0, rng=np.random.default_rng(0), max_steps=1,
        probe_prob=0.0)[0].steps[0]
    assert took.action == "click('2')" and took.action_offered is True

    # **And the case the field exists for**, which had no assertion: a written
    # action that parses and was never on offer. `None` and `True` pin the
    # refusal and the ordinary step; this pins the measurement. It is also the
    # only row shape that distinguishes a reach disagreement -- taken not
    # offered, probe offered -- from a reasoning one, where both were offered and
    # they differ, so nothing would have noticed if it stopped being produced.
    reached = rollout_episodes(
        OffPage(), Writes(), head=None, lambda_=0.0, episodes=1, budget=16,
        escalate_prob=1.0, rng=np.random.default_rng(0), max_steps=1,
        probe_prob=0.0)[0].steps[0]
    assert reached.action == "click('99')"
    assert reached.action_problem is None, "it parsed"
    assert reached.action_offered is False, "and was never offered -- a reach"
    assert reached.probe_mode == "fast" and reached.probe_action_offered is True
    # Which is the reach shape: the two modes differ and only one was on offer.
    assert reached.probe_action != reached.action


def test_a_taken_action_that_did_not_parse_is_not_a_disagreement_either():
    """The mirror of the probe refusal, and the one comparison the canonicalisation
    did not reach: on an escalated step whose written text does not parse, the
    taken action falls back to the raw line while the free probe stores a canonical
    candidate, so the step was compared raw against canonical and counted as a
    disagreement -- with `probe_problem` empty, because that field describes the
    probe's text and not the taken one."""
    class Rambles(WritingMinds):
        def solve(self, context, instructions, budget):
            return "I will think about it some more.", 11, False

    episode = rollout_episodes(
        Canonicalising(), Rambles(), head=None, lambda_=0.0, episodes=1,
        budget=16, escalate_prob=1.0, rng=np.random.default_rng(0), max_steps=3,
        probe_prob=0.0)[0]
    steps = episode.steps
    assert steps and all(s.escalated and s.probe_mode == "fast" for s in steps)
    for step in steps:
        assert step.action_problem == "no action call was written"
        assert step.probe_problem is None, "the probe's own text was fine"
    assert episode.taken_unparsable == len(steps)
    assert episode.modes_disagreed == 0, "nothing comparable was taken"
    assert episode.modes_agreed == 0


def test_no_probed_step_falls_outside_the_four_outcomes():
    """agreed + disagreed + probe refusals + unparsable taken actions must be
    exactly the probed steps.

    **This checks that no step is dropped or double counted, and nothing more.**
    It is blind to any misclassification that preserves the total, which is most
    of them -- removing a `_verdict` branch moves a step into another counted
    bucket and the sum still holds. exps checked that rather than taking my word
    for it. What carries the ordering claim is the per-case tests above.
    """
    class Mixed(WritingMinds):
        def __init__(self): self.n = 0
        def solve(self, context, instructions, budget):
            self.n += 1
            return ["click('1')", "click('2')", "nothing", "also nothing"][
                self.n % 4], 11, False

    for escalate in (0.0, 1.0):
        episode = rollout_episodes(
            Canonicalising(length=12), Mixed(), head=None, lambda_=0.0,
            episodes=1, budget=16, escalate_prob=escalate,
            rng=np.random.default_rng(0), max_steps=12, probe_prob=1.0)[0]
        assert episode.probed_steps > 0, escalate
        assert (episode.modes_agreed + episode.modes_disagreed
                + episode.probe_refusals
                + episode.taken_unparsable) == episode.probed_steps, escalate


def test_both_writers_write_the_same_row(tmp_path):
    """The jsonl corpus and the slime metadata carry one key set, not two.

    They drifted twice. The first time `probe_prob` reached the rollout and no
    caller could switch it on. The second time the probe fired, cost GPU and was
    thrown away: `loop.built` listed eight step keys by hand and none of them
    were the probe's, so a run at `probe_prob` 1.0 wrote a corpus key-for-key
    identical to an unprobed one while its own signature said it had probed --
    and `PROBED-COLLECTION.md`'s gates 1, 2 and 4 read fields that path never
    wrote, so the corpus could not have been gated at all.

    This test **calls both writers** on episodes from the same rollout and
    compares what they produced. It does not re-derive the key list, which is
    what let the drift through: a test that spells out the keys itself is a third
    copy of the thing under test.
    """
    import json

    from litjev._legacy.heads import HeadMetadata
    from slime_hdm.environment import EpisodeResult, StepRecord
    from slime_hdm.loop import LoopConfig, corpus_path, run

    metadata = HeadMetadata(model_id="m", revision="r", hidden_size=4,
                            feature_layers=(-1,))
    config = LoopConfig(rounds=1, episodes_per_round=1, probe_prob=1.0, budget=8,
                        max_steps=4, output=str(tmp_path))
    run(metadata, lambda *_, **__: Corridor(), Scorer(Corridor()), config)
    written = [json.loads(line)
               for line in corpus_path(tmp_path, 0).read_text().splitlines()]
    corpus = [row for row in written if "head_features" in row]
    assert corpus, "the loop wrote no episode row to learn from"

    class Sample:
        class Status:
            COMPLETED, TRUNCATED = "completed", "truncated"

    episodes = rollout_episodes(
        Corridor(), Scorer(Corridor()), None, lambda_=0.0, episodes=1, budget=8,
        max_steps=4, escalate_prob=0.5, rng=np.random.default_rng(0), probe_prob=1.0)
    served = samples_from(episodes, Sample, rollout_id=0, trainable=False)[0][0].metadata

    # Every field of a step is written, by both. This is what goes red when a
    # field is added to `StepRecord` and not to `row`.
    for keys in (set(corpus[0]["head_features"][0]), set(served["head_features"][0])):
        missing = set(StepRecord.__dataclass_fields__) - keys
        assert not missing, f"a step's own fields went unwritten: {sorted(missing)}"

    # And the two paths differ only where they are documented to: the collection
    # records the acting policy's probability, which the serving path has no
    # behaviour to record, and the serving path counts the steps it carries.
    assert (set(corpus[0]["head_features"][0]) - set(served["head_features"][0])
            == {"behaviour_prob"})
    assert not set(served["head_features"][0]) - set(corpus[0]["head_features"][0])
    assert set(corpus[0]) - set(served) == {"index", "attempt"}
    assert set(served) - set(corpus[0]) == {"steps"}

    # The aggregates the gates read are in the corpus and are the episode's own,
    # not a default: at `probe_prob` 1.0 every step is probed, and the four
    # verdicts partition the probed steps (gate 4).
    for row in corpus:
        steps = row["head_features"]
        assert row["probed_steps"] == len(steps) > 0
        assert (row["modes_agreed"] + row["modes_disagreed"] + row["probe_refusals"]
                + row["taken_unparsable"] == row["probed_steps"])
        assert row["rewarded_steps"] <= 1
    assert set(EpisodeResult.row(episodes[0])) <= set(served)


def test_a_confidence_threshold_escalates_on_the_fast_mode_s_top_probability():
    """The arm the single-question venues are compared against, in the loop.

    `Scorer.fast` puts 0.9 on one action at every step, so a cut above that
    escalates every step and a cut below it escalates none. Nothing is trained
    and no head is passed: that is the whole point of the baseline.
    """
    every = rollout_episodes(Corridor(), Scorer(Corridor()), head=None, lambda_=0.0,
                             episodes=1, budget=5, confidence_cut=0.95)
    none = rollout_episodes(Corridor(), Scorer(Corridor()), head=None, lambda_=0.0,
                            episodes=1, budget=5, confidence_cut=0.8)
    assert every[0].escalations == len(every[0].steps)
    assert none[0].escalations == 0


def test_the_confidence_arm_does_not_write_its_cut_into_the_head_s_confidence_field():
    """`confidence` records the head's calibrated P(fast ok), a different quantity.

    Writing the fast distribution's maximum there would make a record of this
    arm indistinguishable from a head run whose c_i happened to equal it.
    """
    (result,) = rollout_episodes(Corridor(), Scorer(Corridor()), head=None, lambda_=0.0,
                                 episodes=1, budget=5, confidence_cut=0.95)
    assert all(step.confidence is None and step.gain is None for step in result.steps)


def test_two_decision_rules_at_once_are_refused():
    with pytest.raises(ValueError, match="two decision rules"):
        rollout_episodes(Corridor(), Scorer(Corridor()), head=None, lambda_=0.0,
                         episodes=1, budget=5, escalate_prob=1.0, confidence_cut=0.95,
                         rng=np.random.default_rng(0))
