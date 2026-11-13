"""Does probing make the gain recoverable? Measured against a known truth.

The claim the probe exists to support is that the gain is not identifiable from a
corpus of episodic steps and is identifiable from a probed one. That is testable
without a model or an environment: build states whose true `P(fast right)` and
`P(slow right)` are chosen, generate episodes under a behaviour policy, and check
whether the fitted head's gain tracks the true one.
"""
import math

import numpy as np
import pytest
import torch

from litjev.core.block import HeadMetadata, escalation_gain_batch
from litjev.core.block_utils import build_features
from slime_hdm.probed_trainer import (
    allowed_classes,
    flatten_probed,
    marginal_cross_entropy,
    train_head_on_probed_episodes,
)

# Four kinds of state. The first is where thinking is worth most and the last
# where it is worth nothing, and the ordering of their true gains is what a
# fitted head has to recover.
STATES = (
    # (name, P(fast right), P(slow right))
    ("thinking rescues", 0.1, 0.9),
    ("thinking helps", 0.4, 0.8),
    ("thinking helps a little", 0.7, 0.8),
    ("thinking is pointless", 0.9, 0.9),
)


def true_gain(fast_right, slow_right):
    return math.log(slow_right) - math.log(fast_right)


def corpus(episodes=400, steps=4, seed=0, probe=True, per_step_reward=False):
    """Episodes over the four states, with the generative story made explicit.

    Each step sits in one of the four states, marked in the feature vector so the
    head can tell them apart. A step's taken mode is right with that state's
    probability for the mode, and the two modes **choose the same action exactly
    when they are both right or both wrong** -- which is what makes agreement the
    observable proxy for the asymmetric classes, and is the generative assumption
    this test is checking the trainer against rather than hiding.
    """
    rng = np.random.default_rng(seed)
    out = []
    for index in range(episodes):
        rows, right_throughout = [], True
        for step in range(steps):
            which = (index + step) % len(STATES)
            _, fast_right, slow_right = STATES[which]
            escalated = bool(rng.random() < 0.5)
            fast_ok = rng.random() < fast_right
            slow_ok = rng.random() < slow_right
            taken_ok = slow_ok if escalated else fast_ok
            right_throughout &= taken_ok
            feature = np.zeros(8, dtype=np.float32)
            feature[which] = 1.0
            row = {
                "step": step, "hidden": feature,
                "probabilities": np.array([0.6, 0.4]), "candidates": 2,
                "escalated": escalated, "generated_tokens": 9 if escalated else 0,
                "action": "taken",
            }
            if per_step_reward:
                # From the **same** draw as the agreement flag below. Drawing it
                # again would make the label and the probe describe different
                # draws of one state, and the fitted gain would then be worse
                # with per-step credit than without -- which is what the first
                # version of this fixture showed, and it was the fixture.
                row["reward"] = 1.0 if taken_ok else 0.0
            if probe:
                agreed = fast_ok == slow_ok
                row |= {"probe_mode": "fast" if escalated else "slow",
                        "probe_action": "taken" if agreed else "other",
                        "probe_probabilities": np.array([0.4, 0.6]),
                        "probe_tokens": 0 if escalated else 9}
            rows.append(row)
        out.append({"terminal_reward": 1.0 if right_throughout else 0.0,
                    "head_features": rows})
    return out


def metadata():
    return HeadMetadata("m", "r", hidden_size=8, feature_layers=(-1,), hidden_width=32)


def state_feature(index):
    """The feature vector for one state, built the way the corpus builds it.

    Not a bare one-hot of length `input_dim`: `build_features` appends statistics
    of the action distribution to the hidden state, so a hand-made vector with
    zeros in those columns is an input the head never saw and its output there
    says nothing. The first version of this test did that and read the resulting
    gain of exactly 0.0 as the trainer failing.
    """
    hidden = np.zeros(8, dtype=np.float32)
    hidden[index] = 1.0
    return build_features(hidden, np.array([0.6, 0.4]), "choice")[None].astype(np.float32)


def fitted_gain(head, index):
    with torch.no_grad():
        outcome = torch.softmax(head(torch.as_tensor(state_feature(index))), dim=-1)
    return float(escalation_gain_batch(outcome)[0])


def test_the_allowed_classes_are_the_ones_the_step_could_have_produced():
    """Agreement pins exactly one class; disagreement leaves the two that agree
    with the mode that acted and say nothing about the other."""
    assert allowed_classes(agreed=True, succeeded=True, escalated=False) == (0,)
    assert allowed_classes(agreed=True, succeeded=False, escalated=True) == (3,)
    # The slow mode acted and the episode succeeded: slow was right, fast unknown.
    assert allowed_classes(agreed=False, succeeded=True, escalated=True) == (0, 2)
    # The fast mode acted and the episode succeeded: fast right, slow unknown.
    assert allowed_classes(agreed=False, succeeded=True, escalated=False) == (0, 1)
    # Failures, the same way round.
    assert allowed_classes(agreed=False, succeeded=False, escalated=True) == (1, 3)
    assert allowed_classes(agreed=False, succeeded=False, escalated=False) == (2, 3)

    # Every allowed set must agree with the mode that acted about that mode.
    for succeeded in (True, False):
        for escalated in (True, False):
            allowed = allowed_classes(False, succeeded, escalated)
            for outcome in allowed:
                acted_right = (outcome in (0, 2)) if escalated else (outcome in (0, 1))
                assert acted_right == succeeded, (outcome, succeeded, escalated)


def test_the_marginal_loss_scores_the_event_and_not_an_invented_label():
    """With two classes allowed, the loss is the negative log of their sum. Any
    split of the pair into one chosen label would assert something unobserved."""
    logits = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
    both = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    assert float(marginal_cross_entropy(logits, both)) == pytest.approx(
        -math.log(0.1 + 0.3), abs=1e-6)
    one = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    assert float(marginal_cross_entropy(logits, one)) == pytest.approx(
        -math.log(0.3), abs=1e-6)
    # A mask allowing everything costs nothing, which is the degenerate case an
    # unprobed step would reduce to.
    everything = torch.ones(1, 4)
    assert float(marginal_cross_entropy(logits, everything)) == pytest.approx(0.0, abs=1e-6)


def test_unprobed_steps_are_dropped_rather_than_given_a_weaker_label():
    rows = corpus(episodes=2, probe=False)
    with pytest.raises(ValueError, match="no probed steps"):
        flatten_probed(rows)
    mixed = corpus(episodes=4, probe=True)
    for row in mixed[0]["head_features"]:
        row["probe_mode"] = None
    features, _, _ = flatten_probed(mixed)
    assert len(features) == 3 * 4, "only the probed episodes' steps survive"


def test_asymmetric_mass_goes_where_the_modes_disagreed():
    """The pathology the probe is for. A head trained this way must not spend
    `fast only` or `slow only` mass at states where the two modes provably acted
    alike, because at those states those classes are impossible."""
    _, report = train_head_on_probed_episodes(metadata(), corpus(), seed=0)
    agreed = report["asymmetric_mass_where_they_agreed"]
    differed = report["asymmetric_mass_where_they_differed"]
    assert agreed < 0.1, agreed
    assert differed > 3 * agreed, (agreed, differed)


def test_the_fitted_gain_recovers_the_true_ordering_when_credit_is_assigned():
    """The claim, end to end, and no further than it goes.

    Four states with true gains 2.197, 0.693, 0.134 and 0.000 nats. With one step
    per episode the terminal reward *is* that step's outcome, so credit assignment
    is out of the way and only identifiability is being tested. There the fitted
    gain recovers the ordering: 1.40, 0.13, 0.004, 0.000.

    What it does not recover is the **scale**. 0.693 fits at 0.13 and 2.197 at
    1.40, and eight times the data does not close the gap (at 6400 episodes the
    second state fits *lower*, at 0.056). So a probed corpus orders the states and
    does not put lambda on the right scale, which matters because lambda is priced
    in these units.
    """
    head, report = train_head_on_probed_episodes(
        metadata(), corpus(episodes=1600, steps=1), seed=0)

    truth = [true_gain(fast, slow) for _, fast, slow in STATES]
    fitted = [fitted_gain(head, index) for index in range(len(STATES))]
    assert truth == sorted(truth, reverse=True), "the states were built in order"
    assert fitted == sorted(fitted, reverse=True), dict(zip(
        [s[0] for s in STATES], [round(f, 3) for f in fitted], strict=True))
    assert fitted[0] > 1.0 and fitted[0] > 5 * fitted[1], fitted
    assert report["gain_spread"] > 0.2, report["gain_spread"]

    # The compression, asserted so it is not quietly fixed or quietly forgotten.
    assert fitted[0] < 0.8 * truth[0], (fitted[0], truth[0])
    assert fitted[1] < 0.3 * truth[1], (fitted[1], truth[1])


def test_smearing_the_credit_over_a_whole_episode_loses_the_small_gains():
    """The limit the probe does not lift. Credit is still not assigned: a step's
    label is the whole episode's reward, so with four steps per episode the two
    smallest gains stop being separated, while the largest survives.

    This is the measurement that says a MiniWoB rerun needs more than probes.
    """
    assigned = train_head_on_probed_episodes(
        metadata(), corpus(episodes=1600, steps=1), seed=0)[0]
    smeared = train_head_on_probed_episodes(
        metadata(), corpus(episodes=400, steps=4), seed=0)[0]

    one = [fitted_gain(assigned, index) for index in range(len(STATES))]
    four = [fitted_gain(smeared, index) for index in range(len(STATES))]

    # The same number of decisions either way, so this is about credit and not
    # about sample size.
    assert one == sorted(one, reverse=True)
    assert four != sorted(four, reverse=True), four
    # The largest gain survives the smearing; the ordering below it does not.
    assert four[0] > 1.0, four
    assert abs(four[2] - four[3]) < 0.01, four


def test_without_probes_the_same_corpus_does_not_identify_the_gain():
    """The other half of the claim, and the reason the probe was added. Trained
    on the same episodes through the path that has no counterfactual, the gain
    collapses: the four states are no longer told apart.
    """
    from slime_hdm.head_trainer import HeadRLConfig, train_head_on_smeared_returns

    rows = corpus(probe=False)
    head, _ = train_head_on_smeared_returns(metadata(), rows, HeadRLConfig(), seed=0)

    fitted = [fitted_gain(head, index) for index in range(len(STATES))]
    probed, _ = train_head_on_probed_episodes(metadata(), corpus(), seed=0)
    probed_fitted = [fitted_gain(probed, index) for index in range(len(STATES))]

    unprobed_spread = max(fitted) - min(fitted)
    probed_spread = max(probed_fitted) - min(probed_fitted)
    assert probed_spread > 2 * unprobed_spread, {
        "unprobed": [round(f, 4) for f in fitted],
        "probed": [round(f, 4) for f in probed_fitted],
    }


def rewarded_corpus(episodes=400, steps=4, seed=0):
    """The four-step corpus, with the environment paying at every step.

    `Environment.step` returns a reward per step and the rollout used to sum it
    and discard the parts. This is the corpus that keeping them makes possible,
    and it is the same episodes as `corpus` -- same draws, same probes, same
    terminal reward -- with the per-step outcome also written down.
    """
    return corpus(episodes=episodes, steps=steps, seed=seed, per_step_reward=True)


def test_a_per_step_reward_is_used_when_the_venue_pays_one():
    """`credit="auto"` should read what the environment gave rather than assume
    it gave nothing. The recorded fact is new: before the per-step reward was
    kept, a venue with intermediate signal was indistinguishable from one
    without."""
    from slime_hdm.probed_trainer import credit_is_per_step

    assert not credit_is_per_step(corpus(episodes=4))
    assert credit_is_per_step(rewarded_corpus(episodes=40))

    _, smeared = train_head_on_probed_episodes(metadata(), corpus(episodes=40), seed=0)
    _, stepwise = train_head_on_probed_episodes(
        metadata(), rewarded_corpus(episodes=40), seed=0)
    assert smeared["credit"] == "episode"
    assert stepwise["credit"] == "step"

    # And the choice can be forced either way on the same corpus.
    _, forced = train_head_on_probed_episodes(
        metadata(), rewarded_corpus(episodes=40), seed=0, credit="episode")
    assert forced["credit"] == "episode"
    with pytest.raises(ValueError, match="credit is auto"):
        flatten_probed(rewarded_corpus(episodes=4), credit="per-step")


def test_a_per_step_reward_recovers_part_of_what_smearing_loses():
    """How much keeping it is worth, and how much is still missing.

    The same episodes, the same probes, the same decision count: only the label
    differs. Per step the four gains fit 1.494 / 0.105 / 0.000 / 0.004 against a
    truth of 2.197 / 0.693 / 0.134 / 0.000; smeared, the second state falls to
    0.058. So per-step credit recovers about **1.8 times** as much of the second
    gain and still leaves the third below resolution -- 0.134 nats fits at 0.000,
    and the bottom two states remain within 0.004 of each other, which is not an
    ordering.

    These numbers moved when `fit_normalizer` stopped dividing by `1e-6` on
    features that never vary. Twelve of this fixture's sixteen columns are
    constant, and for a constant column `m - mean` is not zero in float32 but a
    rounding residue of about 4.2e-7; the old divisor turned that into an input of
    0.417, the same scale as the four columns carrying actual signal. So the
    earlier figures, and the 2x threshold tuned to them, were partly fitted to
    amplified rounding noise. The ratio is 2.21 under the old divisor and 1.81
    under unit scale, and the claim this test makes is the recovery, not the 2.

    What this says about the rerun: keeping the per-step reward is worth doing and
    is free, and it is not sufficient on its own.
    """
    rows = rewarded_corpus(episodes=400, steps=4)
    stepwise = train_head_on_probed_episodes(metadata(), rows, seed=0)[0]
    smeared = train_head_on_probed_episodes(metadata(), rows, seed=0,
                                            credit="episode")[0]

    per_step = [fitted_gain(stepwise, index) for index in range(len(STATES))]
    per_episode = [fitted_gain(smeared, index) for index in range(len(STATES))]

    # The two states where thinking clearly matters are ordered and separated.
    assert per_step[0] > 1.0 > per_step[1] > 10 * per_step[2], per_step
    # And recovered substantially better than by smearing, which is the point of
    # keeping it. The margin is 1.81x; the bound is set below it rather than at it
    # so the test reports a lost recovery and not a drifting third decimal.
    assert per_step[1] > 1.5 * per_episode[1], (per_step, per_episode)
    assert per_step[0] > per_episode[0], (per_step, per_episode)
    # The limit that remains, asserted so it is not mistaken for solved: the
    # third state's 0.134 nats is not resolved from the fourth's zero.
    assert abs(per_step[2] - per_step[3]) < 0.01, per_step
