"""What a hybrid-thinking rollout needs from an environment, and nothing more.

The rollout is the same whatever the task: read a candidate set, decide whether
to think, act, repeat until the environment says the episode is over and hands
back a reward. Keeping that behind a protocol is what stops the loop becoming an
ALFWorld loop, since ALFWorld is one environment and the method is not about it.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class StepRecord:
    """One decision, and everything a head needs to learn from it.

    `hidden` and `probabilities` come from the fast mode, which runs at every
    step whether or not the step escalates. That is what makes the decision free
    and it is also why both are always available to train on.
    """

    step: int
    hidden: np.ndarray
    probabilities: np.ndarray
    candidates: int
    escalated: bool
    generated_tokens: int
    action: str
    gain: float | None = None
    confidence: float | None = None
    # **What the mode not taken would have done at this state**, when the step
    # was probed. `None` when it was not.
    #
    # Why this field exists. The decision head predicts a four-way outcome, and
    # the gain it is thresholded by is a function of the two asymmetric classes --
    # fast right and slow wrong, and its converse. An episodic step observes
    # neither: it records whether the mode it took was right and cannot say what
    # the other would have done, so those two classes are never supervised
    # against each other and the gain is not identifiable from a corpus of these.
    # Measured on the recorded MiniWoB episodes the fitted `o[1]` is 0.0002 and
    # the predicted gain takes three values
    # (`experiments/browser-gate/REWARD-OFFLINE.md`).
    #
    # A probe runs the other mode at the same state **without changing which
    # action is taken**, so the trajectory distribution is untouched and the
    # episodes stay valid for the off-policy update. On an escalated step the
    # fast distribution has already been computed for the decision and was simply
    # being discarded, so that half costs nothing.
    probe_probabilities: np.ndarray | None = None
    probe_action: str | None = None
    # Which mode the probe ran: "fast" on an escalated step, "slow" otherwise.
    probe_mode: str | None = None
    # **Not part of the policy's cost.** A probe's tokens were spent to learn
    # something, not to answer, so they are kept out of `generated_tokens` and
    # reported separately -- otherwise collecting the labels inflates the very
    # cost column the method is judged on.
    probe_tokens: int = 0
    # **And not part of the policy's wall time**, for exactly the reason the
    # tokens are not. `seconds` is the other cost column in Table 1, and a step
    # the policy answered with the fast mode must not be charged for a slow pass
    # run only to produce a label. So `seconds` excludes this and this records it,
    # which also answers what the labels cost -- a thing worth knowing and not
    # recoverable after the fact. exps caught that the first version of the probe
    # separated the tokens and left the seconds inflated.
    probe_seconds: float = 0.0
    # **Why the probe's text was not an action**, where the slow path writes one.
    # A probe that wrote something unexecutable refused; it did not choose
    # differently, and folding the two together is the substitution `act_freely`
    # declines to make for the taken action.
    probe_problem: str | None = None
    # **Was the probe's action one the fast mode was offered?** This is the
    # reach/reasoning split, and it cannot be recovered afterwards: the row stores
    # `candidates` as a count, not the offered list, so a disagreement says
    # nothing about whether the slow mode chose a different element of the same
    # set -- reasoning -- or one that was never on offer -- reach. `parse_action`
    # checks against the whole page's bids while the fast argmax sees only the
    # offered subset under its limit, so a reach disagreement is not hypothetical:
    # it is what Section 4.5's action-interface story is about. exps asked for
    # this before the collection because one boolean closes it permanently and
    # nothing closes it later.
    probe_action_offered: bool | None = None
    # The same question for the action actually taken, which on a written step is
    # also not drawn from the offered set.
    action_offered: bool | None = None
    # **Why the taken text was not an action**, on a written step. The mirror of
    # `probe_problem`, and needed for the same reason: a step whose own text did
    # not parse has no comparable choice either, and without this it was compared
    # raw against a canonical probe and counted as a disagreement.
    action_problem: str | None = None
    # **What the environment paid at this step**, which is `0.0` in a venue that
    # rewards only at the end and need not be. `Environment.step` has always
    # returned a reward per step and the rollout summed it into a total and threw
    # the parts away, so a venue with any intermediate signal looked exactly like
    # one without. Credit assignment is the limit on learning the gain from
    # episodes (`experiments/browser-gate/REWARD-OFFLINE.md`), so a signal that is
    # already being handed to us is not one to discard.
    reward: float = 0.0
    # **Wall time for this step**, decision and environment move together.
    # Added 2026-10-06 because no run had ever recorded it: every time figure in
    # the gate README and in the slow-only cost table was *derived* from 28831's
    # 2.29s fast / 10.95s slow, measured on a different run over a different task
    # set, and the paper's metric is score net of cost. Time cannot be recovered
    # after the fact, so it has to be written while the step happens.
    seconds: float = 0.0


    def row(self, **extra):
        """This step as a corpus row. **The one definition of what gets written.**

        Both writers call it -- `loop.built` for the jsonl corpus and
        `rollout.samples_from` for the slime metadata -- because they drifted
        apart twice. The first time, `probe_prob` reached the rollout and no
        caller could switch it on. The second time the probe fired, cost GPU and
        was discarded, because `built` listed eight keys by hand and none of them
        were the probe's: a run at `probe_prob` 1.0 wrote a corpus key-for-key
        identical to an unprobed one while its own signature said it had probed.
        A field added here reaches both, and a field only one of them lists is
        now impossible.
        """
        return {
            "step": self.step, "hidden": self.hidden,
            "probabilities": self.probabilities, "candidates": self.candidates,
            "escalated": self.escalated, "generated_tokens": self.generated_tokens,
            "seconds": float(self.seconds), "reward": float(self.reward),
            "action": self.action, "action_offered": self.action_offered,
            "action_problem": self.action_problem,
            "gain": self.gain, "confidence": self.confidence,
            "probe_mode": self.probe_mode, "probe_action": self.probe_action,
            "probe_probabilities": self.probe_probabilities,
            "probe_tokens": self.probe_tokens,
            "probe_seconds": float(self.probe_seconds),
            "probe_problem": self.probe_problem,
            "probe_action_offered": self.probe_action_offered,
            **extra,
        }


@dataclass
class EpisodeResult:
    """One trajectory. The reward is terminal, which is the whole difficulty."""

    episode: int
    reward: float
    steps: list[StepRecord] = field(default_factory=list)
    finished: bool = False
    # **Why the episode stopped, in the gate's vocabulary.** The same word for the
    # same event on both paths: `gate.run_episode` has always written `ended`, and
    # this path recorded an unexecutable action in `metadata` instead -- identical
    # behaviour under a different field. exps found it while auditing the two
    # loops and said it was the one thing in the table it had to remember, because
    # a single count of bad actions across both runs would silently miss this
    # path's. One field, so the count is one read.
    ended: str | None = None
    # The episode's own wall clock, not the sum of its steps: `reset` builds a
    # browser and that cost belongs to the episode. The difference is the part a
    # per-step sum would silently drop.
    seconds: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def generated_tokens(self):
        return sum(step.generated_tokens for step in self.steps)

    @property
    def escalations(self):
        return sum(step.escalated for step in self.steps)

    @property
    def rewarded_steps(self):
        """Steps the environment paid something at. Zero in a terminal-only venue.

        What this answers is whether credit can be assigned at all here, which is
        a property of the venue and was previously unanswerable from a record.

        **It is not one per episode.** A failed episode's terminal reward is
        exactly `0.0`, so this is 1 on a success and 0 on a failure. The sharp
        reading of a terminal-only venue is therefore that it is never above 1,
        and is 1 exactly when the episode succeeded; two or more falsifies it.
        """
        return sum(step.reward != 0.0 for step in self.steps)

    @property
    def probe_tokens(self):
        """What the probes cost, kept apart from `generated_tokens` on purpose."""
        return sum(step.probe_tokens for step in self.steps)

    @property
    def probe_seconds(self):
        """The same, in the other currency, and kept out of `seconds` likewise."""
        return sum(step.probe_seconds for step in self.steps)

    @property
    def probed_steps(self):
        return sum(step.probe_mode is not None for step in self.steps)

    @property
    def modes_disagreed(self):
        """Probed steps where thinking would have chosen a different action.

        The quantity the probe exists to supply. A step where the two modes pick
        the same action cannot be one where thinking mattered, whatever the
        episode's outcome, and an episode whose every probed step agreed would
        have ended the same way under either policy.

        **Probes that refused are not counted here.** Both sides of the
        comparison are canonical actions by the time they are stored, and a probe
        whose text was not an action at all has no choice to compare -- see
        `probe_refusals`.
        """
        return sum(self._verdict(step) == "disagreed" for step in self.steps)

    @property
    def modes_agreed(self):
        """Probed steps where both modes chose the same action."""
        return sum(self._verdict(step) == "agreed" for step in self.steps)

    @property
    def probe_refusals(self):
        """Probes whose text was not an executable action.

        Kept apart from a disagreement because it is not one: the slow mode did
        not pick something else, it wrote something that cannot be run. Counting
        it as a disagreement inflates the one number the probe exists to produce.
        """
        return sum(self._verdict(step) == "probe_refused" for step in self.steps)

    @property
    def taken_unparsable(self):
        """Probed steps whose own taken text was not an action.

        The mirror of `probe_refusals`. Such a step has nothing comparable on
        either side, so it belongs in neither the agreed nor the disagreed count.
        """
        return sum(self._verdict(step) == "taken_unparsable" for step in self.steps)

    def row(self, **extra):
        """This episode as a corpus row, aggregates included.

        The aggregates are written rather than left to be recomputed because
        `probed_steps` and the four verdicts are the whole result of a probed
        collection, and a reader that has to rederive them from step rows is a
        reader who can get them wrong. They are also checkable against the steps,
        which is what gate 4 does.
        """
        return {
            "terminal_reward": float(self.reward),
            "finished": bool(self.finished),
            "ended": self.ended,
            "seconds": float(self.seconds),
            "escalations": self.escalations,
            "generated_tokens": self.generated_tokens,
            "rewarded_steps": self.rewarded_steps,
            "probe_tokens": self.probe_tokens,
            "probe_seconds": float(self.probe_seconds),
            "probed_steps": self.probed_steps,
            "modes_agreed": self.modes_agreed,
            "modes_disagreed": self.modes_disagreed,
            "probe_refusals": self.probe_refusals,
            "taken_unparsable": self.taken_unparsable,
            # Why it ended, where the gate recorded a reason: `ended` is one
            # word and this is the text behind it (`unexecutable_action` and the
            # line that did not parse). Neither writer carried it before, so an
            # episode that died on an unexecutable action said only that it had.
            "metadata": dict(self.metadata),
            **extra,
        }

    @staticmethod
    def _verdict(step):
        """One of four, for a probed step: the single place the partition is
        decided, so the four counts cannot overlap or leave a step out.

        Order matters. A step with both texts unparsable is a probe refusal
        first, because that is the side the probe was run to learn about.
        """
        if step.probe_mode is None:
            return None
        if step.probe_problem is not None:
            return "probe_refused"
        if step.action_problem is not None:
            return "taken_unparsable"
        return "disagreed" if step.probe_action != step.action else "agreed"


@runtime_checkable
class Environment(Protocol):
    """A text environment that publishes its admissible actions.

    The fast mode answers by scoring an enumerated candidate set, so an
    environment fits this method when it can say what the available actions are.
    Where it cannot, the fast mode becomes a short generation instead and this
    protocol is the wrong shape.
    """

    def reset(self, episode: int) -> tuple[str, list[str]]:
        """Start episode `episode`. Returns the prompt context and the actions."""

    def step(self, action: str) -> tuple[str, list[str], bool, float]:
        """Take `action`. Returns the context, the next actions, done, and the reward.

        The reward is meaningful only when done is true. An environment that
        rewards intermediate steps can return it as it goes, and the rollout will
        sum it, but nothing here assumes there is any to collect before the end.
        """

    @property
    def episodes(self) -> int:
        """How many episodes this environment can serve."""
