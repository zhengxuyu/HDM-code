"""Train the decision head from episodes whose steps were probed.

`head_trainer` learns from episodes that record, per step, whether the mode it
took was right. That is not enough to learn the gain. The gain is
`log P(slow right) - log P(fast right)`, a function of the two asymmetric
outcomes -- fast right and slow wrong, and its converse -- and a step that ran
one mode observes neither, so nothing separates them and the head is free to put
`o[1]` at 0.0002 and let the gain take three values
(`experiments/browser-gate/REWARD-OFFLINE.md`).

A probe (`StepRecord.probe_probabilities`) records what the mode *not* taken
would have chosen at the same state, without changing what was taken. That buys
one exact constraint and one partial label, and the exact one is the important
half:

**If both modes choose the same action at a state, they cannot differ in
correctness there.** Whatever the environment then does, it does identically, so
`o[1] = o[2] = 0` at that state and the step is `both right` or `both wrong`
according to the episode. This is a fact about the actions, not an inference from
the reward, and it holds whether or not credit can be assigned.

**If they choose differently, the step is one where thinking matters,** and the
episode's outcome speaks for the mode that was taken and is silent about the
other. That is a partial label -- two of the four classes are consistent with it
-- and a partial label has a proper likelihood: the probability of the observed
event, which is the sum of the classes it allows.

So an agreeing step contributes a full cross-entropy and a disagreeing step a
marginal one. The two together are what make `o[1]` and `o[2]` separable: the
head can no longer spend asymmetric mass at states where the modes provably act
alike, so what asymmetric mass it has must go where they do not.

**What this still does not solve.** Credit is not assigned. "The mode taken was
right at this step" is read off the episode's terminal reward and smeared over
every step, exactly as `train_head_on_smeared_returns` does, because one number
arrives at the end. The probe fixes identifiability, not credit. An episode that
failed still does not say which step failed it.
"""

from __future__ import annotations

import numpy as np
import torch

from litjev.core.block import DecisionHead, escalation_gain_batch
from litjev.core.block_utils import build_features

# The four outcome classes, in the order the head emits them.
BOTH_RIGHT, FAST_ONLY, SLOW_ONLY, NEITHER = 0, 1, 2, 3


def allowed_classes(agreed, succeeded, escalated):
    """Which of the four outcomes are consistent with one probed step.

    `agreed` says the two modes chose the same action, `succeeded` is the
    episode's terminal reward read as a per-step label, and `escalated` says
    which mode acted.

    Agreement is the exact case: the modes act alike, so they are right together
    or wrong together and exactly one class is allowed. Disagreement leaves two,
    because the episode speaks only for the mode that acted.
    """
    if agreed:
        return (BOTH_RIGHT,) if succeeded else (NEITHER,)
    if escalated:
        # The slow mode acted. Its correctness is the episode's; the fast mode's
        # is unobserved, so both classes that agree with the slow mode are open.
        return (BOTH_RIGHT, SLOW_ONLY) if succeeded else (FAST_ONLY, NEITHER)
    return (BOTH_RIGHT, FAST_ONLY) if succeeded else (SLOW_ONLY, NEITHER)


def credit_is_per_step(episodes):
    """Did the environment pay at any step rather than only at the end?

    `Environment.step` returns a reward every step and the rollout used to sum it
    and discard the parts, so this was unanswerable from a record. Where the
    answer is yes, a step's label is its own reward and the ordering of small
    gains survives; where it is no, the label is the episode's, smeared, and the
    measurement in `tests/test_probed_trainer.py` is that the two smallest gains
    then stop separating. The probe does not lift that limit -- it says what the
    other mode would have *done*, not whether it was *right*.
    """
    return any(float(step.get("reward", 0.0)) != 0.0
               for episode in episodes for step in episode["head_features"])


def flatten_probed(episodes, question_type="choice", credit="auto"):
    """One row per **probed** step: features, and the classes its label allows.

    Unprobed steps are dropped rather than given a weaker label. A step with no
    probe is the case this module exists to avoid relying on: its label allows
    two classes that straddle the asymmetric pair, which is what leaves the gain
    unidentified. Keeping them would put the pathology back in at a lower weight.

    `credit` is `"auto"`, `"step"` or `"episode"`. Auto uses the per-step reward
    when the venue pays one and the episode's otherwise, which is what a caller
    almost always wants: using what the environment gives rather than assuming it
    gives nothing.
    """
    if credit not in ("auto", "step", "episode"):
        raise ValueError(f"credit is auto, step or episode, not {credit!r}")
    per_step = credit == "step" or (credit == "auto" and credit_is_per_step(episodes))
    features, mask, agreement = [], [], []
    for episode in episodes:
        terminal = float(episode["terminal_reward"]) > 0
        for step in episode["head_features"]:
            succeeded = (float(step.get("reward", 0.0)) > 0) if per_step else terminal
            if step.get("probe_mode") is None:
                continue
            allowed = allowed_classes(
                step["probe_action"] == step.get("action"), succeeded, bool(step["escalated"]))
            row = np.zeros(4, dtype=np.float32)
            row[list(allowed)] = 1.0
            features.append(build_features(
                np.asarray(step["hidden"]),
                np.asarray(step["probabilities"])[: step["candidates"]], question_type))
            mask.append(row)
            agreement.append(step["probe_action"] == step.get("action"))
    if not features:
        raise ValueError(
            "no probed steps to learn from. Collect with probe_prob > 0, or use "
            "head_trainer for a corpus without probes -- and see this module's "
            "docstring for what that costs")
    return (np.stack(features).astype(np.float32), np.stack(mask),
            np.array(agreement, dtype=bool))


def marginal_cross_entropy(logits, allowed):
    """-log of the probability the head assigns to the observed event.

    `allowed` is a 0/1 mask over the four classes. Where one class is allowed
    this is the ordinary cross-entropy; where two are, it is the negative log of
    their sum, which is the likelihood of what was actually seen rather than of
    an invented single label. Choosing one of the two instead -- say, calling a
    successful escalation `slow only` -- would assert the fast mode was wrong,
    which is precisely the observation the step does not contain.
    """
    log_probabilities = torch.log_softmax(logits, dim=-1)
    allowed = allowed.to(log_probabilities.dtype)
    # logsumexp over the allowed classes only, in a numerically safe way.
    masked = log_probabilities.masked_fill(allowed == 0, float("-inf"))
    return -torch.logsumexp(masked, dim=-1).mean()


def train_head_on_probed_episodes(metadata, episodes, *, epochs=60, learning_rate=1e-3,
                                  weight_decay=5e-2, dropout=0.3, batch_size=256,
                                  question_type="choice", seed=0, credit="auto"):
    """Fit a head to probed episodes. Returns (head, report).

    The loss is the marginal cross-entropy above and nothing else: a proper
    scoring rule on what was observed. No price enters, because nothing here is a
    policy term -- the price is applied at serving time, where `escalation_gain`
    is compared against it, and one head serves every price for the same reason
    it does under cross-entropy.
    """
    features, allowed, agreement = flatten_probed(episodes, question_type, credit)

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    head = DecisionHead(metadata, dropout=dropout)
    matrix = torch.as_tensor(features)
    head.fit_normalizer(matrix)
    mask = torch.as_tensor(allowed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)

    history = []
    for _ in range(epochs):
        head.train()
        order = torch.randperm(len(matrix), generator=generator)
        total, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = marginal_cross_entropy(head(matrix[index]), mask[index])
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(index)
            seen += len(index)
        history.append(total / max(seen, 1))

    head.eval()
    with torch.no_grad():
        outcome = torch.softmax(head(matrix), dim=-1)
        gain = escalation_gain_batch(outcome).numpy()
        outcome = outcome.numpy()
    return head, {
        "objective": "probed_marginal_cross_entropy",
        "decisions": int(len(features)),
        "episodes": int(len(episodes)),
        "agreeing_steps": float(agreement.mean()),
        "credit": "step" if (credit == "step"
                             or (credit == "auto" and credit_is_per_step(episodes)))
                  else "episode",
        "final_loss": history[-1] if history else None,
        # The diagnostic this module exists for: asymmetric mass should sit where
        # the modes disagreed and nowhere else.
        "asymmetric_mass_where_they_agreed": float(
            outcome[agreement][:, [FAST_ONLY, SLOW_ONLY]].sum(1).mean())
        if agreement.any() else float("nan"),
        "asymmetric_mass_where_they_differed": float(
            outcome[~agreement][:, [FAST_ONLY, SLOW_ONLY]].sum(1).mean())
        if (~agreement).any() else float("nan"),
        "gain_mean": float(gain.mean()),
        "gain_spread": float(gain.std()),
        "gain_quantiles": {str(q): float(np.quantile(gain, q))
                           for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "history": history,
    }


__all__ = ["allowed_classes", "credit_is_per_step", "flatten_probed",
           "marginal_cross_entropy", "train_head_on_probed_episodes"]
