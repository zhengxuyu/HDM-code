"""Update the decision head from episodes whose reward arrives only at the end.

The supervised setting records both modes' outcomes for every question, so the
expected routed reward is exact and the gradient needs no correction. Here one
trajectory is observed, under whatever policy collected it, and the reward is a
single number for the whole episode. Two things follow, and both are reported
rather than assumed away.

**The data is off-policy.** The head being trained is not the policy that acted,
so every step carries an importance ratio. Full-trajectory ratios multiply over
thirty steps and collapse; this uses the per-decision ratio inside a clipped
surrogate, which is biased in a known direction and stays usable. How usable is
the effective sample size, which `train_head_on_episodes` returns and which
should be read before the reward is.

**Credit is not assigned.** A reward of one says the episode succeeded, not which
escalation earned it, so every step of an episode shares its advantage. With
roughly thirty decisions per episode this is the weakest part of the setup, and no
amount of clipping improves it.

There is no calibration anchor here. Cross-entropy on the four-way outcome is what
kept the predicted distribution honest in the supervised setting, and it has no
counterpart when the outcome of the mode not taken is never seen.
"""

from dataclasses import dataclass, replace

import numpy as np
import torch

from litjev.core.block import DecisionHead, escalation_gain_batch
from litjev.core.block_utils import build_features


@dataclass(frozen=True)
class HeadRLConfig:
    """`lambda_` is the threshold the policy smooths around; the price is drawn.

    These **were** different numbers, and that was the defect: the return was
    priced once at a fixed `cost_weight` while the policy term used a price drawn
    per decision, so the objective asked about one price and valued at another.
    There is one price now, from `lambda_range` when it is set and `lambda_`
    otherwise, and it enters both the policy term and the reward -- see
    `priced_value`.

    `cost_weight` is kept only so an older record can be read back; nothing uses
    it to price a return any more.
    """

    lambda_: float = 0.0
    lambda_range: tuple[float, float] | None = None
    temperature: float = 0.05
    cost_weight: float = 0.0
    clip: float = 0.2
    baseline: str = "mean"
    behaviour_prob: float = 0.5
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 5e-2
    dropout: float = 0.3


def outside_band(ratio, clip):
    """Which ratios left the trust region.

    **Not the same as the clip binding**, which is the confusion that produced
    two wrong claims in one day: a ratio outside the band contributes no gradient
    only on the favourable side, and on the unfavourable side `min` still takes
    `ratio * advantage`. This counts the band, `dropped_decisions` counts the
    vanishing gradient, and they are different numbers.
    """
    return (ratio < 1 - clip) | (ratio > 1 + clip)


def dropped_decisions(ratio, advantages, clip):
    """Which decisions the clipped surrogate contributes **no gradient** for.

    With the loss `-min(rA, clip(r)A)`, `min` takes the bounded branch on the
    favourable side -- the ratio high with a positive advantage, or low with a
    negative one -- and that branch is constant in the parameters, so those
    decisions do not move the head at all. On the unfavourable side `min` takes
    `rA` and the ratio weights the update.

    **The band test is part of the condition, not an afterthought.** Inside the
    band `bounded` *equals* `rA`, so `bounded <= rA` is true there too, and the
    first version of this counter omitted `outside` and called a full-gradient
    decision dropped -- at epoch 0, with every ratio near 1, it read about 1.0
    when nothing was dropped. code-reviewer caught that, and then caught that the
    test for it recomputed this formula inline rather than calling it: a test of
    a copy guards the copy. One definition, used by the trainer and compared
    against autograd by `tests/test_browser_pmpo.py`.
    """
    bounded = ratio.clamp(1 - clip, 1 + clip) * advantages
    return outside_band(ratio, clip) & (bounded <= ratio * advantages)


def effective_sample_size(ratios):
    """(sum w)^2 / sum w^2: how many independent samples the weights are worth.

    Equal weights give back the sample count. One weight dominating gives back
    close to one, whatever the sample count says.
    """
    weights = np.asarray(ratios, dtype=np.float64)
    total = weights.sum()
    if total <= 0:
        return 0.0
    return float(total * total / np.square(weights).sum())


def flatten_episodes(episodes, config, question_type="choice"):
    """One row per decision: features, the action taken, its probability, the return.

    The return is the episode's, minus the price of the escalations it spent, and
    every step of the episode carries it. That is the credit assignment problem
    stated rather than solved.
    """
    features, actions, behaviour, rewards, spending, episode_index = [], [], [], [], [], []
    for index, episode in enumerate(episodes):
        steps = episode["head_features"]
        if not steps:
            continue
        reward = float(episode["terminal_reward"])
        spent = sum(bool(step["escalated"]) for step in steps)
        for step in steps:
            features.append(build_features(
                np.asarray(step["hidden"]), np.asarray(step["probabilities"])[: step["candidates"]],
                question_type))
            actions.append(bool(step["escalated"]))
            behaviour.append(float(step.get("behaviour_prob", config.behaviour_prob)))
            # The reward and what the episode spent, **unpriced**. Pricing happens
            # per drawn price in `priced_value`, because which price a decision is
            # being asked about is what decides whether its escalations were worth
            # it. The first version priced here, once, at a separate fixed
            # `cost_weight`.
            rewards.append(reward)
            spending.append(spent)
            episode_index.append(index)
    if not features:
        raise ValueError("no decisions to learn from")
    return (np.stack(features).astype(np.float32), np.array(actions),
            np.array(behaviour, dtype=np.float64), np.array(rewards, dtype=np.float64),
            np.array(spending, dtype=np.float64), np.array(episode_index))


def priced_value(rewards, spent, prices):
    """What an episode earned net of what its thinking cost, **at this price**.

    This is the correction the user made on 2026-10-05, and it is a correction to
    the objective rather than to a default. The reward a decision is judged by
    should not be "did the episode succeed"; it should be whether the slow mode's
    gain was worth what it cost -- and that depends on the price, so the price has
    to be in the reward.

    It was not. `flatten_episodes` priced the return once, at a fixed
    `cost_weight`, while `draw_prices` redrew a price every epoch for the policy
    term alone. So the objective asked "at price lambda, would you escalate?" and
    answered with an advantage computed at a different, fixed price: escalating on
    a high-advantage episode was rewarded whatever price it was being asked about.
    `HeadRLConfig`'s own docstring said the two numbers were different, and the
    paper says the method has **one** price, "drawn per decision". The paper was
    right and the implementation was not.

    With the price in the reward, a high price makes episodes that escalated
    throughout worth less, so one head is pushed to decline at high prices and
    escalate at low ones -- which is what makes it serve every price, and what
    "price-marginalised" means.

    The discount is multiplicative, not a subtraction: an episode is worth its
    return scaled by what the waiting cost, `R * exp(-lambda * escalations)`,
    which is the single-question reward `u * exp(-lambda * T)` with the sum in the
    exponent. `spent` counts escalations rather than tokens because
    `T_slow - T_fast` is about constant and is absorbed into lambda, exactly as it
    is for one question.

    Two things this assumes, both checked against the 162 recorded MiniWoB
    episodes rather than asserted. The return must be non-negative, or a discount
    makes a failure that waited longer look better than one that did not: every
    recorded episode returns exactly 0.0 or 1.0 (per-round means 0.543 and 0.568),
    so it holds here, and the guard below is there because an environment with a
    penalty for failing would break it silently. And a negative price amplifies
    instead of discounting, by `exp(|lambda| * escalations)`, so the dynamic range
    grows with the episode: the recorded prices are 0, 0.1 and 0.2 over at most 13
    escalations per episode (mean 1.67, longest episode 25 steps), where the
    steepest factor is `exp(-2.6)`, about 0.07.
    """
    if np.any(np.asarray(rewards) < 0):
        raise ValueError(
            "a discounted reward needs a non-negative return: scaling a negative "
            "one by exp(-lambda * escalations) makes a worse episode score higher")
    return rewards * np.exp(-prices * spent)


def draw_prices(count, config, generator):
    """A price per decision, or the one fixed price when no range is given.

    Training at a single price constrains the gain only near that price: the
    policy term cares which side of it each gain falls on and not what the gain
    is, so everything away from the threshold is free to drift. Drawing the price
    instead makes the objective an average over the range, and the head has to
    order the gains across all of it. The range must bracket the prices a
    deployment will use, since a gain above every price drawn escalates on every
    draw and the objective cannot tell those apart.
    """
    if config.lambda_range is None:
        return torch.full((count,), float(config.lambda_))
    low, high = config.lambda_range
    if not high > low:
        raise ValueError(f"lambda_range must be an interval, got {config.lambda_range}")
    return torch.rand(count, generator=generator) * (high - low) + low


def _escalation_logprob(logits, taken, prices, config):
    """log pi(taken | x, lambda) for the soft threshold policy on the head's gain.

    The gain is `escalation_gain_batch`, the same log ratio the rollout escalated
    on and `cdpo_loss` thresholds. It was the difference `o[2] - o[1]` here for one
    commit after the reward became a discount, which made the importance ratio
    divide two different functions of the head: the numerator scored a rule the
    denominator had not followed.
    """
    outcome = torch.softmax(logits, dim=-1)
    gain = escalation_gain_batch(outcome)
    probability = torch.sigmoid((gain - prices) / config.temperature)
    probability = probability.clamp(1e-6, 1 - 1e-6)
    return torch.where(taken, probability, 1 - probability).log()


def train_head_on_episodes(metadata, episodes, config=None, seed=0):
    """Fit a head to the episodes. Returns (head, report).

    The report carries the effective sample size, the share of ratios the clip
    bit, and the advantage spread. A reward that improved while the effective
    sample size collapsed has not been shown to improve.
    """
    config = config or HeadRLConfig()
    features, actions, behaviour, rewards, spent, episode_index = flatten_episodes(
        episodes, config)

    def advantages_at(prices):
        """The advantage of every decision, at the price each was drawn for.

        The baseline is taken at the same prices, so it is the mean value of this
        corpus under that pricing rather than under some other one.
        """
        values = priced_value(rewards, spent, prices)
        baseline = values.mean() if config.baseline == "mean" else 0.0
        return torch.as_tensor(values - baseline, dtype=torch.float32)

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    head = DecisionHead(metadata, dropout=config.dropout)
    matrix = torch.as_tensor(features)
    head.fit_normalizer(matrix)
    taken = torch.as_tensor(actions)
    log_mu = torch.as_tensor(np.log(np.clip(
        np.where(actions, behaviour, 1 - behaviour), 1e-6, 1.0)))
    # The price the behaviour policy acted under is not the price the head is
    # asked about, so a price is drawn per decision and redrawn every epoch.
    prices = draw_prices(len(matrix), config, generator)
    advantages = advantages_at(prices.numpy())
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

    history = []
    for _ in range(config.epochs):
        head.train()
        order = torch.randperm(len(matrix), generator=generator)
        prices = draw_prices(len(matrix), config, generator)
        # Redrawn with the price, not held fixed across draws: the advantage is a
        # function of the price being asked about.
        advantages = advantages_at(prices.numpy())
        total, clipped, stalled, seen = 0.0, 0, 0, 0
        for start in range(0, len(order), config.batch_size):
            index = order[start : start + config.batch_size]
            optimizer.zero_grad()
            log_pi = _escalation_logprob(head(matrix[index]), taken[index],
                                         prices[index], config)
            ratio = torch.exp(log_pi - log_mu[index])
            a = advantages[index]
            unclipped = ratio * a
            bounded = ratio.clamp(1 - config.clip, 1 + config.clip) * a
            loss = -torch.min(unclipped, bounded).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
            clipped += int(outside_band(ratio, config.clip).sum())
            # **Where the gradient actually vanishes**, which `clipped_fraction`
            # does not say. `min` takes the bounded branch on the *favourable*
            # side -- the ratio high with a positive advantage, or low with a
            # negative one -- and that branch is constant in the parameters, so
            # those decisions contribute **nothing** to the update. On the
            # unfavourable side `min` takes `ratio * a` and the ratio does weight
            # it. I had described a high `clipped_fraction` as "the clip constant
            # weighting those updates"; code-reviewer showed it weights nothing,
            # and that half the out-of-band decisions are still ratio-weighted.
            # How much of the batch the surrogate silently dropped, which
            # `clipped_fraction` cannot say. `dropped_decisions` holds the rule
            # and its reasoning, and its test compares **it** against autograd.
            stalled += int(dropped_decisions(ratio, a, config.clip).sum())
            seen += len(index)
        history.append({"surrogate_loss": total / max(seen, 1),
                        "clipped_fraction": clipped / max(seen, 1),
                        "zero_gradient_fraction": stalled / max(seen, 1)})

    head.eval()
    with torch.no_grad():
        final_ratio = torch.exp(
            _escalation_logprob(head(matrix), taken, prices, config) - log_mu).numpy()
        outcome = torch.softmax(head(matrix), dim=-1)
        gain = escalation_gain_batch(outcome).numpy()
        outcome = outcome.numpy()
    report = {
        "decisions": int(len(features)),
        "episodes": int(len(episodes)),
        "decisions_per_episode": float(len(features) / max(len(episodes), 1)),
        "mean_reward": float(rewards.mean()),
        "mean_escalations_per_episode": float(spent.mean()),
        # At the last prices drawn, which is what the final epoch optimised.
        "mean_value": float(priced_value(rewards, spent, prices.numpy()).mean()),
        "value_spread": float(priced_value(rewards, spent, prices.numpy()).std()),
        "advantage_spread": float(advantages.std().item()),
        "price_mean": float(prices.mean().item()),
        # The diagnostic to read first: 400 episodes whose weights are worth 30
        # have not been shown anything, whatever the surrogate says.
        "effective_sample_size": effective_sample_size(final_ratio),
        "effective_fraction": effective_sample_size(final_ratio) / len(final_ratio),
        "ratio_mean": float(final_ratio.mean()),
        "ratio_max": float(final_ratio.max()),
        # `history[-1]` is the **last epoch only**, measured at the converged
        # head, so it says nothing about how much of training ran saturated --
        # also code-reviewer's. The mean over epochs is what does.
        "clipped_fraction": history[-1]["clipped_fraction"] if history else 0.0,
        "clipped_fraction_mean": (float(np.mean([e["clipped_fraction"]
                                                 for e in history]))
                                  if history else 0.0),
        "zero_gradient_fraction": (history[-1]["zero_gradient_fraction"]
                                   if history else 0.0),
        "zero_gradient_fraction_mean": (
            float(np.mean([e["zero_gradient_fraction"] for e in history]))
            if history else 0.0),
        "escalation_rate_learned": float((gain > config.lambda_).mean()),
        "escalation_rate_behaviour": float(actions.mean()),
        "gain_mean": float(gain.mean()),
        "gain_spread": float(gain.std()),
        "history": history,
        "config": vars(config) if hasattr(config, "__dict__") else config.__dict__,
    }
    head.metadata = with_own_provenance(head.metadata, report)
    return head, report


# What a fit records about itself. Taken from the report where present, so one
# function serves both trainers rather than each growing its own copy; a report
# without a field simply does not carry it, which is how a missing number stays
# missing instead of becoming someone else's number.
PROVENANCE_FIELDS = (
    "episodes", "decisions", "decisions_per_episode", "mean_reward",
    "episodes_succeeded", "objective", "price",
    "escalation_rate_behaviour", "escalation_rate_learned", "cloned_escalation_rate",
    # The three to read before treating any of these heads as a trained policy.
    "clipped_fraction_mean", "zero_gradient_fraction_mean", "effective_fraction",
    "final_loss",
)


def with_own_provenance(metadata, report, venue="episodes"):
    """Replace the seed head's `training` block with this fit's own.

    **The structural fields are the seed head's and must stay**: `hidden_size`,
    `feature_layers`, `hidden_width`, `pca_dim` and `readout` are what the loop
    reads `--head` for, and serving depends on them. The `training` block is not
    structural, and carrying it over is how `head-round0.safetensors` came to
    record `train_count 1154`, `test_count 346`, `held_out_categories
    ["chemistry", "economics", "psychology"]` and `auroc_fast_correct 0.8758` --
    the 27B's MMLU-Pro fit, byte for byte, on a head fitted to 150 ALFWorld
    episodes. A venue with no categories and no held-out split has none of those
    quantities.

    That is the `commit: None` fault with its sign flipped, which is the worse
    sign: a null provenance field is noticed and a plausible one gets quoted. exps
    found it in the pre-flight for 29700.
    """
    training = {"venue": venue}
    training.update({name: report[name] for name in PROVENANCE_FIELDS if name in report})
    return replace(metadata, training=training)


def train_head_on_smeared_returns(metadata, episodes, config=None, seed=0, epochs=40,
                                  price=None):
    """Baseline A: the same head, trained by cross-entropy on a smeared return.

    This is the baseline CDPO has to beat, and it is the one worth building
    carefully, because the obvious versions are straw men. It gets **everything**
    the proposed method gets except the loss: the same features from the same
    `flatten_episodes`, the same trunk, the same decision rule at evaluation. The
    repository has made the opposite mistake before -- the single-question
    comparison failed because the baseline carried a class weighting the method
    did not.

    What "smeared" means here, stated because the choice is the baseline:

    The return is one number for the episode, so the only per-step target it can
    supply is "the mode this step took was the right one", with the whole episode
    agreeing. Where the episode succeeded, the behaviour's escalation decision is
    the target; where it failed, its complement is. That is behaviour cloning
    weighted by success, which is what a practitioner with no counterfactual would
    actually do.

    It is **not** given a four-way outcome label, because there is none: the mode
    not taken was never scored. Inventing one -- calling a successful escalation
    "fast wrong, slow right" -- would put a counterfactual in the baseline that
    the data does not contain, and would be a different experiment. So the loss is
    binary cross-entropy on the escalation decision, read off the same
    the log-ratio gain the router uses, which is what keeps the two heads
    comparable at evaluation.
    """
    config = config or HeadRLConfig()
    features, actions, _, rewards, spent, _ = flatten_episodes(episodes, config)

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    def labels_at(prices):
        """The target: clone the behaviour where the advantage was positive.

        Unpriced (`price=None`) means "where the episode succeeded" -- what a
        practitioner with no counterfactual would do, since "worked" for them is
        that the task was completed, not that it was completed cheaply.

        Priced means the sign of **the same advantage CDPO uses**, so the two
        differ in the loss and in nothing else. `"drawn"` takes a price per
        decision from the same `draw_prices` and redraws it every epoch, exactly
        as the CDPO trainer does: a fixed price would leave the two differing in
        the loss **and** in price marginalisation, and then a CDPO win could not
        be credited to the importance ratio and the clip. code-reviewer caught
        that the first version passed one fixed midpoint while the README claimed
        the per-decision draw.
        """
        if price is None:
            good = rewards > 0
        else:
            value = priced_value(rewards, spent, prices)
            good = value > value.mean()
        return torch.as_tensor(np.where(good, actions, ~actions).astype(np.float32))

    drawn = (draw_prices(len(features), config, generator).numpy()
             if price == "drawn" else price)
    succeeded = (rewards > 0 if price is None
                 else priced_value(rewards, spent, drawn)
                 > priced_value(rewards, spent, drawn).mean())
    head = DecisionHead(metadata, dropout=config.dropout)
    matrix = torch.as_tensor(features)
    head.fit_normalizer(matrix)
    labels = labels_at(drawn)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    history = []
    for _ in range(epochs):
        head.train()
        if price == "drawn":
            # Redrawn per epoch, as the CDPO trainer redraws: the marginalisation
            # is over the price, so holding one draw fixed is a different
            # objective.
            labels = labels_at(draw_prices(len(features), config, generator).numpy())
        optimizer.zero_grad()
        logits = head(matrix)
        probabilities = torch.softmax(logits, dim=-1)
        # The same gain the router thresholds, used as the escalation logit, so
        # the two heads differ in how they were trained and in nothing else.
        gain = escalation_gain_batch(probabilities)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            gain / config.temperature, labels)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    head.eval()
    with torch.no_grad():
        final = torch.softmax(head(matrix), dim=-1)
        learned_gain = escalation_gain_batch(final).numpy()
    report = {
        "episodes": int(len(episodes)),
        "decisions": int(len(features)),
        "decisions_per_episode": float(len(features) / max(len(episodes), 1)),
        "price": price,
        "objective": ("smeared_return_cross_entropy" if price is None
                      else "smeared_priced_value_cross_entropy"),
        "episodes_succeeded": float(succeeded.mean()),
        "cloned_escalation_rate": float(labels.mean().item()),
        "escalation_rate_learned": float((learned_gain > config.lambda_).mean()),
        "gain_spread": float(learned_gain.std()),
        "final_loss": history[-1] if history else None,
    }
    # The same reason as the other trainer: `browser_pmpo.py` saves this head, and
    # whatever `--head` it was shaped from would otherwise describe it.
    head.metadata = with_own_provenance(head.metadata, report)
    return head, report
