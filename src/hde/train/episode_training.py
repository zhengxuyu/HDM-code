"""Training the decision head for an episode reward rather than a per-question one.

Single-question routing is separable, and Section 5 of the paper reports that a
decision-focused loss buys nothing there over likelihood training. An episode
that stops at its first wrong step is not separable: the value of step t is
multiplied by the chance of reaching it, and so is its cost, because an
escalation on a step the agent never reaches is never paid for.

Both modes' outcomes are recorded for every question, so the expected progress
under a soft routing policy is exact and differentiable. No sampling, no critic.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from hde.core.block import DecisionHead, escalation_gain_batch

EPS = 1e-9


@dataclass(frozen=True)
class EpisodeCDPOConfig:
    """The episode objective. `anchor_weight` keeps a proper scoring rule underneath.

    The single-question ablation showed that cross-entropy on the four-way
    outcome is what calibrates every functional of the prediction, and that
    down-weighting or class-weighting it is what cost the published recipe its
    calibration. Here the anchor is full and unweighted by default.
    """

    lambda_range: tuple[float, float] = (-0.2, 0.5)
    temperature: float = 0.05
    progress_weight: float = 1.0
    anchor_weight: float = 1.0


def expected_progress_and_cost(escalate_prob, fast_correct, slow_correct, discount=None):
    """Exact expected steps completed, and expected escalations actually paid for.

    `fast_correct` and `slow_correct` are the recorded 0/1 outcomes, so the
    expected correctness of a step under the soft policy is a linear function of
    the escalation probability with no approximation.

    With `discount` set to `exp(-lambda)`, the first return value is instead the
    episodic reward of the paper's eq:return, `E[R exp(-lambda C)]`, where R is
    the steps completed and C the escalations paid for -- including one on the
    step that failed, because the thinking is spent before the answer is scored.

    That expectation does not factor the way the undiscounted one does. Writing

        A_s = p_s slow_s d + (1 - p_s) fast_s        (step s succeeds)
        F_s = p_s (1 - slow_s) d + (1 - p_s) (1 - fast_s)   (step s fails)

    with `d = exp(-lambda)`, an episode that fails first at step k carries R=k-1
    and contributes `(prod_{s<k} A_s) F_k (k-1)`, and the one that never fails
    carries R=n, so

        E[R exp(-lambda C)] = sum_k (k-1) (prod_{s<k} A_s) F_k + n prod_s A_s.

    The obvious alternative, `sum_t prod_{s<=t} A_s`, is **not** this: it
    discounts each step by what was spent reaching it rather than by the episode's
    total, which is a discounted-progress objective and not eq:return. Over 2000
    random episodes of one to five steps the two differ by up to 2.13 steps (worst
    case 6.45 against 4.33, at lambda = -0.29), so the choice is not a detail.
    Which one this is is settled against enumeration of every trajectory in
    `tests/test_episode_training.py` rather than by the derivation above.
    """
    discounted = escalate_prob * slow_correct * (1.0 if discount is None else discount)
    success = discounted + (1 - escalate_prob) * fast_correct
    if discount is None:
        reach = torch.cumprod(success, dim=-1)
        before = torch.cat([torch.ones_like(reach[..., :1]), reach[..., :-1]], dim=-1)
        return reach.sum(-1), (before * escalate_prob).sum(-1)

    failure = (escalate_prob * (1 - slow_correct) * discount
               + (1 - escalate_prob) * (1 - fast_correct))
    reach = torch.cumprod(success, dim=-1)
    before = torch.cat([torch.ones_like(reach[..., :1]), reach[..., :-1]], dim=-1)
    steps = success.shape[-1]
    completed = torch.arange(steps, dtype=success.dtype, device=success.device)
    progress = (before * failure * completed).sum(-1) + steps * reach[..., -1]

    # The cost is the **undiscounted** expected escalations, because what it
    # answers is how many this policy paid for, and that question has no price in
    # it. So it is taken against the undiscounted reach.
    plain = torch.cumprod(
        escalate_prob * slow_correct + (1 - escalate_prob) * fast_correct, dim=-1)
    plain_before = torch.cat([torch.ones_like(plain[..., :1]), plain[..., :-1]], dim=-1)
    return progress, (plain_before * escalate_prob).sum(-1)


def episode_loss(logits, fast_correct, slow_correct, config, generator):
    """Negative discounted expected progress, plus a cross-entropy anchor.

    logits is [episodes, steps, 4]; the outcome arrays have the matching shape.
    A price is drawn per episode, so one head serves every request-time price.
    """
    probs = torch.softmax(logits, dim=-1)
    gain = escalation_gain_batch(probs)
    low, high = config.lambda_range
    price = torch.rand(logits.shape[0], 1, generator=generator, device=logits.device)
    price = low + price * (high - low)
    escalate_prob = torch.sigmoid((gain - price) / config.temperature)
    # The price discounts rather than subtracts, matching the single-question
    # reward `u exp(-lambda T)` and eq:return. The subtraction `progress - price *
    # cost` this replaced let a high enough price push a successful episode below
    # a failed one, which is an ordering no deployment wants: cost alone should
    # not make progress worth less than none.
    progress, _cost = expected_progress_and_cost(
        escalate_prob, fast_correct, slow_correct, discount=torch.exp(-price))
    reward = progress.mean()
    labels = (2 * (1 - fast_correct) + (1 - slow_correct)).long().reshape(-1)
    anchor = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels)
    return -config.progress_weight * reward + config.anchor_weight * anchor


def train_episode_head(
    metadata,
    features,
    fast_correct,
    slow_correct,
    episodes,
    config=None,
    epochs=30,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=5e-2,
    dropout=0.3,
    validation_fraction=0.15,
    patience=6,
    seed=0,
):
    """Same head, same optimiser and early stopping as `train_head`; episode loss.

    Batches are episodes, not questions, because the loss does not factor over
    questions. Validation holds out whole episodes.
    """
    config = config or EpisodeCDPOConfig()
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.as_tensor(np.asarray(features, dtype=np.float32))
    fast = torch.as_tensor(np.asarray(fast_correct, dtype=np.float32))
    slow = torch.as_tensor(np.asarray(slow_correct, dtype=np.float32))
    index = torch.as_tensor(np.asarray([list(e) for e in episodes], dtype=np.int64))
    if index.ndim != 2:
        raise ValueError("episodes must all have the same length")

    order = torch.randperm(len(index), generator=generator)
    n_val = int(len(index) * validation_fraction) if len(index) >= 8 else 0
    val_episodes, train_episodes = index[order[:n_val]], index[order[n_val:]]

    head = DecisionHead(metadata, dropout=dropout)
    head.fit_normalizer(matrix[train_episodes.reshape(-1)])
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def loss_for(batch):
        rows = batch.reshape(-1)
        logits = head(matrix[rows]).reshape(*batch.shape, -1)
        return episode_loss(logits, fast[batch], slow[batch], config, generator)

    history, best_state, best_loss, stale = [], None, float("inf"), 0
    for _ in range(epochs):
        head.train()
        shuffled = train_episodes[torch.randperm(len(train_episodes), generator=generator)]
        total = 0.0
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_for(batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        head.eval()
        row = {"train_loss": total / max(len(shuffled), 1)}
        if n_val:
            with torch.no_grad():
                row["val_loss"] = loss_for(val_episodes).item()
            if row["val_loss"] < best_loss - 1e-4:
                best_loss, stale = row["val_loss"], 0
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        if n_val and stale >= patience:
            break
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head, history


# The name this config carried for one revision, kept so anything written under
# it keeps importing.
EpisodePMPOConfig = EpisodeCDPOConfig
