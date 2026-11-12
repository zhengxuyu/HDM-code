"""\\cdpo{}: the objective that trains the self-routing block, and the loop that applies it.

Three terms. The policy term scores the decision the block is used to make by what
it earns at a price drawn per request; the Brier term keeps the reported confidence
calibrated; the cross-entropy anchor keeps the four-way output a distribution. The
block itself is in :mod:`litjev.core.block`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from litjev.core.block import DecisionHead, escalation_gain_batch
from litjev.core.block_utils import OUTCOME_DIM

__all__ = ["CDPOConfig", "OBJECTIVES", "FORMER_NAMES", "normalise_objective",
           "cdpo_loss", "train_head", "outcome_labels"]

def outcome_labels(fast_correct, slow_correct):
    """Map (fast, slow) correctness to the four outcome classes."""
    fast = np.asarray(fast_correct, dtype=bool)
    slow = np.asarray(slow_correct, dtype=bool)
    return np.where(fast, np.where(slow, 0, 1), np.where(slow, 2, 3)).astype(np.int64)

@dataclass(frozen=True)
class CDPOConfig:
    """Calibrated decision policy optimisation.

    The head's escalation gain g = o[2] - o[1] defines a soft routing policy
    pi(escalate | x, lambda) = sigmoid((g - lambda) / temperature). Both outcomes are
    known for every training question, so the expected routed reward
    E_pi[ a_slow - a_fast - lambda ] is exact and differentiable: no sampling, no
    critic. lambda is drawn per example from `lambda_range` so g stays a usable gain
    estimate for any request-time lambda. A Brier term keeps c = P(fast correct)
    calibrated, and a cross-entropy anchor keeps the four-class output meaningful.

    The anchor is unweighted. Inverse-frequency weighting is the wrong remedy for a
    head whose output feeds a threshold: it optimises a reweighted posterior, so the
    prediction is no longer calibrated, and the class it upweights most is the one
    g = o[2] - o[1] subtracts.
    """

    lambda_range: tuple[float, float] = (-0.2, 0.5)
    temperature: float = 0.05
    calibration_weight: float = 1.0
    anchor_weight: float = 0.25
    # Ablation switches. PMPO changes three things against cross-entropy at once,
    # so each term can be removed on its own to see what it was carrying.
    policy_weight: float = 1.0
    weighted_anchor: bool = False

OBJECTIVES = ("supervised", "cdpo")

FORMER_NAMES = {"pmpo": "cdpo"}

def normalise_objective(name):
    """The canonical objective name, accepting the former one.

    A rename that rejected the old name would make every stored head's record
    unreadable, and a rename that kept both as distinct values would make two
    names for one objective comparable only by accident.
    """
    canonical = FORMER_NAMES.get(name, name)
    if canonical not in OBJECTIVES:
        raise ValueError(
            f"objective must be one of {OBJECTIVES}, or {tuple(FORMER_NAMES)} for the "
            f"former name; got {name!r}")
    return canonical

# Probabilities come from a softmax and so are never exactly zero, but a log of
# one is still worth a floor: without it a saturated head turns one question's
# gradient into an infinity and takes the batch with it.
EPS = 1e-6


def cdpo_loss(logits, labels, class_weight, config, generator):
    """Policy term + calibration term + supervised anchor, for one batch."""
    outcome = torch.softmax(logits, dim=-1)
    fast_correct = (labels <= 1).float()
    slow_correct = ((labels == 0) | (labels == 2)).float()
    low, high = config.lambda_range
    lam = torch.rand(len(labels), generator=generator) * (high - low) + low
    lam = lam.to(logits.device)
    # The reward is the utility discounted by what the wait cost, not the utility
    # with a price subtracted: r = u * exp(-lambda * T), and T = 0 for the fast
    # mode. Escalating pays when P(slow ok) * exp(-lambda) > P(fast ok), which in
    # logarithms is a threshold on the ratio the two modes succeed at rather than
    # on their difference. A difference is bounded by how much the fast mode is
    # already getting wrong, so it shrinks exactly where thinking is worth most:
    # on the sixteen-arrow walk the fast mode's success falls forty-five-fold and
    # the difference falls with it, 0.019 to 0.007, while the ratio rises, 0.005
    # to 0.38 (experiments/depth-walk).
    confidence = outcome[:, 0] + outcome[:, 1]            # P(fast ok)
    gain = escalation_gain_batch(outcome)
    policy = torch.sigmoid((gain - lam) / config.temperature)
    # The realised reward difference, whose expectation is positive exactly when
    # `gain > lam` -- so the policy is trained against the rule it is thresholded
    # by, which is what made the linear version consistent and has to be kept.
    reward = slow_correct * torch.exp(-lam) - fast_correct
    policy_term = -(policy * reward).mean()
    calibration_term = ((confidence - fast_correct) ** 2).mean()
    anchor = nn.functional.cross_entropy(
        logits, labels, weight=class_weight if config.weighted_anchor else None
    )
    return (
        config.policy_weight * policy_term
        + config.calibration_weight * calibration_term
        + config.anchor_weight * anchor
    )

def train_head(
    metadata,
    features,
    labels,
    epochs=30,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=5e-2,
    dropout=0.3,
    validation_fraction=0.15,
    patience=6,
    seed=0,
    device="cpu",
    objective="supervised",
    cdpo=None,
    weighted_supervised=False,
    pmpo=None,
):
    """Train the head with early stopping; the backbone is never touched.

    `objective` is "supervised" (cross-entropy on the four outcomes) or "cdpo"
    (calibrated decision policy optimisation, see `CDPOConfig`). Set
    `weighted_supervised` to reproduce the class-weighted variant we no longer use. A slice of the
    training set is held out for validation; the returned head is the epoch with the
    lowest validation loss. History rows carry train and validation loss.

    `"pmpo"` is the name this objective carried for one revision of the paper and
    is accepted for the heads and scripts written under it. It is normalised to
    `"cdpo"` here, so what a run records is the name the paper uses now, whichever
    name the caller passed --
    a head trained today and a head trained last week are the same head, and their
    records should not disagree about what produced them.
    """
    objective = normalise_objective(objective)
    # One rule for the former argument name, the same as `run_training`'s: both
    # names at once is a mistake, not a precedence question.
    if pmpo is not None and cdpo is not None:
        raise ValueError("pass cdpo=, not both cdpo= and its former name pmpo=")
    config = cdpo if cdpo is not None else (pmpo if pmpo is not None else CDPOConfig())
    torch.manual_seed(seed)
    matrix = torch.as_tensor(np.asarray(features, dtype=np.float32))
    targets = torch.as_tensor(np.asarray(labels, dtype=np.int64))
    if matrix.ndim != 2 or matrix.shape[1] != metadata.input_dim:
        raise ValueError("features must be [N, input_dim]")
    if len(matrix) != len(targets):
        raise ValueError("features and labels length mismatch")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(matrix), generator=generator)
    n_val = int(len(matrix) * validation_fraction) if len(matrix) >= 8 else 0
    val_index, train_index = order[:n_val], order[n_val:]
    head = DecisionHead(metadata, dropout=dropout).to(device)
    head.fit_normalizer(matrix[train_index])
    counts = torch.bincount(targets[train_index], minlength=OUTCOME_DIM).float().clamp_min(1)
    class_weight = (counts.sum() / (OUTCOME_DIM * counts)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Unweighted. Inverse-frequency weighting hurts this head badly: the rarest
    # outcome is "fast right, slow wrong", and the escalation gain g = o2 - o1
    # subtracts exactly that class, so upweighting it puts variance straight into
    # the quantity the router ranks by. Weighting costs about six points of
    # routing AUROC and doubles the gain calibration error; see scripts/table3.py.
    supervised = nn.CrossEntropyLoss(weight=class_weight if weighted_supervised else None)

    def loss_fn(logits, batch_labels):
        if objective == "cdpo":
            return cdpo_loss(logits, batch_labels, class_weight, config, generator)
        return supervised(logits, batch_labels)

    history, best_state, best_loss, stale = [], None, float("inf"), 0
    for _ in range(epochs):
        head.train()
        shuffled = train_index[torch.randperm(len(train_index), generator=generator)]
        total = 0.0
        for start in range(0, len(shuffled), batch_size):
            index = shuffled[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(head(matrix[index].to(device)), targets[index].to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(index)
        head.eval()
        row = {"train_loss": total / max(len(shuffled), 1)}
        if n_val:
            with torch.no_grad():
                val_loss = loss_fn(
                    head(matrix[val_index].to(device)), targets[val_index].to(device)
                ).item()
            row["val_loss"] = val_loss
            if val_loss < best_loss - 1e-4:
                best_loss, stale = val_loss, 0
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


# The names this objective carried for one revision, kept so heads, scripts and
# notebooks written under them keep importing. New code uses CDPOConfig and
# cdpo_loss.
PMPOConfig = CDPOConfig
pmpo_loss = cdpo_loss
