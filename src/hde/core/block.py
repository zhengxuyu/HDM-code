"""The self-routing block: the only trained weights in \\method{}.

A small head read beside ``lm_head`` at the answer boundary, plus the policy that
turns its output into an escalate-or-not decision. The backbone stays frozen; the
head sees only readout hidden states and summary statistics of the candidate
distribution, never question or state text.

The loss that trains it is in :mod:`hde.core.objective`; the metrics that score
it and the feature helpers it is built from are in :mod:`hde.core.block_utils`.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from hde.core.block_utils import (
    FEATURE_FORMAT,
    OUTCOME_DIM,
    QUESTION_TYPES,
    STATS_DIM,
    build_features,
    concentration,
    distribution_stats,
    parse_feature_layers,
)
from hde.formats import SLOT_FORMAT

__all__ = [
    "FEATURE_FORMAT", "QUESTION_TYPES", "STATS_DIM", "OUTCOME_DIM",
    "HeadMetadata", "DecisionHead",
    "OUTCOME_CLASSES", "RoutingPolicy", "MarginPolicy", "MarginRouter",
    "fast_confidence", "escalation_gain", "top_two_margin", "should_escalate",
    "build_features", "concentration", "distribution_stats", "parse_feature_layers",
]

@dataclass(frozen=True)
class HeadMetadata:
    model_id: str
    revision: str
    hidden_size: int
    feature_layers: tuple[int, ...]
    hidden_width: int = 256
    pca_dim: int = 0
    slot_format: str = SLOT_FORMAT
    feature_format: str = FEATURE_FORMAT
    # Which readout produced the distribution this head's features were built
    # from. "coded" scores single-token letters; "content" scores each option's
    # own text, and its softmax over option scores is a different quantity on a
    # different scale. A head fitted to one and served under the other is reading
    # numbers that do not mean what it learned they meant, and nothing in its
    # output would say so. Defaults to coded, which is every head trained so far.
    readout: str = "coded"
    training: dict = field(default_factory=dict)

    @property
    def hidden_dim(self):
        return self.hidden_size * len(self.feature_layers)

    @property
    def input_dim(self):
        return self.hidden_dim + STATS_DIM

    @property
    def model_dim(self):
        """Width the MLP actually sees: PCA components (if any) plus the stats."""
        return (self.pca_dim if self.pca_dim else self.hidden_dim) + STATS_DIM

    def check_serving(self, model_id, revision=None, readout=None):
        if self.model_id != model_id:
            raise ValueError("Decision head was trained for a different model")
        if revision is not None and self.revision != revision:
            raise ValueError("Decision head was trained for a different model revision")
        if self.slot_format != SLOT_FORMAT:
            raise ValueError("Decision head prompt format changed; retrain the head")
        if self.feature_format != FEATURE_FORMAT:
            raise ValueError("Decision head feature format changed; retrain the head")
        if readout is not None and self.readout != readout:
            raise ValueError(
                f"this head's features were built from the {self.readout!r} readout and it "
                f"is being served under {readout!r}. The two produce different quantities "
                f"on different scales -- one a distribution over letter codes, the other a "
                f"softmax over option scores -- so it would be reading numbers that do not "
                f"mean what it learned they meant, and nothing in its output would say so. "
                f"Train a head under {readout!r}, or serve this one under {self.readout!r}")

class DecisionHead(nn.Module):
    def __init__(self, metadata: HeadMetadata, dropout=0.1):
        super().__init__()
        self.metadata = metadata
        dim = metadata.input_dim
        if metadata.pca_dim < 0 or metadata.pca_dim > metadata.hidden_dim:
            raise ValueError("pca_dim must be between 0 and the hidden feature width")
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        if metadata.pca_dim:
            # Whitened principal directions of the standardized hidden features.
            self.register_buffer("projection", torch.zeros(metadata.hidden_dim, metadata.pca_dim))
        self.net = nn.Sequential(
            nn.Linear(metadata.model_dim, metadata.hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(metadata.hidden_width, OUTCOME_DIM),
        )

    def _reduce(self, standardized):
        if not self.metadata.pca_dim:
            return standardized
        split = self.metadata.hidden_dim
        components = standardized[:, :split] @ self.projection
        return torch.cat([components, standardized[:, split:]], dim=1)

    def forward(self, features):
        return self.net(self._reduce((features - self.mean) / self.std))

    def fit_normalizer(self, features):
        matrix = torch.as_tensor(np.asarray(features, dtype=np.float32))
        self.mean.copy_(matrix.mean(0))
        self.std.copy_(constant_safe_scale(matrix.std(0)))
        if self.metadata.pca_dim:
            split = self.metadata.hidden_dim
            hidden = ((matrix - self.mean) / self.std)[:, :split]
            rank = min(self.metadata.pca_dim, len(hidden) - 1, split)
            if rank < 1:
                raise ValueError("Not enough examples to fit the PCA bottleneck")
            _, singular, right = torch.linalg.svd(hidden, full_matrices=False)
            right = _pin_svd_signs(right)
            scale = (singular[:rank] / math.sqrt(max(len(hidden) - 1, 1))).clamp_min(1e-6)
            self.projection.zero_()
            self.projection[:, :rank] = right[:rank].T / scale

    SERVING_SIGMA_LIMIT = 1e3

    def check_standardised(self, matrix):
        """Refuse a feature the normaliser would turn into an enormous input.

        **The check that would have caught the type refusal.** A head fitted before
        `a681033` has `std` at the clamp floor for every feature its training set
        never varied, and a serving value that deviates there standardises to about
        a million. The head does not fail; it saturates and returns the same
        outcome for every such row, which this repository published as a finding
        and worked around once in a test docstring before that.

        Checked per call rather than at load, because a clamped divisor is only a
        fault when a value actually deviates on that feature: the one-hots are
        clamped on every head we have, and on a choice-only benchmark they all
        standardise to exactly zero and nothing is wrong. Refusing at load would
        reject every existing head for a condition most of them never meet.

        The limit is deliberately far above anything real. Standardised features
        span about plus or minus three, ten would be remarkable, and the failure
        arrives at a million.
        """
        z = (matrix - self.mean) / self.std
        worst = z.abs().max().item()
        if worst <= self.SERVING_SIGMA_LIMIT:
            return
        rows, columns = (z.abs() > self.SERVING_SIGMA_LIMIT).nonzero(as_tuple=True)
        bad = sorted(set(columns.tolist()))
        floored = [i for i in bad if self.std[i].item() <= CONSTANT_SCALE_EPS]
        raise ValueError(
            f"feature(s) {bad[:8]} standardise to {worst:,.0f} sigma, so this head "
            f"would saturate rather than route. "
            + (f"The divisor is at the clamp floor for {floored[:8]}, meaning the "
               f"training set never varied them: this head cannot read those "
               f"features and was fitted before they were given unit scale. Refit "
               f"it, or serve only data whose values match the training constants."
               if floored else
               "No divisor is at the clamp floor, so the input itself is far out of "
               "distribution rather than the scaling being at fault.")
        )

    @torch.no_grad()
    def predict(self, features):
        """Outcome probabilities [N, 4] for a feature matrix [N, input_dim]."""
        self.eval()
        matrix = torch.as_tensor(np.asarray(features, dtype=np.float32)).reshape(
            -1, self.metadata.input_dim
        ).to(self.mean.device)
        self.check_standardised(matrix)
        return torch.softmax(self(matrix), dim=-1).cpu().numpy()

    def save(self, path):
        payload = {
            key: value.detach().cpu().contiguous() for key, value in self.state_dict().items()
        }
        # The metadata key keeps the engine's old spelling: it is what the
        # published blocks were written under, and renaming the key here would
        # make every one of them fail to load.
        save_file(payload, str(path), metadata={"litjev": json.dumps(asdict(self.metadata))})

    @classmethod
    def load(cls, path, device="cpu"):
        with safe_open(str(Path(path)), framework="pt") as handle:
            raw = handle.metadata() or {}
        if "litjev" not in raw:
            raise ValueError("Not an HDE decision head file")
        data = json.loads(raw["litjev"])
        data["feature_layers"] = tuple(data["feature_layers"])
        data.setdefault("pca_dim", 0)
        head = cls(HeadMetadata(**data))
        head.load_state_dict(load_file(str(path)))
        return head.to(device).eval()


# ---- the policy that reads the block's output -------------------------------

OUTCOME_CLASSES = (
    "fast_right_slow_right",
    "fast_right_slow_wrong",
    "fast_wrong_slow_right",
    "fast_wrong_slow_wrong",
)

@dataclass(frozen=True)
class RoutingPolicy:
    lambda_: float = 0.0
    budget: int = 0

    def __post_init__(self):
        if not np.isfinite(self.lambda_):
            raise ValueError("lambda must be finite")
        if self.budget < 0:
            raise ValueError("budget must be non-negative")

    @property
    def enabled(self):
        return self.budget > 0

CONSTANT_SCALE_EPS = 1e-6


def constant_safe_scale(spread):
    """Per-feature divisor, with a feature that never varied given unit scale.

    `spread.clamp_min(1e-6)` looks like division-by-zero protection and is an
    amplifier. A feature constant at `c` in training gets `std = 0`, clamped to
    `1e-6`, so a serving-time value of `c + 1` standardises to **one million**
    where every feature the network trained on spans about plus or minus three.
    That does not bias the output, it saturates it.

    It was not hypothetical. The question-type one-hots are the last three
    features, every MMLU-Pro row is `choice`, so in training they are constant at
    `(1, 0, 0)` and all three get `std = 1e-6`. A `noul` question then presents
    `-1e6` at the `choice` position and `+1e6` at its own. Taking the 346 held-out
    MMLU-Pro questions and changing nothing but that one-hot:

        as collected (choice)      gain median +0.196  spread 3.183  314 distinct
        re-typed noul, 1e-6 scale  gain median  0.000  spread 0.000    1 distinct
        re-typed noul, unit scale  gain median +0.197  spread 3.189  317 distinct

    Every gain collapses to exactly zero, so nothing escalates at a non-negative
    price and every item sits at one value. That is the whole of this project's
    reported JevBench type refusal -- "the head escalates none of the 74 `noul`
    and none of the 18 `score` items" -- and it is this line, not the head. At
    unit scale the same head routes a re-typed question exactly as it routes the
    original, to three decimal places.

    Unit scale is the right default rather than zeroing the feature: a constant
    feature carries no information from training, so the natural reading of a
    deviation at serving time is its own magnitude, and for a one-hot that is
    order one. Zeroing would also work for the saturation and would additionally
    discard the fact that the question is out of distribution, which a caller may
    want the head to see.

    This changes newly fitted heads only. A head on disk serves from its stored
    `std` buffer, so every number already measured stands as measured.
    """
    return torch.where(spread > CONSTANT_SCALE_EPS, spread, torch.ones_like(spread))


def _pin_svd_signs(right):
    """Fix each right singular vector's sign, so the projection is the matrix's.

    `U S V^T == (-U) S (-V)^T`, so LAPACK may return either and the choice is the
    implementation's. It differs between our builds: on a byte-identical
    1154x2560 input (same sha1, singular values agreeing to seven figures) the
    arm64 and x86 builds returned **every one of the first ten** right vectors
    with opposite signs. A flipped component negates that coordinate of
    `projection`, the MLP's initialisation does not follow it, and the fit lands
    somewhere else -- so a PCA candidate's AUROC was a property of the machine
    while every raw candidate was bit-identical across both. That is how
    `layer_-1_pca64` came to read 0.843574 on the cluster and 0.838395 on a
    laptop from the same records, the same seed and the same code.

    Seeding cannot fix it, because nothing here is random. Making the
    largest-magnitude entry of each vector positive is a property of the vector,
    so every implementation that returns a correct decomposition now returns the
    same projection.
    """
    pivot = right.abs().argmax(dim=1)
    sign = torch.sign(right[torch.arange(len(right), device=right.device), pivot])
    # A singular vector cannot be all zeros, but a zero pivot would silently zero
    # the component rather than leave it alone.
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return right * sign.unsqueeze(1)


def fast_confidence(outcome):
    """Calibrated P(fast answer is correct)."""
    return float(outcome[0] + outcome[1])

# Probabilities come from a softmax and are never exactly zero, but a log still
# wants a floor: a saturated head would otherwise return an infinity here.
GAIN_EPS = 1e-6


def escalation_gain(outcome):
    """The value of thinking: log P(slow right) - log P(fast right).

    A log ratio, not the difference `P(slow right) - P(fast right)` this used to
    return. The difference is what a linear price `u - lambda * T` makes optimal;
    the reward is a discount, `u * exp(-lambda * T)`, and escalating pays when
    `P(slow right) * exp(-lambda * T) > P(fast right)`, which in logarithms is
    this against the price.

    The difference is not scale-free: it is bounded by how much the fast mode is
    already getting wrong, so it shrinks exactly where thinking is worth most. On
    the sixteen-arrow walk the fast mode's success falls forty-five-fold with
    depth; read as a difference the head's value falls with it, 0.019 to 0.007,
    and read as a ratio it rises, 0.005 to 0.38, which is the direction the task
    actually moves (experiments/depth-walk).

    This must agree with what `cdpo_loss` thresholds, or a head is trained
    against one rule and served under another.
    """
    fast_right = outcome[0] + outcome[1]
    slow_right = outcome[0] + outcome[2]
    return float(math.log(max(slow_right, GAIN_EPS)) - math.log(max(fast_right, GAIN_EPS)))


def escalation_gain_batch(outcome):
    """`escalation_gain` over a batch of outcome distributions, differentiably.

    The same quantity as the scalar version above, kept beside it rather than
    rewritten at each call site. There were three copies of the old difference
    `o[2] - o[1]` when this was introduced -- the CDPO loss, the off-policy head
    update, and its cross-entropy baseline -- and the one in the off-policy update
    had been left behind: rollouts escalated on the log ratio while the loss
    scored the policy by the difference, so the importance ratio divided two
    different functions and the head optimised a rule nothing executed.

    `outcome` is probabilities over (both right, fast only, slow only, neither),
    last axis. Returns one gain per row.
    """
    fast_right = outcome[..., 0] + outcome[..., 1]
    slow_right = outcome[..., 0] + outcome[..., 2]
    return (torch.log(slow_right.clamp_min(GAIN_EPS))
            - torch.log(fast_right.clamp_min(GAIN_EPS)))

@dataclass(frozen=True)
class MarginPolicy:
    """AnyJev's published adaptive rule, reimplemented so we can compare against it.

    Their README states it in two parts, and both are here because the second
    changes the operating point: escalate when "the log-probability gap between
    the top two options is below `tau`" (their default 0.5), and "at most
    `max_cot_share` of the last `cot_window` decisions escalate" (their defaults
    20% of 1000).

    This is the `top-two margin` row of the fitted-nothing table wired to a
    serving path. It fits nothing, reads only the distribution the answering pass
    already produced, and is therefore the floor our trained head has to clear.
    We have that comparison on MMLU-Pro and not on JevBench, which is the gap
    this exists to close.

    **The cap is an absolute count until the window fills, and on our benchmarks it
    never fills.** `MarginRouter.decide` tests `sum(recent) < cap_share * window`,
    which is `0.2 * 1000 = 200` escalations, not 20% of the decisions seen so far.
    Once 1,000 decisions have gone by the two readings agree; below that the cap
    permits 200 escalations whatever the denominator. So on a 346-question
    benchmark the effective ceiling is **200/346 = 57.8%**, not 20%, and the
    numbers that follow from it are:

        at n=346, tau 0.25 / 0.5 / 1.0 / 1.5 escalate 49 / 106 / 165 / 191 items
        on the 4B, all under the cap, so those operating points are the rule's own;
        tau 2 and 3 want 205 and 238 and are truncated to 200, so they collapse
        onto one point rather than giving two.

    That matters for any comparison against this rule rather than for the rule
    itself: **the published rule cannot reach an escalation rate above 57.8% at
    this n**, so a head row above that has no comparable margin row, and a claim
    that the head beats the adaptive rule there would be comparing against
    something the rule is forbidden to do. The behaviour is left as it is rather
    than reinterpreted, because "the last 1,000 decisions" has no meaning at 346
    and an absolute 200 per window is a defensible reading of someone else's rule;
    what is not defensible is describing it as 20%.

    `budget` plays the same role as in `RoutingPolicy`: zero disables routing.
    """

    tau: float = 0.5
    cap_share: float = 0.2
    window: int = 1000
    budget: int = 0

    def __post_init__(self):
        if not np.isfinite(self.tau) or self.tau < 0:
            raise ValueError("tau must be finite and non-negative")
        if not 0.0 <= self.cap_share <= 1.0:
            raise ValueError("cap_share is a share of recent decisions")
        if self.window < 1:
            raise ValueError("window must be at least one decision")
        if self.budget < 0:
            raise ValueError("budget must be non-negative")

    @property
    def enabled(self):
        return self.budget > 0

def top_two_margin(probabilities):
    """The gap between the top two options, in log probability.

    Taken from the normalised distribution rather than the raw logits, so that
    the number means the same thing under either readout: the logits are
    unnormalised and their differences would not be log-probability gaps. With
    one candidate the margin is infinite -- nothing to be unsure between -- which
    makes a single-option question never escalate rather than always.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.size < 2:
        return float("inf")
    top = np.sort(values)[::-1]
    if top[1] <= 0.0:
        return float("inf")
    return float(np.log(top[0]) - np.log(top[1]))

class MarginRouter:
    """A `MarginPolicy` plus the decision history its cap needs.

    The cap is over *recent* decisions, so unlike `should_escalate` this cannot
    be a pure function -- it carries a window. One router per policy per run, and
    `decide` is called once per question in the order they are served, because a
    cap over the last N decisions depends on that order.
    """

    def __init__(self, policy):
        self.policy = policy
        self._recent = deque(maxlen=policy.window)

    @property
    def escalated_share(self):
        return (sum(self._recent) / len(self._recent)) if self._recent else 0.0

    def decide(self, probabilities):
        if not self.policy.enabled:
            self._recent.append(False)
            return False
        unsure = top_two_margin(probabilities) < self.policy.tau
        # The cap is checked against what the window holds *before* this
        # decision, which is what "at most 20% of the last 1000" means when the
        # 1001st arrives. Admitting it first and then testing would let the share
        # exceed the cap by one on every window.
        room = sum(self._recent) < self.policy.cap_share * self.policy.window
        escalate = bool(unsure and room)
        self._recent.append(escalate)
        return escalate

def should_escalate(outcome, policy):
    return bool(policy.enabled and escalation_gain(outcome) > policy.lambda_)
