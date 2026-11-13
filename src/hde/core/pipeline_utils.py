"""The typed objects the pipeline returns, and the readout constants it switches on.

Split from :mod:`hde.core.pipeline` so that file holds the fast-route-slow
sequence alone. Nothing here is parsed from generated text: every answer is
assembled from logits read at the answer boundary.
"""

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

CONFIDENCE_METHOD = "normalized_gini_concentration_v1"

HEAD_CONFIDENCE_METHOD = "decision_head_outcome_v1"

@dataclass(frozen=True)
class RawFieldScores:
    name: str
    logits: np.ndarray
    input_tokens: int
    provenance: dict = field(default_factory=dict)
    hidden: np.ndarray | None = None
    generated_tokens: int = 0
    # What the slow mode reasoned, when it reasoned. The answer is still read
    # from the candidate codes and never parsed out of this text; it is carried
    # so a caller can show the reasoning, which is the only way a viewer can see
    # what escalating bought.
    reasoning: str | None = None

class LogitProvider(Protocol):
    def score(self, state, schema) -> tuple[RawFieldScores, ...]: ...

@dataclass(frozen=True)
class ChoiceAnswer:
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float

@dataclass(frozen=True)
class ScoreAnswer:
    type: Literal["score"]
    score: float
    legend: dict
    probabilities: dict[str, float]
    confidence: float

@dataclass(frozen=True)
class NoulAnswer:
    type: Literal["noul"]
    noul: float

@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int = 0

@dataclass(frozen=True)
class DecisionResponse:
    model: str
    answers: dict[str, ChoiceAnswer | ScoreAnswer | NoulAnswer]
    usage: Usage

@dataclass(frozen=True)
class Evaluation:
    result: DecisionResponse
    diagnostics: dict

def concentration(probabilities):
    """HDE statistic, not a reproduction of Jev's unpublished formula."""
    count = len(probabilities)
    if count == 1:
        return 1.0
    return float(np.clip((count * sum(p * p for p in probabilities) - 1) / (count - 1), 0, 1))

READOUTS = ("coded", "content")

DEFAULT_READOUT = "coded"

RANKED_BY = {"sum": "total", "mean": "mean", "pmi": "pmi"}

NORMALISATIONS = tuple(RANKED_BY)
