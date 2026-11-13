"""Helpers for the self-routing block: its input features and its metrics.

Split from :mod:`hde.core.block` so that file holds the architecture alone.
"""

from __future__ import annotations

import math

import numpy as np

from hde.formats import (  # noqa: F401
    FEATURE_FORMAT,
    OUTCOME_DIM,
    QUESTION_TYPES,
    STATS_DIM,
)


def concentration(probabilities):
    count = len(probabilities)
    if count == 1:
        return 1.0
    return float(np.clip((count * sum(p * p for p in probabilities) - 1) / (count - 1), 0, 1))

def distribution_stats(probabilities, question_type):
    """Summary of the fast candidate distribution; scale-free so it transfers across K."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("probabilities must be a non-empty vector")
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"Unknown question type {question_type!r}")
    order = np.sort(p)[::-1]
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    normalized_entropy = entropy / math.log(len(p)) if len(p) > 1 else 0.0
    margin = float(order[0] - order[1]) if len(p) > 1 else 1.0
    one_hot = [1.0 if question_type == kind else 0.0 for kind in QUESTION_TYPES]
    return np.array(
        [float(order[0]), normalized_entropy, margin, concentration(p), math.log(len(p)), *one_hot],
        dtype=np.float32,
    )

def build_features(hidden, probabilities, question_type):
    """Concatenate readout hidden states (already selected layers) with distribution stats."""
    hidden = np.asarray(hidden, dtype=np.float32).reshape(-1)
    return np.concatenate([hidden, distribution_stats(probabilities, question_type)])

def parse_feature_layers(text):
    """Comma-separated hidden_states indices; -1 is the final normalized output."""
    layers = tuple(int(item) for item in text.split(",") if item.strip())
    if not layers:
        raise ValueError("At least one feature layer is required")
    return layers

def auroc(scores, positives):
    """Rank-based AUROC with tie handling, no sklearn dependency."""
    scores = np.asarray(scores, dtype=np.float64)
    positives = np.asarray(positives, dtype=bool)
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def coverage_accuracy_curve(gain, fast_correct, slow_correct, lambdas):
    """Escalation rate and accuracy per lambda, using the slow answer where escalated."""
    gain = np.asarray(gain, dtype=np.float64)
    fast = np.asarray(fast_correct, dtype=bool)
    slow = np.asarray(slow_correct, dtype=bool)
    rows = []
    for value in lambdas:
        escalate = gain > value
        rows.append(
            {
                "lambda": float(value),
                "escalation_rate": float(escalate.mean()),
                "accuracy": float(np.where(escalate, slow, fast).mean()),
            }
        )
    return rows
