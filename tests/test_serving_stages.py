"""The startup chain's stage 3, and that both deployments go through it.

`cli.serve()` checked the head against the backend; `build_service` did not, so
the decision service would serve a head against another model, or under a readout
whose numbers mean something else, and raise nothing. These fail if either call
site drops the check again.
"""

import ast
import pathlib

import pytest

from litjev.core.block import HeadMetadata
from litjev.serving import attach_head, feature_layers_for, load_head

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


class FakeScorer:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size


def metadata(**overrides):
    base = dict(model_id="org/model", revision="main", hidden_size=8, feature_layers=(-1,))
    return HeadMetadata(**{**base, **overrides})


class FakeHead:
    def __init__(self, **overrides):
        self.metadata = metadata(**overrides)
        self.device = None

    def to(self, device):
        self.device = device
        return self


def test_no_head_is_a_configuration_not_an_error():
    assert load_head(None) is None
    assert load_head("") is None
    assert feature_layers_for(None) == ()
    assert attach_head(None, FakeScorer(8), "org/model") is None


def test_a_head_trained_for_another_model_is_refused():
    with pytest.raises(ValueError, match="different model"):
        attach_head(FakeHead(), FakeScorer(8), "org/other")


def test_a_head_trained_at_another_revision_is_refused():
    with pytest.raises(ValueError, match="different model revision"):
        attach_head(FakeHead(), FakeScorer(8), "org/model", revision="v2")


def test_a_head_whose_hidden_state_is_the_wrong_width_is_refused():
    with pytest.raises(ValueError, match="different backbones"):
        attach_head(FakeHead(), FakeScorer(4096), "org/model")


def test_a_matching_head_is_placed_on_the_device_and_returned():
    head = FakeHead()
    returned = attach_head(head, FakeScorer(8), "org/model", "main", device="cpu")
    assert returned is head and head.device == "cpu"


def test_a_scorer_that_does_not_report_a_hidden_size_is_not_second_guessed():
    """An SGLang server scorer may not expose one; the other checks still run."""
    head = FakeHead()
    assert attach_head(head, object(), "org/model") is head


@pytest.mark.parametrize("path", ["litjev/cli.py", "inference/decision_service.py"])
def test_both_startup_paths_call_the_stage(path):
    """Neither may load a head and reach a serving object without attach_head.

    Source-level because standing up either deployment needs a GPU. The failure
    this guards against is a line being dropped, which is visible here.
    """
    tree = ast.parse((SRC / path).read_text())
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "load_head" in calls, f"{path} loads a head some other way"
    assert "attach_head" in calls, (
        f"{path} builds a serving object without attach_head, so a head and a "
        "backend that disagree would be paired silently"
    )
