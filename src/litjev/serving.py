"""The startup chain, as named stages shared by every way the model is served.

There are two deployments and they build different objects -- `cli.serve()` puts a
`SchemaDecisionEngine` behind FastAPI, `inference.decision_service.build_service`
puts the head beside an SGLang engine and returns only the decision -- but they
pass through the same four stages:

  1. load the self-routing block, if one was asked for
  2. ask it which backbone layers the backend has to expose
  3. attach it to the backend, which must refuse a pairing it was not trained for
  4. assemble the serving object

Stage 3 is here because it used to exist on one path only. `cli.serve()` called
`check_serving` and compared hidden sizes; `build_service` did neither, so the
decision service would serve a head against another model, or under a readout
whose numbers mean something else, and nothing in its output would have said so.

This module knows about the block and about nothing else. Backends are passed in,
so it does not import `inference` or any engine.
"""

from litjev.core.block import DecisionHead


def load_head(path, device=None):
    """Stage 1. No path means no routing: the fast mode answers every request."""
    if not path:
        return None
    return DecisionHead.load(path, device=device) if device else DecisionHead.load(path)


def feature_layers_for(head):
    """Stage 2. Which backbone layers the backend must expose for this head."""
    return head.metadata.feature_layers if head is not None else ()


def attach_head(head, scorer, model_id, revision=None, readout=None, device=None):
    """Stage 3. Refuse any pairing the head was not trained for, then place it.

    Returns the head, so a caller can write ``head = attach_head(...)`` and have
    the checks be impossible to skip by forgetting a line. A None head returns
    None: serving without routing is a configuration, not an error.
    """
    if head is None:
        return None
    head.metadata.check_serving(model_id, revision=revision, readout=readout)
    expected = head.metadata.hidden_size
    actual = getattr(scorer, "hidden_size", None)
    if actual is not None and expected != actual:
        raise ValueError(
            f"this head reads a {expected}-dimensional hidden state and the backend "
            f"produces {actual}. They are different backbones, or the same one at a "
            f"different revision; retrain the head or serve the model it was trained on"
        )
    if device is not None:
        head.to(device)
    return head
