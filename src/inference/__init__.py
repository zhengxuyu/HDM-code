"""SGLang-backed inference for the hybrid-thinking model.

`hde.backend.TransformersScorer` runs the two modes through transformers,
which is what every recorded number so far came from. It is slow where it
matters most: the slow-thinking mode decodes one sequence at a time, and a
reasoning step on a 27B backbone costs minutes. Anything that collects
repeatedly, reinforcement learning above all, is bounded by that.

This package serves the same two modes through SGLang and satisfies the same
interface, so `SchemaDecisionEngine` and everything above it are unchanged. It
is not specific to any benchmark.

Two things make the mapping fit rather than merely work.

The fast mode reads its answer at one position from a set of single-token
candidate codes, which is a request for the logprobs of chosen token ids and no
generation at all. The questions of one request share a state, and the paper's
shared prefill with branched readout is what SGLang's prefix cache does by
itself, so the branching needs no cache surgery here.

The decision head reads a hidden state from a chosen layer at the readout
position. SGLang can return that, with caveats serious enough to check rather
than trust: see `verify.py`.

Two scorers, one interface. `SGLangScorer` runs the engine in this process.
`SGLangServerScorer` talks to one over HTTP and receives the hidden states, which
is how `verify.py` compares the two backends from a machine that holds no GPU.

It is the wrong split for serving. The hidden states of a 900-token prompt are
48 MB of JSON, so a head on the far side of a network pays seconds per step for
features it reduces to two numbers. `decision_service` keeps the head next to
the engine and sends the decision instead; that is what a deployment, and the
browser demo, should use.

Resolved on first use, because `DecisionClient` is the whole of this package
that an environment needs and it is stdlib only. The service it talks to holds
the engine, the tokenizer and the head; an environment sandbox that imported
those to reach the client would be carrying torch and sglang to send JSON.
"""

import importlib

_EXPORTS = {
    "DecisionClient": "inference.client",
    "DecisionService": "inference.decision_service",
    "build_service": "inference.decision_service",
    "SGLangServerScorer": "inference.http_backend",
    "SGLangServerSettings": "inference.http_backend",
    "SGLangScorer": "inference.sglang_backend",
    "SGLangSettings": "inference.sglang_backend",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = ["SGLangScorer", "SGLangSettings", "SGLangServerScorer", "SGLangServerSettings",
           "DecisionService", "DecisionClient", "build_service"]
