"""The decision service, with no engine and no network.

The service exists because sending hidden states is expensive, so the thing to
hold it to is that it sends a decision and not a hidden state: whatever the head
reads has to stay on the serving side. The rest is what a caller needs to be
able to rely on when a step goes wrong halfway through an episode.
"""
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer

import numpy as np
import pytest

from inference.client import DecisionClient
from inference.decision_service import ConcurrentEngine, DecisionService, _handler, head_device


class FakeScorer:
    """A StepScorer that answers without a model."""

    def __init__(self, hidden=None, width=8):
        self.hidden = np.zeros(width) if hidden is None else hidden
        self.calls = []
        self.provider = type("P", (), {"hidden_size": width})()

    def fast(self, state, actions):
        self.calls.append(("fast", state, tuple(actions)))
        probabilities = np.full(len(actions), 1.0 / len(actions))
        return probabilities, self.hidden

    def slow(self, state, actions, budget):
        self.calls.append(("slow", state, tuple(actions), budget))
        probabilities = np.zeros(len(actions))
        probabilities[-1] = 1.0
        return probabilities, 123, "the second option names the German one"


class FakeHead:
    """Carries the identity fields a real head's metadata carries.

    `/health` reports which head is served, not only that one is, so a stub with
    `feature_layers` alone would let the payload be tested against an object
    that cannot distinguish two heads -- which is the failure being fixed.
    """

    def __init__(self, pca_dim=0, hidden_width=256, weight=1.0):
        self.metadata = type("M", (), {
            "feature_layers": (-1,), "pca_dim": pca_dim,
            "hidden_width": hidden_width, "readout": "coded",
            "feature_format": "readout_hidden_plus_stats_v1"})()
        self.weight = weight

    def buffers(self):
        import torch

        yield torch.zeros(1)

    def state_dict(self):
        import torch

        return {"layer.weight": torch.full((2, 2), self.weight)}

    def predict(self, features):
        # 0: both right, 1: fast right slow wrong, 2: fast wrong slow right, 3: both wrong
        return np.array([[0.5, 0.1, 0.3, 0.1]])


def test_the_response_carries_a_decision_and_no_hidden_state():
    """The whole point: 48 MB of features reduce to two numbers before the wire."""
    service = DecisionService(FakeScorer(hidden=np.arange(8.0)), FakeHead())
    answer = service.fast("state", ["a", "b"])
    body = json.dumps(answer)
    assert set(answer) == {"probabilities", "gain", "confidence", "seconds"}
    for value in np.arange(8.0)[1:]:
        assert str(value) not in body
    assert len(body) < 400


def test_gain_and_confidence_are_the_head_read_the_paper_s_way():
    service = DecisionService(FakeScorer(), FakeHead())
    answer = service.fast("state", ["a", "b"])
    # The paper's way is a log ratio now, not a difference: P(slow ok) against
    # P(fast ok), because the reward discounts the wait instead of subtracting it.
    assert answer["gain"] == pytest.approx(math.log(0.8 / 0.6))
    assert answer["confidence"] == pytest.approx(0.5 + 0.1)  # o0 + o1


def test_generate_is_callable_from_several_threads_at_once():
    """The failure this exists for: `this event loop is already running`.

    The real engine wraps an asyncio loop, so two threads calling `generate`
    raced and the second got a 500. The fake here has the same shape -- an
    `async_generate` coroutine and nothing else -- so the wrapper is what is
    being tested, not the engine.
    """
    import threading

    class FakeEngine:
        def __init__(self):
            self.seen = []
            self.lock = threading.Lock()

        async def async_generate(self, prompt, **named):
            with self.lock:
                self.seen.append(prompt)
            return {"text": prompt}

    engine = ConcurrentEngine(FakeEngine())
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(lambda i: engine.generate(f"p{i}"), range(24)))
    assert sorted(a["text"] for a in answers) == sorted(f"p{i}" for i in range(24))


def test_the_service_module_still_has_what_build_service_calls():
    """A guard against an edit removing a helper the GPU path needs.

    Nothing here can call `build_service`: it wants an engine. So the first time
    a missing helper was noticed it was a NameError on a cluster node, minutes
    into loading a 27B model. Naming the helper in a test costs nothing and
    fails on a laptop instead.
    """
    assert head_device("cpu") == "cpu"
    assert head_device("cuda:1") == "cuda:1"
    assert head_device("auto") in ("cpu", "cuda")


def test_health_names_where_the_head_runs():
    """It belongs on the backbone's GPU; a deployment should be able to check."""
    service = DecisionService(FakeScorer(), FakeHead())
    assert service.health()["head_device"] == "cpu"


def test_without_a_head_the_modes_still_serve_but_nothing_routes():
    """A service started without --head must not look like one that routes badly."""
    service = DecisionService(FakeScorer(), None)
    answer = service.fast("state", ["a", "b"])
    assert answer["gain"] is None and answer["confidence"] is None
    assert service.health()["head"] is False


def test_slow_reports_the_reasoning_and_what_it_cost():
    service = DecisionService(FakeScorer(), FakeHead())
    answer = service.slow("state", ["a", "b"], 64)
    assert answer["reasoning"] == "the second option names the German one"
    assert answer["generated_tokens"] == 123
    assert answer["probabilities"][-1] == 1.0
    assert answer["seconds"] >= 0.0


def serve(service):
    server = HTTPServer(("127.0.0.1", 0), _handler(service))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, DecisionClient(f"http://127.0.0.1:{server.server_port}", timeout=10)


def test_client_and_service_agree_over_a_socket():
    scorer = FakeScorer()
    server, client = serve(DecisionService(scorer, FakeHead()))
    try:
        assert client.health()["head"] is True
        answer = client.fast("the page", ["click('a')", "click('b')"])
        assert answer["gain"] == pytest.approx(math.log(0.8 / 0.6))
        assert client.slow("the page", ["click('a')"], 32)["generated_tokens"] == 123
        assert scorer.calls[0][1] == "the page"
        assert scorer.calls[-1][3] == 32
    finally:
        server.shutdown()


def test_a_step_that_fails_is_a_500_and_not_a_closed_connection():
    """An episode has many steps; one bad one must not kill the server."""

    class Broken(FakeScorer):
        def fast(self, state, actions):
            raise ValueError("no candidates are single tokens")

    server, client = serve(DecisionService(Broken(), FakeHead()))
    try:
        with pytest.raises(RuntimeError, match="single tokens"):
            client.fast("state", ["a"])
        assert client.health()["head"] is True  # still up
    finally:
        server.shutdown()


def test_an_unreachable_service_says_how_to_start_one():
    client = DecisionClient("http://127.0.0.1:1", timeout=2)
    with pytest.raises(RuntimeError, match="decision_service"):
        client.health()


def test_health_distinguishes_two_heads_a_structural_check_cannot():
    """The failure this closes, as the two cases that produced it.

    Two live services -- one serving pca 64, one pca 0 -- returned byte-identical
    `/health` payloads, so a driver pointed at the wrong arm of a two-arm
    comparison would apply one head's policy to the other's gain with nothing
    saying so. And structural fields are not enough on their own:
    `head-27b-lastlayer` and `head-27b-29734-forced-layer_-1` agree on layer, pca,
    width and readout and are different fits, which is what the digest is for.
    """
    pca = DecisionService(FakeScorer(), FakeHead(pca_dim=64, hidden_width=256)).health()
    raw = DecisionService(FakeScorer(), FakeHead(pca_dim=0, hidden_width=256)).health()
    assert pca != raw
    assert (pca["head_pca_dim"], raw["head_pca_dim"]) == (64, 0)

    # Same structure, different weights: every reported field agrees except the
    # digest, which is the case the structural fields provably cannot catch.
    one = DecisionService(FakeScorer(), FakeHead(weight=1.0)).health()
    two = DecisionService(FakeScorer(), FakeHead(weight=2.0)).health()
    structural = [k for k in one if k != "head_digest"]
    assert all(one[k] == two[k] for k in structural), "the two should differ only in weights"
    assert one["head_digest"] != two["head_digest"], (
        "two different fits with identical structure reported the same identity")


def test_the_digest_is_computed_once_and_survives_no_head():
    service = DecisionService(FakeScorer(), FakeHead())
    first = service.health()["head_digest"]
    assert service.health()["head_digest"] == first
    headless = DecisionService(FakeScorer(), None).health()
    assert headless["head_digest"] is None and headless["head_pca_dim"] is None
