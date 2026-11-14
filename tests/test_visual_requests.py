"""Requests that carry an image, a prefix, or their own decoding for the reasoning.

Three things a request may add to a text state, none of which needs an engine to be checked:
where the image goes in the prompt (before the text, never after the answer boundary), where
an unchanging prefix goes (first, so consecutive prompts share their beginning), and that a
request's sampling reaches the reasoning call of its own thread and nothing else.
"""
import base64
import io
import json
import threading
from http.server import HTTPServer

import numpy as np
import pytest
from PIL import Image

from hde.core.prompts import PrefixedState, build_messages
from inference.client import DecisionClient
from inference.decision_service import (
    IMAGE_DIGEST_CHARS,
    DecisionService,
    _handler,
    _image_digest,
    _state_of,
    _tagged,
)
from inference.sglang_backend import (
    ANSWER_BOUNDARY,
    GREEDY_TEMPERATURE,
    QWEN_THINK_SAMPLING,
    THINK_SAMPLING,
    VISION_PLACEHOLDERS,
    EncodedVisualState,
    _check_image_rendered,
    _check_rendered,
    _image_of,
    _messages_for,
    _sampling,
    _text_of,
    clean_sampling,
)

BODY = "Which one?\nA. left\nB. right"


def encoded_image(kind="PNG", size=(8, 6)):
    """A small image as base64, the way a request carries one."""
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 30, 60)).save(buffer, format=kind)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class RecordingScorer:
    """A StepScorer that answers without a model and keeps what it was given."""

    def __init__(self):
        self.states = []
        self.sampling_seen = []
        self.provider = type("P", (), {"hidden_size": 4})()

    def fast(self, state, actions):
        self.states.append(state)
        return np.full(len(actions), 1.0 / len(actions)), np.zeros(4)

    def slow(self, state, actions, budget):
        self.states.append(state)
        self.sampling_seen.append(THINK_SAMPLING.get())
        return np.full(len(actions), 1.0 / len(actions)), 7, "thought</think>"

    def solve(self, state, instructions, budget):
        self.states.append(state)
        return "thought</think>\nB", 5, False


# ------------------------------------------------------------------ decoding of the reasoning
def test_sampling_keys_are_checked_and_made_numbers():
    assert clean_sampling(None) is None and clean_sampling({}) is None
    cleaned = clean_sampling({"temperature": "0.6", "top_k": 20.0})
    assert cleaned == {"temperature": 0.6, "top_k": 20} and isinstance(cleaned["top_k"], int)
    with pytest.raises(ValueError, match="unknown sampling keys"):
        clean_sampling({"temprature": 0.6})


def test_a_candidate_read_is_greedy_whatever_the_request_asked_for():
    token = THINK_SAMPLING.set(QWEN_THINK_SAMPLING)
    try:
        assert _sampling(1) == {"max_new_tokens": 1, "temperature": GREEDY_TEMPERATURE}
        assert _sampling(256) == {"max_new_tokens": 256, **QWEN_THINK_SAMPLING}
    finally:
        THINK_SAMPLING.reset(token)


def test_the_reasoning_is_greedy_unless_something_says_otherwise():
    assert THINK_SAMPLING.get() is None
    assert _sampling(256) == {"max_new_tokens": 256, "temperature": GREEDY_TEMPERATURE}


def test_a_request_s_sampling_reaches_its_own_slow_call_and_is_gone_after():
    scorer = RecordingScorer()
    service = DecisionService(scorer)
    answer = service.slow("state", ["a", "b"], 64, sampling={"temperature": 0.6})
    assert scorer.sampling_seen == [{"temperature": 0.6}]
    assert THINK_SAMPLING.get() is None
    assert answer["truncated"] is False
    with pytest.raises(ValueError):
        service.slow("state", ["a", "b"], 64, sampling={"nope": 1})
    assert THINK_SAMPLING.get() is None


# ------------------------------------------------------------------ the prefix
def test_a_prefix_goes_first_and_the_changing_state_last():
    plain = build_messages("the state", BODY)[-1]["content"]
    assert plain == "the state\n\n" + BODY
    prefixed = build_messages(PrefixedState("the state", "what stays"), BODY)[-1]["content"]
    assert prefixed == "what stays\n\n" + BODY + "\n\nthe state"


def test_two_requests_with_one_prefix_share_everything_up_to_the_state():
    first = build_messages(PrefixedState("state one", "what stays"), BODY)[-1]["content"]
    second = build_messages(PrefixedState("state two", "what stays"), BODY)[-1]["content"]
    shared = len("what stays\n\n" + BODY + "\n\n")
    assert first[:shared] == second[:shared] and first[shared:] != second[shared:]


# ------------------------------------------------------------------ the image
def test_the_state_of_a_request_is_text_prefixed_or_visual():
    assert _state_of({"state": "s"}) == "s"
    prefixed = _state_of({"state": "s", "prefix": "p"})
    assert isinstance(prefixed, PrefixedState) and prefixed.prefix == "p"
    visual = _state_of({"state": "s", "image": encoded_image("PNG")})
    assert isinstance(visual, EncodedVisualState) and visual.text == "s"
    assert visual.data_url.startswith("data:image/png;base64,")
    assert _state_of({"state": "s", "image": encoded_image("JPEG")}).data_url.startswith("data:image/jpeg;base64,")
    url = "data:image/png;base64," + encoded_image("PNG")
    assert _state_of({"state": "s", "image": url}).data_url == url
    both = _state_of({"state": "s", "prefix": "p", "image": encoded_image()})
    assert isinstance(both.text, PrefixedState) and both.text.prefix == "p"


def test_an_image_that_is_not_one_is_refused():
    with pytest.raises(ValueError):
        _state_of({"state": "s", "image": base64.b64encode(b"not an image").decode()})
    with pytest.raises(ValueError):
        _state_of({"state": 3, "prefix": "p"})


def test_the_image_is_first_in_the_user_turn_and_the_text_is_untouched():
    state = EncodedVisualState("the state", "data:image/png;base64,AAAA")
    assert _image_of(state) == "data:image/png;base64,AAAA" and _text_of(state) == "the state"
    assert _image_of("the state") is None and _text_of("the state") == "the state"
    parts = _messages_for(state, BODY)[-1]["content"]
    assert parts == [{"type": "image"}, {"type": "text", "text": "the state\n\n" + BODY}]
    assert _messages_for("the state", BODY) == build_messages("the state", BODY)


def test_a_render_that_lost_the_image_is_an_error_not_a_text_forward():
    state = EncodedVisualState("the state", "data:image/png;base64,AAAA")
    with pytest.raises(RuntimeError, match="no vision placeholder"):
        _check_image_rendered("a prompt with no image in it", state)
    _check_image_rendered("a prompt with no image in it", "a text state")      # nothing to lose
    good = "".join(VISION_PLACEHOLDERS) + " the state " + ANSWER_BOUNDARY
    _check_rendered(good, state)
    with pytest.raises(RuntimeError, match="after the answer boundary"):
        _check_rendered("the state " + ANSWER_BOUNDARY + "".join(VISION_PLACEHOLDERS), state)


def test_an_answer_says_whether_it_saw_an_image_and_which():
    image = encoded_image()
    tagged = _tagged({}, {"state": "s", "image": image})
    assert tagged["visual"] is True and len(tagged["image_digest"]) == IMAGE_DIGEST_CHARS
    assert tagged["image_digest"] == _image_digest({"image": image})
    assert _tagged({}, {"state": "s"}) == {"visual": False, "image_digest": None}


def test_all_three_modes_take_an_image_over_a_socket():
    scorer = RecordingScorer()
    server = HTTPServer(("127.0.0.1", 0), _handler(DecisionService(scorer)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = DecisionClient(f"http://127.0.0.1:{server.server_port}")
        image = encoded_image()
        fast = client.fast("s", ["a", "b"], image=image)
        slow = client.slow("s", ["a", "b"], 32, image=image)
        assert fast["visual"] and slow["visual"] and fast["image_digest"] == slow["image_digest"]
        solved = client._call("/solve", {"state": "s", "instructions": "say", "budget": 16, "image": image})
        assert solved["text"].endswith("B")
        assert all(isinstance(state, EncodedVisualState) for state in scorer.states) and len(scorer.states) == 3
        bad = json.dumps({"state": "s", "actions": ["a"], "image": "AAAA"})
        with pytest.raises(Exception):
            client._call("/fast", json.loads(bad))
    finally:
        server.shutdown()
