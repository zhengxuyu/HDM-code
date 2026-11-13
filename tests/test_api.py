from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from litjev.api import create_app
from litjev.core.pipeline import SchemaDecisionEngine
from litjev.core.pipeline_utils import RawFieldScores


class FakeProvider:
    def score(self, state, schema):
        return tuple(RawFieldScores(k, np.zeros(len(v.choices)), 42) for k, v in schema.items())


class FakeEngine(SchemaDecisionEngine):
    def __init__(self):
        # Coded: FakeProvider implements `score` and not `score_options`, which is
        # what these tests are about -- the API, not the readout.
        super().__init__(FakeProvider(), model_id="fake", readout="coded")


def test_playground_serves_without_loading_model():
    def forbidden_load():
        raise AssertionError("Opening the UI must not load weights")

    client = TestClient(create_app(forbidden_load))
    assert client.get("/").status_code == 200
    assert "schema-input" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert not client.get("/health").json()["model_loaded"]


def test_legacy_endpoints_are_not_silently_accepted():
    client = TestClient(create_app(FakeEngine))
    for endpoint in ["/v1/batch-mcq", "/v1/calibrated-schema"]:
        assert client.post(endpoint, json={}).status_code == 404


class ThinkingProvider(FakeProvider):
    """`FakeProvider` plus a slow mode, so a routed request can actually escalate."""

    def think(self, state, schema, names, budget):
        return tuple(
            RawFieldScores(name, np.zeros(len(schema[name].choices)), 42,
                           generated_tokens=7, reasoning="...")
            for name in names
        )


class ThinkingEngine(SchemaDecisionEngine):
    def __init__(self):
        super().__init__(ThinkingProvider(), model_id="fake", readout="coded")


def _routing_page():
    """The markup and the script together: the page's data path lives in the JS."""
    static = Path(__file__).resolve().parents[1] / "src/litjev/static"
    return (static / "routing.html").read_text() + (static / "routing.js").read_text()


def test_the_routing_page_serves_without_loading_weights():
    def forbidden_load():
        raise AssertionError("Opening the UI must not load weights")

    client = TestClient(create_app(forbidden_load))
    page = client.get("/routing")
    assert page.status_code == 200
    assert "/v1/systemtwo/debug" in page.text, "the page must call the routed endpoint"
    assert "/v1/systemone" not in page.text, "`/` is the System One page; this one is not"


def test_the_routing_page_reads_fields_the_api_actually_returns():
    """The contract between the page and the response, which nothing else checks.

    Every path below is one the page indexes. A rename on either side leaves the
    page showing blanks and no test failing, which is how a diagnostics field goes
    unnoticed.
    """
    client = TestClient(create_app(ThinkingEngine))
    reply = client.post("/v1/systemtwo/debug", json={
        "model": "litjev",
        "state": "s",
        "questions": {"q": {"type": "choice", "instructions": "Pick",
                            "criteria": {"A": "1", "B": "2"}}},
        # At or below -1 everything escalates, which needs no head -- and a head
        # is what this stub has not got. A deployment without one answers 422 to
        # every other price; the page renders that `detail` in the row, which the
        # test below checks, because it is the first thing a new server does.
        "routing": {"lambda": -1.0, "budget": 64},
    })
    assert reply.status_code == 200, reply.text
    body = reply.json()

    answers = body["result"]["answers"]
    # The value sits under a key named after the type, not under `value`.
    assert answers["q"]["type"] == "choice" and "choice" in answers["q"]
    assert "answer[answer.type]" in _routing_page(), "the page must read it that way"
    assert "output_tokens" in body["result"]["usage"]

    diagnostics = body["diagnostics"]
    assert "decision_seconds" in diagnostics["timing"]
    assert isinstance(diagnostics["routing"]["escalated"], list)
    assert "lambda" in diagnostics["routing"]

    field = diagnostics["fields"]["q"]
    assert field["system"] in ("one", "two")
    assert "concentration" in field

    page = _routing_page()
    for path in ('result.answers', 'result.usage', 'diagnostics.timing',
                 'diagnostics.routing', 'diagnostics.fields'):
        head, tail = path.split(".")
        assert tail in page, f"the page stopped reading {path}"


def test_a_server_without_a_head_says_so_in_a_field_the_page_renders():
    client = TestClient(create_app(FakeEngine))
    reply = client.post("/v1/systemtwo/debug", json={
        "model": "litjev", "state": "s",
        "questions": {"q": {"type": "choice", "instructions": "Pick",
                            "criteria": {"A": "1", "B": "2"}}},
        "routing": {"lambda": 0.0, "budget": 64},
    })
    assert reply.status_code == 422
    assert "needs a decision head" in reply.json()["detail"]
    assert "payload.detail" in _routing_page(), "the page must show the reason, not a blank row"


def test_preload_builds_the_engine_in_the_callers_thread():
    """An in-process SGLang engine installs signal handlers, which only work on
    the main thread. Built lazily it is built inside a request handler, and a
    `def` endpoint runs in a worker thread, so every request failed with
    `signal only works in main thread of the main interpreter`."""
    import threading

    built_on = []

    def factory():
        built_on.append(threading.current_thread().name)
        return ThinkingEngine()

    app = create_app(factory, preload=True)
    assert built_on == [threading.current_thread().name], \
        "preload must build the engine here, not on whichever thread asks first"

    client = TestClient(app)
    assert client.get("/health").json()["model_loaded"] is True
    assert len(built_on) == 1, "the preloaded engine is reused, not rebuilt"


def test_without_preload_the_playground_still_serves_before_any_weights():
    def forbidden_load():
        raise AssertionError("Opening the UI must not load weights")

    client = TestClient(create_app(forbidden_load))
    assert client.get("/routing").status_code == 200
    assert client.get("/health").json()["model_loaded"] is False


def test_the_script_only_reaches_for_ids_the_markup_defines():
    """A renamed id leaves the page silently half-dead: the handler throws on the
    first null and everything after it never runs. Cheap to check, invisible
    otherwise."""
    import re

    static = Path(__file__).resolve().parents[1] / "src/litjev/static"
    script = (static / "routing.js").read_text()
    markup = (static / "routing.html").read_text()
    wanted = sorted({m.group(1) for m in re.finditer(r'el\("([a-z-]+)"\)', script)})
    assert wanted, "the check itself must find something to check"
    missing = [name for name in wanted if f'id="{name}"' not in markup]
    assert not missing, f"routing.js reads ids the markup does not define: {missing}"


def test_the_slider_never_sends_a_request_by_itself():
    """Only the button sends. `input` fires on every pixel of a drag, so binding
    it would queue dozens of 27B calls; `change` fires on release, so binding
    that makes every adjustment cost a round of inference before the next one
    can be tried."""
    script = (Path(__file__).resolve().parents[1]
              / "src/litjev/static/routing.js").read_text()
    assert 'addEventListener("input", () => {' in script, "the readout still follows the drag"
    for event in ("input", "change"):
        assert f'el("lambda").addEventListener("{event}", run)' not in script
    assert 'el("routing-form").addEventListener("submit"' in script, "the button is the trigger"


def test_the_slider_reaches_past_where_the_gain_commonly_sits():
    """The value of thinking is a log ratio, not a difference of probabilities, so
    it has no upper bound: a question the block rates three times more likely to be
    answered by the slow mode carries v = 1.1. A dial that stopped at 1 escalated
    every such question at every price it could reach, and looked inert."""
    import re

    markup = (Path(__file__).resolve().parents[1]
              / "src/litjev/static/routing.html").read_text()
    slider = re.search(r'<input id="lambda"[^>]*>', markup, re.S).group(0)
    maximum = float(re.search(r'max="(-?[0-9.]+)"', slider).group(1))
    assert 'min="-1"' in slider, slider
    assert maximum > 1.0, slider


def test_the_engine_reports_a_stage_breakdown_and_the_api_keeps_it():
    """`decision_seconds` alone says an escalation was expensive without saying
    which half spent it. The API adds its own wall-clock around the engine's
    numbers, and must update that dict rather than replace it."""
    client = TestClient(create_app(ThinkingEngine))
    reply = client.post("/v1/systemtwo/debug", json={
        "model": "litjev", "state": "s",
        "questions": {"q": {"type": "choice", "instructions": "Pick",
                            "criteria": {"A": "1", "B": "2"}}},
        "routing": {"lambda": -1.0, "budget": 64},
    })
    timing = reply.json()["diagnostics"]["timing"]
    for key in ("policy_seconds", "schema_seconds", "fast_seconds", "slow_seconds",
                "decision_seconds", "model_setup_seconds", "total_seconds"):
        assert key in timing, f"{key} missing from {sorted(timing)}"
        assert timing[key] >= 0

    stages = sum(timing[k] for k in
                 ("policy_seconds", "schema_seconds", "fast_seconds", "slow_seconds"))
    assert stages <= timing["decision_seconds"] + 1e-6, \
        "the stages run inside the decision, so they cannot outlast it"
    assert timing["decision_seconds"] <= timing["total_seconds"] + 1e-6

    page = _routing_page()
    assert "roundtrip - timing.total_seconds" in page, \
        "transport is the only part neither clock covers; the page must derive it"


def test_the_routing_page_is_in_one_language():
    """`/` is Chinese and this page is English; a half-translated page is what a
    string-by-string edit leaves behind."""
    import re

    static = Path(__file__).resolve().parents[1] / "src/litjev/static"
    for name in ("routing.html", "routing.js", "routing.css"):
        text = (static / name).read_text()
        han = re.findall(r"[一-鿿]+", text)
        assert not han, f"{name} still has {han[:3]}"
    assert 'lang="en"' in (static / "routing.html").read_text()
