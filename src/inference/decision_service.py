"""Put the decision head beside the engine, and send only the decision.

The head reads a hidden state, and a hidden state is not small. A 900-token
prompt on the 27B backbone answers with `(1, 900, 5120)` floats, which is 48 MB
of JSON; over a forwarded port that took 29 seconds against 0.75 for the same
request with the hidden states left out. Measured, not estimated: a demo whose
argument is a clock cannot ship its features across a network.

So the split in `http_backend` is the wrong way round for anything but a probe.
This turns it over. The engine, the tokenizer and the head all sit on the
serving side; what crosses the network is the state, the candidate actions, and
the few numbers a decision needs. The head is a few hundred kilobytes on a CPU,
so putting it next to the engine costs the engine nothing.

The service speaks the step level rather than the token level, because that is
what both callers need: `fast` returns the action distribution together with the
gain and the confidence the head predicts, and `slow` reasons first and returns
the distribution and what the reasoning cost. Whoever is driving the browser
never sees a hidden state and never needs to.
"""

import argparse
import asyncio
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from inference.client import DecisionClient  # noqa: F401  re-exported
from litjev.core.block import escalation_gain, fast_confidence
from litjev.core.block_utils import build_features
from litjev.serving import attach_head, feature_layers_for, load_head

CHOICE = "choice"


def encode_hidden(hidden):
    """One hidden state as base64 float32, or None. Compact and exact.

    JSON numbers would cost about four times the bytes and a parse per element;
    base64 of the raw buffer is the same vector with no conversion either way.
    """
    import base64

    import numpy as np

    if hidden is None:
        return None
    buffer = np.ascontiguousarray(np.asarray(hidden, dtype=np.float32))
    return {"dtype": "float32", "shape": list(buffer.shape),
            "data": base64.b64encode(buffer.tobytes()).decode()}


def decode_hidden(encoded):
    """The inverse, used by the client. None stays None."""
    import base64

    import numpy as np

    if not encoded:
        return None
    raw = base64.b64decode(encoded["data"])
    return np.frombuffer(raw, dtype=np.dtype(encoded["dtype"])).reshape(
        encoded["shape"]).copy()


class DecisionService:
    """One step of the hybrid model, with the head applied here.

    Served from several threads at once. One GPU is not a reason to answer one
    request at a time: decoding a 27B model reads 54 GB of weights from memory
    for every step it takes, and that read is the same whether it produces one
    token or thirty-two. At one request in flight the engine was measured
    turning out about 37 tokens a second, which is the batch-of-one figure.
    SGLang batches whatever is in flight, so concurrency is how that read gets
    paid for once and used many times.

    What is shared and what is not follows from that. The engine is shared,
    which is the point. The head is shared: it is a read-only forward pass under
    `no_grad`. The tokenizer is not, because a fast tokenizer's thread safety is
    not a thing to assume, and a copy per thread costs a few megabytes.
    """

    def __init__(self, scorer, head=None, per_thread=None):
        self._shared = scorer
        self._per_thread = per_thread
        self._local = threading.local()
        self.head = head
        self._digest = None

    @property
    def scorer(self):
        """This thread's scorer: its own tokenizer, the shared engine."""
        if self._per_thread is None:
            return self._shared
        mine = getattr(self._local, "scorer", None)
        if mine is None:
            mine = self._local.scorer = self._per_thread()
        return mine

    def fast(self, state, actions, want_hidden=False):
        """The fast distribution, and optionally the state the head reads.

        `want_hidden` exists for training rather than serving: CDPO learns from
        the readout hidden state at every step, and the engine cannot live in the
        browser's process -- Playwright's sync API runs an event loop on the
        calling thread and SGLang's in-process `Engine` drives its own with
        `run_until_complete`, so the two cannot share one. exps hit that at the
        first scoring call of the first episode.

        **This is not the 48 MB the service exists to avoid.** That figure is
        sglang's own server encoding a hidden state for *every prompt position*;
        this service holds the engine in-process, reads one position, and ships
        one vector -- 5128 float32 is about 20 KB, against a step that costs
        seconds. float32 rather than float16 because the head was trained in it
        and a saving of 10 KB is not worth a precision question.
        """
        started = time.perf_counter()
        probabilities, hidden = self.scorer.fast(state, actions)
        seconds = time.perf_counter() - started
        gain = confidence = None
        if self.head is not None and hidden is not None:
            features = build_features(hidden, probabilities, CHOICE)[None]
            outcome = self.head.predict(features)[0]
            gain = float(escalation_gain(outcome))
            confidence = float(fast_confidence(outcome))
        answer = {"probabilities": [float(p) for p in probabilities],
                  "gain": gain, "confidence": confidence,
                  "seconds": round(seconds, 4)}
        if want_hidden:
            answer["hidden"] = encode_hidden(hidden)
        return answer

    def slow(self, state, actions, budget):
        started = time.perf_counter()
        probabilities, generated, reasoning = self.scorer.slow(state, actions, budget)
        return {"probabilities": [float(p) for p in probabilities],
                "generated_tokens": int(generated),
                # Carried so a caller can show what the escalation bought. The
                # answer is still read from the candidate codes; nothing is
                # parsed out of this.
                "reasoning": reasoning,
                "seconds": round(time.perf_counter() - started, 4)}

    def solve(self, state, instructions, budget):
        """The slow mode answering in free form, with no candidate set at all.

        `slow` reasons and then reads the same enumerated codes, so escalating
        buys a longer thought and **not** a larger action space: on a step whose
        right action is text -- type this, choose that, send this answer -- the
        slow mode was as incapable as the fast one, and no price could route
        around it. That made a typing task unwinnable at every lambda and would
        have had the gate report "both near zero" about a venue when it was
        reporting a restriction of ours.

        So the slow mode writes its action instead of picking one. The backends
        already had `solve` for questions whose answers were never enumerated;
        this is the same call, exposed. The caller parses and validates what comes
        back -- strictly, and recording a refusal rather than falling back to a
        click, because an unparsable action is a fact about the step.
        """
        started = time.perf_counter()
        written, generated, truncated = self.scorer.solve(state, instructions, budget)
        return {"text": written,
                "generated_tokens": int(generated),
                "truncated": bool(truncated),
                "seconds": round(time.perf_counter() - started, 4)}

    def head_digest(self):
        """A digest of the served weights, computed once.

        The structural fields do not identify the artefact. `head-27b-lastlayer`
        and `head-27b-29734-forced-layer_-1` are both layer -1, pca 0, width 256,
        coded readout -- and are different fits, recording fast accuracies of
        0.65896 and 0.65029 on the same 346 questions. Only the weights separate
        them.

        Cached rather than recomputed: every driver hits `/health` at startup and
        a rehash per call would be charged to the caller's latency. Float32 on the
        CPU so the digest is a property of the head and not of the device it was
        loaded onto.
        """
        if self._digest is None and self.head is not None:
            digest = hashlib.sha256()
            for name, tensor in sorted(self.head.state_dict().items()):
                digest.update(name.encode())
                digest.update(tensor.detach().to("cpu").float().numpy().tobytes())
            self._digest = digest.hexdigest()[:16]
        return self._digest

    def health(self):
        device = None
        if self.head is not None:
            try:
                device = str(next(self.head.buffers()).device)
            except (StopIteration, AttributeError):
                # A health check reports; it does not fail because it could not
                # name the device.
                device = "unknown"
        meta = self.head.metadata if self.head is not None else None
        return {"head": self.head is not None,
                "head_layers": list(meta.feature_layers) if meta else [],
                "head_device": device,
                "hidden_size": getattr(self.scorer.provider, "hidden_size", None),
                # **Which head, not only whether there is one.** Two services, one
                # serving pca 64 and one pca 0, returned byte-identical payloads
                # here, so a driver pointed at the wrong arm applies one head's
                # policy to another head's gain and nothing says so -- the failure
                # a two-arm comparison cannot survive and cannot detect. These are
                # the fields that differ when the heads do, ending in the digest,
                # which differs whenever anything does.
                "head_pca_dim": meta.pca_dim if meta else None,
                "head_hidden_width": meta.hidden_width if meta else None,
                "head_readout": meta.readout if meta else None,
                "head_feature_format": meta.feature_format if meta else None,
                "head_digest": self.head_digest()}


def _handler(service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            """Quiet: the interesting timings are the ones the caller records."""

        def _send(self, code, body):
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", ""):
                try:
                    self._send(200, service.health())
                except Exception as problem:  # noqa: BLE001  report, do not die
                    self._send(500, {"error": repr(problem)})
            else:
                self._send(404, {"error": f"no such path {self.path}"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as problem:
                self._send(400, {"error": f"not JSON: {problem}"})
                return
            route = self.path.rstrip("/")
            try:
                if route == "/fast":
                    self._send(200, service.fast(
                        body["state"], body["actions"],
                        want_hidden=bool(body.get("want_hidden", False))))
                elif route == "/slow":
                    self._send(200, service.slow(body["state"], body["actions"],
                                                 int(body.get("budget", 256))))
                elif route == "/solve":
                    self._send(200, service.solve(body["state"],
                                                  body.get("instructions", ""),
                                                  int(body.get("budget", 256))))
                else:
                    self._send(404, {"error": f"no such path {self.path}"})
            except KeyError as problem:
                self._send(400, {"error": f"missing field {problem}"})
            except Exception as problem:  # noqa: BLE001  one bad step must not end the run
                self._send(500, {"error": repr(problem)})

    return Handler


class ConcurrentEngine:
    """`generate`, callable from many threads at once.

    `Engine.generate` is a synchronous shell around an asyncio loop, so a second
    thread entering it raises `this event loop is already running`. That is what
    a per-thread tokenizer does not fix and what made the first threaded service
    answer one request and fail the next.

    The engine's own `async_generate` is the way in. One loop runs in a thread of
    its own and every caller submits to it, which leaves the batching where it
    belongs: SGLang decides what to run together, rather than this code queueing
    requests and taking turns. Taking turns would move the waiting without
    removing it, and the reason to do any of this is that a decode step reads
    54 GB of weights whether it serves one sequence or thirty-two.
    """

    def __init__(self, engine):
        self.engine = engine
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True,
                                        name="sglang-loop")
        self._thread.start()

    def generate(self, *arguments, **named):
        submitted = asyncio.run_coroutine_threadsafe(
            self.engine.async_generate(*arguments, **named), self._loop)
        return submitted.result()

    def __getattr__(self, name):
        """Anything else the scorer asks of the engine, unchanged."""
        return getattr(self.engine, name)


def head_device(choice="auto"):
    """Where the decision head runs. The same device as the backbone by default.

    The head is part of the model: 5128 -> 256 -> 4, about 1.3M parameters, and
    it reads a hidden state the backbone has just produced. It ran on the CPU
    here only because `DecisionHead.load` defaults to one and nothing said
    otherwise.
    """
    if choice != "auto":
        return choice
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001  no torch is a CPU deployment by definition
        return "cpu"


def build_service(model_id, head_path, *, sglang_url=None, revision="main",
                  instructions=None, mem_fraction_static=0.8, device="auto"):
    """The service, on top of whichever SGLang the caller points it at.

    `sglang_url` uses a server that is already running, which is the usual case
    on a cluster: one job holds the GPU and this sits beside it. Without one an
    engine is started in this process.
    """
    from slime_htm import StepScorer

    head = load_head(head_path, device=head_device(device))
    layers = feature_layers_for(head)
    if sglang_url:
        from inference import SGLangServerScorer, SGLangServerSettings

        provider = SGLangServerScorer.load(SGLangServerSettings(
            base_url=sglang_url, model_id=model_id, revision=revision,
            feature_layers=layers))
        provider.health()
    else:
        from inference import SGLangScorer, SGLangSettings

        provider = SGLangScorer.load(SGLangSettings(
            model_id=model_id, revision=revision, feature_layers=layers,
            batch_questions=False, mem_fraction_static=mem_fraction_static))
        # Rebuilt around a thread-safe engine rather than mutated, so every
        # per-thread clone below gets the same one.
        provider = type(provider)(ConcurrentEngine(provider.engine), provider.tokenizer,
                                  provider.settings, provider.hidden_size)

    def wrap(backend):
        return StepScorer(backend) if instructions is None else StepScorer(backend, instructions)

    def per_thread():
        """A scorer for this thread: fresh tokenizer, the same engine."""
        from transformers import AutoTokenizer

        clone = type(provider)(provider.engine,
                               AutoTokenizer.from_pretrained(model_id, revision=revision),
                               provider.settings, provider.hidden_size)
        return wrap(clone)

    # The stage this path did not have: a head served against another model, or
    # under a readout whose numbers mean something else, used to raise nothing.
    head = attach_head(head, provider, model_id, revision)
    return DecisionService(wrap(provider), head, per_thread=per_thread)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--head", help="omit to serve the two modes without a router")
    parser.add_argument("--sglang-url", help="an SGLang server to use instead of starting one")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--head-device", default="auto",
                        help="auto puts the head on the GPU the backbone is on")
    arguments = parser.parse_args()

    service = build_service(arguments.model, arguments.head,
                            sglang_url=arguments.sglang_url, revision=arguments.revision,
                            mem_fraction_static=arguments.mem_fraction_static,
                            device=arguments.head_device)
    print(f"decision service on {arguments.host}:{arguments.port}: {service.health()}",
          flush=True)
    ThreadingHTTPServer((arguments.host, arguments.port), _handler(service)).serve_forever()


if __name__ == "__main__":
    main()
