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
from hde.core.block import escalation_gain, fast_confidence
from hde.core.block_utils import build_features
from hde.serving import attach_head, feature_layers_for, load_head

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

    def slow(self, state, actions, budget, sampling=None):
        """The slow distribution: reason for up to `budget` tokens, then read the candidates.

        Args:
            state: the state, text or visual.
            actions: the candidate actions.
            budget: the most tokens the reasoning may take.
            sampling: decoding parameters for this request's reasoning (see
                `sglang_backend.SAMPLING_KEYS`); None uses the service default.

        Returns:
            A dict with `probabilities` (one per action), `generated_tokens`, `truncated`
            (the reasoning did not close within the budget, so the read followed a forced
            close), `reasoning` (the text) and `seconds`.

        Raises:
            ValueError: an unknown key in `sampling`.
        """
        started = time.perf_counter()
        from inference.sglang_backend import THINK_SAMPLING, clean_sampling
        token = THINK_SAMPLING.set(clean_sampling(sampling))
        try:
            probabilities, generated, reasoning = self.scorer.slow(state, actions, budget)
        finally:
            THINK_SAMPLING.reset(token)
        return {"probabilities": [float(p) for p in probabilities],
                "generated_tokens": int(generated),
                # The thinking block never closed within the budget, so the read
                # followed a forced close (see sglang_backend._generate_then_read).
                "truncated": "</think>" not in (reasoning or ""),
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


PNG_SIGNATURE = b"\x89PNG"
IMAGE_HEAD_CHARS = 16          # enough base64 to decode the file signature
IMAGE_DIGEST_CHARS = 16        # hex characters of the image digest carried in an answer


def _state_of(body):
    """The request's state: visual when the body carries an `image`.

    `image` is a base64 PNG/JPEG or a data URL. It is validated once with
    `hde.vision.decode_image` (<= 1 MP, <= 4 MB, PNG/JPEG only) and the original
    string is passed down as a data URL -- SGLang's `image_data` takes one, so the
    image is never decoded and re-encoded on the request path.

    Args:
        body: the request body: `state` (text), and optionally `prefix` (the unchanging
            part of the state, see `hde.core.prompts.PrefixedState`) and `image`.

    Returns:
        The text state (a `PrefixedState` when a prefix was given), or an
        `EncodedVisualState` of it when an image was given.

    Raises:
        KeyError: no `state` in the body.
        ValueError: the prefix or the state is not text, or the image is not a valid
            PNG/JPEG within the size limits.
    """
    image = body.get("image")
    text = body["state"]
    if body.get("prefix"):
        from hde.core.prompts import PrefixedState

        if not isinstance(text, str) or not isinstance(body["prefix"], str):
            raise ValueError("prefix needs a text state")
        text = PrefixedState(text, body["prefix"])
    if not image:
        return text
    from hde.vision import decode_image
    from inference.sglang_backend import EncodedVisualState

    decode_image(image)   # validation only; raises ValueError -> HTTP 400
    if not image.startswith("data:"):
        import base64

        head = base64.b64decode(image[:IMAGE_HEAD_CHARS] + "=" * (-len(image[:IMAGE_HEAD_CHARS]) % 4))
        kind = "png" if head.startswith(PNG_SIGNATURE) else "jpeg"
        image = f"data:image/{kind};base64,{image}"
    return EncodedVisualState(text, image)


def _image_digest(body):
    """A short digest of the request image, so a caller can check /slow saw /fast's image.

    Args:
        body: the request body.

    Returns:
        The first `IMAGE_DIGEST_CHARS` hex characters of the SHA-256 of the image string as
        it was sent, or None for a request without an image.
    """
    image = body.get("image")
    if not image:
        return None
    return hashlib.sha256(image.encode()).hexdigest()[:IMAGE_DIGEST_CHARS]


def _tagged(answer, body):
    """Say in an answer whether the request was visual, and which image it carried.

    Args:
        answer: the answer dict; `visual` and `image_digest` are added to it.
        body: the request body.

    Returns:
        The same `answer`.
    """
    answer["visual"] = bool(body.get("image"))
    answer["image_digest"] = _image_digest(body)
    return answer


def request_handler(service):
    """The HTTP request handler class that serves a service's routes.

    Args:
        service: the object the routes call: `fast`, `slow`, `solve`, `health`.

    Returns:
        A `BaseHTTPRequestHandler` subclass bound to `service`. A caller that adds a route or
        reads an extra request field subclasses it.
    """
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            """Quiet: the interesting timings are the ones the caller records."""

        # The headers and the body used to leave as two small TCP segments; with Nagle's
        # algorithm the second waits for the client's delayed ACK of the first -- 40 ms on every
        # answer, as much as a small model's whole forward pass. One buffered write, no delay.
        wbufsize = 1 << 16

        def setup(self):
            """Open the connection with Nagle's algorithm off (see `wbufsize`)."""
            super().setup()
            import socket
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def _send(self, code, body):
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()

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
                    self._send(200, _tagged(service.fast(
                        _state_of(body), body["actions"],
                        want_hidden=bool(body.get("want_hidden", False))), body))
                elif route == "/slow":
                    self._send(200, _tagged(service.slow(_state_of(body), body["actions"],
                                                         int(body.get("budget", 256)),
                                                         sampling=body.get("sampling")), body))
                elif route == "/solve":
                    # the free-form mode sees the image too; it takes no prefix
                    self._send(200, service.solve(_state_of({"state": body["state"], "image": body.get("image")}),
                                                  body.get("instructions", ""),
                                                  int(body.get("budget", 256))))
                else:
                    self._send(404, {"error": f"no such path {self.path}"})
            except KeyError as problem:
                self._send(400, {"error": f"missing field {problem}"})
            except Exception as problem:  # noqa: BLE001  one bad step must not end the run
                self._send(500, {"error": repr(problem)})

    return Handler


_handler = request_handler      # the name this had before it was part of the interface


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
                  instructions=None, mem_fraction_static=0.8, device="auto",
                  attention_backend=None, torchao_config=None, extra_engine_kwargs=None,
                  head_model_id=None, step_scorer=None):
    """The service, on top of whichever SGLang the caller points it at.

    `sglang_url` uses a server that is already running, which is the usual case
    on a cluster: one job holds the GPU and this sits beside it. Without one an
    engine is started in this process.

    `step_scorer` is the class that turns a backend into the scorer of one step; None uses
    `StepScorer`. A caller that reads the fast distribution another way passes a subclass.
    """
    from slime_hdm import StepScorer

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

        # Some GPU generations only run Qwen3.5's hybrid GDN under the triton or trtllm_mha
        # attention backend; a head fitted under another backend sees slightly different
        # kernels, as with any backend change.
        engine_kwargs = {} if not attention_backend else {"attention_backend": attention_backend}
        if torchao_config:      # online weight quantization at load, e.g. int4wo-128
            engine_kwargs["torchao_config"] = torchao_config
        engine_kwargs.update(extra_engine_kwargs or {})   # e.g. tp_size, quantization, kv_cache_dtype
        provider = SGLangScorer.load(SGLangSettings(
            model_id=model_id, revision=revision, feature_layers=layers,
            batch_questions=False, mem_fraction_static=mem_fraction_static,
            engine_kwargs=engine_kwargs))
        # Rebuilt around a thread-safe engine rather than mutated, so every
        # per-thread clone below gets the same one.
        provider = type(provider)(ConcurrentEngine(provider.engine), provider.tokenizer,
                                  provider.settings, provider.hidden_size)

    scorer_class = step_scorer or StepScorer

    def wrap(backend):
        return scorer_class(backend) if instructions is None else scorer_class(backend, instructions)

    def per_thread():
        """A scorer for this thread: fresh tokenizer, the same engine."""
        from transformers import AutoTokenizer

        clone = type(provider)(provider.engine,
                               AutoTokenizer.from_pretrained(model_id, revision=revision),
                               provider.settings, provider.hidden_size)
        return wrap(clone)

    # The stage this path did not have: a head served against another model, or
    # under a readout whose numbers mean something else, used to raise nothing.
    # head_model_id: serve a head against weights other than those it was trained on (e.g. a
    # quantized copy). Only the identity check uses it; the weights still come from model_id.
    if head_model_id and head_model_id != model_id:
        print(f"WARNING decision head identity check overridden: head trained for {head_model_id}, "
              f"serving {model_id}", flush=True)
    head = attach_head(head, provider, head_model_id or model_id, revision)
    return DecisionService(wrap(provider), head, per_thread=per_thread)


def build_parser():
    """The command line of the service.

    Returns:
        The argument parser; a caller that serves an extended service adds its own options
        to it before parsing.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--head", help="omit to serve the two modes without a router")
    parser.add_argument("--sglang-url", help="an SGLang server to use instead of starting one")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--head-model-id", default=None,
                        help="model id the head was trained for, when serving it on other weights "
                             "(overrides the identity check; the head may be miscalibrated)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--head-device", default="auto",
                        help="auto puts the head on the GPU the backbone is on")
    parser.add_argument("--attention-backend", default=None,
                        help="SGLang attention backend, e.g. triton")
    parser.add_argument("--engine-kwargs", default=None,
                        help='extra sgl.Engine arguments as JSON, e.g. {"tp_size": 2}')
    parser.add_argument("--torchao-config", default=None,
                        help="SGLang online quantization, e.g. int4wo-128 (INT4 weights)")
    return parser


def service_from(arguments, step_scorer=None):
    """Build the service the parsed command line describes.

    Args:
        arguments: the namespace `build_parser().parse_args()` returns.
        step_scorer: the step scorer class (see `build_service`); None uses `StepScorer`.

    Returns:
        The `DecisionService`.
    """
    return build_service(arguments.model, arguments.head,
                         sglang_url=arguments.sglang_url, revision=arguments.revision,
                         mem_fraction_static=arguments.mem_fraction_static,
                         device=arguments.head_device,
                         attention_backend=arguments.attention_backend,
                         torchao_config=arguments.torchao_config,
                         extra_engine_kwargs=json.loads(arguments.engine_kwargs)
                         if arguments.engine_kwargs else None,
                         head_model_id=arguments.head_model_id, step_scorer=step_scorer)


def serve(service, host, port, handler=None):
    """Serve a service over HTTP until the process is stopped.

    Args:
        service: the object the routes call: `fast`, `slow`, `solve`, `health`.
        host: the address to listen on.
        port: the TCP port to listen on.
        handler: a function from a service to its request handler class; None uses the
            routes of this module.
    """
    print(f"decision service on {host}:{port}: {service.health()}", flush=True)
    ThreadingHTTPServer((host, port), (handler or request_handler)(service)).serve_forever()


def main():
    """Parse the command line, build the service and serve it."""
    arguments = build_parser().parse_args()
    serve(service_from(arguments), arguments.host, arguments.port)


if __name__ == "__main__":
    main()
