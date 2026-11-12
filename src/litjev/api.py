"""Jev-compatible standard endpoint and a separate LitJev diagnostics extension."""

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from litjev.core.block import RoutingPolicy
from litjev.core.pipeline_utils import DecisionResponse
from litjev.core.prompts_utils import state_text
from litjev.data.schema import SystemOneRequest
from litjev.vision import MAX_BASE64_LENGTH, VisualState, decode_image


class DebugRequest(SystemOneRequest):
    image: str | None = Field(default=None, max_length=MAX_BASE64_LENGTH)


class Routing(BaseModel):
    """LitJev extension: when the decision head expects thinking to pay off, think."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    lambda_: float = Field(default=0.0, alias="lambda")
    budget: int = Field(default=512, ge=1, le=8192)

    def to_policy(self):
        return RoutingPolicy(self.lambda_, self.budget)


class SystemTwoRequest(SystemOneRequest):
    routing: Routing | None = None


class SystemTwoDebugRequest(SystemTwoRequest):
    image: str | None = Field(default=None, max_length=MAX_BASE64_LENGTH)


def create_app(engine_factory, preload=False):
    """`preload` builds the engine here, in the caller's thread, instead of lazily.

    Lazily is right on a laptop: the playground serves without weights and the
    first request pays for them. It is wrong for an in-process SGLang engine,
    which installs signal handlers at startup and so must be built on the main
    thread -- a FastAPI endpoint defined with `def` runs in a worker thread, so
    the first request died with `signal only works in main thread of the main
    interpreter` and every request after it with the same.
    """
    app = FastAPI(title="LitJev System One")
    get_engine = lru_cache(maxsize=1)(engine_factory)
    load_lock = Lock()
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    @app.get("/routing", include_in_schema=False)
    def routing_playground():
        """The routing page, and the home page.

        It is the one that exercises the decision: the same questions at several
        prices, with the escalation and what it cost. `/routing` stays as an
        alias so links written before this still land.
        """
        return FileResponse(static_dir / "routing.html")

    @app.get("/systemone", include_in_schema=False)
    def playground():
        """The fast mode alone: one request, no price, no escalation."""
        return FileResponse(static_dir / "index.html")

    @app.get("/example", include_in_schema=False)
    def example():
        return FileResponse(static_dir / "example.json")

    def evaluate(request, image=None, routing=None):
        try:
            started = perf_counter()
            schema = request.to_schema()
            state = request.state
            if image is not None:
                state = VisualState(state_text(state), decode_image(image))
            # Validate schemas and images before loading weights.
            with load_lock:
                engine = get_engine()
            if request.model not in {"litjev", engine.model_id}:
                raise ValueError(
                    "Requested model is not loaded; use 'litjev' or the loaded model ID"
                )
            ready = perf_counter()
            evaluation = (
                engine.evaluate(state, schema)
                if routing is None
                else engine.evaluate(state, schema, routing)
            )
            finished = perf_counter()
            result = asdict(evaluation)
            # The engine fills `timing` with its own per-stage breakdown, so this
            # updates it rather than replacing it -- assigning a fresh dict here
            # is what used to drop the stages on the floor.
            result["diagnostics"].setdefault("timing", {}).update({
                "model_setup_seconds": ready - started,
                "decision_seconds": finished - ready,
                "total_seconds": finished - started,
            })
            return result
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": get_engine.cache_info().currsize > 0}

    @app.post("/v1/systemone", response_model=DecisionResponse)
    def systemone(request: SystemOneRequest):
        return evaluate(request)["result"]

    @app.post("/v1/systemone/debug")
    def systemone_debug(request: DebugRequest):
        return evaluate(request, request.image)

    def routing_of(request):
        # Without a routing block the server's default policy applies.
        return request.routing.to_policy() if request.routing is not None else None

    @app.post("/v1/systemtwo", response_model=DecisionResponse)
    def systemtwo(request: SystemTwoRequest):
        return evaluate(request, routing=routing_of(request))["result"]

    @app.post("/v1/systemtwo/debug")
    def systemtwo_debug(request: SystemTwoDebugRequest):
        return evaluate(request, request.image, routing_of(request))

    if preload:
        get_engine()
    return app
