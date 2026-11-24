"""FastAPI service.

GET  /health          liveness, version, engines
POST /ocr?engine=     multipart `image` → lines with confidences and boxes

Build the app with `create_app(settings, registry)`; tests pass their own
so they never depend on the environment or load models. Tracing (telemetry.py) is
on only when an OTLP endpoint is configured, or a `tracer_provider` is passed. `uvicorn
image_to_text.api.app:app` uses the module-level `app`, built from env.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.sdk.trace import TracerProvider
from starlette.concurrency import run_in_threadpool

from image_to_text import SERVICE_VERSION
from image_to_text.config import Settings
from image_to_text.imaging import decode_image
from image_to_text.ocr.base import to_ms
from image_to_text.ocr.router import EngineRegistry
from image_to_text.telemetry import (
    build_tracer_provider,
    engine_attributes,
    instrument,
    record_engine_stages,
)

PACKAGE_LOGGER = "image_to_text"


def create_app(
    settings: Settings | None = None,
    registry: EngineRegistry | None = None,
    tracer_provider: TracerProvider | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    registry = registry or EngineRegistry(settings)
    provider = tracer_provider or build_tracer_provider(settings.otel_endpoint)
    logging.getLogger(PACKAGE_LOGGER).setLevel(settings.log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if settings.warm_up:
            await run_in_threadpool(registry.warm_up)
        yield
        if provider is not None:
            provider.shutdown()  # flush spans still queued

    app = FastAPI(title="image-to-text", version=SERVICE_VERSION, lifespan=lifespan)
    tracer = instrument(app, provider)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "version": SERVICE_VERSION,
            "engines": list(registry.names),
            "default_engine": registry.default,
        }

    @app.post("/ocr")
    async def ocr(
        image: Annotated[UploadFile, File()],
        engine: Annotated[str | None, Query()] = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        with tracer.start_as_current_span("ocr.read_upload") as span:
            raw = await image.read(settings.max_upload_bytes + 1)
            span.set_attribute("upload.bytes", len(raw))
        if len(raw) > settings.max_upload_bytes:
            raise HTTPException(
                413, f"image larger than {settings.max_upload_bytes} bytes"
            )
        try:
            eng = registry.get(engine)
        except KeyError as e:
            raise HTTPException(400, str(e.args[0])) from e
        with tracer.start_as_current_span("ocr.decode") as span:
            arr = decode_image(raw)
            if arr is not None:
                span.set_attribute("image.width", int(arr.shape[1]))
                span.set_attribute("image.height", int(arr.shape[0]))
        if arr is None:
            raise HTTPException(422, "not a readable image")

        with tracer.start_as_current_span("ocr.engine") as span:
            started = time.time_ns()
            result = await run_in_threadpool(eng.read, arr)
            span.set_attributes(engine_attributes(result))
            record_engine_stages(tracer, result, started)
        return {
            "engine": result.engine,
            "image": {"width": int(arr.shape[1]), "height": int(arr.shape[0])},
            "lines": [asdict(ln) for ln in result.lines],
            "timings_ms": {
                **{f"engine_{k}": v for k, v in result.elapsed_ms.items()},
                "total": to_ms(time.perf_counter() - t0),
            },
        }

    return app


app = create_app()
