"""FastAPI service.

GET  /health          liveness, version, engines
POST /ocr?engine=     multipart `image` → lines with confidences and boxes
GET  /v1/journey/photos/{scan_id}/{attempt}
                      a kept scan photo (journey recording on, HTTP Basic auth)

Build the app with `create_app(settings, registry)`; tests pass their own
so they never depend on the environment or load models. Tracing (telemetry.py) is
on only when an OTLP endpoint is configured, or a `tracer_provider` is passed. Journey
records (journey.py) need ITT_JOURNEY_ENABLED too, or a `logger_provider`. `uvicorn
image_to_text.api.app:app` uses the module-level `app`, built from env.
"""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.trace import TracerProvider
from starlette.concurrency import run_in_threadpool

from image_to_text import SERVICE_VERSION
from image_to_text.api.scan_journey import OcrJourney
from image_to_text.config import Settings
from image_to_text.imaging import decode_image
from image_to_text.journey import JourneyIds, JourneyRecorder, build_logger_provider
from image_to_text.ocr.base import to_ms
from image_to_text.ocr.router import EngineRegistry
from image_to_text.photo_store import MEDIA_TYPES, PhotoStore
from image_to_text.telemetry import (
    build_tracer_provider,
    engine_attributes,
    instrument,
    record_engine_stages,
)

PACKAGE_LOGGER = "image_to_text"
SCAN_ID_HEADER = "X-Scan-ID"
PHOTO_REALM = 'Basic realm="scan photos"'
# Module level: route annotations are resolved against module globals.
PHOTO_AUTH = HTTPBasic(auto_error=False)


def _photo_store(settings: Settings, given: PhotoStore | None) -> PhotoStore | None:
    if given is not None:
        return given
    # Photos are kept only where someone can view them (a view password is set).
    if (
        settings.journey_enabled
        and settings.journey_photo_dir
        and settings.journey_view_password
    ):
        return PhotoStore(
            settings.journey_photo_dir,
            settings.journey_photo_retention_days,
            max_bytes=settings.journey_photo_max_mb * 1024 * 1024,
        )
    return None


def _authorised(credentials: HTTPBasicCredentials | None, settings: Settings) -> bool:
    if credentials is None or not settings.journey_view_password:
        return False
    user_ok = secrets.compare_digest(
        credentials.username.encode(), settings.journey_view_user.encode()
    )
    password_ok = secrets.compare_digest(
        credentials.password.encode(), settings.journey_view_password.encode()
    )
    return user_ok and password_ok


def create_app(
    settings: Settings | None = None,
    registry: EngineRegistry | None = None,
    tracer_provider: TracerProvider | None = None,
    logger_provider: LoggerProvider | None = None,
    photo_store: PhotoStore | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    registry = registry or EngineRegistry(settings)
    provider = tracer_provider or build_tracer_provider(settings.otel_endpoint)
    log_provider = logger_provider or build_logger_provider(
        settings.otel_endpoint, settings.journey_enabled
    )
    recorder = JourneyRecorder(log_provider, settings.journey_max_body_bytes)
    photos = _photo_store(settings, photo_store)
    logging.getLogger(PACKAGE_LOGGER).setLevel(settings.log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if settings.warm_up:
            await run_in_threadpool(registry.warm_up)
        if photos is not None:
            await run_in_threadpool(photos.purge)
        yield
        if provider is not None:
            provider.shutdown()  # flush spans still queued
        if log_provider is not None:
            log_provider.shutdown()  # and journey records

    app = FastAPI(title="image-to-text", version=SERVICE_VERSION, lifespan=lifespan)
    tracer = instrument(app, provider)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=[SCAN_ID_HEADER],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "version": SERVICE_VERSION,
            "engines": list(registry.names),
            "default_engine": registry.default,
        }

    async def run_ocr(
        scan: OcrJourney, image: UploadFile, engine: str | None
    ) -> dict[str, Any]:
        started = time.perf_counter()
        with tracer.start_as_current_span("ocr.read_upload") as span:
            raw = await image.read(settings.max_upload_bytes + 1)
            span.set_attribute("upload.bytes", len(raw))
        too_large = len(raw) > settings.max_upload_bytes
        await scan.received(image, raw, engine, too_large, started)
        if too_large:
            raise HTTPException(
                413, f"image larger than {settings.max_upload_bytes} bytes"
            )
        started = time.perf_counter()
        try:
            eng = registry.get(engine)
        except KeyError as e:
            scan.engine_failed(None, engine, "UNKNOWN_ENGINE", e, started)
            raise HTTPException(400, str(e.args[0])) from e
        with tracer.start_as_current_span("ocr.decode") as span:
            arr = decode_image(raw)
            if arr is not None:
                span.set_attribute("image.width", int(arr.shape[1]))
                span.set_attribute("image.height", int(arr.shape[0]))
        await scan.decoded(arr, raw, started)
        if arr is None:
            raise HTTPException(422, "not a readable image")

        with tracer.start_as_current_span("ocr.engine") as span:
            started_ns, started = time.time_ns(), time.perf_counter()
            try:
                result = await run_in_threadpool(eng.read, arr)
            except Exception as e:
                scan.engine_failed(eng.name, engine, "ENGINE_FAILED", e, started)
                raise
            span.set_attributes(engine_attributes(result))
            record_engine_stages(tracer, result, started_ns)
        scan.engine_done(result, engine, started)
        return {
            "engine": result.engine,
            "image": {"width": int(arr.shape[1]), "height": int(arr.shape[0])},
            "lines": [asdict(ln) for ln in result.lines],
            "timings_ms": {f"engine_{k}": v for k, v in result.elapsed_ms.items()},
        }

    @app.post("/ocr")
    async def ocr(
        request: Request,
        response: Response,
        image: Annotated[UploadFile, File()],
        engine: Annotated[str | None, Query()] = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        scan = OcrJourney(recorder, photos, JourneyIds.from_headers(request.headers))
        if recorder.enabled:  # the scan ID only means something when it is recorded
            response.headers[SCAN_ID_HEADER] = scan.ids.scan_id
        try:
            body = await run_ocr(scan, image, engine)
        except HTTPException as e:
            scan.responded(e.status_code, t0, {"detail": e.detail})
            if recorder.enabled:
                e.headers = {**(e.headers or {}), SCAN_ID_HEADER: scan.ids.scan_id}
            raise
        except Exception:
            scan.responded(500, t0)
            raise
        body["timings_ms"]["total"] = to_ms(time.perf_counter() - t0)
        scan.responded(
            200, t0, {"lines": len(body["lines"]), "timings_ms": body["timings_ms"]}
        )
        return body

    @app.get("/v1/journey/photos/{scan_id}/{attempt}")
    def journey_photo(
        scan_id: str,
        attempt: int,
        credentials: Annotated[HTTPBasicCredentials | None, Depends(PHOTO_AUTH)],
    ) -> FileResponse:
        if photos is None or not settings.journey_view_password:
            raise HTTPException(404, "scan photos are not kept here")
        if not _authorised(credentials, settings):
            raise HTTPException(
                401, "credentials required", headers={"WWW-Authenticate": PHOTO_REALM}
            )
        path = photos.find(scan_id, attempt)
        if path is None:
            raise HTTPException(404, "no photo for this scan")
        return FileResponse(
            path,
            media_type=MEDIA_TYPES[path.suffix.lstrip(".")],
            # .bin files are arbitrary bytes: never let a browser sniff them into HTML
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    return app


app = create_app()
