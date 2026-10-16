"""FastAPI service.

GET  /health          liveness, version, engines
POST /ocr?engine=     multipart `image` → lines with confidences and boxes

Build the app with `create_app(settings, registry)`; tests pass their own
so they never depend on the environment or load models. `uvicorn
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
from starlette.concurrency import run_in_threadpool

from image_to_text import SERVICE_VERSION
from image_to_text.config import Settings
from image_to_text.imaging import decode_image
from image_to_text.ocr.base import to_ms
from image_to_text.ocr.router import EngineRegistry

PACKAGE_LOGGER = "image_to_text"


def create_app(
    settings: Settings | None = None, registry: EngineRegistry | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    registry = registry or EngineRegistry(settings)
    logging.getLogger(PACKAGE_LOGGER).setLevel(settings.log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if settings.warm_up:
            await run_in_threadpool(registry.warm_up)
        yield

    app = FastAPI(title="image-to-text", version=SERVICE_VERSION, lifespan=lifespan)
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
        raw = await image.read(settings.max_upload_bytes + 1)
        if len(raw) > settings.max_upload_bytes:
            raise HTTPException(
                413, f"image larger than {settings.max_upload_bytes} bytes"
            )
        try:
            eng = registry.get(engine)
        except KeyError as e:
            raise HTTPException(400, str(e.args[0])) from e
        arr = decode_image(raw)
        if arr is None:
            raise HTTPException(422, "not a readable image")

        result = await run_in_threadpool(eng.read, arr)
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
