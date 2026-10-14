"""FastAPI service.

GET  /health    liveness and version

Build the app with `create_app(settings)`; tests pass their own `Settings`
so they never depend on the environment. `uvicorn
image_to_text.api.app:app` uses the module-level `app`, built from env.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from image_to_text import SERVICE_VERSION
from image_to_text.config import Settings

PACKAGE_LOGGER = "image_to_text"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logging.getLogger(PACKAGE_LOGGER).setLevel(settings.log_level)

    app = FastAPI(title="image-to-text", version=SERVICE_VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": SERVICE_VERSION}

    return app


app = create_app()
