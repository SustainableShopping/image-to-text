"""The journey records of one `/ocr` request (see image_to_text.journey): one record per
stage, `ocr.received` → `ocr.decode` → `ocr.engine` → `ocr.response`, each with its
full input and output. A failed stage is recorded with `journey.status=error` and a
reason code before the request fails.

Every method is a no-op while journey recording is off.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import asdict
from typing import Any

import numpy as np
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from image_to_text.journey import JourneyIds, JourneyRecorder
from image_to_text.ocr.base import EngineResult, to_ms
from image_to_text.photo_store import PhotoStore

log = logging.getLogger("image_to_text.journey")


def _since(started: float) -> float:
    return to_ms(time.perf_counter() - started)


class OcrJourney:
    def __init__(
        self, recorder: JourneyRecorder, photos: PhotoStore | None, ids: JourneyIds
    ):
        self.recorder, self.photos, self.ids = recorder, photos, ids

    async def received(
        self,
        image: UploadFile,
        raw: bytes,
        requested_engine: str | None,
        too_large: bool,
        started: float,
    ) -> None:
        if not self.recorder.enabled:
            return
        self.recorder.record(
            self.ids,
            "ocr.received",
            status="error" if too_large else "ok",
            error="PAYLOAD_TOO_LARGE" if too_large else None,
            duration_ms=_since(started),
            input={
                "filename": image.filename,
                "content_type": image.content_type,
                "bytes": len(raw),
                "engine": requested_engine,
            },
            output={"sha256": hashlib.sha256(raw).hexdigest()},
            attributes={"upload.bytes": len(raw)},
        )

    async def _save_photo(self, raw: bytes) -> str | None:
        if self.photos is None:
            return None
        try:
            return await run_in_threadpool(self.photos.save, self.ids, raw)
        except Exception:
            log.warning(
                "could not keep the photo of scan %s", self.ids.scan_id, exc_info=True
            )
            return None

    async def decoded(self, arr: np.ndarray | None, raw: bytes, started: float) -> None:
        """Records the decode; keeps the photo only if it is an image (bytes that do not
        decode are not kept, so the store holds photos, not arbitrary uploads)."""
        if not self.recorder.enabled:
            return
        if arr is None:
            self.recorder.record(
                self.ids,
                "ocr.decode",
                status="error",
                error="UNREADABLE_IMAGE",
                duration_ms=_since(started),
            )
            return
        duration = _since(started)
        output = {
            "width": int(arr.shape[1]),
            "height": int(arr.shape[0]),
            "photo": await self._save_photo(raw),
        }
        self.recorder.record(
            self.ids, "ocr.decode", duration_ms=duration, output=output
        )

    def engine_done(
        self, result: EngineResult, requested: str | None, started: float
    ) -> None:
        self.recorder.record(
            self.ids,
            "ocr.engine",
            duration_ms=_since(started),
            input={"engine": result.engine, "requested": requested},
            output={
                "lines": [asdict(ln) for ln in result.lines],
                "timings_ms": result.elapsed_ms,
            },
            attributes={"ocr.engine": result.engine, "ocr.lines": len(result.lines)},
        )

    def engine_failed(
        self,
        engine: str | None,
        requested: str | None,
        error: str,
        exc: BaseException,
        started: float,
    ) -> None:
        self.recorder.record(
            self.ids,
            "ocr.engine",
            status="error",
            error=error,
            duration_ms=_since(started),
            input={"engine": engine, "requested": requested},
            output={"exception": type(exc).__name__, "message": str(exc)},
        )

    def responded(self, status_code: int, started: float, output: Any = None) -> None:
        ok = status_code < 400
        self.recorder.record(
            self.ids,
            "ocr.response",
            status="ok" if ok else "error",
            error=None if ok else f"HTTP_{status_code}",
            duration_ms=_since(started),
            output=output,
            attributes={"http.response.status_code": status_code},
        )
