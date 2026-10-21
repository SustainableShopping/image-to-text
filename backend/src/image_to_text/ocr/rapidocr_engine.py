"""RapidOCR (PP-OCR models) on ONNX Runtime: the fast path, and the engine with
Devanagari recognition models. Line-level confidences only."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from image_to_text.config import Settings
from image_to_text.ocr.base import EngineResult, Line, ms

ENGINE_RAPIDOCR = "rapidocr"


class RapidOcrEngine:
    """Sessions are not thread-safe, so reads are serialised; scale with
    worker processes."""

    name = ENGINE_RAPIDOCR

    def __init__(self, settings: Settings, predictor: Any = None):
        self._ocr = predictor or _build_predictor(settings)
        self._lock = threading.Lock()

    def read(self, image: np.ndarray) -> EngineResult:
        with self._lock:
            out = self._ocr(image)
        boxes = out.boxes if out.boxes is not None else ()
        lines = tuple(
            Line(text=t, conf=float(c), box=tuple((float(x), float(y)) for x, y in b))
            for t, c, b in zip(out.txts or (), out.scores or (), boxes, strict=False)
        )
        stages = [*(getattr(out, "elapse_list", None) or ()), None, None, None]
        det, cls, rec = stages[:3]
        return EngineResult(self.name, lines, ms(det=det, cls=cls, rec=rec))


def _build_predictor(s: Settings) -> Any:  # pragma: no cover - loads models
    from rapidocr import RapidOCR

    return RapidOCR(
        params={
            "Global.log_level": "warning",
            "Global.text_score": s.text_score,
            "Global.use_cls": s.use_angle_cls,
            "Det.limit_type": "max",
            "Det.limit_side_len": s.det_limit_side_len,
            "EngineConfig.onnxruntime.intra_op_num_threads": s.onnx_threads,
        }
    )
