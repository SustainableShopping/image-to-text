"""docTR via OnnxTR on ONNX Runtime: the default engine. Word-level confidences;
Latin script only."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from image_to_text.config import Settings
from image_to_text.ocr.base import EngineResult, Line, Word, ms

ENGINE_DOCTR = "doctr"


class DoctrEngine:
    name = ENGINE_DOCTR

    def __init__(self, settings: Settings, predictor: Any = None):
        self._ocr = predictor or _build_predictor(settings)
        self._lock = threading.Lock()

    def read(self, image: np.ndarray) -> EngineResult:
        h, w = image.shape[:2]
        t0 = time.perf_counter()
        with self._lock:
            page = self._ocr([image]).pages[0]
        elapsed = time.perf_counter() - t0
        lines = tuple(
            _to_line(ln, w, h) for blk in page.blocks for ln in blk.lines if ln.words
        )
        return EngineResult(self.name, lines, ms(total=elapsed))


def _to_line(ln: Any, w: int, h: int) -> Line:
    words = tuple(Word(wd.value, float(wd.confidence)) for wd in ln.words)
    (x0, y0), (x1, y1) = ln.geometry  # relative to page size
    box = ((x0 * w, y0 * h), (x1 * w, y0 * h), (x1 * w, y1 * h), (x0 * w, y1 * h))
    return Line(
        text=" ".join(wd.text for wd in words),
        conf=round(min(wd.conf for wd in words), 4),
        box=box,
        words=words,
    )


def _build_predictor(s: Settings) -> Any:  # pragma: no cover - loads models
    import onnxruntime as ort
    from onnxtr.models import EngineConfig, ocr_predictor

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = s.onnx_threads
    cfg = EngineConfig(session_options=opts)
    return ocr_predictor(
        det_arch=s.doctr_det_arch,
        reco_arch=s.doctr_reco_arch,
        det_engine_cfg=cfg,
        reco_engine_cfg=cfg,
        det_bs=1,
        reco_bs=64,
    )
