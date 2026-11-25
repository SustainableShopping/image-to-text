"""Run each engine on each case's photo: time the reads, score the text, summarise."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from image_to_text.evaluation.cases import OcrCase
from image_to_text.evaluation.metrics import pct, section_cer, section_wer, word_recall
from image_to_text.imaging import decode_image
from image_to_text.ocr.base import Engine, EngineResult

WARM_UP_IMAGE_SHAPE = (64, 256, 3)  # blank image, as EngineRegistry.warm_up; no photos


def image_path(case: OcrCase, images: Path | None) -> Path:
    return (images or case.folder) / str(case.image)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _first_photo(cases: Sequence[OcrCase], images: Path | None) -> np.ndarray | None:
    for case in cases:
        path = image_path(case, images)
        if case.image and path.is_file():
            image = decode_image(path.read_bytes())
            if image is not None:
                return image
    return None


def _warm_up(engines: Mapping[str, Engine], image: np.ndarray | None) -> None:
    """First reads load the models and set up every stage; keep them out of the
    timings. A real photo is used when there is one: on a blank image the detector
    finds nothing and recognition would still start cold on the first timed read.
    A broken engine shows up as an error on every case instead."""
    if image is None:
        image = np.full(WARM_UP_IMAGE_SHAPE, 255, dtype=np.uint8)
    for engine in engines.values():
        with contextlib.suppress(Exception):  # reported per case below
            engine.read(image)


def _row(case: OcrCase, engine: str, **fields: Any) -> dict[str, Any]:
    base = {
        "case_id": case.id,
        "engine": engine,
        "tags": list(case.tags),
        "text": None,
        "error": None,
    }
    return {**base, **fields}


def _read(
    case: OcrCase,
    name: str,
    engine: Engine,
    image: np.ndarray,
    decode_ms: float,
    repeat: int,
) -> dict[str, Any]:
    timings: dict[str, Any] = {"decode": decode_ms, "read": []}
    result: EngineResult | None = None
    try:
        for _ in range(repeat):
            started = time.perf_counter()
            result = engine.read(image)
            timings["read"].append(_ms(started))
            for stage, ms in result.elapsed_ms.items():
                timings.setdefault(f"engine_{stage}", []).append(ms)
    except Exception as e:  # one engine failing must not stop the run
        return _row(case, name, error=f"{type(e).__name__}: {e}", timings_ms=timings)
    assert result is not None
    text = "\n".join(line.text for line in result.lines)
    row = _row(
        case,
        name,
        text=text,
        lines=len(result.lines),
        mean_conf=round(mean(line.conf for line in result.lines), 3)
        if result.lines
        else None,
        timings_ms=timings,
    )
    if case.ocr_text:
        row.update(
            cer=section_cer(case.ocr_text, text),
            wer=section_wer(case.ocr_text, text),
            word_recall=word_recall(case.ocr_text, text),
        )
    return row


def evaluate(
    cases: Sequence[OcrCase],
    engines: Mapping[str, Engine],
    images: Path | None,
    repeat: int,
) -> list[dict[str, Any]]:
    """One row per (case with a photo, engine); missing or broken photos: error rows."""
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    _warm_up(engines, _first_photo(cases, images))
    rows = []
    for case in (c for c in cases if c.image):
        path = image_path(case, images)
        if not path.is_file():
            rows.extend(
                _row(case, n, error=f"image not found: {path}") for n in engines
            )
            continue
        started = time.perf_counter()
        image = decode_image(path.read_bytes())
        decode_ms = _ms(started)
        if image is None:
            rows.extend(
                _row(case, n, error="image could not be decoded") for n in engines
            )
            continue
        rows.extend(
            _read(case, n, e, image, decode_ms, repeat) for n, e in engines.items()
        )
    return rows


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [r[field] for r in rows if r.get(field) is not None]
    return round(mean(values), 4) if values else None


def _engine_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reads = [ms for r in rows for ms in (r.get("timings_ms") or {}).get("read", [])]
    decodes = [r["timings_ms"]["decode"] for r in rows if r.get("timings_ms")]
    return {
        "cases": len(rows),
        "errors": sum(r["error"] is not None for r in rows),
        "cer": _mean(rows, "cer"),
        "wer": _mean(rows, "wer"),
        "word_recall": _mean(rows, "word_recall"),
        "read_ms": {"p50": pct(reads, 50), "p95": pct(reads, 95)},
        "decode_ms": {"p50": pct(decodes, 50), "p95": pct(decodes, 95)},
    }


def _per_engine(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({r["engine"] for r in rows})
    return {n: _engine_summary([r for r in rows if r["engine"] == n]) for n in names}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tags = sorted({t for r in rows for t in r["tags"]})
    return {
        "engines": _per_engine(rows),
        "by_tag": {t: _per_engine([r for r in rows if t in r["tags"]]) for t in tags},
    }


_TABLE = (
    ("cases", "cases"),
    ("errors", "errors"),
    ("CER (lower=better)", "cer"),
    ("WER (lower=better)", "wer"),
    ("word recall", "word_recall"),
    ("read p50 ms", "read_ms.p50"),
    ("read p95 ms", "read_ms.p95"),
    ("decode p50 ms", "decode_ms.p50"),
)


def _value(summary: dict[str, Any], path: str) -> Any:
    for part in path.split("."):
        summary = summary.get(part) if isinstance(summary, dict) else None
    return summary


def render_table(summary: dict[str, Any]) -> str:
    names = list(summary["engines"])
    rows = [["metric", *names]]
    for label, path in _TABLE:
        values = [_value(summary["engines"][n], path) for n in names]
        rows.append([label, *("-" if v is None else str(v) for v in values)])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip()
        for r in rows
    )
