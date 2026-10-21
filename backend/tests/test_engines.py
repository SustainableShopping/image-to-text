from types import SimpleNamespace as NS

import numpy as np
import pytest

from image_to_text.config import Settings
from image_to_text.ocr.doctr_engine import DoctrEngine
from image_to_text.ocr.rapidocr_engine import RapidOcrEngine

IMAGE = np.zeros((100, 200, 3), dtype=np.uint8)


def _word(value: str, conf: float) -> NS:
    return NS(value=value, confidence=conf)


def test_doctr_normalises_relative_geometry_and_word_confidences():
    line = NS(
        words=[_word("Wheat", 0.95), _word("Flour", 0.8)],
        geometry=((0.1, 0.2), (0.6, 0.3)),
    )
    page = NS(blocks=[NS(lines=[line, NS(words=[], geometry=None)])])
    engine = DoctrEngine(Settings(), predictor=lambda images: NS(pages=[page]))

    result = engine.read(IMAGE)

    assert result.engine == "doctr"
    assert len(result.lines) == 1
    ln = result.lines[0]
    assert ln.text == "Wheat Flour"
    assert ln.conf == 0.8
    assert ln.box == ((20.0, 20.0), (120.0, 20.0), (120.0, 30.0), (20.0, 30.0))
    assert [w.text for w in ln.words] == ["Wheat", "Flour"]
    assert "total" in result.elapsed_ms


def test_rapidocr_normalises_lines_and_stage_timings():
    out = NS(
        txts=("Sugar", "Salt"),
        scores=(0.91, 0.75),
        boxes=np.array([[[0, 0], [10, 0], [10, 5], [0, 5]]] * 2),
        elapse_list=[0.01, None, 0.02],
    )
    engine = RapidOcrEngine(Settings(), predictor=lambda image: out)

    result = engine.read(IMAGE)

    assert result.engine == "rapidocr"
    assert [ln.text for ln in result.lines] == ["Sugar", "Salt"]
    assert result.lines[0].conf == pytest.approx(0.91)
    assert result.lines[0].box[2] == (10.0, 5.0)
    assert result.lines[0].words == ()
    assert result.elapsed_ms == {"det": 10.0, "cls": 0.0, "rec": 20.0}


def test_rapidocr_handles_no_text():
    out = NS(txts=None, scores=None, boxes=None)
    engine = RapidOcrEngine(Settings(), predictor=lambda image: out)

    result = engine.read(IMAGE)

    assert result.lines == ()
    assert result.elapsed_ms == {"det": 0.0, "cls": 0.0, "rec": 0.0}
