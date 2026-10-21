"""Real engines on a rendered label. Downloads models on first run, so opt in
with ITT_RUN_MODEL_TESTS=1."""

import os

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from image_to_text.config import Settings
from image_to_text.ocr.router import EngineRegistry

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        os.environ.get("ITT_RUN_MODEL_TESTS") != "1",
        reason="set ITT_RUN_MODEL_TESTS=1 to run real OCR models",
    ),
]


def _label() -> np.ndarray:
    img = Image.new("RGB", (900, 160), "white")
    font = ImageFont.load_default(size=48)
    ImageDraw.Draw(img).text((30, 50), "INGREDIENTS: SUGAR, SALT", "black", font)
    return np.asarray(img)


@pytest.mark.parametrize("engine", ["doctr", "rapidocr"])
def test_engine_reads_rendered_label(engine):
    result = EngineRegistry(Settings()).get(engine).read(_label())
    text = " ".join(ln.text for ln in result.lines).upper()
    assert result.engine == engine
    assert "SUGAR" in text
    assert "SALT" in text
