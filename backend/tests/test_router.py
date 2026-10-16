import pytest

from image_to_text.config import Settings
from image_to_text.ocr.base import EngineResult
from image_to_text.ocr.router import FACTORIES, EngineRegistry


class Counting:
    built = 0

    def __init__(self, settings: Settings):
        Counting.built += 1
        self.name = "fake"
        self.reads = 0

    def read(self, image):
        self.reads += 1
        return EngineResult(self.name, (), {})


@pytest.fixture(autouse=True)
def _reset():
    Counting.built = 0


def _registry(default: str = "fake") -> EngineRegistry:
    return EngineRegistry(
        Settings(default_engine=default), {"fake": Counting, "other": Counting}
    )


def test_ships_doctr_and_rapidocr():
    assert tuple(FACTORIES) == ("doctr", "rapidocr")


def test_rejects_unknown_default_engine():
    with pytest.raises(ValueError, match="ITT_DEFAULT_ENGINE"):
        _registry(default="tesseract")


def test_get_without_name_returns_default():
    r = _registry()
    assert r.get() is r.get("fake")
    assert r.get() is not r.get("other")


def test_engines_are_built_lazily_once():
    r = _registry()
    assert Counting.built == 0
    r.get()
    r.get()
    assert Counting.built == 1
    r.get("other")
    assert Counting.built == 2


def test_unknown_engine_raises_key_error():
    with pytest.raises(KeyError, match="unknown engine 'nope'"):
        _registry().get("nope")


def test_warm_up_reads_with_default_engine():
    r = _registry()
    r.warm_up()
    assert r.get().reads == 1
