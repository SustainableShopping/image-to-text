import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from image_to_text import SERVICE_VERSION
from image_to_text.api.app import create_app
from image_to_text.config import Settings
from image_to_text.ocr.base import EngineResult, Line, Word
from image_to_text.ocr.router import EngineRegistry

ALLOWED = "https://allowed.example"
LINE = Line("Sugar", 0.9, ((0, 0), (50, 0), (50, 10), (0, 10)), (Word("Sugar", 0.9),))


class FakeEngine:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    def read(self, image: np.ndarray) -> EngineResult:
        self.calls += 1
        return EngineResult(self.name, (LINE,), {"total": 1.5})


def _registry(settings: Settings) -> EngineRegistry:
    return EngineRegistry(
        settings,
        {
            "doctr": lambda s: FakeEngine("doctr"),
            "rapidocr": lambda s: FakeEngine("rapidocr"),
        },
    )


def _client(**kw) -> TestClient:
    s = Settings(cors_origins=(ALLOWED,), warm_up=False, **kw)
    return TestClient(create_app(s, _registry(s)))


@pytest.fixture
def client() -> TestClient:
    return _client()


def _png(width: int = 40, height: int = 20) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def _upload() -> dict:
    return {"image": ("a.png", _png(), "image/png")}


def test_health_reports_version_and_engines(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "ok": True,
        "version": SERVICE_VERSION,
        "engines": ["doctr", "rapidocr"],
        "default_engine": "doctr",
    }


def test_cors_allows_configured_origin(client):
    r = client.get("/health", headers={"Origin": ALLOWED})
    assert r.headers["access-control-allow-origin"] == ALLOWED


def test_cors_ignores_other_origins(client):
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_ocr_uses_default_engine(client):
    r = client.post("/ocr", files=_upload())
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "doctr"
    assert body["image"] == {"width": 40, "height": 20}
    assert body["lines"][0]["text"] == "Sugar"
    assert body["lines"][0]["words"] == [{"text": "Sugar", "conf": 0.9}]
    assert body["timings_ms"]["engine_total"] == 1.5
    assert "total" in body["timings_ms"]


def test_ocr_engine_selectable_per_request(client):
    r = client.post("/ocr?engine=rapidocr", files=_upload())
    assert r.status_code == 200
    assert r.json()["engine"] == "rapidocr"


def test_ocr_rejects_unknown_engine(client):
    r = client.post("/ocr?engine=nope", files=_upload())
    assert r.status_code == 400
    assert "unknown engine" in r.json()["detail"]


def test_ocr_rejects_oversized_upload():
    r = _client(max_upload_bytes=10).post("/ocr", files=_upload())
    assert r.status_code == 413


def test_ocr_rejects_unreadable_image(client):
    r = client.post("/ocr", files={"image": ("a.txt", b"not an image", "text/plain")})
    assert r.status_code == 422


def test_warm_up_runs_default_engine_on_startup():
    s = Settings(warm_up=True)
    registry = _registry(s)
    with TestClient(create_app(s, registry)):
        assert registry.get().calls == 1
