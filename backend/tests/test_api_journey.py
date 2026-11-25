import base64
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from PIL import Image

from image_to_text.api.app import create_app
from image_to_text.config import Settings
from image_to_text.ocr.base import EngineResult, Line, Word
from image_to_text.ocr.router import EngineRegistry
from image_to_text.photo_store import PhotoStore

LINE = Line("Sugar", 0.9, ((0, 0), (50, 0), (50, 10), (0, 10)), (Word("Sugar", 0.9),))
HEADERS = {"X-Session-ID": "sess-1", "X-Scan-ID": "scan-1", "X-Scan-Attempt": "2"}
PASSWORD = "a-long-password"


class FakeEngine:
    name = "doctr"

    def read(self, image: np.ndarray) -> EngineResult:
        return EngineResult(self.name, (LINE,), {"total": 1.5})


class BrokenEngine:
    name = "rapidocr"

    def read(self, image: np.ndarray) -> EngineResult:
        raise RuntimeError("model crashed")


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buf, format="PNG")
    return buf.getvalue()


def _upload(data: bytes | None = None) -> dict:
    return {"image": ("a.png", data if data is not None else _png(), "image/png")}


def _memory_provider() -> tuple[LoggerProvider, InMemoryLogRecordExporter]:
    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    return provider, exporter


class Journey:
    def __init__(self, tmp_path, **settings):
        provider, self.exporter = _memory_provider()
        s = Settings(
            warm_up=False,
            journey_enabled=True,
            journey_view_password=PASSWORD,
            cors_origins=("https://scan.example",),
            **settings,
        )
        registry = EngineRegistry(
            s, {"doctr": lambda _: FakeEngine(), "rapidocr": lambda _: BrokenEngine()}
        )
        self.photos = PhotoStore(tmp_path, retention_days=14)
        self.client = TestClient(
            create_app(s, registry, logger_provider=provider, photo_store=self.photos),
            raise_server_exceptions=False,
        )

    def records(self) -> dict[str, tuple[dict, dict]]:
        out = {}
        for log in self.exporter.get_finished_logs():
            attrs = dict(log.log_record.attributes)
            out[attrs["journey.step"]] = (attrs, json.loads(log.log_record.body))
        return out


@pytest.fixture
def journey(tmp_path) -> Journey:
    return Journey(tmp_path)


def _auth(user: str = "journey", password: str = PASSWORD) -> dict:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_successful_scan_records_every_stage_with_full_output(journey):
    r = journey.client.post("/ocr", files=_upload(), headers=HEADERS)
    assert r.status_code == 200
    assert r.headers["x-scan-id"] == "scan-1"

    records = journey.records()
    assert list(records) == ["ocr.received", "ocr.decode", "ocr.engine", "ocr.response"]
    for attrs, _ in records.values():
        assert attrs["journey.session_id"] == "sess-1"
        assert attrs["journey.scan_id"] == "scan-1"
        assert attrs["journey.attempt"] == 2
        assert attrs["journey.status"] == "ok"

    attrs, body = records["ocr.received"]
    assert body["input"]["content_type"] == "image/png"
    assert body["input"]["bytes"] == len(_png())
    assert len(body["output"]["sha256"]) == 64
    assert "photo" not in body["output"]
    assert attrs["upload.bytes"] == len(_png())

    _, body = records["ocr.decode"]
    assert body["output"]["width"] == 40
    assert body["output"]["height"] == 20
    assert body["output"]["photo"].endswith("/scan-1-2.png")

    attrs, body = records["ocr.engine"]
    assert body["input"] == {"engine": "doctr", "requested": None}
    assert body["output"]["lines"][0]["text"] == "Sugar"
    assert body["output"]["timings_ms"] == {"total": 1.5}
    assert attrs["ocr.lines"] == 1

    attrs, body = records["ocr.response"]
    assert attrs["http.response.status_code"] == 200
    assert body["output"]["lines"] == 1


def test_scan_without_ids_gets_a_generated_scan_id(journey):
    r = journey.client.post("/ocr", files=_upload())
    scan_id = r.headers["x-scan-id"]
    assert len(scan_id) == 32
    attrs, _ = journey.records()["ocr.response"]
    assert attrs["journey.scan_id"] == scan_id


def test_oversized_upload_is_recorded_as_an_error(tmp_path):
    journey = Journey(tmp_path, max_upload_bytes=10)
    r = journey.client.post("/ocr", files=_upload(), headers=HEADERS)
    assert r.status_code == 413
    records = journey.records()
    attrs, _ = records["ocr.received"]
    assert attrs["journey.status"] == "error"
    assert attrs["journey.error"] == "PAYLOAD_TOO_LARGE"
    assert records["ocr.response"][0]["http.response.status_code"] == 413
    assert list(tmp_path.iterdir()) == []  # oversized uploads are not kept


def test_unknown_engine_is_recorded(journey):
    r = journey.client.post("/ocr?engine=nope", files=_upload(), headers=HEADERS)
    assert r.status_code == 400
    attrs, body = journey.records()["ocr.engine"]
    assert attrs["journey.error"] == "UNKNOWN_ENGINE"
    assert body["input"]["requested"] == "nope"


def test_unreadable_image_is_recorded_but_its_bytes_not_kept(journey, tmp_path):
    r = journey.client.post("/ocr", files=_upload(b"not an image"), headers=HEADERS)
    assert r.status_code == 422
    records = journey.records()
    assert records["ocr.decode"][0]["journey.error"] == "UNREADABLE_IMAGE"
    assert records["ocr.decode"][1]["output"] is None
    assert [f for f in tmp_path.rglob("*") if f.is_file()] == []  # only images are kept
    assert records["ocr.response"][0]["http.response.status_code"] == 422


def test_engine_crash_is_recorded_before_the_500(journey):
    r = journey.client.post("/ocr?engine=rapidocr", files=_upload(), headers=HEADERS)
    assert r.status_code == 500
    records = journey.records()
    attrs, body = records["ocr.engine"]
    assert attrs["journey.error"] == "ENGINE_FAILED"
    assert body["output"] == {"exception": "RuntimeError", "message": "model crashed"}
    assert records["ocr.response"][0]["http.response.status_code"] == 500


def test_photo_can_be_viewed_with_the_password(journey):
    journey.client.post("/ocr", files=_upload(), headers=HEADERS)
    r = journey.client.get("/v1/journey/photos/scan-1/2", headers=_auth())
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.content == _png()


@pytest.mark.parametrize(
    "headers", [{}, _auth(password="wrong-password"), _auth(user="other")]
)
def test_photo_needs_the_right_credentials(journey, headers):
    journey.client.post("/ocr", files=_upload(), headers=HEADERS)
    r = journey.client.get("/v1/journey/photos/scan-1/2", headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Basic")


def test_missing_photo_is_404(journey):
    r = journey.client.get("/v1/journey/photos/nope/1", headers=_auth())
    assert r.status_code == 404


def test_photo_endpoint_is_off_without_a_password(tmp_path):
    photos = PhotoStore(tmp_path, retention_days=14)
    s = Settings(warm_up=False, journey_enabled=True)
    registry = EngineRegistry(s, {"doctr": lambda _: FakeEngine()})
    client = TestClient(create_app(s, registry, photo_store=photos))
    assert client.get("/v1/journey/photos/scan-1/1", headers=_auth()).status_code == 404


def test_journey_off_records_nothing_and_keeps_no_photos(tmp_path):
    s = Settings(warm_up=False, journey_photo_dir=str(tmp_path))
    registry = EngineRegistry(s, {"doctr": lambda _: FakeEngine()})
    client = TestClient(create_app(s, registry))
    r = client.post("/ocr", files=_upload(), headers=HEADERS)
    assert r.status_code == 200
    assert "x-scan-id" not in r.headers  # nothing new while journey is off
    assert list(tmp_path.iterdir()) == []
    assert client.get("/v1/journey/photos/scan-1/2", headers=_auth()).status_code == 404


def test_photo_store_comes_from_settings_when_journey_is_on(tmp_path):
    provider, _ = _memory_provider()
    s = Settings(
        warm_up=False,
        journey_enabled=True,
        journey_photo_dir=str(tmp_path),
        journey_view_password=PASSWORD,
    )
    registry = EngineRegistry(s, {"doctr": lambda _: FakeEngine()})
    with TestClient(create_app(s, registry, logger_provider=provider)) as client:
        client.post("/ocr", files=_upload(), headers=HEADERS)
    assert len(list(tmp_path.rglob("scan-1-2.png"))) == 1


def test_photo_save_failure_does_not_fail_the_scan(journey, monkeypatch):
    def broken(*_):
        raise OSError("disk full")

    monkeypatch.setattr(journey.photos, "save", broken)
    r = journey.client.post("/ocr", files=_upload(), headers=HEADERS)
    assert r.status_code == 200
    assert journey.records()["ocr.decode"][1]["output"]["photo"] is None


def test_scan_id_header_is_exposed_to_browsers(journey):
    r = journey.client.post(
        "/ocr",
        files=_upload(),
        headers={**HEADERS, "Origin": "https://scan.example"},
    )
    assert "x-scan-id" in r.headers["access-control-expose-headers"].lower()


def test_no_photos_are_kept_without_a_view_password(tmp_path):
    provider, _ = _memory_provider()
    s = Settings(warm_up=False, journey_enabled=True, journey_photo_dir=str(tmp_path))
    registry = EngineRegistry(s, {"doctr": lambda _: FakeEngine()})
    with TestClient(create_app(s, registry, logger_provider=provider)) as client:
        assert client.post("/ocr", files=_upload(), headers=HEADERS).status_code == 200
    assert list(tmp_path.rglob("*.png")) == []  # nobody could view them


def test_photos_over_the_size_cap_are_not_kept(tmp_path):
    journey = Journey(tmp_path)
    journey.photos.max_bytes = len(_png()) + 10  # room for exactly one photo
    for scan in ("a", "b"):
        headers = {**HEADERS, "X-Scan-ID": scan}
        assert (
            journey.client.post("/ocr", files=_upload(), headers=headers).status_code
            == 200
        )
    kept = sorted(f.name for f in tmp_path.rglob("*.png"))
    assert kept == ["a-2.png"]
