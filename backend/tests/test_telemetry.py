import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from PIL import Image

from image_to_text.api.app import create_app
from image_to_text.config import Settings
from image_to_text.ocr.base import EngineResult, Line
from image_to_text.ocr.router import EngineRegistry
from image_to_text.telemetry import (
    build_tracer_provider,
    engine_attributes,
)

SECRET_WORD = "Sugarcane"
LINES = (
    Line(SECRET_WORD, 0.8, ((0, 0), (50, 0), (50, 10), (0, 10))),
    Line("Salt", 0.6, ((0, 20), (50, 20), (50, 30), (0, 30))),
)
TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


class StagedEngine:
    """Reports per-stage times the way RapidOCR does."""

    name = "rapidocr"

    def read(self, image: np.ndarray) -> EngineResult:
        return EngineResult(self.name, LINES, {"det": 12.0, "cls": 0.0, "rec": 30.0})


@pytest.fixture
def traced() -> tuple[TestClient, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    s = Settings(warm_up=False, default_engine="rapidocr")
    registry = EngineRegistry(s, {"rapidocr": lambda _: StagedEngine()})
    return TestClient(create_app(s, registry, tracer_provider=provider)), exporter


def _upload() -> dict:
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buf, format="PNG")
    return {"image": ("a.png", buf.getvalue(), "image/png")}


def _by_name(exporter: InMemorySpanExporter) -> dict:
    return {s.name: s for s in exporter.get_finished_spans()}


def test_no_endpoint_means_no_provider():
    assert build_tracer_provider(None) is None
    assert build_tracer_provider("") is None


def test_endpoint_builds_a_provider():
    provider = build_tracer_provider("http://collector:4318/")
    assert isinstance(provider, TracerProvider)
    assert provider.resource.attributes["service.name"] == "image-to-text"
    provider.shutdown()


def test_ocr_records_one_span_per_stage(traced):
    client, exporter = traced
    assert client.post("/ocr", files=_upload()).status_code == 200
    spans = _by_name(exporter)
    for name in ("ocr.read_upload", "ocr.decode", "ocr.engine"):
        assert name in spans
    assert spans["ocr.read_upload"].attributes["upload.bytes"] > 0
    assert spans["ocr.decode"].attributes["image.width"] == 40
    assert spans["ocr.decode"].attributes["image.height"] == 20


def test_engine_span_records_outputs_not_text(traced):
    client, exporter = traced
    client.post("/ocr", files=_upload())
    engine = _by_name(exporter)["ocr.engine"]
    assert engine.attributes["ocr.engine"] == "rapidocr"
    assert engine.attributes["ocr.lines"] == 2
    assert engine.attributes["ocr.mean_conf"] == pytest.approx(0.7)
    assert engine.attributes["ocr.det_ms"] == 12.0
    for span in exporter.get_finished_spans():
        assert all(SECRET_WORD not in str(v) for v in span.attributes.values())


def test_reported_stage_times_become_consecutive_child_spans(traced):
    client, exporter = traced
    client.post("/ocr", files=_upload())
    spans = _by_name(exporter)
    det, rec = spans["ocr.engine.det"], spans["ocr.engine.rec"]
    assert "ocr.engine.cls" not in spans  # 0 ms stages are skipped
    assert det.parent.span_id == spans["ocr.engine"].context.span_id
    assert det.end_time - det.start_time == 12_000_000
    assert rec.start_time == det.end_time
    assert rec.end_time - rec.start_time == 30_000_000


def test_incoming_traceparent_joins_the_callers_trace(traced):
    client, exporter = traced
    client.post("/ocr", files=_upload(), headers={"traceparent": TRACEPARENT})
    trace_ids = {f"{s.context.trace_id:032x}" for s in exporter.get_finished_spans()}
    assert trace_ids == {"0af7651916cd43dd8448eb211c80319c"}


def test_only_the_request_span_and_stage_spans_are_recorded(traced):
    client, exporter = traced
    client.post("/ocr", files=_upload())
    names = {s.name for s in exporter.get_finished_spans()}
    assert not any(n.endswith((" http receive", " http send")) for n in names)


def test_health_is_not_traced(traced):
    client, exporter = traced
    client.get("/health")
    assert exporter.get_finished_spans() == ()


def test_app_shutdown_flushes_queued_spans():
    class Provider(TracerProvider):
        shut_down = False

        def shutdown(self) -> None:
            Provider.shut_down = True
            super().shutdown()

    s = Settings(warm_up=False, default_engine="rapidocr")
    registry = EngineRegistry(s, {"rapidocr": lambda _: StagedEngine()})
    with TestClient(create_app(s, registry, tracer_provider=Provider())):
        assert not Provider.shut_down
    assert Provider.shut_down


def test_engine_attributes_without_lines():
    attrs = engine_attributes(EngineResult("doctr", (), {"total": 5.0}))
    assert attrs["ocr.lines"] == 0
    assert attrs["ocr.mean_conf"] == 0.0
    assert attrs["ocr.total_ms"] == 5.0
