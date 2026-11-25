import json

import pytest
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)

from image_to_text.journey import (
    JourneyIds,
    JourneyRecorder,
    build_logger_provider,
    encode_body,
)

SESSION = "4f1c2a9e-session"
SCAN = "b7e1d2c3-scan"


@pytest.fixture
def memory() -> tuple[JourneyRecorder, InMemoryLogRecordExporter]:
    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    return JourneyRecorder(provider, max_body_bytes=10_000), exporter


def _ids() -> JourneyIds:
    return JourneyIds(session_id=SESSION, scan_id=SCAN, attempt=2)


def test_ids_come_from_headers():
    ids = JourneyIds.from_headers(
        {"x-session-id": SESSION, "x-scan-id": SCAN, "x-scan-attempt": "3"}
    )
    assert ids == JourneyIds(SESSION, SCAN, 3)


def test_missing_scan_id_is_generated_and_session_left_empty():
    ids = JourneyIds.from_headers({})
    assert ids.session_id == ""
    assert len(ids.scan_id) == 32
    assert ids.attempt == 1


@pytest.mark.parametrize(
    "bad", ["../../etc/passwd", "a b", "x" * 65, "", "scan/1", "é"]
)
def test_unsafe_ids_are_replaced(bad):
    ids = JourneyIds.from_headers(
        {"x-session-id": bad, "x-scan-id": bad, "x-scan-attempt": "1"}
    )
    assert ids.session_id == ""
    assert ids.scan_id != bad
    assert len(ids.scan_id) == 32


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "1001", ""])
def test_attempt_outside_range_falls_back_to_one(bad):
    assert JourneyIds.from_headers({"x-scan-attempt": bad}).attempt == 1


def test_record_emits_attributes_and_json_body(memory):
    recorder, exporter = memory
    recorder.record(
        _ids(),
        "ocr.engine",
        duration_ms=12.5,
        input={"engine": "rapidocr"},
        output={"lines": [{"text": "Sugar"}]},
        attributes={"ocr.lines": 1},
    )
    (log,) = exporter.get_finished_logs()
    attrs = dict(log.log_record.attributes)
    assert attrs["journey"] is True
    assert attrs["journey.step"] == "ocr.engine"
    assert attrs["journey.status"] == "ok"
    assert attrs["journey.session_id"] == SESSION
    assert attrs["journey.scan_id"] == SCAN
    assert attrs["journey.attempt"] == 2
    assert attrs["journey.duration_ms"] == 12.5
    assert attrs["journey.truncated"] is False
    assert attrs["ocr.lines"] == 1
    assert attrs["journey.error"] == ""  # always present: dashboards query it
    assert log.log_record.event_name == "journey.ocr.engine"
    body = json.loads(log.log_record.body)
    assert body == {
        "input": {"engine": "rapidocr"},
        "output": {"lines": [{"text": "Sugar"}]},
    }


def test_error_record_carries_status_and_reason(memory):
    recorder, exporter = memory
    recorder.record(_ids(), "ocr.decode", status="error", error="UNREADABLE_IMAGE")
    attrs = dict(exporter.get_finished_logs()[0].log_record.attributes)
    assert attrs["journey.status"] == "error"
    assert attrs["journey.error"] == "UNREADABLE_IMAGE"
    assert "journey.duration_ms" not in attrs


def test_disabled_recorder_records_nothing():
    recorder = JourneyRecorder(None, max_body_bytes=10)
    assert not recorder.enabled
    recorder.record(_ids(), "ocr.engine", output={"x": 1})  # no error


def test_recording_failure_never_raises(memory, caplog):
    recorder, exporter = memory

    class Unserialisable:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    recorder.record(_ids(), "ocr.engine", output=Unserialisable())
    assert exporter.get_finished_logs() == ()
    assert "journey record ocr.engine failed" in caplog.text


def test_body_within_limit_is_plain_json():
    text, truncated = encode_body({"a": 1}, ["é"], max_bytes=100)
    assert not truncated
    assert json.loads(text) == {"input": {"a": 1}, "output": ["é"]}


def test_oversized_body_becomes_a_valid_json_preview():
    text, truncated = encode_body({"text": "x" * 5000}, None, max_bytes=1000)
    assert truncated
    assert len(text.encode()) <= 1000
    preview = json.loads(text)
    assert preview["truncated"] is True
    assert preview["bytes"] > 5000
    assert preview["preview"].startswith('{"input":{"text":"xxx')


def test_no_provider_without_endpoint_or_when_off():
    assert build_logger_provider(None, enabled=True) is None
    assert build_logger_provider("http://collector:4318", enabled=False) is None


def test_endpoint_and_enabled_build_a_provider():
    provider = build_logger_provider("http://collector:4318/", enabled=True)
    assert isinstance(provider, LoggerProvider)
    assert provider.resource.attributes["service.name"] == "image-to-text"
    provider.shutdown()
