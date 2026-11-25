"""Scan journey records: the full input and output of every `/ocr` stage, so one scan
can be followed step by step when troubleshooting (what came in, what each stage
returned, where it failed).

Records are OpenTelemetry *log* records, separate from tracing (telemetry.py, which by
design carries no text). Each record has:

- attributes: `journey=true`, `journey.step`, `journey.status` (ok | error),
  `journey.error`, `journey.session_id`, `journey.scan_id`, `journey.attempt`,
  `journey.duration_ms`, `journey.truncated`, plus step-specific counts;
- body: JSON `{"input": ..., "output": ...}` as a string, capped at `max_body_bytes`.

The collector routes `journey=true` records to their own stream. Off unless
`ITT_JOURNEY_ENABLED=true` and `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Recording never
raises: a failed record is logged and the request carries on.

IDs come from the caller's `X-Session-ID`, `X-Scan-ID` and `X-Scan-Attempt` headers.
Unsafe or missing values are replaced, so IDs are always safe in file names.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

from image_to_text import SERVICE_VERSION

SERVICE_NAME = "image-to-text"
LOGGER_NAME = "image_to_text.journey"
SESSION_HEADER = "x-session-id"
SCAN_HEADER = "x-scan-id"
ATTEMPT_HEADER = "x-scan-attempt"
# Same rule as the ingredient service's X-Request-ID: safe in file names and URLs.
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# A retry loop that reaches this many attempts is broken; clamp instead of trusting it.
MAX_ATTEMPT = 1000
# Room for the preview wrapper's own keys when a body is cut.
_PREVIEW_OVERHEAD = 64
# Records per OTLP export: 16 bodies of at most 512 KiB stay under the collector's
# 16 MiB request limit (the SDK default, 512, could reach 256 MiB and be dropped).
EXPORT_BATCH_SIZE = 16

log = logging.getLogger(LOGGER_NAME)


def safe_id(value: str | None) -> str | None:
    """`value` if it is a safe ID (no path separators, not `.`/`..`), else None."""
    if value and SAFE_ID.match(value) and value not in {".", ".."}:
        return value
    return None


@dataclass(frozen=True)
class JourneyIds:
    session_id: str
    scan_id: str
    attempt: int

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> JourneyIds:
        try:
            attempt = int(headers.get(ATTEMPT_HEADER) or "1")
        except ValueError:
            attempt = 1
        if not 1 <= attempt <= MAX_ATTEMPT:
            attempt = 1
        return cls(
            session_id=safe_id(headers.get(SESSION_HEADER)) or "",
            scan_id=safe_id(headers.get(SCAN_HEADER)) or uuid.uuid4().hex,
            attempt=attempt,
        )


def encode_body(input: Any, output: Any, max_bytes: int) -> tuple[str, bool]:
    """(JSON body, truncated?). An oversized body becomes valid JSON too:
    `{"truncated": true, "bytes": <full size>, "preview": "<start of full JSON>"}`."""
    text = json.dumps(
        {"input": input, "output": output},
        default=str,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    data = text.encode()
    if len(data) <= max_bytes:
        return text, False
    budget = max(max_bytes - _PREVIEW_OVERHEAD, 0)
    while True:
        preview = data[:budget].decode("utf-8", "ignore")
        out = json.dumps(
            {"truncated": True, "bytes": len(data), "preview": preview},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        excess = len(out.encode()) - max_bytes
        if excess <= 0 or budget == 0:
            return out, True
        budget = max(budget - excess, 0)  # escaping made it longer: shrink, retry


class JourneyRecorder:
    def __init__(self, provider: LoggerProvider | None, max_body_bytes: int):
        self._logger = (
            provider.get_logger(LOGGER_NAME, SERVICE_VERSION) if provider else None
        )
        self._max_body_bytes = max_body_bytes

    @property
    def enabled(self) -> bool:
        return self._logger is not None

    def record(
        self,
        ids: JourneyIds,
        step: str,
        *,
        status: str = "ok",
        error: str | None = None,
        duration_ms: float | None = None,
        input: Any = None,
        output: Any = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        if self._logger is None:
            return
        try:
            body, truncated = encode_body(input, output, self._max_body_bytes)
            attrs: dict[str, Any] = {
                **(attributes or {}),
                "journey": True,
                "journey.step": step,
                "journey.status": status,
                "journey.session_id": ids.session_id,
                "journey.scan_id": ids.scan_id,
                "journey.attempt": ids.attempt,
                "journey.truncated": truncated,
            }
            # Always present (empty when ok): dashboards select it before any error.
            attrs["journey.error"] = error or ""
            if duration_ms is not None:
                attrs["journey.duration_ms"] = duration_ms
            self._logger.emit(body=body, attributes=attrs, event_name=f"journey.{step}")
        except Exception:
            log.warning("journey record %s failed", step, exc_info=True)


def build_logger_provider(endpoint: str | None, enabled: bool) -> LoggerProvider | None:
    if not (enabled and endpoint):
        return None
    provider = LoggerProvider(
        resource=Resource.create(
            {"service.name": SERVICE_NAME, "service.version": SERVICE_VERSION}
        )
    )
    exporter = OTLPLogExporter(endpoint=f"{endpoint.rstrip('/')}/v1/logs")
    provider.add_log_record_processor(
        BatchLogRecordProcessor(exporter, max_export_batch_size=EXPORT_BATCH_SIZE)
    )
    return provider
