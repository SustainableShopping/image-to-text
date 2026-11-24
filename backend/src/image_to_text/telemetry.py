"""OpenTelemetry tracing: one span per `/ocr` stage, so the time of each scan can be
followed stage by stage (upload read, decode, engine, and the engine's own stages).

Off unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set (`Settings.otel_endpoint`); then spans
go to that OTLP/HTTP endpoint. An incoming W3C `traceparent` header puts the request's
spans into the caller's trace.

Spans carry sizes, counts, confidences and timings only, never image bytes or OCR text.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from image_to_text import SERVICE_VERSION
from image_to_text.ocr.base import EngineResult

SERVICE_NAME = "image-to-text"
TRACER_NAME = "image_to_text"
# Liveness polling would drown the scan traces
UNTRACED_URLS = "health"
MS_TO_NS = 1_000_000


def build_tracer_provider(endpoint: str | None) -> TracerProvider | None:
    if not endpoint:
        return None
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": SERVICE_NAME, "service.version": SERVICE_VERSION}
        )
    )
    exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def instrument(app: FastAPI, provider: TracerProvider | None) -> trace.Tracer:
    """Server spans for every route but /health; a no-op tracer when tracing is off."""
    if provider is None:
        return trace.NoOpTracer()
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=provider,
        excluded_urls=UNTRACED_URLS,
        exclude_spans=["receive", "send"],  # per-ASGI-message spans: noise
    )
    return provider.get_tracer(TRACER_NAME, SERVICE_VERSION)


def engine_attributes(result: EngineResult) -> dict[str, Any]:
    confs = [ln.conf for ln in result.lines]
    return {
        "ocr.engine": result.engine,
        "ocr.lines": len(result.lines),
        "ocr.mean_conf": round(sum(confs) / len(confs), 3) if confs else 0.0,
        **{f"ocr.{stage}_ms": value for stage, value in result.elapsed_ms.items()},
    }


def record_engine_stages(
    tracer: trace.Tracer, result: EngineResult, start_ns: int
) -> None:
    """Engines report their stage times after the fact; lay them out as consecutive
    child spans of the current span, starting when the engine call started."""
    cursor = start_ns
    for stage, value in result.elapsed_ms.items():
        if stage == "total" or value <= 0:
            continue
        end = cursor + round(value * MS_TO_NS)
        span = tracer.start_span(
            f"ocr.engine.{stage}",
            start_time=cursor,
            attributes={"ocr.stage_ms": value, "ocr.stage_timing": "engine-reported"},
        )
        span.end(end_time=end)
        cursor = end
