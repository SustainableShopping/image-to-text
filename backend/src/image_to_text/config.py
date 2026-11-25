"""Runtime settings, read once from the environment.

Every variable is prefixed `ITT_` and optional; defaults live on `Settings`.
Invalid values fail at startup with a message naming the variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ENV_PREFIX = "ITT_"
# The standard OpenTelemetry variable, not ITT_-prefixed, so the same name works for
# every service and for the OTel SDK itself. Unset: tracing is off.
OTEL_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})
# The photo endpoint is reachable from the internet on dev; refuse guessable passwords.
MIN_VIEW_PASSWORD = 12


def _raw(name: str) -> str | None:
    value = os.environ.get(ENV_PREFIX + name)
    return value.strip() if value and value.strip() else None


def _env_str(name: str, default: str) -> str:
    return _raw(name) or default


def _env_int(name: str, default: int) -> int:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{ENV_PREFIX}{name} must be an integer, got {raw!r}") from e


def _env_float(name: str, default: float) -> float:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{ENV_PREFIX}{name} must be a number, got {raw!r}") from e


def _env_bool(name: str, default: bool) -> bool:
    raw = _raw(name)
    if raw is None:
        return default
    if raw.lower() in _TRUE:
        return True
    if raw.lower() in _FALSE:
        return False
    raise ValueError(f"{ENV_PREFIX}{name} must be true or false, got {raw!r}")


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _raw(name)
    if raw is None:
        return default
    items = tuple(s.strip() for s in raw.split(",") if s.strip())
    return items or default


@dataclass(frozen=True)
class Settings:
    cors_origins: tuple[str, ...] = ()  # API clients by default; browsers opt in
    max_upload_bytes: int = 12 * 1024 * 1024  # a 12 MP phone JPEG is ~3-6 MB
    log_level: str = "INFO"
    default_engine: str = "doctr"
    doctr_det_arch: str = "fast_base"
    doctr_reco_arch: str = "crnn_vgg16_bn"
    det_limit_side_len: int = 960
    use_angle_cls: bool = False
    onnx_threads: int = 8
    text_score: float = 0.5
    warm_up: bool = True
    otel_endpoint: str | None = None  # OTLP/HTTP base URL, e.g. http://collector:4318
    # Scan journey records (journey.py): full input/output of every /ocr stage, sent
    # as OTel logs. Needs otel_endpoint too. Off by default: they hold label text.
    journey_enabled: bool = False
    journey_photo_dir: str | None = None  # keep uploaded photos here; unset: don't
    # 14 days = the dev OpenObserve retention, so photos and records expire together.
    journey_photo_retention_days: int = 14
    # 512 KiB fits the longest real label's OCR output with room to spare, and stays
    # well under the collector's 4 MiB request limit.
    journey_max_body_bytes: int = 512 * 1024
    # Cap on all kept photos together: 2 GiB is ~500-1000 phone photos and a small share
    # of the dev VM disk. Past it, photos are not kept (scans still work).
    journey_photo_max_mb: int = 2048
    journey_view_user: str = "journey"
    journey_view_password: str | None = None  # unset: photo endpoint off

    def __post_init__(self) -> None:
        for name in (
            "max_upload_bytes",
            "det_limit_side_len",
            "onnx_threads",
            "journey_photo_retention_days",
            "journey_max_body_bytes",
            "journey_photo_max_mb",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{ENV_PREFIX}{name.upper()} must be > 0")
        password = self.journey_view_password
        if password is not None and len(password) < MIN_VIEW_PASSWORD:
            raise ValueError(
                f"{ENV_PREFIX}JOURNEY_VIEW_PASSWORD must be at least "
                f"{MIN_VIEW_PASSWORD} characters"
            )
        if not 0.0 <= self.text_score <= 1.0:
            raise ValueError(f"{ENV_PREFIX}TEXT_SCORE must be between 0 and 1")
        if self.log_level not in _LOG_LEVELS:
            raise ValueError(
                f"{ENV_PREFIX}LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}, "
                f"got {self.log_level!r}"
            )

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            cors_origins=_env_list("CORS_ORIGINS", d.cors_origins),
            max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", d.max_upload_bytes),
            log_level=_env_str("LOG_LEVEL", d.log_level).upper(),
            default_engine=_env_str("DEFAULT_ENGINE", d.default_engine).lower(),
            doctr_det_arch=_env_str("DOCTR_DET_ARCH", d.doctr_det_arch),
            doctr_reco_arch=_env_str("DOCTR_RECO_ARCH", d.doctr_reco_arch),
            det_limit_side_len=_env_int("DET_LIMIT_SIDE_LEN", d.det_limit_side_len),
            use_angle_cls=_env_bool("USE_ANGLE_CLS", d.use_angle_cls),
            onnx_threads=_env_int("ONNX_THREADS", d.onnx_threads),
            text_score=_env_float("TEXT_SCORE", d.text_score),
            warm_up=_env_bool("WARM_UP", d.warm_up),
            otel_endpoint=(os.environ.get(OTEL_ENDPOINT_ENV) or "").strip() or None,
            journey_enabled=_env_bool("JOURNEY_ENABLED", d.journey_enabled),
            journey_photo_dir=_raw("JOURNEY_PHOTO_DIR"),
            journey_photo_retention_days=_env_int(
                "JOURNEY_PHOTO_RETENTION_DAYS", d.journey_photo_retention_days
            ),
            journey_max_body_bytes=_env_int(
                "JOURNEY_MAX_BODY_BYTES", d.journey_max_body_bytes
            ),
            journey_photo_max_mb=_env_int(
                "JOURNEY_PHOTO_MAX_MB", d.journey_photo_max_mb
            ),
            journey_view_user=_env_str("JOURNEY_VIEW_USER", d.journey_view_user),
            journey_view_password=_raw("JOURNEY_VIEW_PASSWORD"),
        )
