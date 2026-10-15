"""Runtime settings, read once from the environment.

Every variable is prefixed `ITT_` and optional; defaults live on `Settings`.
Invalid values fail at startup with a message naming the variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ENV_PREFIX = "ITT_"
_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


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

    def __post_init__(self) -> None:
        if self.max_upload_bytes <= 0:
            raise ValueError(f"{ENV_PREFIX}MAX_UPLOAD_BYTES must be > 0")
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
        )
