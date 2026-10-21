"""Picks the engine for a request: `?engine=` or the configured default.
Engines are built lazily, once per process."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping

import numpy as np

from image_to_text.config import ENV_PREFIX, Settings
from image_to_text.ocr.base import Engine
from image_to_text.ocr.doctr_engine import ENGINE_DOCTR, DoctrEngine
from image_to_text.ocr.rapidocr_engine import ENGINE_RAPIDOCR, RapidOcrEngine

log = logging.getLogger(__name__)

EngineFactory = Callable[[Settings], Engine]

FACTORIES: Mapping[str, EngineFactory] = {
    ENGINE_DOCTR: DoctrEngine,
    ENGINE_RAPIDOCR: RapidOcrEngine,
}


class EngineRegistry:
    def __init__(
        self, settings: Settings, factories: Mapping[str, EngineFactory] = FACTORIES
    ):
        if settings.default_engine not in factories:
            raise ValueError(
                f"{ENV_PREFIX}DEFAULT_ENGINE must be one of {tuple(factories)}, "
                f"got {settings.default_engine!r}"
            )
        self._settings = settings
        self._factories = dict(factories)
        self._engines: dict[str, Engine] = {}
        self._lock = threading.Lock()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._factories)

    @property
    def default(self) -> str:
        return self._settings.default_engine

    def get(self, name: str | None = None) -> Engine:
        key = name or self.default
        if key not in self._factories:
            raise KeyError(f"unknown engine {key!r}; choose one of {self.names}")
        with self._lock:
            if key not in self._engines:
                log.info("loading OCR engine %s", key)
                self._engines[key] = self._factories[key](self._settings)
            return self._engines[key]

    def warm_up(self) -> None:
        self.get().read(np.full((64, 256, 3), 255, dtype=np.uint8))
