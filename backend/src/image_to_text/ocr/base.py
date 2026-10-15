"""The engine interface and the shapes every engine is normalised into.

Everything downstream sees only `Line`/`Word` objects, so adding or swapping an
engine is a new adapter module plus a router entry, with no call-site changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Word:
    text: str
    conf: float


@dataclass(frozen=True)
class Line:
    text: str
    conf: float
    box: tuple[tuple[float, float], ...]  # 4 corners, (x, y), image pixels
    words: tuple[Word, ...] = ()  # per-word confidence when the engine gives it

    @property
    def top(self) -> float:
        return min(y for _, y in self.box)

    @property
    def left(self) -> float:
        return min(x for x, _ in self.box)

    @property
    def height(self) -> float:
        return max(y for _, y in self.box) - self.top


@dataclass(frozen=True)
class EngineResult:
    engine: str  # which engine read it; accuracy is tracked per engine
    lines: tuple[Line, ...]
    elapsed_ms: dict[str, float]


class Engine(Protocol):
    name: str

    def read(self, image: np.ndarray) -> EngineResult:
        """`image` is an HxWx3 RGB uint8 array."""
        ...


def to_ms(seconds: float) -> float:
    """Seconds → milliseconds, rounded to 0.1."""
    return round(seconds * 1000, 1)


def ms(**stages: float | None) -> dict[str, float]:
    """Seconds per stage → milliseconds; a missing stage counts as 0."""
    return {k: to_ms(v or 0.0) for k, v in stages.items()}
