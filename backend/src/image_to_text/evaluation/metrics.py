"""Text accuracy metrics for OCR output. Pure functions, standard library only.

The expected text is the ingredient section only, while OCR reads the whole photo, so
the section is scored where it best fits inside the OCR text: text around it costs
nothing. Case and runs of whitespace (line breaks) do not count; the ingredient parser
ignores both.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Hashable, Sequence
from statistics import quantiles

_WORD = re.compile(r"[^\W_]+")


def substring_distance(pattern: Sequence[Hashable], text: Sequence[Hashable]) -> int:
    """Fewest edits turning `pattern` into some part of `text` (Sellers' algorithm)."""
    prev = [0] * (len(text) + 1)
    for i, x in enumerate(pattern, 1):
        cur = [i]
        for j, y in enumerate(text, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return min(prev)


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def section_cer(expected: str, ocr_text: str) -> float:
    """Character error rate of the expected section, aligned inside the OCR text."""
    e = _norm(expected)
    return round(substring_distance(e, _norm(ocr_text)) / max(len(e), 1), 4)


def section_wer(expected: str, ocr_text: str) -> float:
    """The same in whole words: one misread letter makes the whole word wrong."""
    e = _norm(expected).split()
    return round(substring_distance(e, _norm(ocr_text).split()) / max(len(e), 1), 4)


def word_recall(expected: str, ocr_text: str) -> float | None:
    """Share of the expected words (letters and digits only) that OCR read somewhere."""
    want = Counter(_WORD.findall(expected.casefold()))
    if not want:
        return None
    got = Counter(_WORD.findall(ocr_text.casefold()))
    return round(sum((want & got).values()) / sum(want.values()), 4)


def pct(values: Sequence[float], q: int) -> float | None:
    """q-th percentile (inclusive, 1 decimal), as the ingredient service reports it."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return round(quantiles(values, n=100, method="inclusive")[q - 1], 1)
