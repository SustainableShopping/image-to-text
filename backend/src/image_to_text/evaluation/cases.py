"""Evaluation cases: one `<id>.case.json` per label photo.

Only the OCR fields are read here (`id`, `image`, `ocr_text`, `tags`); `text`,
`expect_status` and `ingredients` belong to the later steps and are accepted unread.
Keys starting with "_" are notes. Any other key is an error, so a typo cannot silently
skip a check.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CASE_SUFFIX = ".case.json"
DRAFT_SUFFIX = ".case.draft.json"
_KEYS = {"id", "image", "tags", "ocr_text", "expect_status", "text", "ingredients"}


class CaseError(ValueError):
    """A case file that cannot be used; the message names the file."""


@dataclass(frozen=True)
class OcrCase:
    id: str
    path: Path
    image: str | None = None  # file name, next to the case file or in --images
    tags: tuple[str, ...] = ()
    ocr_text: str | None = None  # the ingredient section as printed, heading included

    @property
    def folder(self) -> Path:
        return self.path.parent


def _optional_str(data: dict[str, Any], name: str, path: Path) -> str | None:
    value = data.get(name)
    if value is not None and not isinstance(value, str):
        raise CaseError(f"{path}: '{name}' must be a string")
    return value


def case_from_json(data: Any, path: Path) -> OcrCase:
    if not isinstance(data, dict):
        raise CaseError(f"{path}: a case must be a JSON object")
    for k in data:
        if k not in _KEYS and not k.startswith("_"):
            raise CaseError(
                f"{path}: unknown field '{k}' (allowed: {', '.join(sorted(_KEYS))})"
            )
    case_id = data.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise CaseError(f"{path}: 'id' is required")
    tags = data.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise CaseError(f"{path}: 'tags' must be a list of strings")
    return OcrCase(
        id=case_id.strip(),
        path=path,
        image=_optional_str(data, "image", path),
        tags=tuple(tags),
        ocr_text=_optional_str(data, "ocr_text", path),
    )


def _read(path: Path) -> OcrCase:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CaseError(f"{path}: not valid JSON ({e})") from None
    except UnicodeDecodeError as e:
        raise CaseError(f"{path}: not UTF-8 text ({e})") from None
    return case_from_json(data, path)


def load_cases(paths: Sequence[Path]) -> list[OcrCase]:
    """Every *.case.json in the given folders (drafts skipped) or the given files."""
    cases: list[OcrCase] = []
    for path in map(Path, paths):
        if path.is_dir():
            cases.extend(_read(p) for p in sorted(path.rglob(f"*{CASE_SUFFIX}")))
        elif path.is_file():
            cases.append(_read(path))
        else:
            raise CaseError(f"{path}: not found")
    seen: dict[str, Path] = {}
    for case in cases:
        if case.id in seen:
            raise CaseError(
                f"{case.path}: duplicate case id '{case.id}' (also in {seen[case.id]})"
            )
        seen[case.id] = case.path
    return cases
