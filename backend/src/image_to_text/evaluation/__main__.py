"""Command line for the OCR evaluation. Run from backend/:

    uv run python -m image_to_text.evaluation           # ../eval/cases, all engines
    uv run python -m image_to_text.evaluation --images P --engine rapidocr --repeat 3
    uv run python -m image_to_text.evaluation --images P --draft  # draft expected text

Writes one JSON line per (case, engine) to --out: the input for scoring the later
pipeline steps on real OCR text. Exit code: 0 done, 1 every read failed (or a draft's
photo could not be read), 2 bad input (case files, unknown engine).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from image_to_text.config import Settings
from image_to_text.evaluation.cases import DRAFT_SUFFIX, CaseError, OcrCase, load_cases
from image_to_text.evaluation.run import evaluate, render_table, summarize
from image_to_text.ocr.base import Engine
from image_to_text.ocr.router import EngineRegistry

DEFAULT_CASES = Path("../eval/cases")
DEFAULT_RESULTS = Path("../eval/results")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
DRAFT_NOTE = (
    "Draft made from OCR output. Keep only the ingredient section as printed "
    "(heading included), correct every misread against the photo, then save as "
    "<id>.case.json."
)


def build_engines(names: Sequence[str] | None) -> dict[str, Engine]:
    registry = EngineRegistry(Settings.from_env())
    return {name: registry.get(name) for name in (names or registry.names)}


def _args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m image_to_text.evaluation",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--cases",
        action="append",
        type=Path,
        help=f"case folder or file (default {DEFAULT_CASES})",
    )
    ap.add_argument(
        "--images",
        type=Path,
        help="folder with the photos (default: next to each case file)",
    )
    ap.add_argument(
        "--engine", action="append", help="engine to run, repeatable (default: all)"
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="timed reads per photo after a warm-up (default 1)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        help=f"results JSONL (default {DEFAULT_RESULTS}/ocr-<time>.jsonl)",
    )
    ap.add_argument(
        "--draft",
        action="store_true",
        help="write <id>.case.draft.json for photos without expected text "
        "(first --engine only; existing drafts are kept)",
    )
    args = ap.parse_args(argv)
    if args.repeat < 1:
        ap.error("--repeat must be at least 1")
    return args


def _write_json(path: Path, data: object) -> None:
    path.write_bytes(
        (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )


def _draft_targets(
    cases: Sequence[OcrCase], folders: Sequence[Path], images: Path | None
) -> list[OcrCase]:
    """Cases with a photo but no expected text, plus photos that have no case yet."""
    targets = [c for c in cases if c.image and not c.ocr_text]
    known = {c.id for c in cases} | {c.image for c in cases if c.image}
    home = folders[0] if folders[0].is_dir() else folders[0].parent
    for folder in [*(f for f in folders if f.is_dir()), *([images] if images else [])]:
        for photo in sorted(
            p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        ):
            if photo.stem not in known and photo.name not in known:
                known.add(photo.stem)
                targets.append(  # full path: read from where it was found
                    OcrCase(
                        id=photo.stem,
                        path=home / f"{photo.stem}.case.json",
                        image=str(photo.resolve()),
                    )
                )
    return targets


def _draft_path(case: OcrCase) -> Path:
    return case.folder / f"{case.id}{DRAFT_SUFFIX}"


def write_drafts(
    cases: Sequence[OcrCase],
    folders: Sequence[Path],
    images: Path | None,
    engines: Mapping[str, Engine],
) -> tuple[list[Path], int]:
    """(drafts written, photos that could not be read). Uses the first engine only.
    A draft that already exists is left alone: it may hold corrections."""
    first = dict(list(engines.items())[:1])
    targets = []
    for case in _draft_targets(cases, folders, images):
        if _draft_path(case).exists():
            print(f"skipped {case.id}: {_draft_path(case)} exists", file=sys.stderr)
        else:
            targets.append(case)
    rows = {r["case_id"]: r for r in evaluate(targets, first, images or None, 1)}
    written, failed = [], 0
    for case in targets:
        row = rows.get(case.id)
        if row is None or row["error"]:
            failed += 1
            print(
                f"skipped {case.id}: {row['error'] if row else 'no photo'}",
                file=sys.stderr,
            )
            continue
        draft = {
            "_note": DRAFT_NOTE + f" Read by {next(iter(first))}.",
            "id": case.id,
            "image": Path(str(case.image)).name,
            "tags": list(case.tags),
            "ocr_text": row["text"],
        }
        _write_json(_draft_path(case), draft)
        written.append(_draft_path(case))
    return written, failed


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    folders = args.cases or [DEFAULT_CASES]
    try:
        cases = load_cases(folders)
    except CaseError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        engines = build_engines(args.engine)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2
    if args.draft:
        written, failed = write_drafts(cases, folders, args.images, engines)
        for path in written:
            print(f"draft: {path}")
        return 1 if failed else 0
    rows = evaluate(cases, engines, args.images, args.repeat)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = args.out or DEFAULT_RESULTS / f"ocr-{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    )
    readable = sum(1 for c in cases if c.image)
    print(
        f"{readable} cases x {len(engines)} engines, {args.repeat} timed reads each\n"
    )
    print(render_table(summarize(rows)))
    print(f"\nresults: {out}")
    if rows and all(r["error"] for r in rows):
        print(
            "error: every read failed (see the error field in the results)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
