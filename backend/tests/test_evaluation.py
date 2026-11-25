"""OCR evaluation: metrics against hand-computed values, and the runner with a fake
engine (no models, no network)."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_to_text.evaluation import __main__ as cli
from image_to_text.evaluation.cases import CaseError, load_cases
from image_to_text.evaluation.metrics import (
    pct,
    section_cer,
    section_wer,
    substring_distance,
    word_recall,
)
from image_to_text.evaluation.run import evaluate, render_table, summarize
from image_to_text.ocr.base import EngineResult, Line


class FakeEngine:
    def __init__(self, name: str, lines: list[str], fail: bool = False):
        self.name = name
        self._lines = lines
        self._fail = fail
        self.calls = 0
        self.shapes: list[tuple[int, ...]] = []

    def read(self, image: np.ndarray) -> EngineResult:
        self.calls += 1
        self.shapes.append(image.shape)
        if self._fail:
            raise RuntimeError("model crashed")
        box = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        lines = tuple(Line(text=t, conf=0.9, box=box) for t in self._lines)
        return EngineResult(engine=self.name, lines=lines, elapsed_ms={"total": 5.0})


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return buf.getvalue()


def write_case(folder: Path, data: dict, image: bytes | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{data['id']}.case.json").write_text(json.dumps(data), encoding="utf-8")
    if image is not None:
        (folder / data["image"]).write_bytes(image)


# ---- metrics ----------------------------------------------------------------


def test_substring_distance():
    assert substring_distance("folic", "tolic") == 1
    assert substring_distance(["a", "b"], ["x", "a", "c", "b", "y"]) == 1
    # the pattern is found inside a longer text: the text around it costs nothing
    assert (
        substring_distance("folic acid", "nutrition facts ... tolic acid, sugar") == 1
    )
    assert substring_distance("", "anything") == 0


def test_section_scores_ignore_case_spacing_and_surrounding_text():
    expected = "Ingredients: Sugar, folic acid."
    ocr = "NUTRITION FACTS\nINGREDIENTS: SUGAR,\nTOLIC ACID.\nNet wt 100 g"
    assert section_cer(expected, ocr) == round(
        1 / len("ingredients: sugar, folic acid."), 4
    )
    assert section_wer(expected, ocr) == 0.25  # 1 of 4 words wrong
    assert (
        word_recall(expected, ocr) == 0.75
    )  # ingredients, sugar, acid found; folic not
    assert section_cer("", "x") == 0.0 and word_recall("", "x") is None


def test_pct_matches_the_ingredient_service():
    assert pct([], 50) is None
    assert pct([4.0], 95) == 4.0
    assert pct([1.0, 2.0, 3.0, 4.0], 50) == 2.5


# ---- cases ------------------------------------------------------------------


def test_loads_the_shared_case_format(tmp_path):
    write_case(
        tmp_path,
        {
            "id": "a",
            "image": "a.png",
            "tags": ["glare"],
            "ocr_text": "Ingredients: salt.",
            "ingredients": [{"text": "salt"}],
            "_note": "x",
        },
    )
    [case] = load_cases([tmp_path])
    assert (case.id, case.image, case.tags, case.ocr_text) == (
        "a",
        "a.png",
        ("glare",),
        "Ingredients: salt.",
    )
    assert case.folder == tmp_path


@pytest.mark.parametrize(
    "data, message",
    [
        ({"image": "a.png"}, "'id'"),
        ({"id": "a", "colour": 1}, "unknown field 'colour'"),
        ({"id": "a", "tags": "x"}, "'tags'"),
        ({"id": "a", "ocr_text": 5}, "'ocr_text'"),
    ],
)
def test_rejects_bad_cases(tmp_path, data, message):
    (tmp_path / "bad.case.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CaseError, match=message):
        load_cases([tmp_path])


def test_rejects_duplicates_and_missing_folders(tmp_path):
    write_case(tmp_path / "x", {"id": "same"})
    write_case(tmp_path / "y", {"id": "same"})
    with pytest.raises(CaseError, match="duplicate case id 'same'"):
        load_cases([tmp_path])
    with pytest.raises(CaseError, match="not found"):
        load_cases([tmp_path / "nope"])


# ---- runner -----------------------------------------------------------------


def test_evaluate_scores_each_engine_and_records_failures(tmp_path):
    write_case(
        tmp_path,
        {"id": "a", "image": "a.png", "tags": ["t"], "ocr_text": "Ingredients: sugar."},
        png(),
    )
    write_case(tmp_path, {"id": "no-photo", "image": "missing.png"})
    write_case(tmp_path, {"id": "text-only", "ocr_text": "x"})
    good = FakeEngine("good", ["Nutrition", "Ingredients: sugar."])
    bad = FakeEngine("bad", [], fail=True)
    rows = evaluate(
        load_cases([tmp_path]), {"good": good, "bad": bad}, images=None, repeat=2
    )

    by = {(r["case_id"], r["engine"]): r for r in rows}
    ok = by[("a", "good")]
    assert ok["text"] == "Nutrition\nIngredients: sugar."
    assert (ok["cer"], ok["wer"], ok["word_recall"], ok["lines"]) == (0.0, 0.0, 1.0, 2)
    assert ok["mean_conf"] == 0.9 and ok["error"] is None
    assert len(ok["timings_ms"]["read"]) == 2 and ok["timings_ms"]["engine_total"] == [
        5.0,
        5.0,
    ]
    assert good.calls == 3  # one warm-up, then two timed reads
    assert by[("a", "bad")]["error"] == "RuntimeError: model crashed"
    assert by[("a", "bad")]["text"] is None
    assert by[("no-photo", "good")]["error"].startswith("image not found")
    assert ("text-only", "good") not in by  # nothing to read

    summary = summarize(rows)
    assert summary["engines"]["good"]["cer"] == 0.0
    assert summary["engines"]["good"]["errors"] == 1
    assert summary["engines"]["bad"]["errors"] == 2
    assert summary["by_tag"]["t"]["good"]["cases"] == 1
    table = render_table(summary)
    assert "good" in table and "CER" in table


def test_warm_up_reads_a_real_photo_so_no_timed_read_starts_cold(tmp_path):
    write_case(tmp_path, {"id": "a", "image": "a.png"}, png())
    engine = FakeEngine("e", ["x"])
    evaluate(load_cases([tmp_path]), {"e": engine}, None, 1)
    assert engine.shapes == [
        (8, 8, 3),
        (8, 8, 3),
    ]  # warm-up on the photo, then the read
    with pytest.raises(ValueError, match="at least 1"):
        evaluate(load_cases([tmp_path]), {"e": engine}, None, 0)


def test_without_any_photo_the_warm_up_uses_a_blank_image(tmp_path):
    write_case(tmp_path, {"id": "a", "image": "missing.png"})
    engine = FakeEngine("e", ["x"])
    evaluate(load_cases([tmp_path]), {"e": engine}, None, 1)
    assert engine.shapes == [(64, 256, 3)]


def test_rejects_case_files_that_are_not_utf8(tmp_path):
    (tmp_path / "a.case.json").write_bytes(b'{"id": "\xff"}')
    with pytest.raises(CaseError, match="not UTF-8"):
        load_cases([tmp_path])


def test_an_undecodable_image_is_an_error_row(tmp_path):
    write_case(tmp_path, {"id": "a", "image": "a.png"}, b"not an image")
    [row] = evaluate(load_cases([tmp_path]), {"e": FakeEngine("e", ["x"])}, None, 1)
    assert row["error"] == "image could not be decoded"


def test_images_folder_overrides_the_case_folder(tmp_path):
    write_case(tmp_path / "cases", {"id": "a", "image": "a.png"})
    (tmp_path / "photos").mkdir()
    (tmp_path / "photos" / "a.png").write_bytes(png())
    [row] = evaluate(
        load_cases([tmp_path / "cases"]),
        {"e": FakeEngine("e", ["x"])},
        tmp_path / "photos",
        1,
    )
    assert row["error"] is None and row["text"] == "x"


# ---- command line -----------------------------------------------------------


@pytest.fixture()
def fake_engines(monkeypatch):
    engines = {"doctr": FakeEngine("doctr", ["Ingredients: sugar."])}
    monkeypatch.setattr(cli, "build_engines", lambda names: engines)
    return engines


def test_cli_writes_results_jsonl(tmp_path, fake_engines, capsys):
    write_case(
        tmp_path / "cases",
        {"id": "a", "image": "a.png", "ocr_text": "Ingredients: sugar."},
        png(),
    )
    out = tmp_path / "out" / "ocr.jsonl"
    code = cli.main(["--cases", str(tmp_path / "cases"), "--out", str(out)])
    assert code == 0
    [row] = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert row["case_id"] == "a" and row["engine"] == "doctr" and row["cer"] == 0.0
    assert b"\r\n" not in out.read_bytes()
    assert "1 cases x 1 engines" in capsys.readouterr().out


def test_cli_drafts_cases_for_new_photos_and_missing_text(
    tmp_path, fake_engines, capsys
):
    cases = tmp_path / "cases"
    write_case(cases, {"id": "known", "image": "known.png"}, png())
    (cases / "new-label.png").write_bytes(png())
    assert cli.main(["--cases", str(cases), "--draft"]) == 0
    drafts = sorted(p.name for p in cases.glob("*.case.draft.json"))
    assert drafts == ["known.case.draft.json", "new-label.case.draft.json"]
    data = json.loads((cases / "new-label.case.draft.json").read_text(encoding="utf-8"))
    assert (
        data["image"] == "new-label.png" and data["ocr_text"] == "Ingredients: sugar."
    )
    assert data["_note"].startswith("Draft")


def test_cli_exit_2_on_bad_input(tmp_path, fake_engines, capsys):
    assert cli.main(["--cases", str(tmp_path / "nope")]) == 2
    assert "not found" in capsys.readouterr().err
