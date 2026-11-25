import time
from datetime import UTC, datetime

import pytest

from image_to_text.journey import JourneyIds
from image_to_text.photo_store import PhotoStore

JPEG = b"\xff\xd8\xff\xe0" + b"jpeg-bytes"
PNG = b"\x89PNG\r\n\x1a\n" + b"png-bytes"
NOW = datetime(2026, 11, 24, 10, 0, tzinfo=UTC).timestamp()


def _store(root, now: float = NOW) -> PhotoStore:
    return PhotoStore(root, retention_days=14, clock=lambda: now)


def test_save_writes_under_the_day_folder_with_the_right_extension(tmp_path):
    ref = _store(tmp_path).save(JourneyIds("s", "scan1", 2), JPEG)
    assert ref == "2026-11-24/scan1-2.jpg"
    assert (tmp_path / ref).read_bytes() == JPEG


def test_png_and_unknown_formats(tmp_path):
    store = _store(tmp_path)
    assert store.save(JourneyIds("s", "p", 1), PNG).endswith("p-1.png")
    assert store.save(JourneyIds("s", "u", 1), b"????").endswith("u-1.bin")


def test_find_returns_the_saved_photo(tmp_path):
    store = _store(tmp_path)
    store.save(JourneyIds("s", "scan1", 1), JPEG)
    found = store.find("scan1", 1)
    assert found is not None
    assert found.read_bytes() == JPEG
    assert store.find("scan1", 2) is None
    assert store.find("other", 1) is None


@pytest.mark.parametrize("bad", ["..", "../x", "a/b", "*", "x" * 65, ""])
def test_find_rejects_unsafe_ids(tmp_path, bad):
    assert _store(tmp_path).find(bad, 1) is None


def test_find_without_any_photos_yet(tmp_path):
    assert _store(tmp_path / "missing").find("scan1", 1) is None


def test_purge_removes_days_older_than_retention(tmp_path):
    old = tmp_path / "2026-11-01"
    old.mkdir()
    (old / "scan-1.jpg").write_bytes(JPEG)
    recent = tmp_path / "2026-11-20"
    recent.mkdir()
    (recent / "scan-1.jpg").write_bytes(JPEG)
    (tmp_path / "not-a-date").mkdir()

    removed = _store(tmp_path).purge()

    assert removed == 1
    assert not old.exists()
    assert recent.exists()
    assert (tmp_path / "not-a-date").exists()


def test_purge_without_root_is_a_no_op(tmp_path):
    assert _store(tmp_path / "missing").purge() == 0


def test_first_save_of_a_day_purges_old_days(tmp_path):
    old = tmp_path / "2026-10-01"
    old.mkdir()
    (old / "scan-1.jpg").write_bytes(JPEG)
    _store(tmp_path).save(JourneyIds("s", "scan2", 1), JPEG)
    assert not old.exists()


def test_store_creates_its_root(tmp_path):
    root = tmp_path / "nested" / "photos"
    _store(root).save(JourneyIds("s", "scan1", 1), JPEG)
    assert root.is_dir()


def test_default_clock_is_wall_time(tmp_path):
    ref = PhotoStore(tmp_path, retention_days=1).save(JourneyIds("s", "x", 1), JPEG)
    assert ref.startswith(time.strftime("%Y-%m-%d/", time.gmtime()))


def test_retention_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        PhotoStore(tmp_path, retention_days=0)


def test_day_boundary_uses_utc(tmp_path):
    late = datetime(2026, 11, 24, 23, 59, tzinfo=UTC).timestamp()
    ref = _store(tmp_path, now=late).save(JourneyIds("s", "x", 1), JPEG)
    assert ref.startswith("2026-11-24/")


def test_save_refuses_photos_beyond_the_total_size_cap(tmp_path):
    store = PhotoStore(
        tmp_path, retention_days=14, clock=lambda: NOW, max_bytes=len(JPEG) * 2
    )
    assert store.save(JourneyIds("s", "a", 1), JPEG) is not None
    assert store.save(JourneyIds("s", "b", 1), JPEG) is not None
    assert store.save(JourneyIds("s", "c", 1), JPEG) is None
    assert store.find("c", 1) is None


def test_max_bytes_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        PhotoStore(tmp_path, retention_days=1, max_bytes=0)
