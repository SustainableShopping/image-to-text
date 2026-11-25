"""Uploaded scan photos, kept for troubleshooting next to the journey records
(journey.py), which reference them by `<day>/<scan_id>-<attempt>.<ext>`.

Layout: `<root>/<YYYY-MM-DD UTC>/<scan_id>-<attempt>.jpg|png|bin`. Day folders older
than `retention_days` are deleted the first time a photo is saved on a new day (and by
`purge()`), so no scheduler is needed. With `max_bytes`, a photo that would take the
folder past that total is not kept (the scan still works), so uploads cannot fill the
disk. IDs are validated (journey.safe_id) before they touch a path, so a crafted ID
cannot leave the root.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from image_to_text.journey import JourneyIds, safe_id

DAY_FORMAT = "%Y-%m-%d"
_SIGNATURES = ((b"\xff\xd8\xff", "jpg"), (b"\x89PNG\r\n\x1a\n", "png"))
MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "png": "image/png",
    "bin": "application/octet-stream",
}

log = logging.getLogger("image_to_text.photo_store")


def _extension(data: bytes) -> str:
    for signature, ext in _SIGNATURES:
        if data.startswith(signature):
            return ext
    return "bin"


class PhotoStore:
    def __init__(
        self,
        root: Path | str,
        retention_days: int,
        clock: Callable[[], float] = time.time,
        max_bytes: int | None = None,
    ):
        if retention_days <= 0:
            raise ValueError("retention_days must be > 0")
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        self.root = Path(root)
        self.retention_days = retention_days
        self.max_bytes = max_bytes
        self._clock = clock

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self._clock(), UTC)

    def used_bytes(self) -> int:
        """Total size of the photos kept now."""
        if not self.root.is_dir():
            return 0
        return sum(f.stat().st_size for f in self.root.glob("*/*") if f.is_file())

    def save(self, ids: JourneyIds, data: bytes) -> str | None:
        """Write the photo; returns its reference relative to the root, or None when
        keeping it would pass `max_bytes`."""
        day = self.root / self._now().strftime(DAY_FORMAT)
        if not day.is_dir():
            day.mkdir(parents=True, exist_ok=True)
            self.purge()
        if (
            self.max_bytes is not None
            and self.used_bytes() + len(data) > self.max_bytes
        ):
            log.warning(
                "scan photo not kept: the photo folder is at its %d-byte cap",
                self.max_bytes,
            )
            return None
        name = f"{ids.scan_id}-{ids.attempt}.{_extension(data)}"
        (day / name).write_bytes(data)
        return f"{day.name}/{name}"

    def find(self, scan_id: str, attempt: int) -> Path | None:
        """The photo of one scan attempt, newest day first, or None."""
        if safe_id(scan_id) is None or not self.root.is_dir():
            return None
        for day in sorted(self.root.iterdir(), reverse=True):
            for ext in MEDIA_TYPES:
                path = day / f"{scan_id}-{attempt}.{ext}"
                if path.is_file():
                    return path
        return None

    def purge(self) -> int:
        """Delete day folders older than the retention; returns how many."""
        if not self.root.is_dir():
            return 0
        cutoff = (self._now() - timedelta(days=self.retention_days)).date()
        removed = 0
        for day in self.root.iterdir():
            try:
                date = datetime.strptime(day.name, DAY_FORMAT).date()
            except ValueError:
                continue  # not ours: leave it alone
            if day.is_dir() and date < cutoff:
                shutil.rmtree(day, ignore_errors=True)
                removed += 1
        if removed:
            log.info("purged %d day(s) of scan photos older than %s", removed, cutoff)
        return removed
