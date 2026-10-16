"""Upload bytes → RGB array."""

from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

log = logging.getLogger(__name__)


def decode_image(raw: bytes) -> np.ndarray | None:
    """HxWx3 RGB uint8 with EXIF rotation applied, or None if not an image."""
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
        arr = np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError) as e:
        log.info("undecodable upload: %s", type(e).__name__)
        return None
    return arr if arr.size else None
