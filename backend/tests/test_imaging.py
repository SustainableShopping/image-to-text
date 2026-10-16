import io

from PIL import Image

from image_to_text.imaging import decode_image

EXIF_ORIENTATION = 0x0112
ROTATE_90_CW = 6


def _encode(img: Image.Image, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", **kw)
    return buf.getvalue()


def test_decodes_to_rgb_array():
    arr = decode_image(_encode(Image.new("L", (40, 20), 128)))
    assert arr is not None
    assert arr.shape == (20, 40, 3)


def test_applies_exif_rotation():
    exif = Image.Exif()
    exif[EXIF_ORIENTATION] = ROTATE_90_CW
    arr = decode_image(_encode(Image.new("RGB", (40, 20)), exif=exif))
    assert arr is not None
    assert arr.shape == (40, 20, 3)


def test_returns_none_for_non_images():
    assert decode_image(b"definitely not an image") is None
    assert decode_image(b"") is None
