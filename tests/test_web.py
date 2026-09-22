from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from qwen35_moderation.web.errors import InputError
from qwen35_moderation.web.image_io import decode_image
from qwen35_moderation.web.monitoring import percentile


def test_percentile_interpolates() -> None:
    assert percentile([], 0.95) is None
    assert percentile([10, 20, 30], 0.5) == 20
    assert percentile([10, 20], 0.95) == pytest.approx(19.5)


def test_decode_image_normalizes_to_rgb() -> None:
    source = Image.new("RGBA", (32, 24), (20, 40, 60, 128))
    buffer = BytesIO()
    source.save(buffer, format="PNG")
    decoded = decode_image(buffer.getvalue())
    assert decoded.mode == "RGB"
    assert decoded.size == (32, 24)


def test_decode_image_rejects_invalid_bytes() -> None:
    with pytest.raises(InputError, match="无法解码图片"):
        decode_image(b"not-an-image")
