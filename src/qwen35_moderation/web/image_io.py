from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import InputError


def decode_image(raw: bytes) -> Image.Image:
    """校验上传图片并统一为 RGB，避免把原文件交给模型处理。"""
    try:
        with Image.open(BytesIO(raw)) as source:
            source.verify()
        with Image.open(BytesIO(raw)) as source:
            if source.width * source.height > 25_000_000:
                raise InputError("图片像素数不能超过 2500 万。", "image-too-large")
            image = ImageOps.exif_transpose(source).convert("RGB")
            return image.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InputError("无法解码图片，请上传 JPEG、PNG 或 WebP。", "invalid-image") from exc
