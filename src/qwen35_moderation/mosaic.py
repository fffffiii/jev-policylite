"""多图拼接压力测试的图像与标签工具。"""

from __future__ import annotations

from collections.abc import Sequence
from math import isqrt

from PIL import Image, ImageOps


MOSAIC_SCENARIOS = ("all_safe", "one_violation", "two_violations", "all_violation")


def scenario_labels(name: str, tile_count: int = 4) -> tuple[int, ...]:
    """返回固定场景下每个格子的二元标签，标签聚合规则为任一违规即违规。"""
    if tile_count != 4:
        raise ValueError("当前 2×2 压力测试只支持 4 张图")
    plans = {
        "all_safe": (0, 0, 0, 0),
        "one_violation": (1, 0, 0, 0),
        "two_violations": (1, 1, 0, 0),
        "all_violation": (1, 1, 1, 1),
    }
    if name not in plans:
        raise ValueError(f"未知拼接场景：{name}；可选值：{', '.join(MOSAIC_SCENARIOS)}")
    return plans[name]


def aggregate_violation(labels: Sequence[int]) -> int:
    """以 OR 规则聚合拼图标签。"""
    if not labels:
        raise ValueError("拼图至少需要一张图片")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("拼图标签只能是 0 或 1")
    return int(any(labels))


def make_mosaic(
    images: Sequence[Image.Image],
    *,
    tile_size: int = 224,
    background: tuple[int, int, int] = (18, 18, 18),
) -> Image.Image:
    """按行优先顺序制作等比留边的方形拼图，不裁剪原图主体。"""
    if tile_size < 32:
        raise ValueError("tile_size 至少为 32")
    side = isqrt(len(images))
    if side * side != len(images) or side < 1:
        raise ValueError("图片数量必须是完全平方数，例如 4 或 9")
    canvas = Image.new("RGB", (side * tile_size, side * tile_size), background)
    for index, image in enumerate(images):
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        fitted = ImageOps.contain(normalized, (tile_size, tile_size), Image.Resampling.LANCZOS)
        x = (index % side) * tile_size + (tile_size - fitted.width) // 2
        y = (index // side) * tile_size + (tile_size - fitted.height) // 2
        canvas.paste(fitted, (x, y))
    return canvas
