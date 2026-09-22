from __future__ import annotations

from PIL import Image

from qwen35_moderation.mosaic import aggregate_violation, make_mosaic, scenario_labels


def test_scenario_labels_and_or_aggregation() -> None:
    assert scenario_labels("all_safe") == (0, 0, 0, 0)
    assert scenario_labels("one_violation") == (1, 0, 0, 0)
    assert aggregate_violation((0, 0, 0, 0)) == 0
    assert aggregate_violation((0, 1, 0, 0)) == 1


def test_make_mosaic_keeps_four_tiles() -> None:
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    images = [Image.new("RGB", (12, 8), color) for color in colors]
    mosaic = make_mosaic(images, tile_size=32)
    assert mosaic.size == (64, 64)
    assert mosaic.getpixel((16, 16)) == colors[0]
    assert mosaic.getpixel((48, 16)) == colors[1]
    assert mosaic.getpixel((16, 48)) == colors[2]
    assert mosaic.getpixel((48, 48)) == colors[3]
