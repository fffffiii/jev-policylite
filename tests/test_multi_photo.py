from __future__ import annotations

import pytest
import torch

from qwen35_moderation.multi_photo import FourPositionHeads, SharedPositionHead, native_inputs, validate_four_image_layout, vision_end_features


def test_four_heads_produce_one_probability_logit_per_position() -> None:
    model = FourPositionHeads(hidden_size=8)
    output = model(torch.randn(3, 8))
    assert output.shape == (3, 4)
    assert model(torch.randn(3, 4, 8)).shape == (3, 4)
    assert sum(parameter.numel() for parameter in model.parameters()) == 4 * (8 * 2 + 8 + 1)
    shared = SharedPositionHead(hidden_size=8)
    assert shared(torch.randn(3, 4, 8)).shape == (3, 4)
    assert sum(parameter.numel() for parameter in shared.parameters()) == 8 * 2 + 8 + 1


def test_layout_checks_position_labels_and_or_label() -> None:
    layout = {
        "sample_id": "sample", "policy_id": "policy", "tile_labels": [1, 0, 0, 0],
        "tiles": [{"group_id": str(i), "label": label} for i, label in enumerate((1, 0, 0, 0))],
    }
    row = {"sample_id": "sample", "policy_id": "policy", "label": 1}
    assert validate_four_image_layout(layout, row) == (1, 0, 0, 0)
    layout["tile_labels"] = [0, 1, 0, 0]
    with pytest.raises(ValueError, match="tile_labels"):
        validate_four_image_layout(layout, row)


def test_native_prompt_keeps_all_four_images_in_order(tmp_path) -> None:
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            self.content = messages[0]["content"]
            return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3)}

    for index in range(4):
        (tmp_path / f"{index}.jpg").write_bytes(b"test")
    processor = Processor()
    native_inputs(
        processor,
        {"tiles": [{"image": f"{index}.jpg"} for index in range(4)]},
        {"policy_text": "测试规则"}, tmp_path, 512,
    )
    assert [item["path"] for item in processor.content if item["type"] == "image"] == [
        str((tmp_path / f"{index}.jpg").resolve()) for index in range(4)
    ]


def test_vision_end_readout_keeps_one_feature_per_image() -> None:
    class Backbone:
        def __call__(self, **kwargs):
            return type("Output", (), {"last_hidden_state": torch.arange(12).reshape(1, 6, 2)})()

    model = type("Model", (), {"backbone": Backbone()})()
    inputs = {"input_ids": torch.tensor([[9, 8, 9, 8, 9, 9]])}
    features = vision_end_features(model, inputs, 9)
    assert features.shape == (1, 4, 2)
    assert features[0, :, 0].tolist() == [0.0, 4.0, 8.0, 10.0]
