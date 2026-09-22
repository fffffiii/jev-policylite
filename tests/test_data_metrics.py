from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from qwen35_moderation.data import MultimodalCollator, configure_max_pixels, find_group_leakage, read_manifest
from qwen35_moderation.metrics import choose_threshold, classification_metrics, fit_temperature, sigmoid


def test_manifest_and_group_leakage(tmp_path: Path) -> None:
    rows = [
        {
            "sample_id": "a:strict",
            "group_id": "a",
            "image": "a.jpg",
            "text": "",
            "policy_id": "strict-v1",
            "policy_text": "规则",
            "label": 0,
            "split": "train",
        },
        {
            "sample_id": "a:contextual",
            "group_id": "a",
            "image": "a.jpg",
            "text": "",
            "policy_id": "contextual-v1",
            "policy_text": "规则",
            "label": 0,
            "split": "validation",
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    loaded = read_manifest(manifest)
    assert find_group_leakage(loaded) == {"a": {"train", "validation"}}


def test_threshold_respects_false_positive_limit() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    probabilities = np.array([0.05, 0.10, 0.40, 0.60, 0.90])
    threshold = choose_threshold(labels, probabilities, max_fpr=0.0)
    metrics = classification_metrics(labels, probabilities, threshold)
    assert metrics["fpr"] == 0.0
    assert metrics["recall"] == 1.0


def test_temperature_is_positive_and_finite() -> None:
    logits = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    temperature = fit_temperature(logits, labels)
    probabilities = sigmoid(logits, temperature)
    assert 0.05 <= temperature <= 20.0
    assert np.isfinite(probabilities).all()


def test_collator_preserves_policy_and_image_when_body_is_long(tmp_path: Path) -> None:
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["processor_kwargs"]["truncation"] is False
            content = messages[0]["content"]
            assert content[0]["type"] == "image"
            assert "完整规则不能删除" in content[1]["text"]
            return {"input_ids": torch.ones(1, len(content[1]["text"]), dtype=torch.long)}

    collator = MultimodalCollator(Processor(), tmp_path, max_length=100)
    row = {"image": "a.jpg", "sample_id": "long", "text": "正文" * 200, "policy_text": "完整规则不能删除"}
    with pytest.warns(UserWarning, match="正文过长"):
        assert collator._encode_one(row)["input_ids"].shape[-1] <= 100
    collator.max_length = 10
    with pytest.raises(ValueError, match="图片和完整规则超过"):
        collator._encode_one(row)


def test_metrics_use_actual_policy_predictions_and_valid_json() -> None:
    result = classification_metrics(np.array([0, 0]), np.array([0.6, 0.7]), .5, np.array([0, 1]))
    assert result["fp"] == 1
    assert result["pr_auc"] is None
    json.dumps(result, allow_nan=False)


def test_image_budget_accepts_non_dict_size_container() -> None:
    class Size:
        longest_edge = 16777216
        def get(self, key):
            return getattr(self, key)
        def __setitem__(self, key, value):
            setattr(self, key, value)
    processor = SimpleNamespace(image_processor=SimpleNamespace(size=Size()))
    configure_max_pixels(processor, 200704)
    assert processor.image_processor.size.longest_edge == 200704


def test_padding_can_use_fixed_online_length() -> None:
    items = [
        {"input_ids": torch.tensor([[1, 2, 3]])},
        {"input_ids": torch.tensor([[4, 5]])},
    ]
    padded = MultimodalCollator._pad_sequence_tensors(items, "input_ids", 0, 5)
    assert padded.tolist() == [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]
    with pytest.raises(ValueError, match="小于实际长度"):
        MultimodalCollator._pad_sequence_tensors(items, "input_ids", 0, 2)
