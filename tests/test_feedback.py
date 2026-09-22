from __future__ import annotations

import pytest

from qwen35_moderation.feedback import build_preference_pair, normalize_action


def test_build_preference_pair_keeps_multimodal_context() -> None:
    pair = build_preference_pair(
        {
            "sample_id": "case-1",
            "image": "images/case-1.jpg",
            "text": "示例正文",
            "policy_id": "strict-v1",
            "policy_text": "禁止明确裸露。",
            "human_action": "复审",
            "model_action": "通过",
            "reward": 0.8,
        }
    )
    assert pair is not None
    assert pair["chosen"] == "review"
    assert pair["rejected"] == "allow"
    assert "禁止明确裸露" in pair["prompt"]
    assert pair["image"] == "images/case-1.jpg"


def test_same_action_is_not_a_preference() -> None:
    assert build_preference_pair(
        {
            "sample_id": "case-2",
            "image": "images/case-2.jpg",
            "policy_id": "strict-v1",
            "policy_text": "规则",
            "human_action": "block",
            "model_action": "block",
        }
    ) is None
    with pytest.raises(ValueError):
        normalize_action("unknown")
