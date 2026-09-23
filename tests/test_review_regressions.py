"""Regression coverage for issues found while reviewing the repository's documentation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qwen35_moderation.feedback import build_preference_pair, preference_group_id
from qwen35_moderation.heads import format_heads
from qwen35_moderation.web.config import Settings
from qwen35_moderation.web.evidence import load_test_metrics


def feedback(**overrides):
    return {
        "sample_id": "image-a:rule-a", "image": "images/a.jpg", "text": "",
        "policy_id": "rule-a", "policy_text": "Example moderation rule",
        "human_action": "review", "model_action": "allow", **overrides,
    }


def test_feedback_preserves_explicit_image_group():
    first = build_preference_pair(feedback(group_id="original-a", feedback_id="review-1"))
    second = build_preference_pair(feedback(group_id="original-a", image="crops/a.jpg", feedback_id="review-2"))
    assert first["group_id"] == second["group_id"] == "original-a"
    assert first["chosen"] == "review" and first["rejected"] == "allow"


def test_legacy_feedback_groups_by_image_not_feedback_or_sample_id():
    first = build_preference_pair(feedback(feedback_id="review-1"))
    second = build_preference_pair(feedback(sample_id="image-a:rule-b", feedback_id="review-2"))
    assert first["group_id"] == second["group_id"] == "images/a.jpg"
    assert preference_group_id({"image": "images/a.jpg", "feedback_id": "another"}) == "images/a.jpg"


@pytest.mark.parametrize("row", [{"group_id": "  ", "image": "a.jpg"}, {"image": ""}])
def test_invalid_group_identity_is_rejected(row):
    with pytest.raises(ValueError):
        preference_group_id(row)


def test_non_object_feedback_has_a_clear_error():
    with pytest.raises(ValueError, match="JSON 对象"):
        build_preference_pair([])


def test_absent_metrics_do_not_fabricate_results(tmp_path):
    report, note = load_test_metrics(tmp_path / "not-provided.json")
    assert report == {}
    assert "未提供" in note


@pytest.mark.parametrize("content", ["{bad", "[]", '{"global": []}', '{"global": {"pr_auc": NaN}}'])
def test_invalid_metrics_are_explained_not_used(tmp_path, content):
    path = tmp_path / "test_metrics.json"; path.write_text(content)
    report, note = load_test_metrics(path)
    assert report == {}
    assert "检查" in note


def test_valid_metrics_keep_missing_values_and_provenance_note(tmp_path):
    original = {"global": {"count": 7, "pr_auc": None, "accuracy": 0.5}, "policy_flip": {"both_correct_rate": None}}
    path = tmp_path / "test_metrics.json"; path.write_text(json.dumps(original))
    report, note = load_test_metrics(path)
    assert report == original
    assert "不会自动验证" in note


def test_service_metrics_are_optional_but_processor_is_required(tmp_path):
    checkpoint = tmp_path / "checkpoint"; checkpoint.mkdir()
    (checkpoint / "adapter").mkdir(); (checkpoint / "processor").mkdir()
    for name in ("metadata.json", "decision_head.pt", "calibration.json"):
        (checkpoint / name).write_text("fixture")
    static = tmp_path / "static"; static.mkdir(); (static / "index.html").write_text("fixture")
    settings = Settings(checkpoint, checkpoint / "calibration.json", tmp_path / "missing-metrics.json", static, None, 10 * 2**20, 30)
    settings.validate_files()
    (checkpoint / "processor").rmdir()
    with pytest.raises(FileNotFoundError, match="processor"):
        settings.validate_files()


def test_head_labels_do_not_change_output_ids_or_probabilities():
    heads = format_heads(violation_probability=0.1, threshold=0.5,
                         attribute_probabilities=np.array([0.2, 0.3, 0.4, 0.5]),
                         decision_probabilities=np.array([0.1, 0.8, 0.1]))
    assert heads["decision"]["action"] == "review"
    assert [row["id"] for row in heads["decision"]["labels"]] == ["block", "review", "allow"]
    assert heads["attribute"]["labels"][2]["id"] == "suggestive"
    assert heads["attribute"]["labels"][2]["probability"] == 0.4


def test_unloaded_policy_head_is_not_presented_as_review_capability():
    heads = format_heads(violation_probability=0.1, threshold=0.5)
    assert heads["decision"]["kind"] == "binary"
    assert heads["decision"]["action"] == "allow"
    assert "不提供复审概率" in heads["decision"]["note"]
    assert all("probability" not in label for label in heads["attribute"]["labels"])
