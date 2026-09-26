"""评测协议的回归测试：阈值不看测试标签，置信区间按原图分组。"""
import importlib.util
import sys
from pathlib import Path

import pytest


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_script("benchmark_nsfw")


@pytest.mark.parametrize("script", ["benchmark_nsfw", "experiment_multi_photo"])
def test_statistical_helpers_import_without_model_dependencies(monkeypatch, script):
    # 模拟 CI 未安装主干依赖，避免本机完整环境掩盖导入错误。
    for name in ("transformers", "peft", "accelerate", "qwen35_moderation.model", "qwen35_moderation.runtime"):
        monkeypatch.setitem(sys.modules, name, None)
    load_script(script)


def test_test_scores_do_not_choose_calibration_threshold():
    records = []
    for split in ("calibration", "test"):
        for level, (label, score) in enumerate(((0, .1), (0, .3), (1, .4), (1, .8))):
            records.append({"group_id": f"{split}-{level}", "split": split, "policy_id": "test-policy",
                            "source_label": level, "label": label, "score": score})
    before = bench.quality_report(records)["test-policy"]
    for row in records:
        if row["split"] == "test":
            row["score"] = 1 - row["score"]
    after = bench.quality_report(records)["test-policy"]
    assert before["calibrated_f1"]["threshold"] == after["calibrated_f1"]["threshold"] == .4
    assert before["test_at_calibration_fpr5"]["threshold"] == after["test_at_calibration_fpr5"]["threshold"]
    assert before["calibrated_f1"]["accuracy"] != after["calibrated_f1"]["accuracy"]


def test_repeating_a_policy_does_not_inflate_bootstrap_sample_size():
    rows = [{"group_id": str(i), "policy_id": "policy", "label": 1, "score": float(i % 3 == 0)} for i in range(12)]
    original = bench.bootstrap_accuracy(rows, {"policy": .5})
    repeated = bench.bootstrap_accuracy([r for row in rows for r in (row, dict(row))], {"policy": .5})
    assert original == repeated


def test_duplicate_manifest_rows_are_rejected_before_four_image_encoding(tmp_path):
    import json
    multi = load_script("experiment_multi_photo")
    folder = tmp_path / "test"
    folder.mkdir()
    rows = [{"sample_id": "a", "group_id": "a", "image": "a.jpg", "text": "", "policy_id": "p",
             "policy_text": "p", "label": 0, "split": "test"}] * 2
    (folder / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (folder / "layouts.jsonl").write_text('\n'.join(json.dumps({"sample_id": s}) for s in ("a", "b")), encoding="utf-8")
    with pytest.raises(ValueError, match="sample_id"):
        multi.split_records(tmp_path, "test")
