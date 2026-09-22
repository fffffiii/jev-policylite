from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from qwen35_moderation.metrics import choose_threshold, fit_temperature, sigmoid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在独立校准集上拟合温度和审核阈值")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("预测文件为空")
    if any(record.get("split") != "calibration" for record in records):
        raise ValueError("只允许使用显式标记为 calibration 的预测数据拟合阈值")
    if not 0 <= args.target_fpr <= 1:
        raise ValueError("target-fpr 必须位于 [0, 1]")
    logits = np.array([record["logit"] for record in records], dtype=np.float32)
    labels = np.array([record["label"] for record in records], dtype=np.int64)
    temperature = fit_temperature(logits, labels)
    probabilities = sigmoid(logits, temperature)
    global_threshold = choose_threshold(labels, probabilities, args.target_fpr)

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped_indices[str(record["policy_id"])].append(index)
    policy_thresholds: dict[str, float] = {}
    for policy_id, indices in grouped_indices.items():
        policy_labels = labels[indices]
        if (policy_labels == 0).sum() == 0:
            continue
        policy_thresholds[policy_id] = choose_threshold(
            policy_labels, probabilities[indices], args.target_fpr
        )

    result = {
        "temperature": temperature,
        "target_fpr": args.target_fpr,
        "global_threshold": global_threshold,
        "policy_thresholds": policy_thresholds,
        "calibration_count": len(records),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
