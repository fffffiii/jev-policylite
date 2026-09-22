"""根据 2×2 拼图布局，评估属性头在多目标缩小场景下的代理表现。"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from qwen35_moderation.heads import ATTRIBUTE_ALIASES, ATTRIBUTE_IDS, SOURCE_LEVEL_ATTRIBUTES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 2×2 拼图中的属性头")
    parser.add_argument("--predictions", required=True, help="evaluate.py 输出的 predictions JSONL")
    parser.add_argument("--layouts", required=True, help="build_mosaic_manifest.py 输出的 layouts.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-source-level-proxy",
        action="store_true",
        help="允许 source_L1-L4 派生属性标签；默认只使用人工属性标签",
    )
    return parser.parse_args()


def targets_for_tiles(
    tiles: list[dict[str, Any]], *, allow_source_level_proxy: bool
) -> tuple[dict[str, int] | None, str | None]:
    """按 OR 规则聚合四个 tile 的属性；真实标签优先于等级代理。"""
    targets = {attribute_id: 0 for attribute_id in ATTRIBUTE_IDS}
    has_actual = False
    has_proxy = False
    for tile in tiles:
        for raw in tile.get("attributes", []):
            name = str(raw).strip()
            if name in SOURCE_LEVEL_ATTRIBUTES:
                if allow_source_level_proxy:
                    has_proxy = True
                    for attribute_id in SOURCE_LEVEL_ATTRIBUTES[name]:
                        targets[attribute_id] = 1
                continue
            attribute_id = ATTRIBUTE_ALIASES.get(name)
            if attribute_id is not None:
                targets[attribute_id] = 1
                has_actual = True
    if has_actual:
        return targets, "human_or_explicit"
    if has_proxy:
        return targets, "source_level_proxy"
    return None, None


def attribute_probability(record: dict[str, Any], attribute_id: str) -> float | None:
    labels = record.get("heads", {}).get("attribute", {}).get("labels", [])
    for label in labels:
        if label.get("id") == attribute_id and label.get("probability") is not None:
            return float(label["probability"])
    return None


def binary_metric(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    values = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    predicted = (scores >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        values, predicted, average="binary", zero_division=0
    )
    result: dict[str, Any] = {
        "count": int(values.size),
        "positive_count": int(values.sum()),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
    }
    if len(np.unique(values)) == 2:
        result["pr_auc"] = float(average_precision_score(values, scores))
        result["roc_auc"] = float(roc_auc_score(values, scores))
    else:
        result["pr_auc"] = None
        result["roc_auc"] = None
    return result


def main() -> None:
    args = parse_args()
    predictions = {
        str(record["sample_id"]): record
        for record in (
            json.loads(line)
            for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    layouts = [
        json.loads(line)
        for line in Path(args.layouts).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels: dict[str, list[int]] = defaultdict(list)
    probabilities: dict[str, list[float]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    scenario_scores: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for layout in layouts:
        record = predictions.get(str(layout["sample_id"]))
        if record is None:
            skipped["prediction_missing"] += 1
            continue
        targets, source = targets_for_tiles(
            list(layout.get("tiles", [])), allow_source_level_proxy=args.allow_source_level_proxy
        )
        if targets is None:
            skipped["attribute_label_missing"] += 1
            continue
        source_counts[source or "unknown"] += 1
        for attribute_id in ATTRIBUTE_IDS:
            probability = attribute_probability(record, attribute_id)
            if probability is None:
                skipped[f"probability_missing:{attribute_id}"] += 1
                continue
            labels[attribute_id].append(targets[attribute_id])
            probabilities[attribute_id].append(probability)
            scenario_scores[str(layout["scenario"])][attribute_id].append((targets[attribute_id], probability))

    result: dict[str, Any] = {
        "input": {"predictions": args.predictions, "layouts": args.layouts},
        "label_source": dict(source_counts),
        "attributes": {
            attribute_id: (
                binary_metric(labels[attribute_id], probabilities[attribute_id])
                if labels[attribute_id]
                else {"count": 0, "reason": "没有可用标签或模型概率"}
            )
            for attribute_id in ATTRIBUTE_IDS
        },
        "by_scenario": {},
        "skipped": dict(skipped),
        "limitations": [
            "四图属性标签使用任一 tile 命中即为正的 OR 规则。",
            "source_L1-L4 是代理标签，只有 --allow-source-level-proxy 时才会被计入。",
            "医学标签在公开试训中没有正样本，不能由该实验验证。",
        ],
    }
    for scenario, scenario_values in scenario_scores.items():
        result["by_scenario"][scenario] = {
            attribute_id: (
                binary_metric(
                    [item[0] for item in scenario_values[attribute_id]],
                    [item[1] for item in scenario_values[attribute_id]],
                )
                if scenario_values[attribute_id]
                else {"count": 0}
            )
            for attribute_id in ATTRIBUTE_IDS
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
