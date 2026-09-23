"""分别评估属性头与策略头，避免把二元违规指标误当作多头指标。"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from qwen35_moderation.heads import (
    ATTRIBUTE_ALIASES,
    ATTRIBUTE_IDS,
    DECISION_ALIASES,
    DECISION_IDS,
    SOURCE_LEVEL_ATTRIBUTES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分别评估属性头与 block/review/allow 策略头")
    parser.add_argument("--predictions", required=True, help="evaluate.py 生成的 predictions JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-source-level-proxy",
        action="store_true",
        help="允许 source_L1-L4 作为属性代理标签；默认只接受人工属性标签",
    )
    return parser.parse_args()


def metric_binary(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
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


def attributes_from_record(
    values: list[Any], *, allow_source_level_proxy: bool
) -> tuple[dict[str, int] | None, str | None]:
    """提取真实属性标签；source_L 标签必须显式授权才可作为代理。"""
    targets = {attribute_id: 0 for attribute_id in ATTRIBUTE_IDS}
    used_actual = False
    used_proxy = False
    for raw in values:
        normalized = str(raw).strip()
        if normalized in SOURCE_LEVEL_ATTRIBUTES:
            if allow_source_level_proxy:
                used_proxy = True
                for attribute_id in SOURCE_LEVEL_ATTRIBUTES[normalized]:
                    targets[attribute_id] = 1
            continue
        attribute_id = ATTRIBUTE_ALIASES.get(normalized)
        if attribute_id is not None:
            targets[attribute_id] = 1
            used_actual = True
    if used_actual:
        return targets, "human_or_explicit"
    if used_proxy:
        return targets, "source_level_proxy"
    return None, None


def probability_map(head: dict[str, Any]) -> dict[str, float] | None:
    labels = head.get("labels", []) if isinstance(head, dict) else []
    values = {
        str(item.get("id")): float(item["probability"])
        for item in labels
        if isinstance(item, dict) and item.get("probability") is not None
    }
    return values or None


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("预测文件为空")

    attribute_labels: dict[str, list[int]] = defaultdict(list)
    attribute_scores: dict[str, list[float]] = defaultdict(list)
    attribute_sources: Counter[str] = Counter()
    decision_labels: list[int] = []
    decision_probabilities: list[list[float]] = []
    skipped = Counter()

    for record in records:
        heads = record.get("heads", {})
        attribute_scores_map = probability_map(heads.get("attribute", {})) if isinstance(heads, dict) else None
        targets, source = attributes_from_record(
            list(record.get("attributes", [])),
            allow_source_level_proxy=args.allow_source_level_proxy,
        )
        if targets is None:
            skipped["attribute_without_accepted_label"] += 1
        elif attribute_scores_map is None:
            skipped["attribute_head_probability_missing"] += 1
        else:
            attribute_sources[source or "unknown"] += 1
            for attribute_id in ATTRIBUTE_IDS:
                if attribute_id not in attribute_scores_map:
                    skipped[f"attribute_probability_missing:{attribute_id}"] += 1
                    continue
                attribute_labels[attribute_id].append(targets[attribute_id])
                attribute_scores[attribute_id].append(attribute_scores_map[attribute_id])

        raw_action = record.get("decision_label")
        decision_scores_map = probability_map(heads.get("decision", {})) if isinstance(heads, dict) else None
        action = DECISION_ALIASES.get(str(raw_action).strip()) if raw_action is not None else None
        if action is None:
            skipped["decision_without_human_label"] += 1
        elif decision_scores_map is None or any(item not in decision_scores_map for item in DECISION_IDS):
            skipped["decision_head_probability_missing"] += 1
        else:
            decision_labels.append(DECISION_IDS.index(action))
            decision_probabilities.append([decision_scores_map[item] for item in DECISION_IDS])

    attributes: dict[str, Any] = {}
    for attribute_id in ATTRIBUTE_IDS:
        labels = attribute_labels[attribute_id]
        scores = attribute_scores[attribute_id]
        attributes[attribute_id] = metric_binary(labels, scores) if labels else {"count": 0, "reason": "没有可用标注"}

    decision: dict[str, Any]
    if decision_labels:
        actual = np.asarray(decision_labels, dtype=np.int64)
        probability_matrix = np.asarray(decision_probabilities, dtype=np.float64)
        predicted = probability_matrix.argmax(axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, predicted, labels=range(len(DECISION_IDS)), zero_division=0
        )
        decision = {
            "count": int(actual.size),
            "accuracy": float((actual == predicted).mean()),
            "classes": {
                action: {
                    "support": int((actual == index).sum()),
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                }
                for index, action in enumerate(DECISION_IDS)
            },
        }
    else:
        decision = {"count": 0, "reason": "预测记录没有人工 block/review/allow 标签"}

    result = {
        "input": str(Path(args.predictions)),
        "attribute_label_source": dict(attribute_sources),
        "attributes": attributes,
        "decision": decision,
        "skipped": dict(skipped),
        "limitations": [
            "属性头指标仅在有人工属性标签时可作为正式结论。",
            "source_L1-L4 代理标签需要 --allow-source-level-proxy，输出只能用于开发诊断。",
            "决策头需要显式 decision_label；二元违规 label 不能证明 review 类有效。",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
