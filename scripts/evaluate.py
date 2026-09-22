from __future__ import annotations

import argparse
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from qwen35_moderation.data import ModerationDataset, read_manifest
from qwen35_moderation.heads import bundle_probabilities, format_heads
from qwen35_moderation.metrics import classification_metrics, sigmoid
from qwen35_moderation.model import model_inputs
from qwen35_moderation.runtime import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线评分并计算审核指标")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["validation", "calibration", "test"], required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--calibration")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, metadata, collator = load_checkpoint(args.checkpoint, device)
    if args.image_root:
        collator.image_root = Path(args.image_root)
    rows = read_manifest(args.manifest)
    dataset = ModerationDataset(rows, args.split, collator.image_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    calibration = {"temperature": 1.0, "global_threshold": 0.5, "policy_thresholds": {}}
    if args.calibration:
        calibration.update(json.loads(Path(args.calibration).read_text(encoding="utf-8")))
    temperature = float(calibration["temperature"])

    records: list[dict[str, object]] = []
    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with torch.no_grad(), context:
        for batch in loader:
            inputs = {key: value.to(device) for key, value in model_inputs(batch).items()}
            bundle = model(return_heads=True, **inputs)
            logits = bundle.violation_logit.float().cpu().numpy()
            attribute_probabilities, decision_probabilities = bundle_probabilities(bundle)
            probabilities = sigmoid(logits, temperature)
            for index, sample_id in enumerate(batch["sample_ids"]):
                policy_id = batch["policy_ids"][index]
                threshold = float(
                    calibration["policy_thresholds"].get(policy_id, calibration["global_threshold"])
                )
                records.append(
                    {
                        "sample_id": sample_id,
                        "split": args.split,
                        "group_id": batch["group_ids"][index],
                        "policy_id": policy_id,
                        "attributes": batch["attributes"][index],
                        "decision_label": batch["decisions"][index],
                        "label": int(batch["labels"][index].item()),
                        "logit": float(logits[index]),
                        "probability": float(probabilities[index]),
                        "threshold": threshold,
                        "prediction": int(probabilities[index] >= threshold),
                        "heads": format_heads(
                            violation_probability=float(probabilities[index]),
                            threshold=threshold,
                            attribute_probabilities=(
                                None if attribute_probabilities is None else attribute_probabilities[index]
                            ),
                            decision_probabilities=(
                                None if decision_probabilities is None else decision_probabilities[index]
                            ),
                        ),
                    }
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{args.split}_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["policy_id"])].append(record)
    metrics: dict[str, object] = {}
    all_labels = np.array([record["label"] for record in records])
    all_probabilities = np.array([record["probability"] for record in records])
    metrics["global"] = classification_metrics(
        all_labels, all_probabilities, float(calibration["global_threshold"]),
        predictions=np.array([record["prediction"] for record in records]),
    )
    metrics["by_policy"] = {}
    for policy_id, policy_records in grouped.items():
        labels = np.array([record["label"] for record in policy_records])
        probabilities = np.array([record["probability"] for record in policy_records])
        threshold = float(
            calibration["policy_thresholds"].get(policy_id, calibration["global_threshold"])
        )
        metrics["by_policy"][policy_id] = classification_metrics(labels, probabilities, threshold)
    attribute_records: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        for attribute in record["attributes"]:
            attribute_records[str(attribute)].append(record)
    metrics["by_attribute"] = {}
    for attribute, subset in attribute_records.items():
        labels = np.array([record["label"] for record in subset])
        probabilities = np.array([record["probability"] for record in subset])
        predictions = np.array([record["prediction"] for record in subset])
        subset_metrics = classification_metrics(labels, probabilities, threshold=0.5, predictions=predictions)
        metrics["by_attribute"][attribute] = subset_metrics

    group_records: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        group_records[str(record["group_id"])].append(record)
    flip_groups = [
        subset
        for subset in group_records.values()
        if len({record["policy_id"] for record in subset}) >= 2
        and len({record["label"] for record in subset}) >= 2
    ]
    metrics["policy_flip"] = {
        "group_count": len(flip_groups),
        "both_correct_rate": (
            sum(
                all(record["prediction"] == record["label"] for record in subset)
                for subset in flip_groups
            )
            / len(flip_groups)
            if flip_groups
            else None
        ),
    }
    metrics["checkpoint_validation"] = metadata.get("validation_metrics")
    (output_dir / f"{args.split}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
