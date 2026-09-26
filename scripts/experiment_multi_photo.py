"""四图定位与延迟实验：冻结编码器，分别训练四个位置检测头。"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from collections import Counter, defaultdict, deque
from contextlib import nullcontext
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from qwen35_moderation.data import MultimodalCollator, configure_max_pixels, read_manifest
from qwen35_moderation.mosaic import make_mosaic
from qwen35_moderation.multi_photo import FourPositionHeads, POSITION_NAMES, SharedPositionHead, load_four_position_heads, load_shared_position_head, native_inputs, validate_four_image_layout, vision_end_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较原生四图、2×2 拼图及逐张审核的速度和逐图准确率")
    parser.add_argument("--stage", required=True, choices=["encode", "fit", "benchmark"])
    parser.add_argument("--mode", choices=["native", "native_local", "native_shared", "mosaic"], help="encode/fit 阶段必填")
    parser.add_argument("--checkpoint", default="outputs/public-pilot-multihead")
    parser.add_argument("--dataset-root", default="outputs/multi-photo-data-v1")
    parser.add_argument("--source-image-root", default=".")
    parser.add_argument("--output-root", default="outputs/multi-photo-v1")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--native-max-pixels", type=int, default=50176)
    parser.add_argument("--mosaic-max-pixels", type=int, default=200704)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--limit", type=int, help="仅供编码链路调试，正式结果不要使用")
    parser.add_argument("--benchmark-cases", type=int, default=50)
    return parser.parse_args()


def split_records(root: Path, split: str, limit: int | None = None) -> list[tuple[dict[str, Any], dict[str, Any], tuple[int, int, int, int]]]:
    directory = root / split
    rows = read_manifest(directory / "manifest.jsonl")
    layouts = [json.loads(line) for line in (directory / "layouts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(layout["sample_id"]): layout for layout in layouts}
    row_ids = [str(row["sample_id"]) for row in rows]
    if len(by_id) != len(layouts) or len(set(row_ids)) != len(rows) or set(row_ids) != set(by_id):
        raise ValueError(f"{split} 清单和布局数量或 sample_id 不一致")
    result = []
    for row in rows[:limit]:
        if row["split"] != split:
            raise ValueError(f"{split} 清单包含其他 split")
        layout = by_id[str(row["sample_id"])]
        result.append((row, layout, validate_four_image_layout(layout, row)))
    return result


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_forward(
    model: nn.Module, encoded: dict[str, torch.Tensor], device: torch.device,
    vision_end_id: int | None = None,
) -> tuple[torch.Tensor, float]:
    synchronize(device)
    started = time.perf_counter()
    inputs = {key: value.to(device) for key, value in encoded.items()}
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with torch.inference_mode(), context:
        features = model.encode(**inputs) if vision_end_id is None else vision_end_features(model, inputs, vision_end_id)
    synchronize(device)
    return features.detach().float().cpu(), (time.perf_counter() - started) * 1000


def percentiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("没有可汇总的耗时")
    return {"mean_ms": float(data.mean()), "p50_ms": float(np.percentile(data, 50)), "p95_ms": float(np.percentile(data, 95))}


def encode(args: argparse.Namespace) -> None:
    if args.mode is None:
        raise ValueError("encode 必须指定 --mode")
    if args.mode == "native_shared":
        raise ValueError("native_shared 复用 native_local 特征；请只运行 --stage fit")
    # 数据校验和缓存特征训练无需加载主干模型依赖。
    from qwen35_moderation.runtime import load_checkpoint

    device = torch.device(args.device)
    model, processor, _, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    configure_max_pixels(processor, args.native_max_pixels if args.mode.startswith("native") else args.mosaic_max_pixels)
    vision_end_id = (
        processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        if args.mode == "native_local" else None
    )
    if args.mode == "native_local" and not isinstance(vision_end_id, int):
        raise ValueError("处理器缺少 vision_end token，无法使用局部读出")
    output = Path(args.output_root) / args.mode
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        records = split_records(Path(args.dataset_root), split, args.limit)
        mosaic_collator = MultimodalCollator(processor, Path(args.dataset_root) / split, args.max_length)
        features: list[torch.Tensor] = []
        labels: list[tuple[int, int, int, int]] = []
        timings: list[dict[str, Any]] = []
        # 首个前向会触发内核初始化，先预热以免污染正式延迟分布。
        warmup_row, warmup_layout, _ = records[0]
        warmup_inputs = (
            native_inputs(processor, warmup_layout, warmup_row, Path(args.source_image_root), args.max_length)
            if args.mode.startswith("native") else mosaic_collator._encode_one(warmup_row)
        )
        timed_forward(model, warmup_inputs, device, vision_end_id)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for index, (row, layout, target) in enumerate(records, start=1):
            started = time.perf_counter()
            if args.mode.startswith("native"):
                encoded = native_inputs(processor, layout, row, Path(args.source_image_root), args.max_length)
            else:
                encoded = mosaic_collator._encode_one(row)
            preprocessing_ms = (time.perf_counter() - started) * 1000
            embedding, model_ms = timed_forward(model, encoded, device, vision_end_id)
            features.append(embedding.squeeze(0))
            labels.append(target)
            timings.append({
                "sample_id": row["sample_id"], "preprocess_ms": preprocessing_ms,
                "model_ms": model_ms, "total_ms": preprocessing_ms + model_ms,
                "input_tokens": int(encoded["attention_mask"].sum().item()),
            })
            if index % 50 == 0:
                print(f"{args.mode} {split}: {index}/{len(records)}", flush=True)
        payload = {
            "features": torch.stack(features), "labels": torch.tensor(labels, dtype=torch.float32),
            "sample_ids": [row["sample_id"] for row, _, _ in records],
        }
        torch.save(payload, output / f"{split}_features.pt")
        report = {
            "mode": args.mode, "split": split, "count": len(records), "device": str(device),
            "preprocess": percentiles([item["preprocess_ms"] for item in timings]),
            "model": percentiles([item["model_ms"] for item in timings]),
            "end_to_end": percentiles([item["total_ms"] for item in timings]),
            "input_tokens": {
                "mean": float(np.mean([item["input_tokens"] for item in timings])),
                "p50": float(np.percentile([item["input_tokens"] for item in timings], 50)),
                "p95": float(np.percentile([item["input_tokens"] for item in timings], 95)),
            },
            "peak_allocated_gb": (torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None),
            "notes": "单组四图、batch=1；已加载模型，包含图像读取和预处理，不含模型加载/网络传输；预制拼图生成未计入",
            "samples": timings,
        }
        (output / f"{split}_timing.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "samples"}, ensure_ascii=False), flush=True)


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    predicted = (scores >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
    negative_count = int((labels == 0).sum())
    return {
        "count": int(labels.size), "positive_count": int(labels.sum()),
        "accuracy": float(np.mean(labels == predicted)), "precision": float(precision),
        "recall": float(recall), "f1": float(f1),
        "fpr": float(((predicted == 1) & (labels == 0)).sum() / negative_count) if negative_count else None,
        "pr_auc": float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
    }


def four_image_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    predictions = (scores >= 0.5).astype(np.int64)
    one_positive = labels.sum(axis=1) == 1
    positions = {
        name: binary_metrics(labels[:, index], scores[:, index])
        for index, name in enumerate(POSITION_NAMES)
    }
    return {
        "per_position": positions,
        "micro": binary_metrics(labels.reshape(-1), scores.reshape(-1)),
        "exact_match": float(np.mean(np.all(labels == predictions, axis=1))),
        "any_violation_accuracy": float(np.mean(labels.max(axis=1) == predictions.max(axis=1))),
        "one_violation_count": int(one_positive.sum()),
        "one_violation_exact_match": (
            float(np.mean(np.all(labels[one_positive] == predictions[one_positive], axis=1)))
            if one_positive.any() else None
        ),
        "one_violation_top1": (
            float(np.mean(np.argmax(scores[one_positive], axis=1) == np.argmax(labels[one_positive], axis=1)))
            if one_positive.any() else None
        ),
    }


def cached_split(output: Path, split: str) -> dict[str, Any]:
    return torch.load(output / f"{split}_features.pt", map_location="cpu", weights_only=True)


def fit(args: argparse.Namespace) -> None:
    if args.mode is None:
        raise ValueError("fit 必须指定 --mode")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_root) / args.mode
    feature_dir = Path(args.output_root) / ("native_local" if args.mode == "native_shared" else args.mode)
    train = cached_split(feature_dir, "train")
    validation = cached_split(feature_dir, "validation")
    test = cached_split(feature_dir, "test")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    head_class = SharedPositionHead if args.mode == "native_shared" else FourPositionHeads
    model = head_class(int(train["features"].shape[-1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    loader = DataLoader(TensorDataset(train["features"], train["labels"]), batch_size=args.batch_size, shuffle=True)
    val_features = validation["features"].to(device)
    val_labels = validation["labels"].to(device)
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = float(F.binary_cross_entropy_with_logits(model(val_features), val_labels).item())
        history.append({"epoch": epoch, "validation_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch = val_loss, epoch
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, output / "four_position_heads.pt")
        if epoch - best_epoch >= 12:
            break
    loader = load_shared_position_head if args.mode == "native_shared" else load_four_position_heads
    model = loader(output / "four_position_heads.pt", int(train["features"].shape[-1]), device)
    report: dict[str, Any] = {
        "mode": args.mode, "seed": args.seed, "best_epoch": best_epoch,
        "best_validation_loss": best_loss, "head_parameters": sum(p.numel() for p in model.parameters()),
        "training": (
            "冻结主干和 LoRA；每组图只编码一次；一个共享 LayerNorm+Linear 头作用于四张图的局部表征"
            if args.mode == "native_shared" else
            "冻结主干和 LoRA；每组图只编码一次；四个 LayerNorm+Linear 头在缓存特征上训练"
        ),
        "splits": {}, "history": history,
    }
    for split, payload in (("train", train), ("validation", validation), ("test", test)):
        with torch.inference_mode():
            scores = torch.sigmoid(model(payload["features"].to(device))).cpu().numpy()
        labels = payload["labels"].numpy().astype(np.int64)
        report["splits"][split] = four_image_metrics(labels, scores)
        with (output / f"{split}_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for sample_id, target, probabilities in zip(payload["sample_ids"], labels, scores):
                handle.write(json.dumps({
                    "sample_id": sample_id, "tile_labels": target.tolist(),
                    "tile_probabilities": [float(value) for value in probabilities],
                }, ensure_ascii=False) + "\n")
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, ensure_ascii=False, indent=2), flush=True)


def benchmark(args: argparse.Namespace) -> None:
    from qwen35_moderation.model import model_inputs
    from qwen35_moderation.runtime import load_checkpoint

    if args.benchmark_cases < 1:
        raise ValueError("benchmark-cases 必须大于 0")
    device = torch.device(args.device)
    model, processor, _, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    configure_max_pixels(processor, args.native_max_pixels)
    collator = MultimodalCollator(processor, Path(args.source_image_root), args.max_length)
    all_records = split_records(Path(args.dataset_root), "test")
    buckets: dict[tuple[str, str], deque[tuple[dict[str, Any], dict[str, Any], tuple[int, int, int, int]]]] = defaultdict(deque)
    for record in all_records:
        row, layout, _ = record
        buckets[(str(row["policy_id"]), str(layout["scenario"]))].append(record)
    records = []
    while len(records) < min(args.benchmark_cases, len(all_records)):
        for key in sorted(buckets):
            if buckets[key] and len(records) < args.benchmark_cases:
                records.append(buckets[key].popleft())
    hidden_size = int(model.backbone.config.text_config.hidden_size)
    position_heads = {
        mode: load_four_position_heads(Path(args.output_root) / mode / "four_position_heads.pt", hidden_size, device)
        for mode in ("native", "native_local", "mosaic")
    }
    position_heads["native_shared"] = load_shared_position_head(
        Path(args.output_root) / "native_shared" / "four_position_heads.pt", hidden_size, device
    )
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    if not isinstance(vision_end_id, int):
        raise ValueError("处理器缺少 vision_end token")
    warmup_row, warmup_layout, _ = records[0]
    warmup_tile = warmup_layout["tiles"][0]
    warmup_inputs = collator._encode_one({
        **warmup_row, "image": warmup_tile["image"], "sample_id": warmup_tile["sample_id"],
    })
    timed_forward(model, warmup_inputs, device)
    warmup_batch = collator([
        {**warmup_row, "image": tile["image"], "sample_id": tile["sample_id"], "group_id": tile["group_id"], "label": tile["label"]}
        for tile in warmup_layout["tiles"]
    ])
    timed_forward(model, model_inputs(warmup_batch), device)
    # 三种输入形态各自预热，避免首次四图内核初始化进入均值。
    configure_max_pixels(processor, args.native_max_pixels)
    timed_forward(
        model,
        native_inputs(processor, warmup_layout, warmup_row, Path(args.source_image_root), args.max_length),
        device,
    )
    timed_forward(
        model,
        native_inputs(processor, warmup_layout, warmup_row, Path(args.source_image_root), args.max_length),
        device, vision_end_id,
    )
    configure_max_pixels(processor, args.mosaic_max_pixels)
    warmup_mosaic = MultimodalCollator(processor, Path(args.dataset_root) / "test", args.max_length)
    timed_forward(model, warmup_mosaic._encode_one(warmup_row), device)
    group_times: dict[str, list[float]] = {"sequential_four": [], "batched_four": [], "native": [], "native_local": [], "native_shared": [], "mosaic_online": []}
    scores: dict[str, list[list[float]]] = {"sequential_four": [], "batched_four": [], "native": [], "native_local": [], "native_shared": [], "mosaic_online": []}
    labels: list[tuple[int, int, int, int]] = []
    with tempfile.TemporaryDirectory(prefix="jev-four-photo-") as temporary:
      temporary_image = Path(temporary) / "mosaic.jpg"
      for index, (row, layout, target) in enumerate(records, start=1):
        configure_max_pixels(processor, args.native_max_pixels)
        started = time.perf_counter()
        group_scores = []
        for tile in layout["tiles"]:
            single = {**row, "image": tile["image"], "sample_id": tile["sample_id"], "group_id": tile["group_id"], "label": tile["label"]}
            encoded = collator._encode_one(single)
            synchronize(device)
            inputs = {key: value.to(device) for key, value in model_inputs(encoded).items()}
            context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with torch.inference_mode(), context:
                value = model(**inputs)
            synchronize(device)
            group_scores.append(float(torch.sigmoid(value.float()).item()))
        group_times["sequential_four"].append((time.perf_counter() - started) * 1000)
        scores["sequential_four"].append(group_scores)

        started = time.perf_counter()
        batch = collator([
            {**row, "image": tile["image"], "sample_id": tile["sample_id"], "group_id": tile["group_id"], "label": tile["label"]}
            for tile in layout["tiles"]
        ])
        synchronize(device)
        inputs = {key: value.to(device) for key, value in model_inputs(batch).items()}
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with torch.inference_mode(), context:
            values = model(**inputs)
        synchronize(device)
        group_times["batched_four"].append((time.perf_counter() - started) * 1000)
        scores["batched_four"].append(torch.sigmoid(values.float()).cpu().tolist())

        configure_max_pixels(processor, args.native_max_pixels)
        started = time.perf_counter()
        encoded = native_inputs(processor, layout, row, Path(args.source_image_root), args.max_length)
        feature, _ = timed_forward(model, encoded, device)
        with torch.inference_mode():
            probabilities = torch.sigmoid(position_heads["native"](feature.to(device))).cpu().squeeze(0).tolist()
        synchronize(device)
        group_times["native"].append((time.perf_counter() - started) * 1000)
        scores["native"].append(probabilities)

        started = time.perf_counter()
        encoded = native_inputs(processor, layout, row, Path(args.source_image_root), args.max_length)
        feature, _ = timed_forward(model, encoded, device, vision_end_id)
        with torch.inference_mode():
            probabilities = torch.sigmoid(position_heads["native_local"](feature.to(device))).cpu().squeeze(0).tolist()
        synchronize(device)
        group_times["native_local"].append((time.perf_counter() - started) * 1000)
        scores["native_local"].append(probabilities)

        started = time.perf_counter()
        encoded = native_inputs(processor, layout, row, Path(args.source_image_root), args.max_length)
        feature, _ = timed_forward(model, encoded, device, vision_end_id)
        with torch.inference_mode():
            probabilities = torch.sigmoid(position_heads["native_shared"](feature.to(device))).cpu().squeeze(0).tolist()
        synchronize(device)
        group_times["native_shared"].append((time.perf_counter() - started) * 1000)
        scores["native_shared"].append(probabilities)

        configure_max_pixels(processor, args.mosaic_max_pixels)
        started = time.perf_counter()
        images: list[Image.Image] = []
        try:
            for tile in layout["tiles"]:
                with Image.open(Path(args.source_image_root) / str(tile["image"])) as source:
                    images.append(source.copy())
            mosaic = make_mosaic(images, tile_size=224)
            mosaic.save(temporary_image, format="JPEG", quality=95, subsampling=0)
            mosaic.close()
        finally:
            for source in images:
                source.close()
        encoded = collator._encode_one({**row, "image": str(temporary_image)})
        feature, _ = timed_forward(model, encoded, device)
        with torch.inference_mode():
            probabilities = torch.sigmoid(position_heads["mosaic"](feature.to(device))).cpu().squeeze(0).tolist()
        synchronize(device)
        group_times["mosaic_online"].append((time.perf_counter() - started) * 1000)
        scores["mosaic_online"].append(probabilities)
        labels.append(target)
        if index % 10 == 0:
            print(f"matched benchmark: {index}/{len(records)}", flush=True)
    result = {
        "device": str(device), "cases": len(records),
        "policy_counts": dict(Counter(str(row["policy_id"]) for row, _, _ in records)),
        "scenario_counts": dict(Counter(str(layout["scenario"]) for _, layout, _ in records)),
        "unique_original_images": len({str(tile["group_id"]) for _, layout, _ in records for tile in layout["tiles"]}),
        "notes": "同一批 test 四图组；batched_four 的模型 batch=4，其余每组调用一次。均已加载并预热模型。包含图像读取和预处理；mosaic_online 还包含实时拼图与 JPEG 编码；不含网络传输和模型加载。",
    }
    for mode in group_times:
        result[mode] = {
            "end_to_end": percentiles(group_times[mode]),
            "metrics": four_image_metrics(np.asarray(labels), np.asarray(scores[mode])),
            "speedup_vs_sequential_mean": float(np.mean(group_times["sequential_four"]) / np.mean(group_times[mode])),
            "speedup_vs_batched_mean": float(np.mean(group_times["batched_four"]) / np.mean(group_times[mode])),
        }
    output_path = Path(args.output_root) / "comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # 保存逐组输出和原始计时，支持复算指标，不能把重复原图当成独立样本。
    with output_path.with_name("comparison_samples.jsonl").open("w", encoding="utf-8") as handle:
        for index, (row, layout, target) in enumerate(records):
            handle.write(json.dumps({
                "sample_id": row["sample_id"], "policy_id": row["policy_id"], "scenario": layout["scenario"],
                "group_ids": [tile["group_id"] for tile in layout["tiles"]], "labels": list(target),
                "scores": {mode: values[index] for mode, values in scores.items()},
                "latency_ms": {mode: values[index] for mode, values in group_times.items()},
            }, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def main() -> None:
    args = parse_args()
    if args.stage == "encode":
        encode(args)
    elif args.stage == "fit":
        fit(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
