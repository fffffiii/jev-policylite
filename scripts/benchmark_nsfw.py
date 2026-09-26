"""统一评测鉴黄系统：独立校准、逐图质量、预热延迟、完整类别输出。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import f1_score

from qwen35_moderation.data import MultimodalCollator, configure_max_pixels, find_group_leakage, read_manifest
from qwen35_moderation.metrics import choose_threshold, classification_metrics


EXPOSED = {"BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_EXPOSED"}


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {"mean_ms": float(values.mean()), "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95))}


def metrics(labels, scores, threshold=0.5):
    result = classification_metrics(np.asarray(labels), np.asarray(scores), threshold)
    result["f1"] = float(f1_score(labels, np.asarray(scores) >= threshold, zero_division=0))
    return result


def best_f1_threshold(labels, scores):
    """只在校准集择阈值；并列时选择较高阈值。"""
    scores = np.asarray(scores, dtype=float)
    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    return float(max(candidates, key=lambda t: (f1_score(labels, scores >= t, zero_division=0), t)))


def bootstrap_accuracy(records, thresholds, repeats=1000):
    """以原图为重采样单位，保留同图的两个政策判断。"""
    groups = defaultdict(list)
    for row in records:
        groups[row["group_id"]].append(int((row["score"] >= thresholds[row["policy_id"]]) == row["label"]))
    counts = np.asarray([len(v) for v in groups.values()])
    correct = np.asarray([sum(v) for v in groups.values()])
    rng = np.random.default_rng(20260926)
    draws = rng.integers(0, len(counts), size=(repeats, len(counts)))
    values = correct[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return [float(x) for x in np.quantile(values, [.025, .975])]


def quality_report(records):
    result = {}
    policies = sorted({r["policy_id"] for r in records})
    for policy in policies:
        calibration = [r for r in records if r["split"] == "calibration" and r["policy_id"] == policy]
        test = [r for r in records if r["split"] == "test" and r["policy_id"] == policy]
        cy, cp = [r["label"] for r in calibration], [r["score"] for r in calibration]
        ty, tp = [r["label"] for r in test], [r["score"] for r in test]
        threshold = best_f1_threshold(cy, cp)
        low_fpr = choose_threshold(np.asarray(cy), np.asarray(cp), max_fpr=.05)
        result[policy] = {
            "default": metrics(ty, tp), "calibrated_f1": metrics(ty, tp, threshold),
            "calibration_f1": metrics(cy, cp, threshold),
            "calibration_fpr5": metrics(cy, cp, low_fpr), "test_at_calibration_fpr5": metrics(ty, tp, low_fpr),
            "source_levels": {str(level): metrics([r["label"] for r in test if r["source_label"] == level],
                                                    [r["score"] for r in test if r["source_label"] == level], threshold)
                              for level in range(4)},
        }
    test = [r for r in records if r["split"] == "test"]
    default_thresholds = {p: .5 for p in policies}
    calibrated_thresholds = {p: result[p]["calibrated_f1"]["threshold"] for p in policies}
    for name, thresholds in (("default", default_thresholds), ("calibrated_f1", calibrated_thresholds)):
        labels = np.asarray([r["label"] for r in test])
        scores = np.asarray([r["score"] for r in test])
        predicted = np.asarray([int(r["score"] >= thresholds[r["policy_id"]]) for r in test])
        pooled = classification_metrics(labels, scores, .5, predictions=predicted)
        pooled["f1"] = float(f1_score(labels, predicted, zero_division=0))
        pooled["accuracy_cluster_bootstrap_95ci"] = bootstrap_accuracy(test, thresholds)
        result[f"pooled_{name}"] = pooled
    return result


def select_latency_rows(rows, count=120):
    """按原图去重、按等级平衡，固定两个政策的交替顺序。"""
    by_group = defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            by_group[row["group_id"]].append(row)
    buckets = defaultdict(list)
    for group in sorted(by_group):
        pair = sorted(by_group[group], key=lambda r: r["policy_id"])
        buckets[pair[0]["source_label"]].append(pair)
    rng = random.Random(20260926)
    result = []
    for level in sorted(buckets):
        rng.shuffle(buckets[level])
        for index, pair in enumerate(buckets[level][:count//4]):
            result.append(pair[index % len(pair)])
    rng.shuffle(result)
    return result


class Runner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.meta = {}
        if args.model.startswith("jev"):
            # 仅运行 Jev 时加载模型依赖，统计工具可在轻量 CPU 环境导入。
            from qwen35_moderation.model import model_inputs
            from qwen35_moderation.runtime import load_checkpoint
            self.model_inputs = model_inputs
            self.model, processor, _, _ = load_checkpoint(args.checkpoint, self.device)
            pixels = 50176 if args.model == "jev_224" else 200704
            configure_max_pixels(processor, pixels)
            self.collator = MultimodalCollator(processor, Path(args.image_root), 1024)
            self.meta = {"max_pixels": pixels, "precision": "bfloat16 backbone + autocast; three heads enabled"}
        elif args.model == "nudenet":
            import nudenet.nudenet as nude
            import onnxruntime as ort
            self.nude = nude
            self.labels = list(getattr(nude, "__labels"))
            model_path = Path(nude.__file__).with_name("320n.onnx")
            options = ort.SessionOptions()
            options.intra_op_num_threads = args.threads
            options.inter_op_num_threads = 1
            providers = [("CUDAExecutionProvider", {"device_id": self.device.index or 0})] if self.device.type == "cuda" else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
            if self.device.type == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
                raise RuntimeError("NudeNet CUDA provider 未加载，拒绝把 CPU 结果写成 GPU 结果")
            self.meta = {"resolution": 320, "precision": "float32 ONNX", "providers": self.session.get_providers(),
                         "labels": self.labels, "onnx_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                         "weight_bytes": model_path.stat().st_size, "exposed_mapping": sorted(EXPOSED),
                         "onnx_input_shape": self.session.get_inputs()[0].shape}
        else:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            folder = Path(args.baselines) / args.model
            processor_config = json.loads((folder / "preprocessor_config.json").read_text(encoding="utf-8"))
            if processor_config.get("image_processor_type") == "ViTFeatureExtractor":
                # 兼容旧模型卡的类名；保留原 resize、均值、方差和插值配置。
                from transformers import ViTImageProcessor
                self.processor = ViTImageProcessor.from_dict(processor_config)
            else:
                self.processor = AutoImageProcessor.from_pretrained(folder, use_fast=False, local_files_only=True)
            self.model = AutoModelForImageClassification.from_pretrained(folder, local_files_only=True).to(self.device).eval()
            self.labels = [str(self.model.config.id2label[i]).lower() for i in range(self.model.config.num_labels)]
            self.meta = {"labels": self.labels, "precision": "float32 weights + bfloat16 autocast", "processor": self.processor.to_dict(),
                         "weight_bytes": sum(p.stat().st_size for p in folder.glob("*.safetensors"))}
        if args.model != "nudenet":
            self.meta["parameters"] = sum(p.numel() for p in self.model.parameters())

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def run(self, rows):
        self.sync()
        start = time.perf_counter()
        args = self.args
        if args.model.startswith("jev"):
            encoded = self.model_inputs(self.collator(rows))
        elif args.model == "nudenet":
            prepared = [self.nude._read_image(str(Path(args.image_root) / r["image"]), 320) for r in rows]
            encoded = np.vstack([p[0] for p in prepared])
        else:
            images = []
            try:
                for row in rows:
                    with Image.open(Path(args.image_root) / row["image"]) as source:
                        images.append(ImageOps.exif_transpose(source).convert("RGB"))
                encoded = self.processor(images=images, return_tensors="pt")
            finally:
                for image in images:
                    image.close()
        prep_end = time.perf_counter()
        if args.model == "nudenet":
            outputs = self.session.run(None, {self.session.get_inputs()[0].name: encoded})
            probabilities = []
            extra = []
            for i, prepared_row in enumerate(prepared):
                detections = self.nude._postprocess([outputs[0][i:i+1]], *prepared_row[1:], 320, 320)
                values = {label: 0.0 for label in self.labels}
                for detection in detections:
                    label = detection["class"]
                    values[label] = max(values[label], detection["score"])
                probabilities.append(max(values[c] for c in EXPOSED))
                extra.append({"classes": values, "detections": detections})
        else:
            inputs = {k: v.to(self.device) for k, v in encoded.items()}
            context = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
            with torch.inference_mode(), context:
                if args.model.startswith("jev"):
                    bundle = self.model(return_heads=True, **inputs)
                    probabilities = torch.sigmoid(bundle.violation_logit.float()).cpu().tolist()
                    attributes = torch.sigmoid(bundle.attribute_logits.float()).cpu().tolist()
                    actions = torch.softmax(bundle.decision_logits.float(), -1).cpu().tolist()
                    extra = [{"attributes": a, "actions": d} for a, d in zip(attributes, actions)]
                else:
                    values = torch.softmax(self.model(**inputs).logits.float(), -1).cpu().tolist()
                    extra = [{"classes": dict(zip(self.labels, v))} for v in values]
                    probabilities = []
                    for row, values_row in zip(rows, extra):
                        p = values_row["classes"]
                        if args.model == "falconsai":
                            probabilities.append(p["nsfw"])
                        else:
                            names = ["porn", "hentai"] + (["sexy"] if row["policy_id"] == "pilot-suggestive-v1" else [])
                            probabilities.append(sum(p[name] for name in names))
        self.sync()
        stop = time.perf_counter()
        if len(probabilities) != len(rows) or not np.isfinite(probabilities).all():
            raise RuntimeError("预测数量或数值错误")
        return probabilities, extra, {"total_ms": (stop-start)*1000, "preprocess_ms": (prep_end-start)*1000,
                                      "forward_postprocess_ms": (stop-prep_end)*1000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["jev_224", "jev_448", "falconsai", "five_class", "nudenet"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default="outputs/public-pilot-multihead")
    parser.add_argument("--manifest", type=Path, default=Path("data/public-pilot/manifest.jsonl"))
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--baselines", default="outputs/nsfw-baselines")
    parser.add_argument("--output", type=Path, default=Path("outputs/nsfw-comparison-v1"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--latency-images", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timing-only", action="store_true", help="复用已验证的同清单预测，仅重新计时")
    args = parser.parse_args()
    import resource
    run_started = datetime.now(timezone.utc).isoformat()
    previous = None
    if args.timing_only:
        previous = json.loads((args.output / f"{args.model}.json").read_text(encoding="utf-8"))
        if previous["source_manifest_sha256"] != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
            raise ValueError("计时复用的预测不属于当前清单")
        if previous["model"] != args.model or previous["args"]["checkpoint"] != args.checkpoint:
            raise ValueError("计时复用的模型配置不一致")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260926)
    rows = read_manifest(args.manifest)
    if find_group_leakage(rows):
        raise ValueError("数据存在跨集合原图")
    evaluation = [r for r in rows if r["split"] in ("calibration", "test")]
    # 测试集顺序固定洗牌，避免同等级图片集中影响批处理计时。
    random.Random(20260926).shuffle(evaluation)
    if args.smoke:
        evaluation = evaluation[:8]
    args.output.mkdir(parents=True, exist_ok=True)
    start_load = time.perf_counter()
    runner = Runner(args)
    load_seconds = time.perf_counter() - start_load
    for batch_size in (1, 4):
        for _ in range(3):
            runner.run(evaluation[:batch_size])
    if runner.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runner.device)
    path = args.output / f"{args.model}-predictions.jsonl"
    predictions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if previous else []
    if previous and (len(predictions) != len(evaluation) or {r["sample_id"] for r in predictions} != {r["sample_id"] for r in evaluation}):
        raise ValueError("计时复用的预测缺少样本")
    for start in ([] if previous else range(0, len(evaluation), 4)):
        batch = evaluation[start:start+4]
        scores, extra, _ = runner.run(batch)
        for row, score, output in zip(batch, scores, extra):
            predictions.append({**{k: row[k] for k in ("sample_id", "group_id", "split", "policy_id", "source_label", "label")},
                                "score": float(score), **output})
        if start % 200 == 0:
            print(f"{args.model}: quality {start+len(batch)}/{len(evaluation)}", flush=True)
    if not previous:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions), encoding="utf-8")
    latency_rows = select_latency_rows(rows, args.latency_images)
    if args.smoke:
        latency_rows = latency_rows[:4]
    latency = {}
    traces = []
    # 覆盖全部输入及 batch 形状，避免首见尺寸的内核初始化进入正式计时。
    for batch_size in (1, 4):
        for start in range(0, len(latency_rows), batch_size):
            runner.run(latency_rows[start:start+batch_size])
    for repeat in range(1 if args.smoke else args.repeats):
        # 每轮交替 batch 次序，保留逐批原始计时供复算。
        for batch_size in ((1, 4) if repeat % 2 == 0 else (4, 1)):
            for _ in range(3):
                runner.run(latency_rows[:batch_size])
            for start in range(0, len(latency_rows), batch_size):
                batch = latency_rows[start:start+batch_size]
                _, _, timing = runner.run(batch)
                traces.append({"repeat": repeat, "batch_size": batch_size, "sample_ids": [r["sample_id"] for r in batch], **timing})
            print(f"{args.model}: timing repeat={repeat+1} batch={batch_size}", flush=True)
    for batch_size in (1, 4):
        data = [r for r in traces if r["batch_size"] == batch_size]
        latency[str(batch_size)] = {"batches": len(data), "images": sum(len(r["sample_ids"]) for r in data),
                                   "end_to_end": summarize([r["total_ms"] for r in data]),
                                   "preprocess": summarize([r["preprocess_ms"] for r in data]),
                                   "forward_postprocess": summarize([r["forward_postprocess_ms"] for r in data]),
                                   "images_per_second": sum(len(r["sample_ids"]) for r in data)/sum(r["total_ms"] for r in data)*1000}
    packages = {}
    for package in ("torch", "torchvision", "transformers", "peft", "numpy", "Pillow", "onnxruntime-gpu", "nudenet"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        gpu_status = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,memory.used,utilization.gpu", "--format=csv"], text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        gpu_status = None
    report = {"model": args.model, "run_started_utc": run_started, "metadata": runner.meta, "args": vars(args) | {"manifest": str(args.manifest), "output": str(args.output)},
              "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "environment": {"platform": platform.platform(), "python": platform.python_version(), "packages": packages,
                              "cpu_threads": args.threads, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                              "gpu": torch.cuda.get_device_name(runner.device) if runner.device.type == "cuda" else None, "gpu_snapshot_after_run": gpu_status},
              "load_seconds": load_seconds, "quality": quality_report(predictions) if not args.smoke else {},
              "latency": latency, "latency_unique_images": len(latency_rows),
              "memory": {"process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                         "torch_peak_allocated_mib": torch.cuda.max_memory_allocated(runner.device)/2**20 if runner.device.type == "cuda" and args.model != "nudenet" else None},
              "timing_scope": "warm process: disk image read, preprocessing, H2D, forward, all heads/boxes, CPU output; excludes loading, HTTP/queue/network; page cache warm",
              "traces": traces}
    if previous:
        report["quality_environment"] = previous.get("quality_environment", previous["environment"])
    (args.output / f"{args.model}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"model": args.model, "quality": report["quality"].get("pooled_calibrated_f1"), "latency": latency, "memory": report["memory"]}), flush=True)


if __name__ == "__main__":
    main()
