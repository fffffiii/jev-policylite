"""完整预热后交错计时，减轻主机负载漂移对不同系统的比较影响。"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark_nsfw import Runner, select_latency_rows, summarize
from qwen35_moderation.data import read_manifest

MODELS = ("falconsai", "five_class", "nudenet", "jev_224", "jev_448")


def gpu_snapshot():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv"], text=True, timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("outputs/nsfw-comparison-v1"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reports = {name: json.loads((args.results / f"{name}.json").read_text(encoding="utf-8")) for name in MODELS}
    manifests = {r["args"]["manifest"] for r in reports.values()}
    if len(manifests) != 1:
        raise ValueError("模型清单不匹配")
    manifest = Path(next(iter(manifests)))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if any(r["source_manifest_sha256"] != digest for r in reports.values()):
        raise ValueError("模型报告与当前清单散列不匹配")
    rows = select_latency_rows(read_manifest(manifest), 120)
    before = gpu_snapshot()
    started = datetime.now(timezone.utc).isoformat()
    runners = {}
    for name in MODELS:
        runners[name] = Runner(SimpleNamespace(**reports[name]["args"]))
        print(f"loaded {name}", flush=True)
    # 对每个待测输入都执行一次未计时前向，涵盖动态尺寸和不同 batch。
    warmup_max = {}
    for name, runner in runners.items():
        observed = []
        for batch_size in (1, 4):
            for start in range(0, len(rows), batch_size):
                _, _, timing = runner.run(rows[start:start+batch_size])
                observed.append(timing["total_ms"])
        warmup_max[name] = max(observed)
        print(f"fully warmed {name}", flush=True)
    traces = {name: [] for name in MODELS}
    schedule = []
    rng = random.Random(20260926)
    for repeat in range(args.repeats):
        for batch_size in ((1, 4) if repeat % 2 == 0 else (4, 1)):
            starts = list(range(0, len(rows), batch_size))
            rng.shuffle(starts)
            for start in starts:
                batch = rows[start:start+batch_size]
                order = list(MODELS)
                rng.shuffle(order)
                schedule.append({"repeat": repeat, "batch_size": batch_size, "start": start, "order": order})
                for name in order:
                    _, _, timing = runners[name].run(batch)
                    traces[name].append({"repeat": repeat, "batch_size": batch_size,
                                         "sample_ids": [r["sample_id"] for r in batch], **timing})
            print(f"paired repeat={repeat+1} batch={batch_size} done", flush=True)
    protocol = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "all models resident, serial interleaved calls, full-input warmup, no outlier removal",
                "seed": 20260926, "repeats": args.repeats, "warmup_calls_per_model": len(rows)+len(rows)//4,
                "warmup_max_ms": warmup_max, "gpu_before": before, "gpu_after": gpu_snapshot(),
                "unique_images": len({r["group_id"] for r in rows}),
                "policy_counts": dict(Counter(r["policy_id"] for r in rows)),
                "source_counts": dict(Counter(r["source_label"] for r in rows)), "schedule": schedule,
                "memory_note": "Per-model memory comes from the previous isolated-process run, not this multi-model resident process."}
    (args.results / "paired_timing_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = args.results / "before-full-warmup"
    archive.mkdir(exist_ok=True)
    for name, report in reports.items():
        # 保留上一轮未覆盖全部尺寸的原始记录，不删除异常值来美化均值。
        (archive / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["isolated_memory_environment"] = report["environment"]
        report["latency_protocol"] = "paired_timing_protocol.json"
        report["traces"] = traces[name]
        for batch_size in (1, 4):
            data = [r for r in traces[name] if r["batch_size"] == batch_size]
            report["latency"][str(batch_size)] = {
                "batches": len(data), "images": sum(len(r["sample_ids"]) for r in data),
                "end_to_end": summarize([r["total_ms"] for r in data]),
                "preprocess": summarize([r["preprocess_ms"] for r in data]),
                "forward_postprocess": summarize([r["forward_postprocess_ms"] for r in data]),
                "images_per_second": sum(len(r["sample_ids"]) for r in data)/sum(r["total_ms"] for r in data)*1000,
            }
        (args.results / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"model": name, "latency": report["latency"]}), flush=True)


if __name__ == "__main__":
    main()
