"""从原始预测与计时生成对比表、属性诊断和可复现数据图。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

ORDER = ["falconsai", "five_class", "nudenet", "jev_224", "jev_448"]
NAMES = {"falconsai": "Falconsai ViT", "five_class": "Giacomo ViT (5-class)", "nudenet": "NudeNet 320n",
         "jev_224": "Jev-PolicyLite 50k", "jev_448": "Jev-PolicyLite 201k"}
COLORS = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00"]
MULTI_NAMES = {"sequential_four": "Serial x4", "batched_four": "Batch=4", "native": "Global heads",
               "native_local": "Local heads", "native_shared": "Shared local head", "mosaic_online": "Online mosaic"}


def save_figure(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    # Matplotlib 在 SVG 路径行尾保留空格；去掉后便于通过仓库空白检查。
    svg = folder / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)


def read_predictions(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generated_block(text, name, body):
    """替换有明确边界的生成区域，保留手工撰写的协议说明。"""
    start, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
    block = f"{start}\n\n{body.strip()}\n\n{end}"
    pattern = re.escape(start) + r".*?" + re.escape(end)
    if start in text:
        return re.sub(pattern, lambda _: block, text, flags=re.S)
    marker = f"<!-- {name} -->"
    if marker not in text:
        raise ValueError(f"文档缺少生成区域：{name}")
    return text.replace(marker, block)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("docs/results/nsfw-comparison-v1"))
    parser.add_argument("--figures", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    reports = {name: json.loads((args.results / f"{name}.json").read_text(encoding="utf-8")) for name in ORDER}
    source_hashes = {r["source_manifest_sha256"] for r in reports.values()}
    if len(source_hashes) != 1:
        raise ValueError("对照模型使用的源清单不一致")
    keys = None
    timing_keys = None
    for name in ORDER:
        rows = read_predictions(args.results / f"{name}-predictions.jsonl")
        current = {(r["split"], r["sample_id"], r["label"]) for r in rows}
        if len(current) != len(rows) or (keys is not None and current != keys):
            raise ValueError(f"{name} 评测样本重复或未与其他方法匹配")
        keys = current
        report = reports[name]
        if report.get("latency_protocol") != "paired_timing_protocol.json":
            raise ValueError(f"{name} 尚未完成全输入预热后的配对计时")
        measured = {(t["repeat"], t["batch_size"], tuple(t["sample_ids"])) for t in report["traces"]}
        if len(measured) != len(report["traces"]) or (timing_keys is not None and measured != timing_keys):
            raise ValueError(f"{name} 计时样本或重复轮次未匹配")
        timing_keys = measured
        for batch_size in (1, 4):
            trace = [r for r in report["traces"] if r["batch_size"] == batch_size]
            stored = report["latency"][str(batch_size)]
            times = np.asarray([t["total_ms"] for t in trace])
            expected = [times.mean(), *np.percentile(times, [50, 95])]
            if not np.allclose(expected, [stored["end_to_end"][k] for k in ("mean_ms", "p50_ms", "p95_ms")], atol=1e-9):
                raise ValueError(f"{name} 汇总延迟与原始计时不一致")
            if stored["batches"] != len(trace) or stored["images"] != sum(len(t["sample_ids"]) for t in trace):
                raise ValueError(f"{name} 吞吐样本数错误")
    summary = {"models": {}, "attributes": {}}
    for name, report in reports.items():
        summary["models"][name] = {"name": NAMES[name], "quality": report["quality"], "latency": report["latency"], "memory": report["memory"]}
        if not name.startswith("jev"):
            continue
        rows = [r for r in read_predictions(args.results / f"{name}-predictions.jsonl") if r["split"] == "test"]
        attr = {}
        for index, label in enumerate(("nudity", "sexual_act", "suggestive", "medical")):
            actual = [int(r["source_label"] >= 2) if index == 0 else int(r["source_label"] == 3) if index == 1
                      else int(r["source_label"] == 1) if index == 2 else 0 for r in rows]
            predicted = [r["attributes"][index] >= .5 for r in rows]
            attr[label] = {"positive_policy_rows": sum(actual), "positive_unique_images": len({r["group_id"] for r, y in zip(rows, actual) if y}),
                           "predicted_positive_rows": sum(predicted), "f1": float(f1_score(actual, predicted)) if sum(actual) else None,
                           "precision": float(precision_score(actual, predicted, zero_division=0)) if sum(actual) else None,
                           "recall": float(recall_score(actual, predicted, zero_division=0)) if sum(actual) else None}
        summary["attributes"][name] = attr
    (args.results / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    quality = ["| Model | Acc @0.5 | Acc calibrated | F1 | Recall | FPR | Accuracy 95% CI |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    speed = ["| Model | B=1 mean / P95 (ms) | B=4 mean / P95 (ms) | B=4 images/s | Peak RSS (MiB) | Torch allocated (MiB) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    policy = ["| Model | Policy | Threshold | Accuracy | F1 | Recall | FPR | AP | Recall / FPR at calibration FPR≤5% |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name in ORDER:
        report = reports[name]
        q = report["quality"]["pooled_calibrated_f1"]
        default = report["quality"]["pooled_default"]
        ci = q["accuracy_cluster_bootstrap_95ci"]
        quality.append(f"| {NAMES[name]} | {default['accuracy']:.2%} | {q['accuracy']:.2%} | {q['f1']:.2%} | {q['recall']:.2%} | {q['fpr']:.2%} | {ci[0]:.2%}–{ci[1]:.2%} |")
        l1, l4 = (report["latency"][str(b)]["end_to_end"] for b in (1, 4))
        memory = report["memory"]
        torch_mem = memory["torch_peak_allocated_mib"]
        speed.append(f"| {NAMES[name]} | {l1['mean_ms']:.2f} / {l1['p95_ms']:.2f} | {l4['mean_ms']:.2f} / {l4['p95_ms']:.2f} | {report['latency']['4']['images_per_second']:.1f} | {memory['process_peak_rss_mib']:.0f} | {torch_mem:.0f} |" if torch_mem is not None else
                     f"| {NAMES[name]} | {l1['mean_ms']:.2f} / {l1['p95_ms']:.2f} | {l4['mean_ms']:.2f} / {l4['p95_ms']:.2f} | {report['latency']['4']['images_per_second']:.1f} | {memory['process_peak_rss_mib']:.0f} | N/A (ONNX) |")
        for p in ("pilot-suggestive-v1", "pilot-explicit-v1"):
            q = report["quality"][p]["calibrated_f1"]
            low = report["quality"][p]["test_at_calibration_fpr5"]
            policy.append(f"| {NAMES[name]} | {p} | {q['threshold']:.6f} | {q['accuracy']:.2%} | {q['f1']:.2%} | {q['recall']:.2%} | {q['fpr']:.2%} | {q['pr_auc']:.2%} | {low['recall']:.2%} / {low['fpr']:.2%} |")
    for name, lines in (("quality", quality), ("speed", speed), ("policies", policy)):
        (args.results / f"{name}-table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    attr_lines = ["| 属性 | 测试原图正例 | 50k F1 @0.5 | 201k F1 @0.5 |", "| --- | ---: | ---: | ---: |"]
    for key, name in (("nudity", "裸露"), ("sexual_act", "性行为"), ("suggestive", "性暗示"), ("medical", "医学")):
        low, high = [summary["attributes"][n][key] for n in ("jev_224", "jev_448")]
        low_f1 = f"{low['f1']:.2%}" if low["f1"] is not None else "不可评估"
        high_f1 = f"{high['f1']:.2%}" if high["f1"] is not None else "不可评估"
        attr_lines.append(f"| {name} | {low['positive_unique_images']} | {low_f1} | {high_f1} |")
    cn_quality = "\n".join(quality).replace("| Model | Acc @0.5 | Acc calibrated | F1 | Recall | FPR | Accuracy 95% CI |", "| 模型 | 准确率 @0.5 | 校准后准确率 | F1 | 召回率 | 误报率 | 准确率 95% CI |")
    cn_speed = "\n".join(speed).replace("| Model | B=1 mean / P95 (ms) | B=4 mean / P95 (ms) | B=4 images/s | Peak RSS (MiB) | Torch allocated (MiB) |", "| 模型 | B=1 均值 / P95 (ms) | B=4 均值 / P95 (ms) | B=4 图片/秒 | 峰值 RSS (MiB) | Torch 分配显存 (MiB) |")
    cn_policy = "\n".join(policy).replace("| Model | Policy | Threshold | Accuracy | F1 | Recall | FPR | AP | Recall / FPR at calibration FPR≤5% |", "| 模型 | 政策 | 阈值 | 准确率 | F1 | 召回率 | FPR | AP | 校准 FPR≤5% 工作点的测试召回 / FPR |")
    doc = Path("docs/NSFW_COMPARISON_V1.md")
    content = doc.read_text(encoding="utf-8")
    for marker, body in (("QUALITY_TABLE", cn_quality), ("SPEED_TABLE", cn_speed), ("POLICY_TABLE", cn_policy), ("ATTRIBUTE_TABLE", "\n".join(attr_lines))):
        content = generated_block(content, marker, body)
    doc.write_text(content, encoding="utf-8")

    multi = json.loads(Path("docs/results/multi-photo-v1/comparison.json").read_text(encoding="utf-8"))
    cn_multi_names = ["逐张串行", "单图 batch=4", "四图，共享末 token", "四图，局部位置头", "四图，共享局部头", "实时 2×2 拼图"]
    for locale in ("zh", "en"):
        is_cn = locale == "zh"
        header = "| 方法 | 逐图准确率 | F1 | 召回率 | 四张全对 | 均值/组 | P95/组 |" if is_cn else "| Method | Per-image accuracy | F1 | Recall | All four correct | Mean/group | P95/group |"
        lines = [header, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for i, (key, name) in enumerate(MULTI_NAMES.items()):
            q, latency = multi[key]["metrics"], multi[key]["end_to_end"]
            m = q["micro"]
            lines.append(f"| {cn_multi_names[i] if is_cn else name} | {m['accuracy']:.2%} | {m['f1']:.2%} | {m['recall']:.2%} | {q['exact_match']:.1%} | {latency['mean_ms']:.2f} ms | {latency['p95_ms']:.2f} ms |")
        readme = Path("README.md" if is_cn else "README.en.md")
        content = readme.read_text(encoding="utf-8")
        for marker, body in (("MULTI_PHOTO_TABLE", "\n".join(lines)), ("NSFW_QUALITY_TABLE", cn_quality if is_cn else "\n".join(quality)), ("NSFW_SPEED_TABLE", cn_speed if is_cn else "\n".join(speed))):
            content = generated_block(content, marker, body)
        readme.write_text(content, encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.axisbelow": True,
                         "pdf.fonttype": 42, "svg.hashsalt": "jev-policylite-benchmark-v1"})
    y = np.arange(len(ORDER))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.8, 4.3), gridspec_kw={"width_ratios": [1.1, 1]})
    for i, name in enumerate(ORDER):
        values = [reports[name]["latency"][str(b)]["end_to_end"]["mean_ms"] for b in (1, 4)]
        left.barh(i-.17, values[0], .28, color=COLORS[i], alpha=.4, edgecolor=COLORS[i])
        left.barh(i+.17, values[1], .28, color=COLORS[i])
        for offset, value in zip((-.17, .17), values):
            left.text(value*1.06, i+offset, f"{value:.1f}", va="center", fontsize=8)
        vals = [100*reports[name]["quality"][key]["f1"] for key in ("pooled_default", "pooled_calibrated_f1")]
        right.barh(i-.17, vals[0], .28, color=COLORS[i], alpha=.4)
        right.barh(i+.17, vals[1], .28, color=COLORS[i])
        for offset, value in zip((-.17, .17), vals):
            right.text(value+1, i+offset, f"{value:.1f}", va="center", fontsize=8)
    left.set_yticks(y, [NAMES[k] for k in ORDER]); left.invert_yaxis()
    left.set_xscale("log"); left.set_xlim(.8, max(r["latency"]["4"]["end_to_end"]["mean_ms"] for r in reports.values())*2)
    left.set_xlabel("Mean latency per call (ms, log scale)")
    left.set_title("Speed  ·  light: B=1 / solid: B=4", fontsize=11)
    right.set_yticks(y, []); right.invert_yaxis(); right.set_xlim(0, 106)
    right.set_xlabel("Test F1 (%)")
    right.set_title("Quality  ·  light: t=0.5 / solid: calibrated", fontsize=11)
    for ax in (left, right):
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Moderation systems on RTX 3090", fontsize=15, fontweight="bold")
    fig.text(.5, .01, "600 test images × 2 policies; calibration uses 300 separate images. Latency includes image I/O and preprocessing.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .94))
    save_figure(fig, args.figures, "nsfw-speed-quality")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    for ax, policy_id, title in zip(axes, ("pilot-suggestive-v1", "pilot-explicit-v1"), ("Suggestive policy (L2–L4 positive)", "Explicit policy (L3–L4 positive)")):
        matrix = np.asarray([[reports[n]["quality"][policy_id]["source_levels"][str(level)]["accuracy"]*100 for level in range(4)] for n in ORDER])
        im = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
        for row in range(5):
            for col in range(4):
                ax.text(col, row, f"{matrix[row, col]:.1f}", ha="center", va="center", color="white" if matrix[row, col] > 64 else "#18242e")
        ax.set_xticks(range(4), ["L1", "L2", "L3", "L4"])
        ax.set_yticks(y, [NAMES[n] for n in ORDER] if ax is axes[0] else [])
        ax.set_title(title, fontsize=11); ax.set_xlabel("Source level (150 unique images each)")
    fig.colorbar(im, ax=axes, label="Correct decisions (%)", fraction=.025, pad=.02)
    fig.suptitle("Where each system makes errors", fontweight="bold", fontsize=14)
    fig.subplots_adjust(left=.2, right=.86, top=.83, bottom=.19, wspace=.12)
    save_figure(fig, args.figures, "nsfw-source-levels")

    multi = json.loads(Path("docs/results/multi-photo-v1/comparison.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for index, (key, title) in enumerate(MULTI_NAMES.items()):
        x = multi[key]["end_to_end"]["mean_ms"]
        value = multi[key]["metrics"]["micro"]["accuracy"]*100
        ax.scatter(x, value, s=65, color=(COLORS+["#CC79A7"])[index], marker=("o", "s", "^", "D", "P", "v")[index], label=title)
        ax.annotate(f"{x:.0f} ms / {value:.2f}%", (x, value), xytext=((8 if key != "sequential_four" else -110), (9 if key != "native_shared" else -17)), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean latency per four-image group (ms)"); ax.set_ylabel("Per-image accuracy (%)")
    ax.set_xlim(130, 570); ax.set_ylim(79, 92); ax.grid(alpha=.2)
    ax.set_title("Four-image readouts: speed–accuracy trade-off", pad=12)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.text(.5, .01, "Historical paired run: 200 groups, 800 labels, 426 unique source images. Threshold = 0.5.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, 1))
    save_figure(fig, args.figures, "multi-photo-tradeoff")


if __name__ == "__main__":
    main()
