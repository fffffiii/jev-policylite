"""从同一数据划分构造 2×2 四图拼接压力测试集。"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from qwen35_moderation.data import read_manifest
from qwen35_moderation.mosaic import MOSAIC_SCENARIOS, aggregate_violation, make_mosaic, scenario_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造 2×2 四图拼接审核压力测试集")
    parser.add_argument("--manifest", required=True, help="源 JSONL 清单")
    parser.add_argument("--image-root", required=True, help="源图片根目录")
    parser.add_argument("--split", default="test", choices=["train", "validation", "calibration", "test"])
    parser.add_argument("--output-dir", required=True, help="拼图和布局清单的输出目录")
    parser.add_argument("--output-manifest", required=True, help="生成的 JSONL 清单")
    parser.add_argument("--cases-per-scenario", type=int, default=20)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--policies", nargs="*", help="默认使用源清单中的全部政策")
    return parser.parse_args()


def unique_rows_by_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """每个 group 只保留一条样本，避免同一原图在一张拼图中重复。"""
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        group_id = str(row["group_id"])
        if group_id in unique:
            raise ValueError(f"同一 group 在同一 split/policy 中重复：{group_id}")
        unique[group_id] = row
    return list(unique.values())


def choose_tiles(
    pools: dict[int, list[dict[str, Any]]], labels: tuple[int, ...], rng: random.Random
) -> list[dict[str, Any]]:
    """从正负样本池抽取互不重复的四个 group。"""
    selected: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    for label in labels:
        candidates = [row for row in pools[label] if str(row["group_id"]) not in used_groups]
        if not candidates:
            raise ValueError(f"标签 {label} 的可用原图不足，无法组成不重复四图拼接")
        row = rng.choice(candidates)
        selected.append(row)
        used_groups.add(str(row["group_id"]))
    rng.shuffle(selected)
    return selected


def main() -> None:
    args = parse_args()
    if args.cases_per_scenario < 1:
        raise ValueError("cases-per-scenario 必须大于 0")
    rows = [row for row in read_manifest(args.manifest) if row["split"] == args.split]
    if not rows:
        raise ValueError(f"源清单没有 {args.split} 数据")
    source_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_manifest = Path(args.output_manifest)
    if output_manifest.exists():
        raise FileExistsError(f"输出清单已存在：{output_manifest}")
    if image_dir.exists() and any(image_dir.iterdir()):
        raise FileExistsError(f"输出图片目录非空：{image_dir}")
    image_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    policies = args.policies or sorted({str(row["policy_id"]) for row in rows})
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for policy_id in policies:
        policy_rows = unique_rows_by_group([row for row in rows if row["policy_id"] == policy_id])
        pools: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in policy_rows:
            pools[int(row["label"])].append(row)
        if not pools[0] or not pools[1]:
            raise ValueError(f"政策 {policy_id} 缺少正样本或负样本")
        grouped[policy_id] = pools

    rng = random.Random(args.seed)
    manifests: list[dict[str, Any]] = []
    layouts: list[dict[str, Any]] = []
    for policy_id in policies:
        for scenario in MOSAIC_SCENARIOS:
            plan = scenario_labels(scenario)
            for case_index in range(args.cases_per_scenario):
                tiles = choose_tiles(grouped[policy_id], plan, rng)
                images: list[Image.Image] = []
                try:
                    for tile in tiles:
                        image_path = source_root / str(tile["image"])
                        with Image.open(image_path) as source:
                            images.append(source.copy())
                    mosaic = make_mosaic(images, tile_size=args.tile_size)
                finally:
                    for image in images:
                        image.close()
                slug_policy = "".join(char if char.isalnum() else "-" for char in policy_id).strip("-")
                stem = f"mosaic-{args.split}-{slug_policy}-{scenario}-{case_index:03d}"
                image_name = f"{stem}.jpg"
                mosaic.save(image_dir / image_name, format="JPEG", quality=95, subsampling=0)
                mosaic.close()
                labels = [int(tile["label"]) for tile in tiles]
                expected = aggregate_violation(labels)
                if expected != aggregate_violation(plan):
                    raise AssertionError("抽样标签与场景计划不一致")
                first = tiles[0]
                manifests.append(
                    {
                        "sample_id": stem,
                        "group_id": stem,
                        "image": f"images/{image_name}",
                        "text": "",
                        "policy_id": policy_id,
                        "policy_text": first["policy_text"],
                        "label": expected,
                        "split": args.split,
                        "attributes": ["mosaic_2x2", f"mosaic_{scenario}"],
                        "source": "derived-mosaic",
                        "annotation_source": "derived-or",
                        "mosaic_scenario": scenario,
                    }
                )
                layouts.append(
                    {
                        "sample_id": stem,
                        "policy_id": policy_id,
                        "scenario": scenario,
                        "expected_label": expected,
                        "tile_labels": labels,
                        "tiles": [
                            {
                                "group_id": tile["group_id"],
                                "sample_id": tile["sample_id"],
                                "image": tile["image"],
                                "label": int(tile["label"]),
                                "attributes": tile.get("attributes", []),
                            }
                            for tile in tiles
                        ],
                    }
                )

    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in manifests:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "layouts.jsonl").open("w", encoding="utf-8") as handle:
        for row in layouts:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source_manifest": str(Path(args.manifest).resolve()),
        "source_split": args.split,
        "seed": args.seed,
        "tile_size": args.tile_size,
        "policies": policies,
        "cases_per_scenario": args.cases_per_scenario,
        "scenario_count": len(MOSAIC_SCENARIOS),
        "record_count": len(manifests),
        "label_rule": "OR: 任一 tile 在同一 policy 下违规，则整张 2×2 拼图违规",
        "limitations": [
            "这是缩小目标与多目标干扰压力测试，不等价于四张独立图片逐张审核。",
            "标签由源样本按 OR 规则派生，不能替代人工审核拼图。",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
