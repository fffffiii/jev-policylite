"""核对源图划分、四图标签与已发布权重；不读取或导出图片内容。"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from qwen35_moderation.data import find_group_leakage, read_manifest
from qwen35_moderation.multi_photo import validate_four_image_layout


def audit(manifest: Path, multi_root: Path) -> dict:
    rows = read_manifest(manifest)
    by_id = {row["sample_id"]: row for row in rows}
    errors = []
    if len(by_id) != len(rows):
        errors.append("源清单 sample_id 重复")
    leakage = find_group_leakage(rows)
    if leakage:
        errors.append(f"源图跨集合：{len(leakage)}")
    hashes = defaultdict(set)
    for row in rows:
        hashes[row.get("pixel_sha256", row["group_id"])].add(row["split"])
        expected = int(row["source_label"] >= (1 if row["policy_id"] == "pilot-suggestive-v1" else 2))
        if row["label"] != expected:
            errors.append(f"源图规则标签错误：{row['sample_id']}")
    hash_overlap = sum(len(v) > 1 for v in hashes.values())
    if hash_overlap:
        errors.append(f"像素散列跨集合：{hash_overlap}")
    multi = {}
    original_splits = defaultdict(set)
    sample_splits = defaultdict(set)
    for split in ("train", "validation", "test"):
        folder = multi_root / split
        records = read_manifest(folder / "manifest.jsonl")
        layouts = [json.loads(line) for line in (folder / "layouts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        lookup = {x["sample_id"]: x for x in layouts}
        if len(lookup) != len(layouts) or len(records) != len(layouts):
            errors.append(f"{split} 清单和布局未一一对应")
        originals = set()
        for row in records:
            sample_splits[row["sample_id"]].add(split)
            if row["split"] != split:
                errors.append(f"{split} 目录包含其他划分")
            layout = lookup[row["sample_id"]]
            validate_four_image_layout(layout, row)
            for tile in layout["tiles"]:
                original = by_id[tile["sample_id"]]
                for key in ("image", "group_id", "label"):
                    if tile[key] != original[key]:
                        errors.append(f"{split} 原图 {key} 不一致")
                if original["split"] != split or original["policy_id"] != row["policy_id"]:
                    errors.append(f"{split} 原图政策或划分不一致")
                originals.add(tile["group_id"])
                original_splits[tile["group_id"]].add(split)
        multi[split] = {"groups": len(records), "image_occurrences": 4 * len(records), "unique_originals": len(originals)}
    overlap = sum(len(v) > 1 for v in original_splits.values())
    if overlap:
        errors.append(f"四图原图跨集合：{overlap}")
    return {
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_rows": len(rows), "source_unique_images": len(hashes),
        "source_split_rows": dict(Counter(x["split"] for x in rows)),
        "source_group_overlap": len(leakage), "source_pixel_hash_overlap": hash_overlap,
        "multi_photo": multi, "multi_original_overlap": overlap,
        "multi_sample_id_reused_across_splits": sum(len(v) > 1 for v in sample_splits.values()),
        "notes": ["旧四图 sample_id 在不同 split 文件夹复用；以 (split, sample_id) 为主键。", "像素散列与分组校验不能证明不同来源、人或未知预训练集之间没有重叠。"],
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/public-pilot/manifest.jsonl"))
    parser.add_argument("--multi-root", type=Path, default=Path("outputs/multi-photo-data-v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/nsfw-comparison-v1/audit.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/public-pilot-multihead"))
    args = parser.parse_args()
    result = audit(args.manifest, args.multi_root)
    result["checkpoint_sha256"] = {
        path.relative_to(args.checkpoint).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.checkpoint.rglob("*")) if path.suffix in (".pt", ".safetensors")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
