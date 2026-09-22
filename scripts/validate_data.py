from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from qwen35_moderation.data import find_group_leakage, read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验审核数据清单")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--skip-image-decode", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_manifest(args.manifest)
    image_root = Path(args.image_root)
    errors: list[str] = []

    sample_counts = Counter(str(row["sample_id"]) for row in rows)
    duplicate_samples = [sample_id for sample_id, count in sample_counts.items() if count > 1]
    if duplicate_samples:
        errors.append(f"sample_id 重复：{duplicate_samples[:10]}")

    leakage = find_group_leakage(rows)
    if leakage:
        preview = list(leakage.items())[:10]
        errors.append(f"group_id 跨数据划分：{preview}")

    labels_by_group_policy: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for row in rows:
        labels_by_group_policy[(str(row["group_id"]), str(row["policy_id"]), str(row["text"]))].add(int(row["label"]))
    conflicts = [key for key, labels in labels_by_group_policy.items() if len(labels) > 1]
    if conflicts:
        errors.append(f"同一原图与政策存在冲突标签：{conflicts[:10]}")

    decoded_paths: set[Path] = set()
    hashes: dict[str, list[Path]] = defaultdict(list)
    path_splits: dict[Path, set[str]] = defaultdict(set)
    for row in rows:
        path_splits[(image_root / row["image"]).resolve()].add(row["split"])
    for path, splits in path_splits.items():
        if len(splits) > 1:
            errors.append(f"同一图片路径跨数据划分：{path}")
    for row in rows:
        image_path = (image_root / row["image"]).resolve()
        if image_path in decoded_paths:
            continue
        decoded_paths.add(image_path)
        if not image_path.is_file():
            errors.append(f"图片不存在：{image_path}")
            continue
        if not args.skip_image_decode:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                errors.append(f"图片损坏：{image_path}（{exc}）")
                continue
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        hashes[digest].append(image_path)

    exact_duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    for paths in exact_duplicates:
        if len(set().union(*(path_splits[path] for path in paths))) > 1:
            errors.append(f"完全相同图片跨数据划分：{paths[:3]}")
    split_counts = Counter(str(row["split"]) for row in rows)
    label_counts = Counter((str(row["split"]), int(row["label"])) for row in rows)
    policy_counts = Counter(str(row["policy_id"]) for row in rows)

    print(f"样本数：{len(rows)}；原图组：{len(set(str(row['group_id']) for row in rows))}")
    print(f"划分：{dict(split_counts)}")
    print(f"划分/标签：{dict(label_counts)}")
    print(f"政策：{dict(policy_counts)}")
    print(f"完全重复图片组：{len(exact_duplicates)}（需要人工确认 group_id 是否一致）")
    if errors:
        print("校验失败：", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("清单校验通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
