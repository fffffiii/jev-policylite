from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把人工标注表转换为训练 JSONL")
    parser.add_argument("--annotations", required=True, help="UTF-8 CSV 标注表")
    parser.add_argument("--strict-policy", default="policies/strict.txt")
    parser.add_argument("--contextual-policy", default="policies/contextual.txt")
    parser.add_argument("--output", default="data/manifest.jsonl")
    parser.add_argument("--seed", default="jev-policylite-v1")
    return parser.parse_args()


def stable_split(group_id: str, seed: str) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 60:
        return "train"
    if bucket < 70:
        return "validation"
    if bucket < 80:
        return "calibration"
    return "test"


def parse_label(value: str, row_number: int, column: str) -> int:
    normalized = value.strip()
    if normalized not in {"0", "1"}:
        raise ValueError(f"第 {row_number} 行 {column} 必须填写 0 或 1")
    return int(normalized)


def main() -> None:
    args = parse_args()
    policies = {
        "strict-v1": Path(args.strict_policy).read_text(encoding="utf-8").strip(),
        "contextual-v1": Path(args.contextual_policy).read_text(encoding="utf-8").strip(),
    }
    required = {
        "source_id",
        "image",
        "text",
        "strict_label",
        "contextual_label",
        "source",
        "annotation_source",
    }
    output_rows: list[dict[str, object]] = []
    with Path(args.annotations).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"标注表缺少列：{sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            group_id = row["source_id"].strip()
            if not group_id:
                raise ValueError(f"第 {row_number} 行 source_id 为空")
            split = stable_split(group_id, args.seed)
            attributes = [item.strip() for item in row.get("attributes", "").split("|") if item.strip()]
            for policy_id, label_column in (
                ("strict-v1", "strict_label"),
                ("contextual-v1", "contextual_label"),
            ):
                output_rows.append(
                    {
                        "sample_id": f"{group_id}:{policy_id}",
                        "group_id": group_id,
                        "image": row["image"].strip(),
                        "text": row["text"].strip(),
                        "policy_id": policy_id,
                        "policy_text": policies[policy_id],
                        "label": parse_label(row[label_column], row_number, label_column),
                        "split": split,
                        "attributes": attributes,
                        "source": row["source"].strip(),
                        "annotation_source": row["annotation_source"].strip(),
                        "notes": row.get("notes", "").strip(),
                    }
                )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已写入 {len(output_rows)} 条政策样本：{output_path}")


if __name__ == "__main__":
    main()
