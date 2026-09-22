"""用二元清单构造仅供流程验证的 Block/Allow 偏好对。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from qwen35_moderation.data import read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从二元清单构造 DPO 冒烟测试数据")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="train", choices=["train", "validation", "calibration", "test"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, default=0, help="0 表示使用该划分全部记录")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"输出文件已存在：{output}")
    rows = [row for row in read_manifest(args.manifest) if row["split"] == args.split]
    if not rows:
        raise ValueError(f"清单中没有 {args.split!r} 数据")
    random.Random(args.seed).shuffle(rows)
    if args.max_records > 0:
        rows = rows[: args.max_records]
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"block": 0, "allow": 0}
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            chosen = "block" if int(row["label"]) == 1 else "allow"
            rejected = "allow" if chosen == "block" else "block"
            counts[chosen] += 1
            pair = {
                "sample_id": str(row["sample_id"]),
                "group_id": str(row.get("group_id", row["sample_id"])),
                "image": str(row["image"]),
                "policy_id": str(row["policy_id"]),
                "policy_text": str(row["policy_text"]),
                "text": str(row.get("text", "")),
                "chosen": chosen,
                "rejected": rejected,
                "reward": 1.0,
                "feedback_id": f"bootstrap:{row['sample_id']}",
                "provenance": "binary_bootstrap",
            }
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    summary = {
        "input": str(args.manifest),
        "split": args.split,
        "output": str(output),
        "records": len(rows),
        "chosen_counts": counts,
        "provenance": "binary_bootstrap",
        "warning": "这些偏好由原二元标签自动生成，只能验证训练链路；它不包含 review，也不构成新的人工监督。",
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
