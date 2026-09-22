"""将人工复核 JSONL 转换为偏好优化数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen35_moderation.feedback import build_preference_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把人工审核反馈转换为 DPO/GRPO 偏好对")
    parser.add_argument("--feedback", required=True, help="人工复核 JSONL")
    parser.add_argument("--output", required=True, help="偏好对 JSONL")
    parser.add_argument("--strict", action="store_true", help="遇到无效记录时立即失败")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.feedback)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"输出文件已存在：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_same_action = 0
    invalid: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8") as reader, output.open("w", encoding="utf-8") as writer:
        for line_number, raw in enumerate(reader, start=1):
            if not raw.strip():
                continue
            try:
                pair = build_preference_pair(json.loads(raw))
            except (ValueError, json.JSONDecodeError) as exc:
                if args.strict:
                    raise ValueError(f"第 {line_number} 行无效：{exc}") from exc
                invalid.append({"line": line_number, "error": str(exc)})
                continue
            if pair is None:
                skipped_same_action += 1
                continue
            writer.write(json.dumps(pair, ensure_ascii=False) + "\n")
            written += 1
    summary = {
        "input": str(source),
        "output": str(output),
        "written": written,
        "skipped_same_action": skipped_same_action,
        "invalid": invalid,
        "note": "输出是多模态偏好数据；训练时必须在 prompt 前拼接对应 image，不要把图片路径当作文本特征。",
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
