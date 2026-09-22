from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成带合并 LoRA 的完整多模态生成检查点")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空：{args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    # 训练时 LoRA 只挂在语言骨干，因此在相同层级加载后合并。
    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model,
        args.checkpoint / "adapter",
        is_trainable=False,
    ).merge_and_unload(safe_merge=True)
    model.tie_weights()
    model.save_pretrained(args.output, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(args.checkpoint / "processor")
    processor.save_pretrained(args.output)
    for name in ("decision_head.pt", "calibration.json", "metadata.json"):
        shutil.copy2(args.checkpoint / name, args.output / name)

    export_info = {
        "format": "transformers-generation-merged-lora",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "base_model": str(args.base_model.resolve()),
        "requires_retraining": False,
        "warning": "生成模型包含合并 LoRA；独立审核头仍需目标运行时接入。",
    }
    (args.output / "merge_export.json").write_text(
        json.dumps(export_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(export_info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
