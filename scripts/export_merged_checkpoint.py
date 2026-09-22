from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from qwen35_moderation.model import model_inputs
from qwen35_moderation.runtime import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并审核模型 LoRA，生成可继续量化的完整检查点")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path, help="用于核对合并前后输出的 JSON 文件")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空：{args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, processor, metadata, collator = load_checkpoint(args.checkpoint, device)

    before: float | None = None
    batch = None
    if args.sample:
        sample = json.loads(args.sample.read_text(encoding="utf-8"))
        batch = collator([sample])
        inputs = {key: value.to(device) for key, value in model_inputs(batch).items()}
        with torch.inference_mode():
            before = float(model(**inputs).item())

    # 将 LoRA 增量写回语言模型；视觉塔与审核头保持原样。
    model.backbone.language_model = model.backbone.language_model.merge_and_unload(
        safe_merge=True
    )

    after: float | None = None
    if batch is not None:
        inputs = {key: value.to(device) for key, value in model_inputs(batch).items()}
        with torch.inference_mode():
            after = float(model(**inputs).item())

    model.backbone.save_pretrained(args.output / "backbone", safe_serialization=True)
    processor.save_pretrained(args.output / "processor")
    torch.save(model.head.state_dict(), args.output / "decision_head.pt")
    shutil.copy2(args.checkpoint / "calibration.json", args.output / "calibration.json")

    export_metadata = {
        **metadata,
        "export": {
            "format": "transformers-merged-lora",
            "source_checkpoint": str(args.checkpoint.resolve()),
            "requires_retraining": False,
            "decision_head": "decision_head.pt",
            "verification_logit_before": before,
            "verification_logit_after": after,
            "verification_abs_diff": None if before is None else abs(before - after),
        },
    }
    (args.output / "metadata.json").write_text(
        json.dumps(export_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(export_metadata["export"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
