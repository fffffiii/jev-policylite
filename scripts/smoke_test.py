from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoProcessor

from qwen35_moderation.data import MultimodalCollator, configure_max_pixels, read_manifest
from qwen35_moderation.model import ModerationModel, model_inputs
from qwen35_moderation.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单批次前向、反向和显存 smoke test")
    parser.add_argument("--config", default="configs/train.yaml")
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("smoke test 需要 CUDA GPU")
    args = parse_args()
    config = load_yaml(args.config)
    model_config = config["model"]
    data_config = config["data"]
    lora_config = model_config["lora"]
    rows = read_manifest(data_config["manifest"])
    row = next((item for item in rows if item["split"] == "train"), None)
    if row is None:
        raise ValueError("清单没有 train 样本")

    processor = AutoProcessor.from_pretrained(model_config["name_or_path"])
    processor.tokenizer.padding_side = "right"
    configure_max_pixels(processor, int(data_config["max_pixels"]))
    collator = MultimodalCollator(
        processor,
        Path(data_config["image_root"]),
        int(data_config["max_length"]),
    )
    batch = collator([row])
    model = ModerationModel.create(
        model_config["name_or_path"],
        dtype=torch.bfloat16,
        lora_rank=int(lora_config["rank"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        target_modules=lora_config["target_modules"],
        gradient_checkpointing=True,
    ).cuda()
    inputs = {key: value.cuda() for key, value in model_inputs(batch).items()}
    labels = batch["labels"].cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(**inputs)
        loss = F.binary_cross_entropy_with_logits(logits.float(), labels)
    loss.backward()
    report = {
        "loss": float(loss.detach().cpu()),
        "input_tokens": int(batch["attention_mask"].sum().item()),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "trainable_parameters": model.trainable_parameter_summary()["trainable"],
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
