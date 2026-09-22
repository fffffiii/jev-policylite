from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from .data import MultimodalCollator, configure_max_pixels
from .model import ModerationModel


def load_checkpoint(
    checkpoint: str | Path, device: torch.device
) -> tuple[ModerationModel, Any, dict[str, Any], MultimodalCollator]:
    checkpoint_path = Path(checkpoint)
    metadata = json.loads((checkpoint_path / "metadata.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    model_config = config["model"]
    data_config = config["data"]
    lora_config = model_config["lora"]
    dtype = torch.bfloat16 if model_config["dtype"] == "bfloat16" else torch.float16

    processor_path = checkpoint_path / "processor"
    processor = AutoProcessor.from_pretrained(processor_path)
    processor.tokenizer.padding_side = "right"
    configure_max_pixels(processor, int(data_config["max_pixels"]))

    model = ModerationModel.create(
        model_config["name_or_path"],
        dtype=dtype,
        lora_rank=int(lora_config["rank"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        target_modules=lora_config["target_modules"],
        gradient_checkpointing=False,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        adapter_path=checkpoint_path / "adapter",
    )
    model.load_head(checkpoint_path)
    model.to(device).eval()
    collator = MultimodalCollator(
        processor=processor,
        image_root=Path(data_config["image_root"]),
        max_length=int(data_config["max_length"]),
    )
    return model, processor, metadata, collator
