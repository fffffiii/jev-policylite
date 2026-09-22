from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

from qwen35_moderation.data import (
    ModerationDataset,
    MultimodalCollator,
    configure_max_pixels,
    find_group_leakage,
    read_manifest,
)
from qwen35_moderation.heads import encode_attribute_batch, encode_decision_batch, masked_multilabel_bce, masked_softmax_ce
from qwen35_moderation.metrics import classification_metrics, sigmoid
from qwen35_moderation.model import ModerationModel, model_inputs
from qwen35_moderation.utils import load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Qwen3.5-0.8B 图文审核决策模型")
    parser.add_argument("--config", default="configs/train.yaml")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, accelerator: Accelerator) -> dict[str, Any]:
    model.eval()
    gathered_logits: list[torch.Tensor] = []
    gathered_labels: list[torch.Tensor] = []
    for batch in loader:
        logits = model(**model_inputs(batch))
        labels = batch["labels"]
        logits, labels = accelerator.gather_for_metrics((logits, labels))
        gathered_logits.append(logits.detach().float().cpu())
        gathered_labels.append(labels.detach().float().cpu())
    logits_np = torch.cat(gathered_logits).numpy()
    labels_np = torch.cat(gathered_labels).numpy()
    return classification_metrics(labels_np, sigmoid(logits_np), threshold=0.5)


def save_best(
    accelerator: Accelerator,
    model: torch.nn.Module,
    processor: Any,
    output_dir: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        metadata = {
            "base_model": config["model"]["name_or_path"],
            "config": config,
            "validation_metrics": metrics,
        }
        unwrapped.save_artifacts(output_dir, metadata)
        processor.save_pretrained(output_dir / "processor")
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    training = config["training"]
    data_config = config["data"]
    model_config = config["model"]
    set_seed(int(training["seed"]))

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
    )
    rows = read_manifest(data_config["manifest"])
    leakage = find_group_leakage(rows)
    if leakage:
        raise ValueError(f"发现 group_id 跨数据划分，示例：{list(leakage.items())[:5]}")
    for split in ("train", "validation"):
        if {row['label'] for row in rows if row['split'] == split} != {0, 1}:
            raise ValueError(f"{split} 必须同时包含正常与违规标签")

    processor = AutoProcessor.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    processor.tokenizer.padding_side = "right"
    configure_max_pixels(processor, int(data_config["max_pixels"]))

    collator = MultimodalCollator(
        processor=processor,
        image_root=Path(data_config["image_root"]),
        max_length=int(data_config["max_length"]),
    )
    train_dataset = ModerationDataset(rows, "train", data_config["image_root"])
    validation_dataset = ModerationDataset(rows, "validation", data_config["image_root"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["per_device_batch_size"]),
        shuffle=True,
        num_workers=int(data_config["num_workers"]),
        pin_memory=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training["per_device_batch_size"]),
        shuffle=False,
        num_workers=int(data_config["num_workers"]),
        pin_memory=True,
        collate_fn=collator,
    )

    dtype = torch.bfloat16 if model_config["dtype"] == "bfloat16" else torch.float16
    lora_config = model_config["lora"]
    model = ModerationModel.create(
        model_config["name_or_path"],
        dtype=dtype,
        lora_rank=int(lora_config["rank"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        target_modules=lora_config["target_modules"],
        gradient_checkpointing=bool(model_config["gradient_checkpointing"]),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    multi_head = bool(model_config.get("multi_head", False))
    decision_from_label = bool(model_config.get("decision_from_label", False))
    if multi_head:
        model.enable_multi_heads()
    summary = model.trainable_parameter_summary()
    accelerator.print(f"参数量：{json.dumps(summary, ensure_ascii=False)}")

    lora_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    ]
    head_parameters = list(model.head.parameters())
    if multi_head:
        head_parameters.extend(model.attribute_head.parameters())
        head_parameters.extend(model.policy_head.parameters())
    optimizer = AdamW(
        [
            {"params": lora_parameters, "lr": float(training["lora_learning_rate"])},
            {"params": head_parameters, "lr": float(training["head_learning_rate"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    update_steps_per_epoch = math.ceil(
        len(train_loader) / int(training["gradient_accumulation_steps"])
    )
    total_epochs = int(training["head_warmup_epochs"]) + int(training["epochs"])
    total_steps = update_steps_per_epoch * total_epochs
    warmup_steps = int(total_steps * float(training["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, scheduler
    )

    output_dir = Path(training["output_dir"])
    best_pr_auc = -1.0
    stale_evaluations = 0
    global_step = 0
    stop_training = False
    optimizer.zero_grad(set_to_none=True)

    started = time.monotonic()
    for epoch in range(total_epochs):
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.set_lora_trainable(epoch >= int(training["head_warmup_epochs"]))
        accelerator.print(f"epoch_start={epoch + 1}/{total_epochs} phase={'joint' if epoch >= int(training['head_warmup_epochs']) else 'head'}")
        model.train()
        for batch_index, batch in enumerate(train_loader, start=1):
            with accelerator.accumulate(model):
                if multi_head:
                    bundle = model(return_heads=True, **model_inputs(batch))
                    logits = bundle.violation_logit
                    loss = F.binary_cross_entropy_with_logits(logits.float(), batch["labels"].float())
                    attribute_targets, attribute_mask = encode_attribute_batch(batch["attributes"])
                    attribute_targets = attribute_targets.to(logits.device)
                    attribute_mask = attribute_mask.to(logits.device)
                    loss = loss + masked_multilabel_bce(bundle.attribute_logits, attribute_targets, attribute_mask)
                    decisions = list(batch["decisions"])
                    if decision_from_label:
                        decisions = [
                            item if item else ("block" if float(label) >= 0.5 else "allow")
                            for item, label in zip(decisions, batch["labels"].tolist())
                        ]
                    decision_targets, decision_mask = encode_decision_batch(decisions)
                    decision_targets = decision_targets.to(logits.device)
                    decision_mask = decision_mask.to(logits.device)
                    loss = loss + masked_softmax_ce(bundle.decision_logits, decision_targets, decision_mask)
                    trained = accelerator.unwrap_model(model)
                    if bool(attribute_mask.detach().sum().item()):
                        trained.attribute_head_supervised = True
                    if bool(decision_mask.detach().sum().item()):
                        trained.policy_head_supervised = True
                else:
                    logits = model(**model_inputs(batch))
                    loss = F.binary_cross_entropy_with_logits(logits.float(), batch["labels"].float())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"训练 loss 非有限值：epoch={epoch + 1} batch={batch_index}")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(training["max_grad_norm"]))
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % int(training["log_every"]) == 0:
                    accelerator.print(
                        f"epoch={epoch + 1} step={global_step} loss={loss.detach().float().item():.6f} elapsed_s={time.monotonic() - started:.1f} peak_gb={torch.cuda.max_memory_allocated() / 2**30:.2f}"
                    )
                if global_step % int(training["eval_every"]) == 0:
                    metrics = evaluate(model, validation_loader, accelerator)
                    accelerator.print(f"validation={json.dumps(metrics, ensure_ascii=False)}")
                    score = float(metrics["pr_auc"])
                    if np.isfinite(score) and score > best_pr_auc:
                        best_pr_auc = score
                        stale_evaluations = 0
                        save_best(accelerator, model, processor, output_dir, config, metrics)
                    else:
                        stale_evaluations += 1
                    model.train()
                    if epoch >= int(training["head_warmup_epochs"]) and stale_evaluations >= int(training["early_stopping_patience"]):
                        stop_training = True
                        break
        if stop_training:
            break
        metrics = evaluate(model, validation_loader, accelerator)
        accelerator.print(f"epoch_end_validation={json.dumps(metrics, ensure_ascii=False)}")
        score = float(metrics["pr_auc"])
        if np.isfinite(score) and score > best_pr_auc:
            best_pr_auc = score
            stale_evaluations = 0
            save_best(accelerator, model, processor, output_dir, config, metrics)
        model.train()

    if best_pr_auc < 0:
        metrics = evaluate(model, validation_loader, accelerator)
        save_best(accelerator, model, processor, output_dir, config, metrics)
    accelerator.print(f"训练完成，最佳 validation PR-AUC={best_pr_auc:.6f}")


if __name__ == "__main__":
    main()
