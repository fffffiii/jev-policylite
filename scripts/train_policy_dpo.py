"""冻结 Qwen 与其他审核头，只对离散策略头做 DPO。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset

from qwen35_moderation.feedback import normalize_action
from qwen35_moderation.heads import DECISION_IDS
from qwen35_moderation.model import model_inputs
from qwen35_moderation.preference import discrete_dpo_loss
from qwen35_moderation.runtime import load_checkpoint
from qwen35_moderation.utils import set_seed


ACTION_INDEX = {action: index for index, action in enumerate(DECISION_IDS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 Block / Review / Allow 策略头执行离散 DPO")
    parser.add_argument("--checkpoint", required=True, help="包含 policy_head.pt 的起始检查点")
    parser.add_argument("--preferences", required=True, help="偏好对 JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root", help="覆盖检查点中的图片根目录")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="抽取视觉特征时的批量")
    parser.add_argument("--head-batch-size", type=int, default=64, help="训练小策略头时的批量")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_preferences(path: str | Path) -> list[dict[str, Any]]:
    required = {"sample_id", "image", "policy_id", "policy_text", "chosen", "rejected"}
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                missing = sorted(required - row.keys())
                if missing:
                    raise ValueError(f"缺少字段：{missing}")
                chosen = normalize_action(row["chosen"])
                rejected = normalize_action(row["rejected"])
                if chosen == rejected:
                    raise ValueError("chosen 与 rejected 不能相同")
                reward = abs(float(row.get("reward", 1.0)))
                if not math.isfinite(reward) or reward <= 0:
                    raise ValueError("reward 的绝对值必须是有限正数")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"偏好文件第 {line_number} 行无效：{exc}") from exc
            normalized = dict(row)
            normalized.update(
                chosen=chosen,
                rejected=rejected,
                reward=reward,
                text=str(row.get("text", "")),
                group_id=str(row.get("group_id", row.get("feedback_id", row["sample_id"]))),
            )
            rows.append(normalized)
    if len(rows) < 2:
        raise ValueError("偏好数据至少需要两条记录，才能拆分训练集和验证集")
    return rows


def split_preferences(
    rows: list[dict[str, Any]], validation_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation-ratio 必须在 0 与 1 之间")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["group_id"]), []).append(row)
    if len(groups) < 2:
        raise ValueError("偏好数据至少需要两个 group_id，才能避免训练/验证泄漏")
    ordered = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest(),
    )
    validation_group_count = min(
        len(ordered) - 1, max(1, round(len(ordered) * validation_ratio))
    )
    validation_groups = set(ordered[:validation_group_count])
    train = [row for row in rows if row["group_id"] not in validation_groups]
    validation = [row for row in rows if row["group_id"] in validation_groups]
    return train, validation


class PreferenceDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class PreferenceCollator:
    def __init__(self, base_collator: Any):
        self.base_collator = base_collator

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        model_rows = [
            {
                **row,
                "label": 0,
                "group_id": row["group_id"],
            }
            for row in rows
        ]
        batch = self.base_collator(model_rows)
        batch["chosen"] = torch.tensor([ACTION_INDEX[row["chosen"]] for row in rows])
        batch["rejected"] = torch.tensor([ACTION_INDEX[row["rejected"]] for row in rows])
        batch["weights"] = torch.tensor([row["reward"] for row in rows], dtype=torch.float32)
        return batch


@torch.no_grad()
def extract_features(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[TensorDataset, float]:
    features: list[torch.Tensor] = []
    chosen: list[torch.Tensor] = []
    rejected: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    started = time.monotonic()
    model.eval()
    for batch in loader:
        inputs = {key: value.to(device) for key, value in model_inputs(batch).items()}
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with context:
            pooled = model.encode(**inputs)
        features.append(pooled.detach().float().cpu())
        chosen.append(batch["chosen"])
        rejected.append(batch["rejected"])
        weights.append(batch["weights"])
    dataset = TensorDataset(
        torch.cat(features), torch.cat(chosen), torch.cat(rejected), torch.cat(weights)
    )
    return dataset, time.monotonic() - started


@torch.no_grad()
def evaluate_head(
    policy_head: torch.nn.Module,
    reference_head: torch.nn.Module,
    dataset: TensorDataset,
    device: torch.device,
    beta: float,
    batch_size: int,
) -> dict[str, float | int]:
    policy_head.eval()
    reference_head.eval()
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    margins: list[float] = []
    reference_margins: list[float] = []
    advantages: list[float] = []
    kls: list[float] = []
    for features, chosen, rejected, weights in DataLoader(dataset, batch_size=batch_size):
        features = features.to(device)
        chosen = chosen.to(device)
        rejected = rejected.to(device)
        weights = weights.to(device)
        policy_logits = policy_head(features)
        reference_logits = reference_head(features)
        result = discrete_dpo_loss(
            policy_logits, reference_logits, chosen, rejected, beta=beta, weights=weights
        )
        policy_log_probs = F.log_softmax(policy_logits.float(), dim=-1)
        reference_log_probs = F.log_softmax(reference_logits.float(), dim=-1)
        policy_probs = policy_log_probs.exp()
        batch_kl = (policy_probs * (policy_log_probs - reference_log_probs)).sum(dim=-1)
        per_sample_loss = -F.logsigmoid(beta * result.advantages)
        weighted_loss_sum += float((per_sample_loss * weights).sum().item())
        weight_sum += float(weights.sum().item())
        margins.extend(result.policy_margins.cpu().tolist())
        reference_margins.extend(result.reference_margins.cpu().tolist())
        advantages.extend(result.advantages.cpu().tolist())
        kls.extend(batch_kl.cpu().tolist())
    count = len(margins)
    return {
        "samples": count,
        "dpo_loss": weighted_loss_sum / weight_sum,
        "preference_accuracy": float(sum(value > 0 for value in margins) / count),
        "mean_policy_margin": float(sum(margins) / count),
        "mean_reference_margin": float(sum(reference_margins) / count),
        "mean_advantage": float(sum(advantages) / count),
        "mean_kl": float(sum(kls) / count),
    }


def save_checkpoint(
    source: Path,
    output: Path,
    model: torch.nn.Module,
    metadata: dict[str, Any],
    report: dict[str, Any],
) -> None:
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    output.mkdir(parents=True)
    for directory in ("adapter", "processor"):
        source_directory = source / directory
        if not source_directory.is_dir():
            raise FileNotFoundError(f"起始检查点缺少目录：{source_directory}")
        shutil.copytree(source_directory, output / directory)
    for filename in ("decision_head.pt", "attribute_head.pt", "calibration.json"):
        source_file = source / filename
        if source_file.exists():
            shutil.copy2(source_file, output / filename)
    torch.save(model.policy_head.state_dict(), output / "policy_head.pt")
    updated_metadata = copy.deepcopy(metadata)
    updated_metadata["preference_optimization"] = report
    heads = updated_metadata.setdefault("heads", {})
    files = heads.setdefault("files", {})
    files["policy"] = "policy_head.pt"
    (output / "metadata.json").write_text(
        json.dumps(updated_metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "dpo_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.head_batch_size <= 0:
        raise ValueError("epochs 与 batch size 必须大于 0")
    if args.learning_rate <= 0 or args.beta <= 0:
        raise ValueError("learning-rate 与 beta 必须大于 0")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    set_seed(args.seed)
    random.seed(args.seed)
    rows = read_preferences(args.preferences)
    train_rows, validation_rows = split_preferences(rows, args.validation_ratio, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, _, metadata, base_collator = load_checkpoint(args.checkpoint, device)
    if not model.policy_head_ready:
        raise ValueError("起始检查点没有已训练的 policy_head.pt，不能建立 DPO 参考策略")
    if args.image_root:
        base_collator.image_root = Path(args.image_root)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.policy_head.parameters():
        parameter.requires_grad = True
    model.eval()
    model.policy_head.train()
    reference_head = copy.deepcopy(model.policy_head).to(device).eval()
    for parameter in reference_head.parameters():
        parameter.requires_grad = False

    collator = PreferenceCollator(base_collator)
    train_source = DataLoader(
        PreferenceDataset(train_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )
    validation_source = DataLoader(
        PreferenceDataset(validation_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )
    train_features, train_extract_seconds = extract_features(model, train_source, device)
    validation_features, validation_extract_seconds = extract_features(model, validation_source, device)

    baseline = {
        "train": evaluate_head(
            model.policy_head, reference_head, train_features, device, args.beta, args.head_batch_size
        ),
        "validation": evaluate_head(
            model.policy_head,
            reference_head,
            validation_features,
            device,
            args.beta,
            args.head_batch_size,
        ),
    }
    optimizer = AdamW(
        model.policy_head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_features, batch_size=args.head_batch_size, shuffle=True, generator=generator
    )
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.policy_head.train()
        epoch_losses: list[float] = []
        for features, chosen, rejected, weights in train_loader:
            features = features.to(device)
            chosen = chosen.to(device)
            rejected = rejected.to(device)
            weights = weights.to(device)
            policy_logits = model.policy_head(features)
            with torch.no_grad():
                reference_logits = reference_head(features)
            result = discrete_dpo_loss(
                policy_logits, reference_logits, chosen, rejected, beta=args.beta, weights=weights
            )
            if not torch.isfinite(result.loss):
                raise FloatingPointError(f"第 {epoch} 轮出现非有限 DPO loss")
            optimizer.zero_grad(set_to_none=True)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy_head.parameters(), args.max_grad_norm)
            optimizer.step()
            epoch_losses.append(result.loss.detach().item())
        metrics = evaluate_head(
            model.policy_head,
            reference_head,
            validation_features,
            device,
            args.beta,
            args.head_batch_size,
        )
        row = {"epoch": epoch, "mean_train_batch_loss": sum(epoch_losses) / len(epoch_losses), **metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    final = {
        "train": evaluate_head(
            model.policy_head, reference_head, train_features, device, args.beta, args.head_batch_size
        ),
        "validation": evaluate_head(
            model.policy_head,
            reference_head,
            validation_features,
            device,
            args.beta,
            args.head_batch_size,
        ),
    }
    elapsed = time.monotonic() - started
    report = {
        "algorithm": "discrete_dpo_policy_head_only",
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "preferences": str(Path(args.preferences).resolve()),
        "actions": list(DECISION_IDS),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "head_batch_size": args.head_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "beta": args.beta,
            "validation_ratio": args.validation_ratio,
            "max_grad_norm": args.max_grad_norm,
            "seed": args.seed,
        },
        "split": {"train": len(train_rows), "validation": len(validation_rows)},
        "baseline": baseline,
        "history": history,
        "final": final,
        "performance": {
            "feature_extraction_seconds": train_extract_seconds + validation_extract_seconds,
            "head_training_seconds": elapsed,
            "peak_cuda_memory_gb": (
                torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            ),
        },
        "scope": "仅更新 policy_head；视觉骨干、LoRA、二元头和属性头保持不变。",
    }
    save_checkpoint(Path(args.checkpoint), output, model, metadata, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
