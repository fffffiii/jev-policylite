"""同一份隐藏特征上的 Attribute Head 与 Decision Head。

Backbone 只负责把输入收成 h。两个 head 都很小，各自把 h 变成任务结果：

- Attribute Head：多标签 sigmoid，回答画面里有什么。
- Decision Head：三类 softmax，回答当前 policy 下 Block / Review / Allow。

已经训好的二元审核头继续单独存放，不把未训练的新头概率写成结果。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ATTRIBUTE_IDS = ("nudity", "sexual_act", "suggestive", "medical")
ATTRIBUTE_NAMES = {
    "nudity": "裸露",
    "sexual_act": "性行为",
    "suggestive": "擦边",
    "medical": "医学",
}
ATTRIBUTE_ALIASES = {
    "nudity": "nudity",
    "裸露": "nudity",
    "sexual_act": "sexual_act",
    "性行为": "sexual_act",
    "suggestive": "suggestive",
    "擦边": "suggestive",
    "medical": "medical",
    "医学": "medical",
}

DECISION_IDS = ("block", "review", "allow")
DECISION_NAMES = {"block": "Block", "review": "Review", "allow": "Allow"}
DECISION_ALIASES = {
    "block": "block",
    "review": "review",
    "allow": "allow",
    "阻断": "block",
    "复审": "review",
    "通过": "allow",
}
# 公开试训只有发布方的 L1–L4，没有单独的属性标注。这是等级到属性的代理，不是人工属性复核。
SOURCE_LEVEL_ATTRIBUTES = {
    "source_L1": (),
    "source_L2": ("suggestive",),
    "source_L3": ("nudity",),
    "source_L4": ("nudity", "sexual_act"),
}

BINARY_DECISION_NOTE = "当前权重是二元决策头。Block / Review / Allow 需要已训练的 policy_head.pt。"


def make_task_head(hidden_size: int, outputs: int) -> nn.Sequential:
    """与现有审核头相同的 LayerNorm + Linear，方便端侧用同一套读法。"""
    return nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, outputs))


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    scaled = np.clip(np.asarray(logits, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def softmax_np(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def encode_attribute_batch(attributes: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
    """把标注里的属性名变成多热标签。没有任何已知属性的样本不参与损失。"""
    targets = torch.zeros(len(attributes), len(ATTRIBUTE_IDS), dtype=torch.float32)
    mask = torch.zeros(len(attributes), dtype=torch.float32)
    index = {name: position for position, name in enumerate(ATTRIBUTE_IDS)}
    for row_index, tags in enumerate(attributes):
        recognized = False
        for tag in tags:
            normalized = str(tag).strip()
            if normalized in SOURCE_LEVEL_ATTRIBUTES:
                recognized = True
                for attribute_id in SOURCE_LEVEL_ATTRIBUTES[normalized]:
                    targets[row_index, index[attribute_id]] = 1.0
                continue
            attribute_id = ATTRIBUTE_ALIASES.get(normalized)
            if attribute_id is None:
                continue
            targets[row_index, index[attribute_id]] = 1.0
            recognized = True
        if recognized:
            mask[row_index] = 1.0
    return targets, mask


def encode_decision_batch(decisions: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """把 block / review / allow 变成类别下标。空标签不参与损失。"""
    targets = torch.zeros(len(decisions), dtype=torch.long)
    mask = torch.zeros(len(decisions), dtype=torch.float32)
    index = {name: position for position, name in enumerate(DECISION_IDS)}
    for row_index, decision in enumerate(decisions):
        if decision is None or str(decision).strip() == "":
            continue
        decision_id = DECISION_ALIASES.get(str(decision).strip())
        if decision_id is None:
            raise ValueError(f"无法识别的决策标签：{decision}")
        targets[row_index] = index[decision_id]
        mask[row_index] = 1.0
    return targets, mask


def masked_multilabel_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_label = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    weighted = per_label * mask.float().unsqueeze(-1)
    return weighted.sum() / mask.float().sum().clamp(min=1.0)


def masked_softmax_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe_targets = targets.long().clamp(min=0, max=logits.shape[-1] - 1)
    per_sample = F.cross_entropy(logits.float(), safe_targets, reduction="none")
    weighted = per_sample * mask.float()
    return weighted.sum() / mask.float().sum().clamp(min=1.0)


def _label_rows(ids: tuple[str, ...], names: dict[str, str], probabilities: np.ndarray | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, label_id in enumerate(ids):
        row: dict[str, Any] = {"id": label_id, "name": names[label_id]}
        if probabilities is not None:
            row["probability"] = float(probabilities[position])
        rows.append(row)
    return rows


def bundle_probabilities(bundle: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    """把 HeadBundle 里的 logit 变成概率。未加载的头返回 None。"""
    attribute = None
    decision = None
    if getattr(bundle, "attribute_logits", None) is not None:
        attribute = torch.sigmoid(bundle.attribute_logits.detach().float()).cpu().numpy()
    if getattr(bundle, "decision_logits", None) is not None:
        decision = torch.softmax(bundle.decision_logits.detach().float(), dim=-1).cpu().numpy()
    return attribute, decision


def format_heads(
    *,
    violation_probability: float,
    threshold: float,
    attribute_probabilities: np.ndarray | None = None,
    decision_probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    """把两只 head 收成稳定的 JSON 结构。没有权重时不填概率。"""
    attribute_ready = attribute_probabilities is not None
    attribute = {
        "id": "attribute",
        "title": "Attribute Head",
        "question": "画面里有什么",
        "ready": attribute_ready,
        "kind": "multilabel_sigmoid",
        "labels": _label_rows(ATTRIBUTE_IDS, ATTRIBUTE_NAMES, attribute_probabilities),
    }
    if not attribute_ready:
        attribute["note"] = "检查点里还没有 attribute_head.pt，属性概率要等这只头训练后再输出。"
    if decision_probabilities is not None:
        action = DECISION_IDS[int(np.argmax(decision_probabilities))]
        decision = {
            "id": "decision",
            "title": "Decision Head",
            "question": "按照当前 policy 应该怎么处理",
            "ready": True,
            "kind": "softmax",
            "action": action,
            "labels": _label_rows(DECISION_IDS, DECISION_NAMES, decision_probabilities),
        }
    else:
        action = "block" if violation_probability >= threshold else "allow"
        decision = {
            "id": "decision",
            "title": "Decision Head",
            "question": "按照当前 policy 应该怎么处理",
            "ready": True,
            "kind": "binary",
            "action": action,
            "note": BINARY_DECISION_NOTE,
            "labels": [{"id": "violation", "name": "违规", "probability": float(violation_probability)}],
        }
    return {"attribute": attribute, "decision": decision}


def sequential_head_arrays(state: dict[str, torch.Tensor]) -> tuple[np.ndarray, int, int]:
    """把 LayerNorm + Linear 打成端侧使用的小端 float32 数组。"""
    norm_weight = state["0.weight"].detach().float().cpu().numpy()
    norm_bias = state["0.bias"].detach().float().cpu().numpy()
    weight = state["1.weight"].detach().float().cpu().numpy()
    bias = state["1.bias"].detach().float().cpu().numpy()
    outputs, hidden_size = weight.shape
    if norm_weight.shape != (hidden_size,) or norm_bias.shape != (hidden_size,) or bias.shape != (outputs,):
        raise ValueError("head 的 LayerNorm 与 Linear 维度不一致")
    flat = np.concatenate([norm_weight, norm_bias, weight.reshape(-1), bias]).astype("<f4", copy=False)
    return flat, hidden_size, outputs


def apply_sequential_head(hidden: np.ndarray, flat: np.ndarray, outputs: int) -> np.ndarray:
    """与端侧 C++ 相同的 LayerNorm + Linear。hidden 是最后一层向量。"""
    values = np.asarray(hidden, dtype=np.float64)
    hidden_size = int(values.shape[-1])
    expected = hidden_size * (2 + outputs) + outputs
    weights = np.asarray(flat, dtype=np.float64)
    if weights.size != expected:
        raise ValueError(f"head 长度应为 {expected}，实际 {weights.size}")
    norm_weight = weights[:hidden_size]
    norm_bias = weights[hidden_size:hidden_size * 2]
    linear_weight = weights[hidden_size * 2:hidden_size * 2 + outputs * hidden_size].reshape(outputs, hidden_size)
    linear_bias = weights[-outputs:]
    mean = values.mean()
    variance = np.mean((values - mean) ** 2)
    normalized = (values - mean) / np.sqrt(variance + 1e-5)
    normalized = normalized * norm_weight + norm_bias
    return normalized @ linear_weight.T + linear_bias
