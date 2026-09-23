"""把人工复核结果转换为偏好学习数据。"""

from __future__ import annotations

from typing import Any

from .heads import DECISION_ALIASES


def normalize_action(value: Any) -> str:
    """归一化 Block / Review / Allow 动作标签。"""
    normalized = str(value or "").strip()
    action = DECISION_ALIASES.get(normalized)
    if action is None:
        raise ValueError("动作必须是 block、review、allow 或对应中文标签")
    return action


def preference_group_id(row: dict[str, Any]) -> str:
    """保留原图组；旧记录缺少 group_id 时仅按图片路径分组。"""
    value = row.get("group_id")
    if value is not None:
        group = str(value).strip()
        if not group:
            raise ValueError("group_id 不能为空")
        return group
    image = str(row.get("image", "")).strip()
    if not image:
        raise ValueError("缺少 group_id 时，image 必须是非空图片路径")
    return image


def build_preference_pair(row: dict[str, Any]) -> dict[str, Any] | None:
    """将一条人工复核记录转换为离散 DPO 的动作偏好对。"""
    if not isinstance(row, dict):
        raise ValueError("反馈记录必须是 JSON 对象")
    required = {"sample_id", "image", "policy_id", "policy_text", "human_action", "model_action"}
    missing = sorted(key for key in required if key not in row)
    if missing:
        raise ValueError(f"反馈记录缺少字段：{missing}")
    chosen_action = normalize_action(row["human_action"])
    rejected_action = normalize_action(row["model_action"])
    if chosen_action == rejected_action:
        return None
    policy_text = str(row["policy_text"]).strip()
    if not policy_text:
        raise ValueError("policy_text 不能为空")
    text = str(row.get("text", ""))
    return {
        "sample_id": str(row["sample_id"]),
        "group_id": preference_group_id(row),
        "image": str(row["image"]),
        "policy_id": str(row["policy_id"]),
        "policy_text": policy_text,
        "text": text,
        "prompt": (
            "你是内容审核模型。根据图片、正文和审核规则，从 block、review、allow 中选择一个动作。\n"
            f"审核规则：{policy_text}\n内容正文：{text}\n动作："
        ),
        "chosen": chosen_action,
        "rejected": rejected_action,
        "reward": float(row.get("reward", 1.0)),
        "reviewer": str(row.get("reviewer", "")),
        "feedback_id": str(row.get("feedback_id", row["sample_id"])),
    }
