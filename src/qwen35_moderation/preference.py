"""离散审核动作的偏好优化损失。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class PreferenceLoss:
    """DPO 损失及便于记录的逐样本量。"""

    loss: torch.Tensor
    advantages: torch.Tensor
    policy_margins: torch.Tensor
    reference_margins: torch.Tensor


def discrete_dpo_loss(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    *,
    beta: float,
    weights: torch.Tensor | None = None,
) -> PreferenceLoss:
    """计算 Block / Review / Allow 离散策略的 DPO 损失。"""
    if policy_logits.ndim != 2 or reference_logits.shape != policy_logits.shape:
        raise ValueError("policy_logits 与 reference_logits 必须是形状相同的二维张量")
    batch_size, action_count = policy_logits.shape
    if chosen.shape != (batch_size,) or rejected.shape != (batch_size,):
        raise ValueError("chosen 与 rejected 必须是一维批次索引")
    if beta <= 0:
        raise ValueError("beta 必须大于 0")
    chosen = chosen.long()
    rejected = rejected.long()
    if torch.any(chosen == rejected):
        raise ValueError("chosen 与 rejected 不能相同")
    if torch.any(chosen < 0) or torch.any(chosen >= action_count):
        raise ValueError("chosen 超出动作类别范围")
    if torch.any(rejected < 0) or torch.any(rejected >= action_count):
        raise ValueError("rejected 超出动作类别范围")

    policy_log_probs = F.log_softmax(policy_logits.float(), dim=-1)
    reference_log_probs = F.log_softmax(reference_logits.float(), dim=-1)
    gather_chosen = chosen.unsqueeze(-1)
    gather_rejected = rejected.unsqueeze(-1)
    policy_margins = (
        policy_log_probs.gather(1, gather_chosen) - policy_log_probs.gather(1, gather_rejected)
    ).squeeze(-1)
    reference_margins = (
        reference_log_probs.gather(1, gather_chosen)
        - reference_log_probs.gather(1, gather_rejected)
    ).squeeze(-1)
    advantages = policy_margins - reference_margins
    losses = -F.logsigmoid(float(beta) * advantages)

    if weights is None:
        loss = losses.mean()
    else:
        if weights.shape != (batch_size,):
            raise ValueError("weights 必须是一维批次权重")
        weights = weights.float()
        if not torch.all(torch.isfinite(weights)) or torch.any(weights <= 0):
            raise ValueError("weights 必须是有限正数")
        loss = (losses * weights).sum() / weights.sum()
    return PreferenceLoss(loss, advantages, policy_margins, reference_margins)
