from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel

from .heads import ATTRIBUTE_IDS, DECISION_IDS, make_task_head


@dataclass
class HeadBundle:
    """一次前向同时给出的审核 logit。未加载的头保持 None。"""

    violation_logit: torch.Tensor
    attribute_logits: torch.Tensor | None = None
    decision_logits: torch.Tensor | None = None


class ModerationModel(nn.Module):
    """Qwen3.5 多模态骨干，后面挂审核头。

    `head` 是已有的二元违规头。`attribute_head` 与 `policy_head` 接在同一份
    池化隐藏特征上，分别回答画面属性和 Block / Review / Allow。
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        hidden_size = int(backbone.config.text_config.hidden_size)
        self.head = make_task_head(hidden_size, 1)
        self.attribute_head = make_task_head(hidden_size, len(ATTRIBUTE_IDS))
        self.policy_head = make_task_head(hidden_size, len(DECISION_IDS))
        self.multi_head_enabled = False
        self.attribute_head_ready = False
        self.policy_head_ready = False
        self.attribute_head_supervised = False
        self.policy_head_supervised = False
        self._set_extra_head_trainable(False)
        self.head.float()
        self.attribute_head.float()
        self.policy_head.float()

    @classmethod
    def create(
        cls,
        model_name: str,
        *,
        dtype: torch.dtype,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: str | list[str] = "all-linear",
        gradient_checkpointing: bool = True,
        trust_remote_code: bool = False,
        adapter_path: str | Path | None = None,
    ) -> "ModerationModel":
        backbone = AutoModel.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        for parameter in backbone.parameters():
            parameter.requires_grad = False

        from peft import LoraConfig, PeftModel, get_peft_model

        if adapter_path is None:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            backbone.language_model = get_peft_model(backbone.language_model, lora_config)
        else:
            backbone.language_model = PeftModel.from_pretrained(
                backbone.language_model, str(adapter_path), is_trainable=False
            )

        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable()
            if hasattr(backbone, "enable_input_require_grads"):
                backbone.enable_input_require_grads()
        backbone.config.use_cache = False
        model = cls(backbone)
        model.head.float()
        model.attribute_head.float()
        model.policy_head.float()
        return model

    def _set_extra_head_trainable(self, trainable: bool) -> None:
        for parameter in list(self.attribute_head.parameters()) + list(self.policy_head.parameters()):
            parameter.requires_grad = trainable

    def enable_multi_heads(self) -> None:
        """打开两只新头的训练。没有监督样本的头不会被写成权重文件。"""
        self.multi_head_enabled = True
        self._set_extra_head_trainable(True)

    def set_lora_trainable(self, trainable: bool) -> None:
        for name, parameter in self.backbone.language_model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = trainable

    def train(self, mode: bool = True) -> "ModerationModel":
        super().train(mode)
        # 视觉塔冻结后固定为推理模式，避免随机层影响同图对照。
        self.backbone.visual.eval()
        return self

    def trainable_parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total": total, "trainable": trainable}

    def _pool(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        attention_mask = inputs["attention_mask"]
        outputs = self.backbone(**inputs, return_dict=True, use_cache=False)
        hidden = outputs.last_hidden_state
        last_positions = attention_mask.long().sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_positions]
        return pooled.float()

    def encode(self, **inputs: torch.Tensor) -> torch.Tensor:
        """把图文输入编码为各审核头共享的池化特征。"""
        return self._pool(inputs)

    def forward(self, return_heads: bool = False, **inputs: torch.Tensor) -> torch.Tensor | HeadBundle:
        pooled = self.encode(**inputs)
        violation_logit = self.head(pooled).squeeze(-1)
        if not return_heads:
            return violation_logit
        attribute_logits = self.attribute_head(pooled) if self._attribute_active() else None
        decision_logits = self.policy_head(pooled) if self._policy_active() else None
        return HeadBundle(violation_logit, attribute_logits, decision_logits)

    def _attribute_active(self) -> bool:
        return self.multi_head_enabled or self.attribute_head_ready

    def _policy_active(self) -> bool:
        return self.multi_head_enabled or self.policy_head_ready

    def save_artifacts(self, output_dir: str | Path, metadata: dict[str, Any]) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.backbone.language_model.save_pretrained(output_path / "adapter")
        torch.save(self.head.state_dict(), output_path / "decision_head.pt")
        files = {"violation": "decision_head.pt"}
        if self.attribute_head_ready or self.attribute_head_supervised:
            torch.save(self.attribute_head.state_dict(), output_path / "attribute_head.pt")
            files["attribute"] = "attribute_head.pt"
        if self.policy_head_ready or self.policy_head_supervised:
            torch.save(self.policy_head.state_dict(), output_path / "policy_head.pt")
            files["policy"] = "policy_head.pt"
        payload = {
            **metadata,
            "heads": {
                "attribute_labels": list(ATTRIBUTE_IDS),
                "decision_labels": list(DECISION_IDS),
                "files": files,
            },
        }
        (output_path / "metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load_head(self, checkpoint: str | Path) -> None:
        checkpoint_path = Path(checkpoint)
        state = torch.load(checkpoint_path / "decision_head.pt", map_location="cpu", weights_only=True)
        self.head.load_state_dict(state)
        attribute_path = checkpoint_path / "attribute_head.pt"
        if attribute_path.exists():
            attribute_state = torch.load(attribute_path, map_location="cpu", weights_only=True)
            self.attribute_head.load_state_dict(attribute_state)
            self.attribute_head_ready = True
            self.attribute_head_supervised = True
        policy_path = checkpoint_path / "policy_head.pt"
        if policy_path.exists():
            policy_state = torch.load(policy_path, map_location="cpu", weights_only=True)
            self.policy_head.load_state_dict(policy_state)
            self.policy_head_ready = True
            self.policy_head_supervised = True
        self._set_extra_head_trainable(False)


def model_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    ignored = {"labels", "sample_ids", "group_ids", "policy_ids", "attributes", "decisions"}
    return {key: value for key, value in batch.items() if key not in ignored and torch.is_tensor(value)}
