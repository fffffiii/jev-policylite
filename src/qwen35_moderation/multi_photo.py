"""四图定位实验：一个共享表征接四个按位置区分的二元检测头。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .heads import make_task_head


POSITION_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")


class FourPositionHeads(nn.Module):
    """四个独立的小头分别预测 2×2 布局中的一张图片。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.heads = nn.ModuleList(make_task_head(hidden_size, 1) for _ in POSITION_NAMES)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            return torch.cat([head(features.float()) for head in self.heads], dim=-1)
        if features.ndim == 3 and features.shape[1] == len(POSITION_NAMES):
            return torch.cat([head(features[:, index].float()) for index, head in enumerate(self.heads)], dim=-1)
        raise ValueError("features 必须是 [batch, hidden_size] 或 [batch, 4, hidden_size]")


class SharedPositionHead(nn.Module):
    """一个共享小头分别作用于四张图的局部表征。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = make_task_head(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != len(POSITION_NAMES):
            raise ValueError("共享头要求 [batch, 4, hidden_size]")
        return self.head(features.float()).squeeze(-1)


def load_four_position_heads(path: str | Path, hidden_size: int, device: torch.device) -> FourPositionHeads:
    """加载独立于基础检查点保存的四图头。"""
    model = FourPositionHeads(hidden_size)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def load_shared_position_head(path: str | Path, hidden_size: int, device: torch.device) -> SharedPositionHead:
    model = SharedPositionHead(hidden_size)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def native_inputs(
    processor: Any, layout: dict[str, Any], row: dict[str, Any], image_root: Path, max_length: int
) -> dict[str, torch.Tensor]:
    """把四张独立图片和同一审核规则放进单次 prefill。"""
    content: list[dict[str, str]] = [
        {"type": "text", "text": f"你是内容审核模型。审核规则：{row['policy_text']}\n请按顺序查看四张图片并分别判断是否违规。"}
    ]
    for index, tile in enumerate(layout["tiles"], start=1):
        path = (image_root / str(tile["image"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        content.extend([{"type": "text", "text": f"图片{index}："}, {"type": "image", "path": str(path)}])
    content.append({"type": "text", "text": "只需进行内部判断。"})
    encoded = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
        processor_kwargs={"truncation": False},
    )
    if encoded["input_ids"].shape[-1] > max_length:
        raise ValueError(f"四图 token 数 {encoded['input_ids'].shape[-1]} 超过 max_length={max_length}")
    return {key: value for key, value in encoded.items() if torch.is_tensor(value)}


def vision_end_features(model: nn.Module, inputs: dict[str, torch.Tensor], token_id: int) -> torch.Tensor:
    """读取每张图视觉片段末端的隐藏状态，四张图各得到一个表征。"""
    ids = inputs["input_ids"]
    if ids.shape[0] != 1:
        raise ValueError("局部读出目前只支持 batch=1")
    positions = torch.nonzero(ids[0] == token_id, as_tuple=False).flatten()
    if positions.numel() != len(POSITION_NAMES):
        raise ValueError(f"预期四个 vision_end token，实际 {positions.numel()} 个")
    outputs = model.backbone(**inputs, return_dict=True, use_cache=False)
    return outputs.last_hidden_state[:, positions, :].float()


def validate_four_image_layout(layout: dict[str, Any], row: dict[str, Any]) -> tuple[int, int, int, int]:
    """校验标签顺序与拼图位置一致，防止训练时位置错位。"""
    if str(layout["sample_id"]) != str(row["sample_id"]):
        raise ValueError("布局与清单的 sample_id 不一致")
    if str(layout["policy_id"]) != str(row["policy_id"]):
        raise ValueError("布局与清单的 policy_id 不一致")
    labels = tuple(int(value) for value in layout["tile_labels"])
    tiles = layout["tiles"]
    if len(labels) != 4 or len(tiles) != 4:
        raise ValueError("四图头要求恰好四张原图")
    if labels != tuple(int(tile["label"]) for tile in tiles):
        raise ValueError("tile_labels 与 tiles 中的标签不一致")
    if any(value not in (0, 1) for value in labels):
        raise ValueError("每个位置的标签只能是 0 或 1")
    if max(labels) != int(row["label"]):
        raise ValueError("整图 OR 标签与位置标签不一致")
    if len({str(tile["group_id"]) for tile in tiles}) != 4:
        raise ValueError("同一组内包含重复原图")
    return labels  # type: ignore[return-value]
