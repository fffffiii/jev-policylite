from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


VALID_SPLITS = {"train", "validation", "calibration", "test"}
REQUIRED_FIELDS = {
    "sample_id",
    "group_id",
    "image",
    "text",
    "policy_id",
    "policy_text",
    "label",
    "split",
}


def configure_max_pixels(processor: Any, max_pixels: int) -> None:
    """兼容不同 Transformers 版本的 Qwen 图像面积上限字段。"""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("处理器缺少 image_processor，无法处理图片")
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels
    size = getattr(image_processor, "size", None)
    if size is not None and hasattr(size, "get") and size.get("longest_edge") is not None:
        size["longest_edge"] = max_pixels


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL 清单并进行基础字段校验。"""
    manifest_path = Path(path)
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是合法 JSON：{exc}") from exc
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"第 {line_number} 行缺少字段：{sorted(missing)}")
            if row["split"] not in VALID_SPLITS:
                raise ValueError(f"第 {line_number} 行 split 非法：{row['split']}")
            if row["label"] not in (0, 1):
                raise ValueError(f"第 {line_number} 行 label 必须是 0 或 1")
            rows.append(row)
    if not rows:
        raise ValueError(f"清单为空：{manifest_path}")
    return rows


def find_group_leakage(rows: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    """返回跨数据划分出现的 group_id。"""
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        group_splits.setdefault(str(row["group_id"]), set()).add(str(row["split"]))
    return {group: splits for group, splits in group_splits.items() if len(splits) > 1}


class ModerationDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], split: str, image_root: str | Path):
        self.rows = [row for row in rows if row["split"] == split]
        self.image_root = Path(image_root)
        if not self.rows:
            raise ValueError(f"清单中没有 {split!r} 数据")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


@dataclass
class MultimodalCollator:
    processor: Any
    image_root: Path
    max_length: int = 1024
    pad_to_max_length: bool = False

    def _encode_one(self, row: dict[str, Any]) -> dict[str, torch.Tensor]:
        image_path = (self.image_root / row["image"]).resolve()
        body = str(row["text"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(image_path)},
                    {
                        "type": "text",
                        "text": (
                            "你是内容审核模型。请依据给定规则判断内容是否违规。\n"
                            f"审核规则：{row['policy_text']}\n"
                            f"内容正文：{body}\n"
                            "只需进行内部判断。"
                        ),
                    },
                ],
            }
        ]
        while True:
            encoded = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
                processor_kwargs={"truncation": False},
            )
            if encoded["input_ids"].shape[-1] <= self.max_length:
                if body != str(row["text"]):
                    warnings.warn(f"正文过长已截短：{row['sample_id']}；图片与规则完整保留", stacklevel=2)
                return {key: value for key, value in encoded.items() if torch.is_tensor(value)}
            if not body:
                raise ValueError(f"图片和完整规则超过 max_length={self.max_length}：{row['sample_id']}")
            # 只裁正文，完整保留图片 token、规则和最终判断位置。
            body = body[:len(body) // 2]
            messages[0]["content"][1]["text"] = (
                "你是内容审核模型。请依据给定规则判断内容是否违规。\n"
                f"审核规则：{row['policy_text']}\n内容正文：{body}\n只需进行内部判断。"
            )

    @staticmethod
    def _pad_sequence_tensors(
        items: list[dict[str, torch.Tensor]], key: str, pad_value: int,
        target_length: int | None = None,
    ) -> torch.Tensor:
        tensors = [item[key].squeeze(0) for item in items]
        observed_length = max(tensor.shape[0] for tensor in tensors)
        max_length = target_length or observed_length
        if max_length < observed_length:
            raise ValueError(f"目标 padding 长度 {max_length} 小于实际长度 {observed_length}")
        padded = []
        for tensor in tensors:
            pad_shape = (max_length - tensor.shape[0], *tensor.shape[1:])
            padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype)
            padded.append(torch.cat([tensor, padding], dim=0))
        return torch.stack(padded)

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = [self._encode_one(row) for row in rows]
        batch: dict[str, Any] = {}
        sequence_keys = {"input_ids", "attention_mask", "mm_token_type_ids"}
        concat_keys = set().union(*(item.keys() for item in encoded)) - sequence_keys

        batch["input_ids"] = self._pad_sequence_tensors(
            encoded, "input_ids", int(self.processor.tokenizer.pad_token_id),
            self.max_length if self.pad_to_max_length else None,
        )
        batch["attention_mask"] = self._pad_sequence_tensors(
            encoded, "attention_mask", 0, self.max_length if self.pad_to_max_length else None
        )
        if all("mm_token_type_ids" in item for item in encoded):
            batch["mm_token_type_ids"] = self._pad_sequence_tensors(
                encoded, "mm_token_type_ids", 0,
                self.max_length if self.pad_to_max_length else None,
            )
        for key in concat_keys:
            if all(key in item for item in encoded):
                batch[key] = torch.cat([item[key] for item in encoded], dim=0)

        batch["labels"] = torch.tensor([row["label"] for row in rows], dtype=torch.float32)
        batch["sample_ids"] = [str(row["sample_id"]) for row in rows]
        batch["group_ids"] = [str(row.get("group_id", row["sample_id"])) for row in rows]
        batch["policy_ids"] = [str(row["policy_id"]) for row in rows]
        batch["attributes"] = [list(row.get("attributes", [])) for row in rows]
        batch["decisions"] = [row.get("decision") for row in rows]
        return batch
