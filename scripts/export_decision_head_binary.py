from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="将二元违规头导出为端侧格式的 FP32 文件（不含属性头或策略头）")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    state = torch.load(
        args.checkpoint / "decision_head.pt", map_location="cpu", weights_only=True
    )
    arrays = [
        state["0.weight"].float().numpy(),
        state["0.bias"].float().numpy(),
        state["1.weight"].float().reshape(-1).numpy(),
        state["1.bias"].float().reshape(-1).numpy(),
    ]
    hidden_size = int(arrays[0].size)
    if [array.size for array in arrays] != [hidden_size, hidden_size, hidden_size, 1]:
        raise ValueError("审核头结构与端侧读取格式不匹配")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.concatenate(arrays).astype("<f4", copy=False).tofile(args.output)
    metadata = {
        "dtype": "float32-little-endian",
        "hidden_size": hidden_size,
        "layout": ["layer_norm_weight", "layer_norm_bias", "linear_weight", "linear_bias"],
        "layer_norm_eps": 1e-5,
        "float_count": hidden_size * 3 + 1,
        "byte_count": (hidden_size * 3 + 1) * 4,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
