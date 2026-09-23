"""一次审核四张图片，输出每张图片的违规概率和端到端本机耗时。"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image

from qwen35_moderation.data import MultimodalCollator, configure_max_pixels
from qwen35_moderation.mosaic import make_mosaic
from qwen35_moderation.multi_photo import POSITION_NAMES, load_four_position_heads, load_shared_position_head, native_inputs, vision_end_features
from qwen35_moderation.runtime import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="四图批量审核；顺序是左上、右上、左下、右下")
    parser.add_argument("--checkpoint", required=True, help="含 adapter、processor 和基础审核头的检查点")
    parser.add_argument("--heads", required=True, help="对应 native 或 mosaic 模式训练得到的四位置头")
    parser.add_argument("--mode", required=True, choices=["native", "native_local", "native_shared", "mosaic"])
    parser.add_argument("--images", nargs=4, required=True, type=Path)
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--policy-text")
    policy.add_argument("--policy-file", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold 必须在 [0, 1] 内")
    images = [path.resolve() for path in args.images]
    for path in images:
        if not path.is_file():
            raise FileNotFoundError(path)
    policy_text = args.policy_text if args.policy_text is not None else args.policy_file.read_text(encoding="utf-8")
    device = torch.device(args.device)
    model, processor, _, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    hidden_size = int(model.backbone.config.text_config.hidden_size)
    head_loader = load_shared_position_head if args.mode == "native_shared" else load_four_position_heads
    heads = head_loader(args.heads, hidden_size, device)
    row = {"sample_id": "inference", "image": "", "text": "", "policy_text": policy_text, "label": 0}
    layout = {"tiles": [{"image": str(path)} for path in images]}
    configure_max_pixels(processor, 50176 if args.mode.startswith("native") else 200704)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if args.mode.startswith("native"):
        encoded = native_inputs(processor, layout, row, Path("."), args.max_length)
    else:
        with tempfile.TemporaryDirectory(prefix="jev-four-photo-") as temporary:
            sources = []
            try:
                for path in images:
                    with Image.open(path) as source:
                        sources.append(source.copy())
                mosaic = make_mosaic(sources, tile_size=224)
                path = Path(temporary) / "mosaic.jpg"
                mosaic.save(path, format="JPEG", quality=95, subsampling=0)
                mosaic.close()
            finally:
                for source in sources:
                    source.close()
            collator = MultimodalCollator(processor, Path("."), args.max_length)
            encoded = collator._encode_one({**row, "image": str(path)})
    inputs = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if args.mode in ("native_local", "native_shared"):
                token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
                if not isinstance(token_id, int):
                    raise ValueError("处理器缺少 vision_end token")
                features = vision_end_features(model, inputs, token_id)
            else:
                features = model.encode(**inputs)
        probabilities = torch.sigmoid(heads(features.float())).squeeze(0).cpu().tolist()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(json.dumps({
        "mode": args.mode,
        "images": [
            {"position": name, "path": str(path), "violation_probability": float(probability),
             "violation": bool(probability >= args.threshold)}
            for name, path, probability in zip(POSITION_NAMES, images, probabilities)
        ],
        "threshold": args.threshold, "elapsed_ms": elapsed_ms,
        "peak_allocated_gb": (torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None),
        "timing_scope": "模型已加载；包含读图、预处理、前向和四个检测头；不含网络传输",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
