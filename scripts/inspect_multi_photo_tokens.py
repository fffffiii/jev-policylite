"""检查四图模板中的视觉 token 位置，用于开发局部读出头。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoProcessor

from qwen35_moderation.data import configure_max_pixels
from qwen35_moderation.multi_photo import native_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", nargs=4, required=True)
    parser.add_argument("--policy-file", required=True)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(Path(args.checkpoint) / "processor")
    configure_max_pixels(processor, 50176)
    encoded = native_inputs(
        processor,
        {"tiles": [{"image": str(Path(path).resolve())} for path in args.images]},
        {"policy_text": Path(args.policy_file).read_text(encoding="utf-8")}, Path("."), 2048,
    )
    ids = encoded["input_ids"][0].tolist()
    markers = {}
    for token in ("<|image|>", "<|vision_start|>", "<|vision_end|>"):
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        markers[token] = {"id": token_id, "positions": [i for i, value in enumerate(ids) if value == token_id]}
    print(json.dumps({"length": len(ids), "markers": markers}, ensure_ascii=False))


if __name__ == "__main__":
    main()
