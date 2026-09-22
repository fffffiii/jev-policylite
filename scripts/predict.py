from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from qwen35_moderation.metrics import sigmoid
from qwen35_moderation.model import model_inputs
from qwen35_moderation.runtime import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对单条图文内容执行审核")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", default="")
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--policy-file", required=True)
    parser.add_argument("--calibration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, metadata, collator = load_checkpoint(args.checkpoint, device)
    image_path = Path(args.image).resolve()
    collator.image_root = image_path.parent
    row = {
        "sample_id": "request",
        "image": image_path.name,
        "text": args.text,
        "policy_id": args.policy_id,
        "policy_text": Path(args.policy_file).read_text(encoding="utf-8").strip(),
        "label": 0,
    }
    calibration = {"temperature": 1.0, "global_threshold": 0.5, "policy_thresholds": {}}
    if args.calibration:
        calibration.update(json.loads(Path(args.calibration).read_text(encoding="utf-8")))
    threshold = float(
        calibration["policy_thresholds"].get(args.policy_id, calibration["global_threshold"])
    )
    batch = collator([row])
    inputs = {key: value.to(device) for key, value in model_inputs(batch).items()}
    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with torch.no_grad(), context:
        logit = float(model(**inputs).float().cpu().item())
    probability = float(sigmoid(np.array([logit]), calibration["temperature"])[0])
    result = {
        "violation_score": probability,
        "decision": "violation" if probability >= threshold else "pass",
        "threshold": threshold,
        "policy_version": args.policy_id,
        "model_version": str(Path(args.checkpoint).resolve()),
        "base_model": metadata["base_model"],
        "input_status": "ok",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
