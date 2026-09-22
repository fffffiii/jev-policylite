from __future__ import annotations

import json
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from qwen35_moderation.heads import bundle_probabilities, format_heads
from qwen35_moderation.metrics import sigmoid
from qwen35_moderation.model import model_inputs
from qwen35_moderation.runtime import load_checkpoint

from .errors import InputError
from .image_io import decode_image


POLICIES = {
    "pilot-suggestive-v1": {
        "name": "强性暗示管控",
        "description": "L2 及以上判为违规",
        "text": "允许正常姿态且无性暗示的轻度性感穿着；禁止强烈性暗示、诱惑性姿势、私密部位裸露及明确性行为。",
        "trained": True,
    },
    "pilot-explicit-v1": {
        "name": "裸露与色情管控",
        "description": "L3 及以上判为违规",
        "text": "允许有衣物遮挡的性感穿着和性暗示姿势；禁止私密部位裸露、自慰、明确性行为及性器官特写。",
        "trained": True,
    },
}


class ModerationEngine:
    def __init__(self, checkpoint: Path, calibration_file: Path, device: torch.device):
        started = time.perf_counter()
        self.checkpoint = checkpoint
        self.device = device
        self.model, _, self.metadata, self.collator = load_checkpoint(checkpoint, device)
        # 固定在线序列形状，避免规则长度变化触发新的 Triton 内核编译。
        self.collator.pad_to_max_length = True
        self.calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
        self.load_ms = (time.perf_counter() - started) * 1000

    def moderate(
        self,
        image: Image.Image,
        text: str,
        policy_id: str,
        policy_text: str,
        threshold_mode: str,
    ) -> dict[str, Any]:
        if policy_id in POLICIES and not policy_text.strip():
            resolved_text = POLICIES[policy_id]["text"]
            support = "trained"
        elif policy_text.strip():
            resolved_text = policy_text.strip()
            support = "experimental"
        else:
            raise InputError("请选择预设规则或填写自定义规则。", "missing-policy")
        if len(text) > 10_000 or len(resolved_text) > 4_000:
            raise InputError("正文最多 10000 字，规则最多 4000 字。", "text-too-long")
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="jev-policylite-") as directory:
            image_path = Path(directory) / "input.jpg"
            image.save(image_path, format="JPEG", quality=95)
            row = {
                "sample_id": "request", "image": str(image_path), "text": text,
                "policy_id": policy_id, "policy_text": resolved_text, "label": 0,
            }
            batch = self.collator([row])
        preprocessing_ms = (time.perf_counter() - started) * 1000
        inputs = {key: value.to(self.device) for key, value in model_inputs(batch).items()}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_started = time.perf_counter()
        context = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
        with torch.inference_mode(), context:
            bundle = self.model(return_heads=True, **inputs)
            logit = float(bundle.violation_logit.float().cpu().item())
            attribute_probabilities, decision_probabilities = bundle_probabilities(bundle)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - inference_started) * 1000
        raw_score = float(sigmoid(np.array([logit]))[0])
        calibrated_score = float(sigmoid(np.array([logit]), self.calibration["temperature"])[0])
        if threshold_mode == "balanced":
            score, threshold = raw_score, 0.5
        else:
            score = calibrated_score
            threshold = float(
                self.calibration["policy_thresholds"].get(
                    policy_id, self.calibration["global_threshold"]
                ) if support == "trained" else self.calibration["global_threshold"]
            )
        return {
            "decision": "violation" if score >= threshold else "pass",
            "violation_score": score,
            "raw_score": raw_score,
            "calibrated_score": calibrated_score,
            "threshold": threshold,
            "threshold_mode": threshold_mode,
            "policy_id": policy_id,
            "policy_support": support,
            "preprocessing_ms": preprocessing_ms,
            "inference_ms": inference_ms,
            "input_tokens": int(batch["attention_mask"].sum().item()),
            "image_size": {"width": image.width, "height": image.height},
            "base_model": self.metadata["base_model"],
            "heads": format_heads(
                violation_probability=score,
                threshold=threshold,
                attribute_probabilities=None if attribute_probabilities is None else attribute_probabilities[0],
                decision_probabilities=None if decision_probabilities is None else decision_probabilities[0],
            ),
        }
