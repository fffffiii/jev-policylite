# Edge status

The project investigates local, offline moderation for constrained hardware. This document separates completed prototype measurements from planned targets.

## Current Q4 prototype

The current Linux x86_64 CPU prototype quantizes a merged Qwen3.5-0.8B generation checkpoint to a 4-bit MNN path and reads the existing **binary** moderation head from `hidden_states` rather than generating an answer.

| Measurement | Current observation |
| --- | ---: |
| Q4 model directory | about 479 MB |
| Binary head file | 12,292 bytes, FP32 |
| Peak RSS, four CPU threads | about 859 MiB |
| Visual encoding | about 1.6–1.8 s |
| Quantized backbone forward | about 4.0–4.3 s |
| Cold single-process run | about 10 s |
| Stratified sample decision agreement | 8 / 8 |

The sample showed probability movement after quantization, so production thresholds must be recalibrated against a dedicated calibration set. The sample agreement is not a full regression evaluation.

## What still needs work

- Run a full held-out quantized regression evaluation.
- Add attribute and three-way policy-head execution to the edge runtime.
- Package and measure Windows x86_64, Android ARM64 and iOS builds.
- Measure real mobile memory, latency, thermal behavior and battery draw.

The exported small heads are deliberately simple `LayerNorm + Linear` modules. The current `scripts/export_decision_head_binary.py` exports the binary head only; multi-head runtime support needs an explicit follow-up implementation.
