# Training guide

This guide covers data preparation, supervised training, calibration, and evaluation. All paths are examples; keep model caches, real data and experiment outputs outside a public Git repository.

## 1. Prepare data

Start with [data/annotations.example.csv](../data/annotations.example.csv), or create a JSONL manifest matching [data/README.md](../data/README.md). A record has a stable `sample_id`, `group_id`, image path, policy id, policy text, binary label and split. Every variant of one original image must use the same `group_id` and stay in one split.

```bash
python scripts/build_manifest.py \
  --annotations data/annotations.csv \
  --output data/manifest.jsonl

python scripts/validate_data.py \
  --manifest data/manifest.jsonl \
  --image-root .
```

Run a smoke test before a full experiment. Memory use depends on image size, sequence length, batch size, and the installed runtime. Check the smoke-test result on your own GPU before increasing the workload; the example configuration is not a memory-fit guarantee.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test.py --config configs/train.yaml
```

## 2. Supervised adaptation

The normal supervised trainer freezes the visual tower, warms up task heads, then trains language LoRA and task heads. Checkpoints contain an adapter, processor, binary head and metadata. A multi-head configuration can additionally save attribute and policy heads when a supervised label exists.

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  scripts/train.py --config configs/train.yaml
```

The public-pilot configuration uses `decision_from_label: true` only because the open pilot lacks decision annotations. It maps a binary label to block or allow and cannot establish a useful `review` action.

## 3. Calibrate and test

Fit temperature and threshold only with a calibration split. Do not choose thresholds on the test split.

```bash
python scripts/evaluate.py \
  --checkpoint outputs/your-checkpoint \
  --manifest data/manifest.jsonl \
  --image-root . \
  --split calibration \
  --output-dir outputs/evaluation

python scripts/calibrate.py \
  --predictions outputs/evaluation/calibration_predictions.jsonl \
  --target-fpr 0.01 \
  --output outputs/evaluation/calibration.json

python scripts/evaluate.py \
  --checkpoint outputs/your-checkpoint \
  --manifest data/manifest.jsonl \
  --image-root . \
  --split test \
  --calibration outputs/evaluation/calibration.json \
  --output-dir outputs/evaluation
```

Report PR-AUC, recall and achieved FPR for each policy. Small test sets cannot prove a low target FPR. Keep a failure taxonomy for policy flips, art, medicine, small targets, multi-image inputs and text/image conflicts.

## 4. Add human feedback and DPO

Continue with [Human feedback and discrete DPO](HUMAN_FEEDBACK_AND_RL.md). The DPO trainer needs a checkpoint containing `policy_head.pt`; it freezes all other model parameters, caches features once, and trains only that head.

## 5. Verify before release

```bash
python -m pytest -q
python scripts/build_mosaic_manifest.py --help
python scripts/evaluate.py --help
```

Use the [release checklist](../RELEASE_CHECKLIST.md) to check staged files. The reviewed pilot adapter under `models/pilot-multihead-v0.1/` is intentional; other weights, source data, feedback, credentials, and browser profiles should not be committed.

## 6. Run the local web interface

Follow the model/processor preparation and service command in the [README](../README.en.md#local-service). The service requires a checkpoint, processor, and binary calibration file. Offline test metrics are optional: set `TEST_METRICS_FILE` to the report for the same checkpoint, data split, and threshold. Missing or invalid reports are identified on the page instead of being replaced with pilot numbers.

The UI shows the binary threshold verdict separately from policy-head action suggestions. It does not explain the model’s reasoning. Service-processing time excludes upload and image decoding; it is not browser-to-server round-trip latency. Uploaded images are temporarily written during preprocessing, then removed; configure access controls and review temporary storage before using sensitive data.
