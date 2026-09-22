#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/public-pilot-multihead}"
cd "$PROJECT_DIR"
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null || true)" ]]; then
  echo "OUTPUT_NONEMPTY"
  exit 3
fi
mkdir -p logs
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
setsid nohup python scripts/train.py \
  --config configs/public-pilot-multihead.yaml \
  > logs/public-pilot-multihead-train.log 2>&1 < /dev/null &
echo "STARTED:$!"
