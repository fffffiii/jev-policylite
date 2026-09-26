#!/usr/bin/env bash
# 各模型顺序运行，避免同时争用 GPU 导致计时失真。
set -euo pipefail
if [ "$#" -eq 0 ]; then
  set -- falconsai five_class nudenet jev_224 jev_448
fi
mkdir -p outputs/nsfw-comparison-v1
for benchmark_model in "$@"; do
  "${PYTHON:-python}" scripts/benchmark_nsfw.py --model "$benchmark_model" \
    ${TIMING_ONLY:+--timing-only} \
    > "outputs/nsfw-comparison-v1/${benchmark_model}.log" 2>&1
done
