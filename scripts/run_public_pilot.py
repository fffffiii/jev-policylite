"""依次执行数据检查、训练、独立校准和测试，逐阶段保存状态。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    state = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": os.getpid()}

    def save(stage, **values):
        state.update(stage=stage, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **values)
        (logs / "public-pilot-status.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(stage, args):
        save(stage)
        with (logs / f"public-pilot-{stage}.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", *args], stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        save("waiting_for_data")
        deadline = time.monotonic() + 3600
        while not Path("data/public-pilot/provenance.json").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("数据准备超过一小时，请检查 prepare-public.log")
            time.sleep(15)
        run("checks", ["-m", "pytest", "-q"])
        run("data_validation", ["scripts/validate_data.py", "--manifest", "data/public-pilot/manifest.jsonl"])
        run("train", ["scripts/train.py", "--config", "configs/public-pilot.yaml"])
        common = ["scripts/evaluate.py", "--checkpoint", "outputs/public-pilot", "--manifest", "data/public-pilot/manifest.jsonl", "--batch-size", "4"]
        run("calibration_predictions", common + ["--split", "calibration", "--output-dir", "outputs/public-pilot/calibration"])
        run("calibrate", ["scripts/calibrate.py", "--predictions", "outputs/public-pilot/calibration/calibration_predictions.jsonl", "--output", "outputs/public-pilot/calibration.json", "--target-fpr", "0.01"])
        run("test_default", common + ["--split", "test", "--output-dir", "outputs/public-pilot/test-default"])
        run("test_calibrated", common + ["--split", "test", "--calibration", "outputs/public-pilot/calibration.json", "--output-dir", "outputs/public-pilot/test-calibrated"])
        save("complete")
    except Exception as exc:
        save("failed", failed_stage=state["stage"], error=repr(exc))
        raise


if __name__ == "__main__":
    main()
