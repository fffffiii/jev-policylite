from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    checkpoint: Path
    calibration_file: Path
    test_metrics_file: Path
    static_dir: Path
    edge_bundle: Path | None
    max_upload_bytes: int
    requests_per_minute: int

    @classmethod
    def from_env(cls) -> "Settings":
        package_dir = Path(__file__).resolve().parent
        checkpoint = Path(os.getenv("MODEL_CHECKPOINT", "outputs/public-pilot")).resolve()
        calibration = Path(
            os.getenv("CALIBRATION_FILE", str(checkpoint / "calibration.json"))
        ).resolve()
        metrics = Path(
            os.getenv(
                "TEST_METRICS_FILE",
                str(checkpoint / "test-default/test_metrics.json"),
            )
        ).resolve()
        max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "10"))
        rate_limit = int(os.getenv("REQUESTS_PER_MINUTE", "30"))
        edge_bundle_value = os.getenv("EDGE_BUNDLE_PATH", "").strip()
        edge_bundle = Path(edge_bundle_value).resolve() if edge_bundle_value else None
        if max_upload_mb < 1 or max_upload_mb > 50:
            raise ValueError("MAX_UPLOAD_MB 必须位于 1 到 50")
        if rate_limit < 1:
            raise ValueError("REQUESTS_PER_MINUTE 必须大于 0")
        return cls(
            checkpoint=checkpoint,
            calibration_file=calibration,
            test_metrics_file=metrics,
            static_dir=package_dir / "static",
            edge_bundle=edge_bundle,
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            requests_per_minute=rate_limit,
        )

    def validate_files(self) -> None:
        required = [
            self.checkpoint / "metadata.json",
            self.checkpoint / "decision_head.pt",
            self.checkpoint / "adapter",
            self.calibration_file,
            self.test_metrics_file,
            self.static_dir / "index.html",
        ]
        if self.edge_bundle is not None:
            required.append(self.edge_bundle)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"服务缺少必需文件：{missing}")
