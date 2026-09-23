"""Load optional offline metrics without presenting them as a verified model benchmark."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value: {value}")


def load_test_metrics(path: Path) -> tuple[dict[str, Any], str]:
    """Return the supplied report and a display note; missing/invalid reports stay empty."""
    if not path.is_file():
        return {}, "未提供离线评测文件；可通过 TEST_METRICS_FILE 指定。"
    try:
        report = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeError, ValueError):
        return {}, "离线评测文件无法读取或不是有效 JSON，请检查 TEST_METRICS_FILE。"
    if not isinstance(report, dict) or not isinstance(report.get("global"), dict):
        return {}, "离线评测文件缺少 global 指标对象，请检查文件格式。"
    return report, "指标来自导入文件；请核对检查点、测试集与阈值，服务不会自动验证它们是否匹配。"
