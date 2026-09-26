"""流程图校验应兼容 Git 换行转换，并继续拒绝过期 PNG。"""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("newline,changed", [(b"\n", False), (b"\r\n", False), (b"\n", True)])
def test_diagram_check_handles_checkout_line_endings(tmp_path, newline, changed):
    # 在临时仓库布局内执行实际命令，使用已发布图片及其摘要。
    scripts = tmp_path / "scripts"
    assets = tmp_path / "site" / "assets"
    scripts.mkdir()
    assets.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts" / "render_diagrams.py", scripts / "render_diagrams.py")
    for name in ("architecture", "post-training"):
        svg = (ROOT / "site" / "assets" / f"{name}.svg").read_bytes().replace(b"\r\n", b"\n")
        if changed and name == "architecture":
            assert b"</title>" in svg
            svg = svg.replace(b"</title>", b" updated</title>", 1)
        (assets / f"{name}.svg").write_bytes(svg.replace(b"\n", newline))
        shutil.copyfile(ROOT / "site" / "assets" / f"{name}.png", assets / f"{name}.png")
    result = subprocess.run(
        [sys.executable, str(scripts / "render_diagrams.py"), "--check"],
        capture_output=True,
    )
    if changed:
        assert result.returncode != 0, "源码变化后必须拒绝旧 PNG"
    else:
        assert result.returncode == 0, result.stderr
        assert b"OK: architecture.png" in result.stdout
        assert b"OK: post-training.png" in result.stdout
