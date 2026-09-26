"""从可编辑 SVG 导出 PNG，并验证 PNG 对应的 SVG 版本。"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
from PIL.PngImagePlugin import PngInfo


def render_svg(source: bytes, width: int, height: int) -> bytes:
    """使用浏览器原生 SVG 渲染器，避免依赖系统 Cairo 库。"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("请安装浏览器绘图依赖：pip install -e '.[browser]'") from exc
    data_url = "data:image/svg+xml;base64," + base64.b64encode(source).decode("ascii")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        page.set_content(
            '<html><head><style>html,body{margin:0;padding:0}</style></head>'
            f'<body><img id="figure" width="{width}" height="{height}" src="{data_url}"></body></html>'
        )
        page.locator("#figure").wait_for(state="visible")
        result = page.locator("#figure").screenshot()
        browser.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="从可编辑 SVG 导出项目流程图 PNG")
    parser.add_argument("--check", action="store_true", help="检查 PNG 尺寸与 SVG 源码版本")
    args = parser.parse_args()
    assets = Path(__file__).resolve().parents[1] / "site" / "assets"
    for name in ("architecture", "post-training"):
        source, output = assets / f"{name}.svg", assets / f"{name}.png"
        # Git 在 Windows 与 Linux 间转换换行；渲染和新摘要统一使用 LF。
        svg = source.read_bytes().replace(b"\r\n", b"\n")
        root = ET.fromstring(svg)
        width, height = int(root.attrib["width"]), int(root.attrib["height"])
        digest = hashlib.sha256(svg).hexdigest()
        if args.check:
            if not output.is_file():
                raise SystemExit(f"缺少 PNG：{output}")
            with Image.open(output) as image:
                # 兼容已发布 PNG 中按 Windows CRLF 源码保存的旧摘要。
                legacy_digest = hashlib.sha256(svg.replace(b"\n", b"\r\n")).hexdigest()
                if image.size != (width, height) or image.info.get("svg_sha256") not in {digest, legacy_digest}:
                    raise SystemExit(f"PNG 与 SVG 源码不匹配：{output}")
            print(f"OK: {output.name}")
        else:
            rendered = render_svg(svg, width, height)
            with Image.open(io.BytesIO(rendered)) as image:
                metadata = PngInfo()
                metadata.add_text("svg_sha256", digest)
                image.save(output, format="PNG", pnginfo=metadata)
            print(f"Rendered {source.name} -> {output.name}")


if __name__ == "__main__":
    main()
