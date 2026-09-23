"""离线检查文档链接、可编辑流程图与静态项目页。"""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids = []; self.refs = []; self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs: self.ids.append(attrs["id"])
        if tag in {"a", "link"} and attrs.get("href"): self.refs.append(attrs["href"])
        if tag in {"script", "img"} and attrs.get("src"): self.refs.append(attrs["src"])
        if tag == "img": self.images.append(attrs)


def test_markdown_links_resolve_locally():
    tracked = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    for relative in (Path(name.decode("utf-8")) for name in tracked.split(b"\0") if name):
        if relative.suffix != ".md":
            continue
        document = ROOT / relative
        # 跳过命令示例，只检查 Markdown 链接与图片路径。
        content = re.sub(r"```.*?```", "", document.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", content):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path: continue
            path = (document.parent / unquote(parsed.path)).resolve()
            assert path.exists(), f"Broken link in {document.relative_to(ROOT)}: {target}"


def test_project_page_ids_links_and_image_alternatives():
    page = ROOT / "site" / "index.html"
    parser = PageParser(); parser.feed(page.read_text(encoding="utf-8"))
    assert len(parser.ids) == len(set(parser.ids)), "Duplicate HTML ids"
    for target in parser.refs:
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc: continue
        if parsed.path:
            assert (page.parent / unquote(parsed.path)).exists(), f"Missing site file: {target}"
        elif parsed.fragment:
            assert parsed.fragment in parser.ids, f"Missing anchor: {target}"
    for image in parser.images:
        if image.get("id") == "dialog-image": continue  # 放大图在打开弹窗时填充。
        assert image.get("alt"), "Static figures need descriptive alt text"


def test_diagrams_have_editable_text_and_no_embedded_raster():
    ns = {"svg": "http://www.w3.org/2000/svg"}
    for name in ("architecture", "post-training"):
        svg = ET.parse(ROOT / "site" / "assets" / f"{name}.svg").getroot()
        assert svg.find("svg:title", ns) is not None
        assert svg.find("svg:desc", ns) is not None
        assert len(svg.findall(".//svg:text", ns)) > 15
        assert not svg.findall(".//svg:image", ns)
        assert not svg.findall(".//svg:script", ns)
        assert (ROOT / "site" / "assets" / f"{name}.png").is_file()
