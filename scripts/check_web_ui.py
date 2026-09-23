"""Exercise both static UIs in Chromium. API fixtures are explicit; this does not run a model."""
from __future__ import annotations

import argparse
import base64
import re
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def load_local_page(page, *, site: bool, fixtures=None):
    """Load local files directly; no HTTP server or external network is required."""
    directory = ROOT / "site" if site else ROOT / "src/qwen35_moderation/web/static"
    html = (directory / "index.html").read_text(encoding="utf-8")
    html = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.S)
    html = re.sub(r"<link\b[^>]*>", "", html)
    if site:
        # set_content 没有站点基址，内联页面实际引用的本地图片供弹窗检查。
        for asset in set(re.findall(r"assets/[A-Za-z0-9_.-]+\.(?:svg|png)", html)):
            source = directory / asset
            if not source.is_file():
                continue
            mime = "image/svg+xml" if source.suffix == ".svg" else "image/png"
            data_url = f"data:{mime};base64," + base64.b64encode(source.read_bytes()).decode()
            html = html.replace(asset, data_url)
    page.set_content(html, wait_until="load")
    page.add_style_tag(path=str(directory / ("assets/style.css" if site else "styles.css")))
    if site:
        page.add_script_tag(path=str(directory / "assets/config.js"))
        page.add_script_tag(path=str(directory / "assets/main.js"))
    else:
        page.evaluate("""fixtures => {
            window.__fixtureRequests = [];
            window.fetch = async (path, options = {}) => {
                let result;
                if (path === '/api/v1/model') result = fixtures.info;
                else if (path === '/api/v1/status') result = fixtures.status;
                else if (path === '/api/v1/moderations') {
                    window.__fixtureRequests.push(options.body.get('policy_text'));
                    result = fixtures.result;
                } else throw new Error('Unexpected fixture request: ' + path);
                return new Response(JSON.stringify(result), {status: 200,
                    headers: {'Content-Type': 'application/json'}});
            };
        }""", fixtures)
        page.add_script_tag(path=str(directory / "app.js"))
        page.wait_for_function("document.getElementById('heroReady').textContent === '已就绪'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check static-page interactions and the service UI with mocked API responses")
    parser.add_argument("--output-dir", default="outputs/web-review")
    parser.add_argument("--chromium-executable", default=shutil.which("chromium"))
    args = parser.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("Install Playwright, then run: playwright install chromium") from exc
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def layout(page, name):
        overflow = page.evaluate("document.documentElement.scrollWidth > innerWidth + 1")
        check(f"{name}: no horizontal page overflow", not overflow)

    info = {
        "test_metrics": {"global": {"count": 12, "pr_auc": None, "accuracy": .75, "recall": .8},
                         "policy_flip": {"both_correct_rate": None}},
        "test_metrics_available": True, "test_metrics_note": "Browser fixture, not a model benchmark.",
        "max_upload_bytes": 5 * 2**20,
        "limitations": ["<img src=x onerror=alert(1)>", "Fixture data; no inference was performed."],
    }
    status = {
        "model_ready": True, "uptime_seconds": 61, "total_requests": 1, "failed_requests": 0,
        "requests_per_minute": 1, "latency_ms": {"p50": 70., "p95": 75., "latest": 70.},
        "gpus": [], "process": {"rss_mb": 512.}, "cuda": {"allocated_mb": 0.}, "recent_requests": [],
    }
    result = {
        "decision": "pass", "violation_score": .2, "threshold": .5, "threshold_mode": "balanced",
        "policy_support": "trained", "total_ms": 70., "inference_ms": 45., "preprocessing_ms": 25., "input_tokens": 100,
        "heads": {
            "attribute": {"id": "attribute", "title": "属性头", "note": "未加载属性头", "labels": []},
            "decision": {"id": "decision", "title": "处置建议（Policy head）", "action": "review",
                         "question": "Independent fixture output", "labels": [
                             {"id": "block", "name": "Block", "probability": .1},
                             {"id": "review", "name": "Review", "probability": .8},
                             {"id": "allow", "name": "Allow", "probability": .1}]},
        },
    }
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium_executable, headless=True, args=["--no-sandbox"])
        browser_version = browser.version
        for width in (1440, 768, 390, 320):
            context = browser.new_context(viewport={"width": width, "height": 960}, device_scale_factor=1)
            page = context.new_page(); errors = []; page.on("pageerror", lambda error: errors.append(str(error)))
            load_local_page(page, site=True)
            layout(page, f"site {width}px")
            for index in range(4):
                page.locator(f"#stage-tab-{index}").click()
                check(f"site {width}px: stage {index}", page.locator(f"#stage-tab-{index}").get_attribute("aria-selected") == "true")
            page.locator("#stage-tab-3").press("Home")
            check(f"site {width}px: keyboard tab navigation", page.locator("#stage-tab-0").get_attribute("aria-selected") == "true")
            for index in range(3):
                page.locator(f"#code-tab-{index}").click()
                check(f"site {width}px: code {index}", page.locator("#code-panel").get_attribute("aria-labelledby") == f"code-tab-{index}")
            # 模拟剪贴板失败分支，避免依赖宿主系统剪贴板。
            page.evaluate("Object.defineProperty(navigator, 'clipboard', {configurable: true, value: {writeText: () => Promise.reject(new Error('fixture'))}})")
            page.locator("#copy-code").click()
            page.wait_for_function("document.getElementById('copy-code').textContent === '手动复制'")
            check(f"site {width}px: clipboard fallback", "手动复制" in page.locator("#copy-status").inner_text())
            for figure in page.locator("[data-zoom]").all():
                figure.click()
                page.wait_for_function("document.getElementById('dialog-image').complete && document.getElementById('dialog-image').naturalWidth > 0")
                check(f"site {width}px: image dialog", page.locator("#figure-dialog").evaluate("e => e.open"))
                page.keyboard.press("Escape")
                check(f"site {width}px: dialog closes", not page.locator("#figure-dialog").evaluate("e => e.open"))
            if width in (1440, 390):
                page.locator("#stage-tab-0").click(); page.locator("#code-tab-0").click()
                page.evaluate("document.activeElement.blur(); scrollTo(0, 0)")
                page.wait_for_timeout(150)
                page.screenshot(path=str(output / f"site-{width}.png"), full_page=True)
            check(f"site {width}px: no JavaScript errors", not errors)
            context.close()

        for width in (1440, 390, 320):
            context = browser.new_context(viewport={"width": width, "height": 1000}, device_scale_factor=1)
            page = context.new_page(); errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            load_local_page(page, site=False, fixtures={"info": info, "status": status, "result": result})
            layout(page, f"service {width}px")
            check(f"service {width}px: null metrics render", page.locator("#metricPrauc").inner_text() == "--")
            check(f"service {width}px: sample count from API", "12" in page.locator("#evidenceCount").inner_text())
            check(f"service {width}px: API upload limit", "5 MiB" in page.locator("#uploadLimit").inner_text())
            check(f"service {width}px: HTML evidence escaped", page.locator("#limitationsList img").count() == 0)
            check(f"service {width}px: no-GPU state", "未获得" in page.locator("#gpuList").inner_text())
            page.evaluate("data => { renderStatus(data); renderStatus(data); }", status)
            check(f"service {width}px: polls are not latency samples", page.evaluate("state.latency.length") == 1)
            page.locator("#imageInput").set_input_files(str(ROOT / "data/smoke/giraffes.jpg"))
            page.locator("#policySelect").select_option("custom")
            page.locator("#policyText").fill("CUSTOM_RULE_MUST_NOT_LEAK")
            page.locator("#policySelect").select_option("pilot-explicit-v1")
            page.locator("#submitButton").click()
            page.wait_for_selector("#resultContent:not(.hidden)")
            check(f"service {width}px: old custom policy cleared", page.evaluate("window.__fixtureRequests.length === 1 && window.__fixtureRequests[0] === ''"))
            check(f"service {width}px: binary verdict stays separate", page.locator("#decisionText").inner_text() == "未判违规")
            check(f"service {width}px: review suggestion separately shown", "Review" in page.locator(".head-row.selected").inner_text())
            layout(page, f"service result {width}px")
            if width in (1440, 390):
                page.evaluate("document.activeElement.blur(); scrollTo(0, 0)")
                page.wait_for_timeout(150)
                page.screenshot(path=str(output / f"service-fixture-{width}.png"), full_page=True)
            page.evaluate("renderModelInfo({test_metrics: {}, test_metrics_note: '未提供评测文件', limitations: []})")
            check(f"service {width}px: missing metrics state", page.locator("#metricPrauc").inner_text() == "--" and "未提供" in page.locator("#evidenceNote").inner_text())
            gpu_status = {**status, "gpus": [{"index": 2, "name": "GPU <b>fixture</b>", "memory_used_mb": 1024., "memory_total_mb": 8192., "utilization_percent": 3., "temperature_c": 40., "power_w": 10.}]}
            page.evaluate("renderStatus", gpu_status)
            check(f"service {width}px: actual GPU index", "GPU 2" in page.locator("#gpuMemoryTitle").inner_text())
            check(f"service {width}px: GPU name escaped", page.locator("#gpuList b").count() == 0)
            page.locator("#imageInput").set_input_files({"name": "bad.txt", "mimeType": "text/plain", "buffer": b"fixture"})
            check(f"service {width}px: invalid upload cleared", page.locator("#imageInput").evaluate("e => e.files.length") == 0)
            check(f"service {width}px: no JavaScript errors", not errors)
            context.close()
        browser.close()
    report = {"mode": "Offline DOM/JavaScript checks with mocked API; no HTTP server, model inference or GPU test", "browser": browser_version, "passed_checks": len(checks), "checks": checks}
    (output / "browser-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: {len(checks)} browser checks. Screenshots and report: {output}")


if __name__ == "__main__":
    main()
