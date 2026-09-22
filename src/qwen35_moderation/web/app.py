from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .config import Settings
from .errors import AppError, BusyError, InputError
from .image_io import decode_image
from .inference import POLICIES, ModerationEngine
from .monitoring import RequestRecord, ServiceMetrics, log_event

settings = Settings.from_env()
metrics = ServiceMetrics()
engine: ModerationEngine | None = None
inference_lock = asyncio.Lock()
queue_guard = asyncio.Lock()
rate_windows: dict[str, deque[float]] = defaultdict(deque)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    settings.validate_files()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # FLA 在导入内核时探测当前设备，先建立 CUDA 上下文避免误判为 CPU。
        torch.cuda.init()
    log_event("info", "model_loading", checkpoint=str(settings.checkpoint), device=str(device))
    # Triton/FLA 必须在服务主线程初始化，否则会错误识别为 CPU 后端。
    engine = ModerationEngine(settings.checkpoint, settings.calibration_file, device)
    warmup_started = time.perf_counter()
    engine.moderate(
        Image.new("RGB", (500, 374), (112, 122, 105)), "", "pilot-suggestive-v1", "", "balanced"
    )
    metrics.model_warmup_ms = (time.perf_counter() - warmup_started) * 1000
    metrics.model_load_ms = engine.load_ms
    metrics.model_loaded_at = time.time()
    log_event(
        "info", "model_ready", load_ms=engine.load_ms,
        warmup_ms=metrics.model_warmup_ms, device=str(device),
    )
    yield
    log_event("info", "service_stopping")


app = FastAPI(
    title="Qwen3.5 审核实验台",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)
app.mount("/assets", StaticFiles(directory=settings.static_dir), name="assets")


def problem(request: Request, title: str, status: int, detail: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        media_type="application/problem+json",
        content={
            "type": f"/errors/{code}", "title": title, "status": status,
            "detail": detail, "instance": request.url.path,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))[:100]
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; connect-src 'self'"
    )
    log_event(
        "info", "http_request", request_id=request_id, method=request.method,
        path=request.url.path, status=response.status_code,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    log_event("warn", "application_error", request_id=request.state.request_id, code=exc.code)
    return problem(request, exc.title, exc.status, exc.detail, exc.code)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, _: RequestValidationError):
    return problem(request, "请求格式无效", 422, "请检查表单字段和文件。", "validation-error")


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    log_event(
        "error", "unexpected_error", request_id=request.state.request_id,
        error_type=type(exc).__name__, detail=str(exc)[:500],
    )
    return problem(request, "服务内部错误", 500, "推理失败，请使用请求 ID 查询日志。", "internal-error")


def check_rate_limit(client: str) -> tuple[int, int]:
    now = time.time()
    window = rate_windows[client]
    while window and window[0] < now - 60:
        window.popleft()
    if len(window) >= settings.requests_per_minute:
        raise AppError(
            "请求过多", 429,
            f"每个客户端每分钟最多提交 {settings.requests_per_minute} 次。",
            "rate-limit",
        )
    window.append(now)
    return settings.requests_per_minute, settings.requests_per_minute - len(window)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(settings.static_dir / "index.html")


@app.get("/downloads/jev-policylite-mnn-q4-linux-x86_64.tar.gz", include_in_schema=False)
async def download_edge_bundle(request: Request):
    if settings.edge_bundle is None or not settings.edge_bundle.is_file():
        return problem(request, "文件不存在", 404, "端侧离线包尚未就绪", "not-found")
    return FileResponse(
        settings.edge_bundle,
        filename="jev-policylite-mnn-q4-linux-x86_64.tar.gz",
        media_type="application/gzip",
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> JSONResponse:
    is_ready = engine is not None and metrics.model_loaded_at is not None
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={"status": "ready" if is_ready else "loading", "model_ready": is_ready},
    )


@app.get("/api/v1/status")
async def status() -> dict[str, Any]:
    snapshot = metrics.snapshot()
    snapshot["model"] = {
        "name": engine.metadata["base_model"] if engine else "loading",
        "checkpoint": settings.checkpoint.name,
        "device": str(engine.device) if engine else None,
    }
    return snapshot


@app.get("/api/v1/model")
async def model_info() -> dict[str, Any]:
    test_metrics = json.loads(settings.test_metrics_file.read_text(encoding="utf-8"))
    return {
        "base_model": engine.metadata["base_model"] if engine else "loading",
        "checkpoint": settings.checkpoint.name,
        "policies": POLICIES,
        "test_metrics": test_metrics,
        "calibration": engine.calibration if engine else {},
        "limitations": [
            "试训数据只覆盖 L1-L4 性感、裸露与色情分级。",
            "自定义规则、医疗、艺术、日常安全图片和图文冲突尚未专项验证。",
            "校准集目标误报率为 1%，独立测试集实际误报率为 2.89%。",
        ],
    }


@app.post("/api/v1/moderations")
async def create_moderation(
    request: Request,
    image: UploadFile = File(...),
    text: str = Form(""),
    policy_id: str = Form("pilot-suggestive-v1"),
    policy_text: str = Form(""),
    threshold_mode: str = Form("balanced"),
) -> JSONResponse:
    client = request.client.host if request.client else "unknown"
    limit, remaining = check_rate_limit(client)
    if threshold_mode not in {"balanced", "low_fpr"}:
        raise InputError("threshold_mode 只能是 balanced 或 low_fpr。")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", policy_id):
        raise InputError("policy_id 格式无效。", "invalid-policy-id")
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise InputError("仅支持 JPEG、PNG 和 WebP 图片。", "unsupported-media-type")
    raw = await image.read(settings.max_upload_bytes + 1)
    if not raw:
        raise InputError("图片不能为空。", "empty-image")
    if len(raw) > settings.max_upload_bytes:
        raise AppError("文件过大", 413, "图片不能超过 10 MB。", "upload-too-large")
    if engine is None:
        raise AppError("模型未就绪", 503, "模型仍在加载。", "model-not-ready")
    picture = await asyncio.to_thread(decode_image, raw)
    async with queue_guard:
        if metrics.queued_requests >= 8:
            raise BusyError()
        metrics.queued(1)
    metrics.begin()
    request_started = time.perf_counter()
    try:
        async with inference_lock:
            metrics.queued(-1)
            # 单 worker 串行执行 GPU 推理，避免跨线程 CUDA 上下文和重复模型实例。
            result = engine.moderate(picture, text, policy_id, policy_text, threshold_mode)
        total_ms = (time.perf_counter() - request_started) * 1000
        result["total_ms"] = total_ms
        result["request_id"] = request.state.request_id
        metrics.finish(
            RequestRecord(
                request_id=request.state.request_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                policy_id=policy_id,
                decision=result["decision"],
                score=result["violation_score"],
                latency_ms=total_ms,
                inference_ms=result["inference_ms"],
            )
        )
    except Exception:
        metrics.queued(-1)
        metrics.fail()
        raise
    response = JSONResponse(content=result)
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    return response
