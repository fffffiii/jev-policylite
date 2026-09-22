from __future__ import annotations

import json
import os
import statistics
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import psutil
import torch


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@dataclass
class RequestRecord:
    request_id: str
    timestamp: str
    policy_id: str
    decision: str
    score: float
    latency_ms: float
    inference_ms: float


class ServiceMetrics:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.model_loaded_at: float | None = None
        self.model_load_ms: float | None = None
        self.model_warmup_ms: float | None = None
        self.total_requests = 0
        self.failed_requests = 0
        self.active_requests = 0
        self.queued_requests = 0
        self.latencies: deque[float] = deque(maxlen=500)
        self.inference_times: deque[float] = deque(maxlen=500)
        self.request_times: deque[float] = deque(maxlen=1000)
        self.recent: deque[RequestRecord] = deque(maxlen=20)
        self._lock = threading.Lock()
        self._gpu_cache: tuple[float, list[dict[str, Any]]] = (0.0, [])

    def begin(self) -> None:
        with self._lock:
            self.active_requests += 1

    def queued(self, delta: int) -> None:
        with self._lock:
            self.queued_requests = max(0, self.queued_requests + delta)

    def finish(self, record: RequestRecord) -> None:
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
            self.total_requests += 1
            self.latencies.append(record.latency_ms)
            self.inference_times.append(record.inference_ms)
            self.request_times.append(time.time())
            self.recent.appendleft(record)

    def fail(self) -> None:
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
            self.total_requests += 1
            self.failed_requests += 1
            self.request_times.append(time.time())

    def gpu_status(self) -> list[dict[str, Any]]:
        cached_at, cached = self._gpu_cache
        if time.time() - cached_at < 1.0:
            return cached
        fields = "index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit"
        try:
            output = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout
            result = []
            for line in output.splitlines():
                parts = [part.strip() for part in line.split(",")]
                result.append(
                    {
                        "index": int(parts[0]), "name": parts[1],
                        "memory_used_mb": float(parts[2]), "memory_total_mb": float(parts[3]),
                        "utilization_percent": float(parts[4]), "temperature_c": float(parts[5]),
                        "power_w": float(parts[6]), "power_limit_w": float(parts[7]),
                    }
                )
            self._gpu_cache = (time.time(), result)
            return result
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return []

    def snapshot(self) -> dict[str, Any]:
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()
        system_memory = psutil.virtual_memory()
        with self._lock:
            latencies = list(self.latencies)
            inference = list(self.inference_times)
            cutoff = time.time() - 60
            rpm = sum(timestamp >= cutoff for timestamp in self.request_times)
            recent = [asdict(item) for item in self.recent]
            counters = {
                "total_requests": self.total_requests,
                "failed_requests": self.failed_requests,
                "active_requests": self.active_requests,
                "queued_requests": self.queued_requests,
            }
        cuda = {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_allocated_mb": 0.0}
        if torch.cuda.is_available():
            cuda = {
                "allocated_mb": torch.cuda.memory_allocated() / 2**20,
                "reserved_mb": torch.cuda.memory_reserved() / 2**20,
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            }
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": time.time() - self.started_at,
            "model_ready": self.model_loaded_at is not None,
            "model_load_ms": self.model_load_ms,
            "model_warmup_ms": self.model_warmup_ms,
            **counters,
            "requests_per_minute": rpm,
            "latency_ms": {
                "average": statistics.fmean(latencies) if latencies else None,
                "p50": percentile(latencies, 0.5),
                "p95": percentile(latencies, 0.95),
                "latest": latencies[-1] if latencies else None,
            },
            "inference_ms": {
                "average": statistics.fmean(inference) if inference else None,
                "p50": percentile(inference, 0.5),
                "p95": percentile(inference, 0.95),
                "latest": inference[-1] if inference else None,
            },
            "process": {
                "rss_mb": process_memory.rss / 2**20,
                "vms_mb": process_memory.vms / 2**20,
                "cpu_percent": process.cpu_percent(interval=None),
                "threads": process.num_threads(),
            },
            "system": {
                "memory_used_gb": system_memory.used / 2**30,
                "memory_total_gb": system_memory.total / 2**30,
                "memory_percent": system_memory.percent,
                "cpu_percent": psutil.cpu_percent(interval=None),
            },
            "cuda": cuda,
            "gpus": self.gpu_status(),
            "recent_requests": recent,
        }


def log_event(level: str, event: str, **fields: Any) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)
