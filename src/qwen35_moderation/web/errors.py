from __future__ import annotations


class AppError(Exception):
    def __init__(self, title: str, status: int, detail: str, code: str):
        super().__init__(detail)
        self.title = title
        self.status = status
        self.detail = detail
        self.code = code


class InputError(AppError):
    def __init__(self, detail: str, code: str = "invalid-input"):
        super().__init__("输入无效", 422, detail, code)


class BusyError(AppError):
    def __init__(self):
        super().__init__("请求过多", 429, "推理队列已满，请稍后重试。", "queue-full")
