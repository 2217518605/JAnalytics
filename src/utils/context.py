"""轻量运行时上下文 — 替代 coze_coding_utils.runtime_ctx.context"""
import uuid
import threading


class Context:
    """替代 coze_coding_utils 的 Context"""

    def __init__(self, method: str = "", headers: dict = None):
        self.run_id = uuid.uuid4().hex
        self.method = method
        self.headers = headers or {}


def new_context(method: str = "", headers: dict = None) -> Context:
    """创建新的运行时上下文，替代 coze 的 new_context"""
    return Context(method=method, headers=headers)


# 线程本地存储，替代 coze 的 request_context
request_context = threading.local()
