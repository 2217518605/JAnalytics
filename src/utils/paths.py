"""项目路径 — 替代 COZE_WORKSPACE_PATH"""
import os

PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# 兼容旧代码中直接读环境变量的场景
os.environ.setdefault("COZE_WORKSPACE_PATH", PROJECT_ROOT)
