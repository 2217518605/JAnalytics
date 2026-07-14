"""标准日志配置 — 替代 coze_coding_utils.log.*"""
import logging
import sys

LOG_LEVEL = logging.INFO
LOG_FILE = None


def setup_logging(
    log_file: str = None,
    max_bytes: int = 100 * 1024 * 1024,
    backup_count: int = 5,
    log_level: int = logging.INFO,
    use_json_format: bool = False,
    console_output: bool = True,
):
    """配置标准 Python logging，兼容 coze 的 setup_logging 接口"""
    global LOG_LEVEL, LOG_FILE
    LOG_LEVEL = log_level
    LOG_FILE = log_file

    root = logging.getLogger()
    root.setLevel(log_level)

    # 清除已有 handler，避免重复
    root.handlers.clear()

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    if console_output:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(log_level)
        console.setFormatter(formatter)
        root.addHandler(console)

    if log_file:
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
