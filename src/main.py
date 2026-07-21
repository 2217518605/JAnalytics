import argparse
import asyncio
import glob
import inspect
import importlib
import json
import os
import sys
import threading
import traceback
import logging
import uuid

# Windows: psycopg 异步模式需要 SelectorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterable, AsyncIterable, AsyncGenerator, Optional
import uvicorn
import time
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from storage.database.db import get_session, get_engine
from storage.memory.memory_saver import get_memory_saver
from storage.database.shared.model import Base
from sqlalchemy import event, func
from datetime import datetime, date

from utils.context import new_context, Context, request_context
from utils.paths import PROJECT_ROOT
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# ---- 内联 graph_helper (替代 coze_coding_utils.helper.graph_helper) ----
def _is_dev_env() -> bool:
    return os.getenv("COZE_PROJECT_ENV", "") == "DEV"

def _is_agent_proj() -> bool:
    return os.getenv("COZE_PROJECT_TYPE", "workflow") == "agent"

def _get_graph_instance(module_name: str):
    module = importlib.import_module(module_name)
    for _, obj in inspect.getmembers(module):
        if isinstance(obj, CompiledStateGraph):
            return obj
    return None

def _get_agent_instance(module_name: str, ctx=None):
    module = importlib.import_module(module_name)
    return module.build_agent(ctx)

def _get_graph_node_func_with_inout(graph, node_name: str):
    for node_id, node in graph.nodes.items():
        if node_id == START or node_id == END:
            continue
        if node.data:
            _func = node.data.func
            if _func.__name__ != node_name:
                continue
            sig = inspect.signature(_func)
            params = list(sig.parameters.values())
            input_cls = params[0].annotation if params else None
            return _func, input_cls, None
    return None, None, None

# ---- 内联 _extract_core_stack ----
def _extract_core_stack() -> str:
    return traceback.format_exc()

# ---- 内联 init_run_config / init_agent_config ----
def _init_run_config(graph: CompiledStateGraph, ctx: Context) -> RunnableConfig:
    return RunnableConfig(
        run_id=ctx.run_id,
        configurable={"thread_id": ctx.run_id},
    )

def _init_agent_config(graph: CompiledStateGraph, ctx: Context) -> RunnableConfig:
    return RunnableConfig(
        run_id=ctx.run_id,
        configurable={"thread_id": ctx.run_id},
    )

# ---- 内联 _LangGraphParser ----
class _LangGraphParser:
    def __init__(self, graph: CompiledStateGraph):
        self._graph = graph
    def get_node_metadata(self, node_id: str):
        return {}

# ---- 内联 ErrorClassifier ----
from dataclasses import dataclass, field
from enum import Enum, auto

class ErrorCategory(Enum):
    UNKNOWN = auto()
    VALIDATION = auto()
    TIMEOUT = auto()
    LLM_ERROR = auto()
    INTERNAL = auto()

@dataclass
class ClassifiedError:
    code: str = "UNKNOWN"
    message: str = ""
    category: ErrorCategory = ErrorCategory.UNKNOWN

class ErrorClassifier:
    def classify(self, e: Exception, context: dict = None) -> ClassifiedError:
        return ClassifiedError(code=type(e).__name__, message=str(e), category=ErrorCategory.INTERNAL)
    def get_error_response(self, e: Exception, context: dict = None) -> dict:
        return {"error_code": type(e).__name__, "error_message": str(e), "category": "INTERNAL"}

# ---- 内联 RunOpt ----
@dataclass
class RunOpt:
    workflow_debug: bool = False

# ---- 内联 StreamRunner ----
class _StreamRunner:
    def __init__(self):
        pass

    def stream(self, payload: dict, graph: CompiledStateGraph, run_config: RunnableConfig, ctx: Context):
        for chunk in graph.stream(payload, run_config):
            yield chunk

    async def astream(self, payload: dict, graph: CompiledStateGraph, run_config: RunnableConfig, ctx: Context, run_opt: RunOpt = None):
        async for chunk in graph.astream(payload, run_config):
            yield chunk

# ---- 内联 stream handlers ----
async def _agent_stream_handler(payload, ctx, run_id, stream_sse_func, sse_event_func, error_classifier, register_task_func):
    task = asyncio.current_task()
    if task:
        register_task_func(run_id, task)
    try:
        async for chunk in stream_sse_func(payload, ctx, RunOpt()):
            yield chunk
    except Exception as e:
        err = error_classifier.get_error_response(e, {"run_id": run_id})
        yield sse_event_func({"error": err})

async def _workflow_stream_handler(payload, ctx, run_id, stream_sse_func, sse_event_func, error_classifier, register_task_func, run_opt=None):
    task = asyncio.current_task()
    if task:
        register_task_func(run_id, task)
    try:
        async for chunk in stream_sse_func(payload, ctx, run_opt or RunOpt()):
            yield chunk
    except Exception as e:
        err = error_classifier.get_error_response(e, {"run_id": run_id})
        yield sse_event_func({"error": err})

# ---- 内联 _OpenAIChatHandler ----
class _OpenAIChatHandler:
    def __init__(self, service):
        self._service = service
    async def handle(self, payload: dict, ctx: Context):
        messages = payload.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="messages is required")
        last_msg = messages[-1].get("content", "") if isinstance(messages[-1], dict) else str(messages[-1])
        return await self._service.run({"text": last_msg}, ctx)

# ---- 内联 async task 基础设施 ----
_ASYNC_HEADER_X_RUN_ID = "x-run-id"
_ASYNC_RECURSION_LIMIT = 100

def _parse_deadline_sec(headers: dict) -> int:
    """从请求头解析超时时间(秒)，默认 900"""
    try:
        val = headers.get("x-deadline", "900")
        return int(val)
    except (ValueError, TypeError):
        return 900

def _extract_biz_context(headers: dict) -> dict:
    """从请求头提取业务上下文"""
    ctx = {}
    for key in ("x-tenant-id", "x-user-id", "x-trace-id"):
        val = headers.get(key)
        if val:
            ctx[key] = val
    return ctx

class _AsyncTaskStorageError(Exception):
    pass

class _AsyncTaskRuntime:
    """最小化异步任务管理器（内存存储）"""
    def __init__(self, session_factory=None, engine=None, graph=None, checkpointer=None):
        self._graph = graph
        self._checkpointer = checkpointer
        self._tasks: dict = {}

    async def submit(self, task_id: str, payload: dict, biz_context: dict = None,
                     deadline_sec: int = 900, run_config: RunnableConfig = None, ctx: Context = None) -> dict:
        self._tasks[task_id] = {"status": "queued", "task_id": task_id, "payload": payload}
        return {"status": "queued", "task_id": task_id}

    async def get(self, task_id: str) -> dict:
        return self._tasks.get(task_id)

    async def shutdown(self):
        self._tasks.clear()

# ---- 日志配置 ----
setup_logging(log_level=logging.INFO, console_output=True)

WORKSPACE_DIR = PROJECT_ROOT
# 兼容旧代码: 确保 COZE_WORKSPACE_PATH 也被设置
os.environ.setdefault("COZE_WORKSPACE_PATH", PROJECT_ROOT)


# 超时配置常量
TIMEOUT_SECONDS = 900  # 15分钟

class GraphService:
    def __init__(self):
        # 用于跟踪正在运行的任务（使用asyncio.Task）
        self.running_tasks: Dict[str, asyncio.Task] = {}
        # 错误分类器
        self.error_classifier = ErrorClassifier()
        # stream runner
        self._agent_stream_runner = _StreamRunner()
        self._workflow_stream_runner = _StreamRunner()
        self._graph = None
        self._graph_lock = threading.Lock()

    def set_graph(self, graph) -> None:
        """Inject the compiled graph used by sync endpoints. Called once from
        lifespan with a no-checkpointer build, so /run /stream_run /node_run
        never hit the checkpoint DB."""
        self._graph = graph

    def _get_graph(self, ctx=Context):
        if self._graph is not None:
            return self._graph
        with self._graph_lock:
            if self._graph is not None:
                return self._graph
            if _is_agent_proj():
                self._graph = _get_agent_instance("agents.agent", ctx)
            else:
                self._graph = _get_graph_instance("graphs.graph")
            return self._graph

    @staticmethod
    def _sse_event(data: Any, event_id: Any = None) -> str:
        id_line = f"id: {event_id}\n" if event_id else ""
        return f"{id_line}event: message\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

    def _get_stream_runner(self):
        if _is_agent_proj():
            return self._agent_stream_runner
        else:
            return self._workflow_stream_runner

    # 流式运行（原始迭代器）：本地调用使用
    def stream(self, payload: Dict[str, Any], run_config: RunnableConfig, ctx=Context) -> Iterable[Any]:
        graph = self._get_graph(ctx)
        stream_runner = self._get_stream_runner()
        for chunk in stream_runner.stream(payload, graph, run_config, ctx):
            yield chunk

    # 同步运行：本地/HTTP 通用
    async def run(self, payload: Dict[str, Any], ctx=None) -> Dict[str, Any]:
        if ctx is None:
            ctx = new_context("run")

        run_id = ctx.run_id
        logger.info(f"Starting run with run_id: {run_id}")

        try:
            graph = self._get_graph(ctx)
            # custom tracer
            run_config = _init_run_config(graph, ctx)
            run_config.setdefault("configurable", {})["thread_id"] = ctx.run_id

            # 直接调用，LangGraph会在当前任务上下文中执行
            # 如果当前任务被取消，LangGraph的执行也会被取消
            return await graph.ainvoke(payload, config=run_config, context=ctx)

        except asyncio.CancelledError:
            logger.info(f"Run {run_id} was cancelled")
            return {"status": "cancelled", "run_id": run_id, "message": "Execution was cancelled"}
        except Exception as e:
            # 使用错误分类器分类错误
            err = self.error_classifier.classify(e, {"node_name": "run", "run_id": run_id})
            # 记录详细的错误信息和堆栈跟踪
            logger.error(
                f"Error in GraphService.run: [{err.code}] {err.message}\n"
                f"Category: {err.category.name}\n"
                f"Traceback:\n{_extract_core_stack()}"
            )
            # 保留原始异常堆栈，便于上层返回真正的报错位置
            raise
        finally:
            # 清理任务记录
            self.running_tasks.pop(run_id, None)

    # 流式运行（SSE 格式化）：HTTP 路由使用
    async def stream_sse(self, payload: Dict[str, Any], ctx=None, run_opt: Optional[RunOpt] = None) -> AsyncGenerator[str, None]:
        if ctx is None:
            ctx = new_context(method="stream_sse")
        if run_opt is None:
            run_opt = RunOpt()

        run_id = ctx.run_id
        logger.info(f"Starting stream with run_id: {run_id}")
        graph = self._get_graph(ctx)
        if _is_agent_proj():
            run_config = _init_agent_config(graph, ctx)
        else:
            run_config = _init_run_config(graph, ctx)  # vibeflow

        is_workflow = not _is_agent_proj()

        try:
            async for chunk in self.astream(payload, graph, run_config=run_config, ctx=ctx, run_opt=run_opt):
                if is_workflow and isinstance(chunk, tuple):
                    event_id, data = chunk
                    yield self._sse_event(data, event_id)
                else:
                    yield self._sse_event(chunk)
        finally:
            # 清理任务记录
            self.running_tasks.pop(run_id, None)
            pass  # cozeloop.flush() removed

    # 取消执行 - 使用asyncio的标准方式
    def cancel_run(self, run_id: str, ctx: Optional[Context] = None) -> Dict[str, Any]:
        """
        取消指定run_id的执行

        使用asyncio.Task.cancel()来取消任务,这是标准的Python异步取消机制。
        LangGraph会在节点之间检查CancelledError,实现优雅的取消。
        """
        logger.info(f"Attempting to cancel run_id: {run_id}")

        # 查找对应的任务
        if run_id in self.running_tasks:
            task = self.running_tasks[run_id]
            if not task.done():
                # 使用asyncio的标准取消机制
                # 这会在下一个await点抛出CancelledError
                task.cancel()
                logger.info(f"Cancellation requested for run_id: {run_id}")
                return {
                    "status": "success",
                    "run_id": run_id,
                    "message": "Cancellation signal sent, task will be cancelled at next await point"
                }
            else:
                logger.info(f"Task already completed for run_id: {run_id}")
                return {
                    "status": "already_completed",
                    "run_id": run_id,
                    "message": "Task has already completed"
                }
        else:
            logger.warning(f"No active task found for run_id: {run_id}")
            return {
                "status": "not_found",
                "run_id": run_id,
                "message": "No active task found with this run_id. Task may have already completed or run_id is invalid."
            }

    # 运行指定节点：本地/HTTP 通用
    async def run_node(self, node_id: str, payload: Dict[str, Any], ctx=None) -> Any:
        if ctx is None or Context.run_id == "":
            ctx = new_context(method="node_run")

        _graph = self._get_graph()
        node_func, input_cls, output_cls = _get_graph_node_func_with_inout(_graph.get_graph(), node_id)
        if node_func is None or input_cls is None:
            raise KeyError(f"node_id '{node_id}' not found")

        parser = _LangGraphParser(_graph)
        metadata = parser.get_node_metadata(node_id) or {}

        _g = StateGraph(input_cls, input_schema=input_cls, output_schema=output_cls)
        _g.add_node("sn", node_func, metadata=metadata)
        _g.set_entry_point("sn")
        _g.add_edge("sn", END)
        _graph = _g.compile()

        run_config = _init_run_config(_graph, ctx)
        return await _graph.ainvoke(payload, config=run_config)

    def graph_inout_schema(self) -> Any:
        if _is_agent_proj():
            return {"input_schema": {}, "output_schema": {}}
        builder = getattr(self._get_graph(), 'builder', None)
        if builder is not None:
            input_cls = getattr(builder, 'input_schema', None) or self.graph.get_input_schema()
            output_cls = getattr(builder, 'output_schema', None) or self.graph.get_output_schema()
        else:
            logger.warning(f"No builder input schema found for graph_inout_schema, using graph input schema instead")
            input_cls = self.graph.get_input_schema()
            output_cls = self.graph.get_output_schema()

        return {
            "input_schema": input_cls.model_json_schema(), 
            "output_schema": output_cls.model_json_schema(),
            "code":0,
            "msg":""
        }

    async def astream(self, payload: Dict[str, Any], graph: CompiledStateGraph, run_config: RunnableConfig, ctx=Context, run_opt: Optional[RunOpt] = None) -> AsyncIterable[Any]:
        stream_runner = self._get_stream_runner()
        async for chunk in stream_runner.astream(payload, graph, run_config, ctx, run_opt):
            yield chunk


service = GraphService()

async_runtime: Optional[_AsyncTaskRuntime] = None
async_graph: Optional[CompiledStateGraph] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_engine()
    @event.listens_for(engine, "connect")
    def _set_utc(dbapi_conn, _):
        with dbapi_conn.cursor() as cur:
            cur.execute("SET TIME ZONE 'UTC'")
    checkpointer = get_memory_saver()
    if _is_agent_proj():
        base = _get_agent_instance("agents.agent", None)
        sync_graph = base.builder.compile(checkpointer=checkpointer)
    else:
        base = _get_graph_instance("graphs.graph")
        sync_graph = base.builder.compile()
    global async_graph, async_runtime
    async_graph = base.builder.compile(checkpointer=checkpointer)
    service.set_graph(sync_graph)
    # 创建电商数据表
    try:
        EcomBase.metadata.create_all(bind=engine)
        logger.info("电商智能助手数据表创建完成")
    except Exception as e:
        logger.warning(f"电商表创建(可能已存在): {e}")

    async_runtime = _AsyncTaskRuntime(
        session_factory=get_session, engine=engine,
        graph=async_graph, checkpointer=checkpointer,
    )
    yield
    if async_runtime is not None:
        await async_runtime.shutdown()

app = FastAPI(lifespan=lifespan)

# CORS - 允许浏览器跨域请求
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 用户身份中间件 ----
from contextvars import ContextVar

_current_user_id: ContextVar[int] = ContextVar("current_user_id", default=0)


def _uid() -> int:
    """获取当前请求的用户 ID"""
    return _current_user_id.get()


@app.middleware("http")
async def _user_id_middleware(request: Request, call_next):
    uid = request.headers.get("x-user-id", "0")
    try:
        uid_int = int(uid)
    except (ValueError, TypeError):
        uid_int = 0
    _current_user_id.set(uid_int)
    return await call_next(request)

# OpenAI 兼容接口处理器
openai_handler = _OpenAIChatHandler(service)


@app.post("/async_run")
async def http_async_run(request: Request) -> dict:
    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in http_async_run: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {_extract_core_stack()}")
    try:
        deadline_sec = _parse_deadline_sec(request.headers)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 一个 ID 走到底：task_id == run_id == thread_id == ctx.run_id == coze_run_id。
    # 优先用上游 x-run-id；没传就生成 UUID。
    run_id = request.headers.get(_ASYNC_HEADER_X_RUN_ID) or uuid.uuid4().hex

    # ctx 在 handler scope 构造，与同步 /run 路径一致；后面 new_context 默认会
    # 给 run_id 一个新 UUID，同步路径也是显式覆盖（main.py /run 处），这里同理。
    ctx = new_context(method="async_run", headers=request.headers)
    ctx.run_id = run_id
    request_context.set(ctx)  # 与其他 HTTP endpoint 一致：让日志组件拿到 run_id 等信息
    run_config = _init_run_config(async_graph, ctx)
    run_config["recursion_limit"] = _ASYNC_RECURSION_LIMIT
    run_config.setdefault("configurable", {})["thread_id"] = run_id

    biz_context = _extract_biz_context(request.headers) or {}
    biz_context[_ASYNC_HEADER_X_RUN_ID] = run_id  # 也留 DB 一份方便审计/排查

    try:
        return await async_runtime.submit(
            task_id=run_id,
            payload=payload,
            biz_context=biz_context,
            deadline_sec=deadline_sec,
            run_config=run_config,
            ctx=ctx,
        )
    except _AsyncTaskStorageError as e:
        raise HTTPException(status_code=503,
                            detail=f"async-task storage unavailable: {e}")


@app.get("/task/{task_id}")
async def http_get_task(task_id: str) -> dict:
    try:
        row = await async_runtime.get(task_id)
    except _AsyncTaskStorageError as e:
        raise HTTPException(status_code=503,
                            detail=f"async-task storage unavailable: {e}")
    if row is None:
        raise HTTPException(status_code=404, detail="task not found")
    return row


HEADER_X_RUN_ID = "x-run-id"
@app.post("/run")
async def http_run(request: Request) -> Dict[str, Any]:
    global result
    raw_body = await request.body()
    try:
        body_text = raw_body.decode("utf-8")
    except Exception as e:
        body_text = str(raw_body)
        raise HTTPException(status_code=400,
                            detail=f"Invalid JSON format: {body_text}, traceback: {traceback.format_exc()}, error: {e}")

    ctx = new_context(method="run", headers=request.headers)
    # 优先使用上游指定的 run_id，保证 cancel 能精确匹配
    upstream_run_id = request.headers.get(HEADER_X_RUN_ID)
    if upstream_run_id:
        ctx.run_id = upstream_run_id
    run_id = ctx.run_id
    request_context.set(ctx)

    logger.info(
        f"Received request for /run: "
        f"run_id={run_id}, "
        f"query={dict(request.query_params)}, "
        f"body={body_text}"
    )

    try:
        payload = await request.json()

        # 创建任务并记录 - 这是关键，让我们可以通过run_id取消任务
        task = asyncio.create_task(service.run(payload, ctx))
        service.running_tasks[run_id] = task

        try:
            result = await asyncio.wait_for(task, timeout=float(TIMEOUT_SECONDS))
        except asyncio.TimeoutError:
            logger.error(f"Run execution timeout after {TIMEOUT_SECONDS}s for run_id: {run_id}")
            task.cancel()
            try:
                result = await task
            except asyncio.CancelledError:
                return {
                    "status": "timeout",
                    "run_id": run_id,
                    "message": f"Execution timeout: exceeded {TIMEOUT_SECONDS} seconds"
                }

        if not result:
            result = {}
        if isinstance(result, dict):
            result["run_id"] = run_id
        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in http_run: {e}, traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format, {_extract_core_stack()}")

    except asyncio.CancelledError:
        logger.info(f"Request cancelled for run_id: {run_id}")
        result = {"status": "cancelled", "run_id": run_id, "message": "Execution was cancelled"}
        return result

    except Exception as e:
        # 使用错误分类器获取错误信息
        error_response = service.error_classifier.get_error_response(e, {"node_name": "http_run", "run_id": run_id})
        logger.error(
            f"Unexpected error in http_run: [{error_response['error_code']}] {error_response['error_message']}, "
            f"traceback: {traceback.format_exc()}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": error_response["error_code"],
                "error_message": error_response["error_message"],
                "stack_trace": _extract_core_stack(),
            }
        )
    finally:
        _flush()


HEADER_X_WORKFLOW_STREAM_MODE = "x-workflow-stream-mode"


def _register_task(run_id: str, task: asyncio.Task):
    service.running_tasks[run_id] = task


@app.post("/stream_run")
async def http_stream_run(request: Request):
    ctx = new_context(method="stream_run", headers=request.headers)
    # 优先使用上游指定的 run_id，保证 cancel 能精确匹配
    upstream_run_id = request.headers.get(HEADER_X_RUN_ID)
    if upstream_run_id:
        ctx.run_id = upstream_run_id
    workflow_stream_mode = request.headers.get(HEADER_X_WORKFLOW_STREAM_MODE, "").lower()
    workflow_debug = workflow_stream_mode == "debug"
    request_context.set(ctx)
    raw_body = await request.body()
    try:
        body_text = raw_body.decode("utf-8")
    except Exception as e:
        body_text = str(raw_body)
        raise HTTPException(status_code=400,
                            detail=f"Invalid JSON format: {body_text}, traceback: {_extract_core_stack()}, error: {e}")
    run_id = ctx.run_id
    is_agent = _is_agent_proj()
    logger.info(
        f"Received request for /stream_run: "
        f"run_id={run_id}, "
        f"is_agent_project={is_agent}, "
        f"query={dict(request.query_params)}, "
        f"body={body_text}"
    )
    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in http_stream_run: {e}, traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format:{_extract_core_stack()}")

    if is_agent:
        stream_generator = _agent_stream_handler(
            payload=payload,
            ctx=ctx,
            run_id=run_id,
            stream_sse_func=service.stream_sse,
            sse_event_func=service._sse_event,
            error_classifier=service.error_classifier,
            register_task_func=_register_task,
        )
    else:
        stream_generator = _workflow_stream_handler(
            payload=payload,
            ctx=ctx,
            run_id=run_id,
            stream_sse_func=service.stream_sse,
            sse_event_func=service._sse_event,
            error_classifier=service.error_classifier,
            register_task_func=_register_task,
            run_opt=RunOpt(workflow_debug=workflow_debug),
        )

    response = StreamingResponse(stream_generator, media_type="text/event-stream")
    return response

@app.post("/cancel/{run_id}")
async def http_cancel(run_id: str, request: Request):
    """
    取消指定run_id的执行

    使用asyncio.Task.cancel()实现取消,这是Python标准的异步任务取消机制。
    LangGraph会在节点之间的await点检查CancelledError,实现优雅取消。
    """
    ctx = new_context(method="cancel", headers=request.headers)
    request_context.set(ctx)
    logger.info(f"Received cancel request for run_id: {run_id}")
    result = service.cancel_run(run_id, ctx)
    return result


@app.post(path="/node_run/{node_id}")
async def http_node_run(node_id: str, request: Request):
    raw_body = await request.body()
    try:
        body_text = raw_body.decode("utf-8")
    except UnicodeDecodeError:
        body_text = str(raw_body)
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {body_text}")
    ctx = new_context(method="node_run", headers=request.headers)
    request_context.set(ctx)
    logger.info(
        f"Received request for /node_run/{node_id}: "
        f"query={dict(request.query_params)}, "
        f"body={body_text}",
    )

    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in http_node_run: {e}, traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format:{_extract_core_stack()}")
    try:
        return await service.run_node(node_id, payload, ctx)
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"node_id '{node_id}' not found or input miss required fields, traceback: {_extract_core_stack()}")
    except Exception as e:
        # 使用错误分类器获取错误信息
        error_response = service.error_classifier.get_error_response(e, {"node_name": node_id})
        logger.error(
            f"Unexpected error in http_node_run: [{error_response['error_code']}] {error_response['error_message']}, "
            f"traceback: {traceback.format_exc()}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": error_response["error_code"],
                "error_message": error_response["error_message"],
                "stack_trace": _extract_core_stack(),
            }
        )
    finally:
        _flush()


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """OpenAI Chat Completions API 兼容接口"""
    ctx = new_context(method="openai_chat", headers=request.headers)
    request_context.set(ctx)

    logger.info(f"Received request for /v1/chat/completions: run_id={ctx.run_id}")

    try:
        payload = await request.json()
        return await openai_handler.handle(payload, ctx)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in openai_chat_completions: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    finally:
        _flush()


@app.get("/health")
async def health_check():
    try:
        # 这里可以添加更多的健康检查逻辑
        return {
            "status": "ok",
            "message": "Service is running",
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get(path="/graph_parameter")
async def http_graph_inout_parameter(request: Request):
    return service.graph_inout_schema()

# ============================================================
# 电商智能助手路由 (挂载到主应用)
# ============================================================
import hashlib
import io
import uuid
import threading
import pandas as pd
from fastapi import UploadFile, File, Form, Body, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ecommerce.models import (
    Base as EcomBase, User, SalesData, ReportFile,
    Conversation, GeneratedImage, DashboardCache
)

# 请求模型
class LoginRequest(BaseModel):
    username: str
    password: str
class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: str = ""
class AskRequest(BaseModel):
    question: str
    session_id: str = ""
class CopywritingRequest(BaseModel):
    product_name: str = "高腰显瘦牛仔裤"
    style: str = "直筒"
    scene: str = "抖音短视频"
    target_audience: str = "年轻女性"
    key_selling_points: str = "显瘦、高腰、百搭"
    tone: str = "亲和"
class ScriptRequest(BaseModel):
    script_type: str = "种草"
    product_name: str = "高腰显瘦牛仔裤"
    duration: str = "30s"
    target_audience: str = "年轻女性"
class StrategyRequest(BaseModel):
    focus: str = "综合"
    period: str = "本月"
class ImageGenRequest(BaseModel):
    prompt: str
    style: str = "写实"
    size: str = "2K"
class DataEditRequest(BaseModel):
    id: int
    field: str
    value: Any
class SessionHistoryRequest(BaseModel):
    session_id: str
    msg_type: str = ""

# 静态文件路径
EC_STATIC_DIR = os.path.join(PROJECT_ROOT, "assets", "ecommerce")

# 挂载头像静态目录
_avatar_dir = os.path.join(PROJECT_ROOT, "assets", "avatars")
os.makedirs(_avatar_dir, exist_ok=True)
app.mount("/avatars", StaticFiles(directory=_avatar_dir), name="avatars")

# 电商前端静态文件
_ec_static = os.path.join(PROJECT_ROOT, "assets", "ecommerce")
if os.path.isdir(_ec_static):
    app.mount("/ecommerce", StaticFiles(directory=_ec_static, html=True), name="ecommerce")


@app.get("/")
async def root():
    """根路径重定向到电商面板"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ecommerce")

def _ec_db():
    return get_session()


def _user_uploads_dir(uid: int = 0) -> str:
    """用户专属上传目录 (uid=0 返回根目录)"""
    d = os.path.join(PROJECT_ROOT, "assets", "uploads", str(uid)) if uid else os.path.join(PROJECT_ROOT, "assets", "uploads")
    os.makedirs(d, exist_ok=True)
    return d


def _user_comments_dir(uid: int = 0) -> str:
    """用户专属评论目录 (uid=0 返回根目录)"""
    d = os.path.join(PROJECT_ROOT, "assets", "comments", str(uid)) if uid else os.path.join(PROJECT_ROOT, "assets", "comments")
    os.makedirs(d, exist_ok=True)
    return d


def _scan_user_files(uid: int) -> list:
    """扫描用户专属目录，返回文件信息列表"""
    files = []
    dirs = [_user_uploads_dir(uid), _user_comments_dir(uid)]
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.startswith("."):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext not in (".csv", ".xlsx", ".xls"):
                continue
            fp = os.path.join(d, fn)
            try:
                row_count = None
                if ext == ".csv":
                    with open(fp, "rb") as f:
                        row_count = sum(1 for _ in f)
                mtime = os.path.getmtime(fp)
                files.append({
                    "id": 0,
                    "file_name": fn,
                    "file_size": os.path.getsize(fp),
                    "file_type": ext.upper().lstrip("."),
                    "row_count": row_count,
                    "data_year": date.today().year,
                    "status": "uploaded",
                    "report_period": str(date.today().month),
                    "created_at": datetime.fromtimestamp(mtime).isoformat()
                })
            except Exception:
                pass
    return files

def _make_pwd(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _call_llm(prompt: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 8192) -> str:
    """调用大模型 — 使用 OpenAI 兼容接口"""
    from utils.llm import call_llm
    return call_llm(prompt=prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens)

def _parse_kv(ed, k):
    """从 extra_data dict 提取数值，支持多个候选key"""
    if not ed or not isinstance(ed, dict): return 0
    for key in (k if isinstance(k, (list,tuple)) else [k]):
        v = ed.get(key)
        if v is not None:
            try: return float(str(v).replace(",","").replace("¥","").replace("￥","").strip())
            except: pass
    return 0

def _parse_kv_text(ed, keys, strip_suffix=True):
    """从 extra_data dict 提取文本值，支持多个候选key"""
    if not ed or not isinstance(ed, dict):
        return ""
    for key in (keys if isinstance(keys, (list,tuple)) else [keys]):
        v = ed.get(key)
        if v is not None and not (isinstance(v, float) and (v != v or v == 0.0)):
            txt = str(v).strip().replace("\t","")
            if txt.lower() in ("nan", "none", "null", ""):
                continue
            # 去掉前缀数字和分隔符，如 "2267 - 大华在北漂" → "大华在北漂"
            import re
            if strip_suffix:
                txt = re.sub(r'^\d+\s*[-—]\s*', '', txt)
            return txt
    return ""

def _get_sales_summary(db, year=None, period: str = ""):
    """获取销售数据摘要 - 优先 CSV 直接聚合，DB 兜底。

    period: "本月" / "本季度" / "本年度" — 用于筛选目标月份的数据文件
    """
    from ecommerce.analysis import get_uploaded_files, analyze_csv_text, get_analysis_for_llm
    import re as _re

    today = date.today()
    current_year = today.year

    # 根据 period 确定目标月份列表
    if period == "本月":
        target_months = [today.month]
    elif period == "本季度":
        q = (today.month - 1) // 3 + 1
        target_months = list(range((q - 1) * 3 + 1, q * 3 + 1))
    elif period == "本年度":
        target_months = list(range(1, 13))
    else:
        target_months = []  # 不限月份

    # ── CSV 路径：匹配目标月份的文件 ──
    all_files = get_uploaded_files("uploads", user_id=_uid())
    if all_files:
        if target_months:
            matched = []
            for fp in all_files:
                fname = os.path.basename(fp)
                m = _re.search(r'(?<!\d)(\d{1,2})\s*月', fname)
                if m and int(m.group(1)) in target_months:
                    matched.append(fp)
            if matched:
                if len(matched) == 1:
                    return analyze_csv_text(matched[0])
                else:
                    text, _ = get_analysis_for_llm(matched)
                    return text
            # 指定了月份但没匹配到 → 返回明确提示
            month_list = "、".join(f"{m}月" for m in target_months)
            return f"暂无{month_list}的数据文件，请先上传对应月份的销售报表。"
        # 未指定月份 → 取最新文件
        if len(all_files) == 1:
            return analyze_csv_text(all_files[0])
        else:
            max_files = 5
            selected = all_files[:max_files]
            text, _ = get_analysis_for_llm(selected)
            if len(all_files) > max_files:
                text += f"\n\n⚠️ 共 {len(all_files)} 个文件，仅分析了最近 {max_files} 个。如需分析特定文件请明确指定。"
            return text

    # ── DB 兜底：按月份过滤（仅当前用户） ──
    q = db.query(SalesData).filter(SalesData.user_id == _uid())
    if year:
        q = q.filter(SalesData.data_year == year)
    else:
        q = q.filter(SalesData.data_year == current_year)
    if target_months:
        q = q.filter(SalesData.data_month.in_(target_months))
    rows = q.all()
    if not rows:
        return "暂无销售数据。"
    total_vol = sum(float(r.sales_volume or 0) for r in rows)
    total_amt = sum(float(r.sales_amount or 0) for r in rows)
    total_profit = sum(float(r.profit or 0) for r in rows)
    nl = chr(10)
    month_label = period if period else "全部"
    return f"=== 销售数据总览（{month_label}） ==={nl}总销量: {total_vol:,.0f}{nl}总销售额: {chr(165)}{total_amt:,.2f}{nl}总利润: {chr(165)}{total_profit:,.2f}"


def _get_zhubo_markdown_table(db, target_month=None, top_n=10, filter_zhubo=None):
    """生成主播对比/排行Markdown表格
    - target_month=None: 对比最近两个月TOP N
    - target_month=3: 仅显示该月销量TOP N
    - filter_zhubo="与辉同行": 单主播详情（多维度）
    """
    import logging
    logging.info(f"[_get_zhubo_markdown_table] filter_zhubo={filter_zhubo!r}, target_month={target_month}")
    sales = db.query(SalesData).filter(SalesData.user_id == _uid())
    total = sales.count()
    if total == 0: return ""
    year = sales.first().data_year

    rows = sales.filter(SalesData.data_year == year).all()
    zmdata = {}
    for r in rows:
        ed = r.extra_data if isinstance(r.extra_data, dict) else {}
        zname = (ed.get("主播") or "").strip()
        if not zname or zname.lower() == "nan": continue
        m = r.data_month
        vol = _parse_kv(ed, ["利润-销售数量(扣退)","销售数量(扣退)","sales_volume"]) or (float(r.sales_volume or 0))
        amt = _parse_kv(ed, ["利润-销售金额(扣退)","销售金额(扣退)","sales_amount"]) or (float(r.sales_amount or 0))
        ship = _parse_kv(ed, ["商品数据-实发数量","实发数量"])
        profit = _parse_kv(ed, ["利润-毛利额","毛利额","profit"]) or (float(r.profit or 0))
        if vol == 0 and amt == 0: continue
        zmdata.setdefault(m, {}).setdefault(zname, {"vol":0,"amt":0,"ship":0,"profit":0})
        zmdata[m][zname]["vol"] += vol
        zmdata[m][zname]["amt"] += amt
        zmdata[m][zname]["ship"] += (ship or 0)
        zmdata[m][zname]["profit"] += profit

    # 指定月份 + 指定主播 → 单主播详情（多维度）
    if target_month and target_month in zmdata and filter_zhubo:
        for z, d in zmdata[target_month].items():
            if filter_zhubo.lower() in z.lower():
                lines = [""]
                lines.append(f"| 指标 | {target_month}月数据 |")
                lines.append("| --- | --- |")
                v = f"{d['vol']:,.0f}" if d['vol'] > 0 else "—"
                a = f"¥{d['amt']:,.2f}" if d['amt'] > 0 else "—"
                s = f"{d['ship']:,.0f}" if d['ship'] > 0 else "—"
                p = f"¥{d['profit']:,.2f}" if d['profit'] > 0 else "—"
                lines.append(f"| 主播 | {z} |")
                lines.append(f"| 实际成交件数 | {v} |")
                lines.append(f"| 销售额 | {a} |")
                lines.append(f"| 实发件数 | {s} |")
                lines.append(f"| 毛利额 | {p} |")
                return "\n".join(lines)
        return ""  # 未找到匹配主播

    # 指定月份 → 仅展示该月TOP N按销量排序（多列）
    if target_month and target_month in zmdata:
        zhubos = zmdata[target_month]
        sorted_zhubos = sorted(zhubos.items(), key=lambda x: -x[1]["vol"])
        top = sorted_zhubos[:top_n]
        lines = [""]
        lines.append(f"| 主播名称 | {target_month}月销量 | {target_month}月销售额 | 实发件数 | 毛利额 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for z, d in top:
            v = f"{d['vol']:,.0f}" if d['vol'] > 0 else "—"
            a = f"¥{d['amt']:,.2f}" if d['amt'] > 0 else "—"
            s = f"{d['ship']:,.0f}" if d['ship'] > 0 else "—"
            p = f"¥{d['profit']:,.2f}" if d['profit'] > 0 else "—"
            lines.append(f"| {z} | {v} | {a} | {s} | {p} |")
        return "\n".join(lines)

    # 无指定月份 → 对比最近两个月
    months = sorted(zmdata.keys())
    if len(months) < 2: return ""

    m1, m2 = months[0], months[1]
    m1_zhubos = {z: d for z, d in sorted(zmdata[m1].items(), key=lambda x: -x[1]["vol"])}
    m2_zhubos = {z: d for z, d in sorted(zmdata[m2].items(), key=lambda x: -x[1]["vol"])}
    
    # 如果指定了主播名称，只保留该主播
    if filter_zhubo:
        m1_zhubos = {z:d for z,d in m1_zhubos.items() if filter_zhubo.lower() in z.lower()}
        m2_zhubos = {z:d for z,d in m2_zhubos.items() if filter_zhubo.lower() in z.lower()}
        all_names = set(list(m1_zhubos.keys()) + list(m2_zhubos.keys()))
    else:
        all_names = set(list(m1_zhubos.keys())[:top_n*2] + list(m2_zhubos.keys())[:top_n*2])

    lines = [""]
    lines.append(f"| 主播名称 | {m1}月销量 | {m1}月销售额 | {m2}月销量 | {m2}月销售额 | 变化 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for z in sorted(all_names, key=lambda x: -(m2_zhubos.get(x,{})|m1_zhubos.get(x,{})).get("vol",0)):
        d1 = m1_zhubos.get(z, {"vol":0,"amt":0,"ship":0,"profit":0})
        d2 = m2_zhubos.get(z, {"vol":0,"amt":0,"ship":0,"profit":0})
        v1 = f"{d1['vol']:,.0f}" if d1["vol"] > 0 else "—"
        a1 = f"¥{d1['amt']:,.2f}" if d1["amt"] > 0 else "—"
        v2 = f"{d2['vol']:,.0f}" if d2["vol"] > 0 else "—"
        a2 = f"¥{d2['amt']:,.2f}" if d2["amt"] > 0 else "—"
        if d1["vol"] > 0 and d2["vol"] > 0:
            pct = f"+{(d2['vol']/d1['vol']-1)*100:.0f}%" if d2['vol'] > d1['vol'] else f"-{(1-d2['vol']/d1['vol'])*100:.0f}%"
        elif d1["vol"] == 0 and d2["vol"] > 0:
            pct = "新上榜"
        elif d1["vol"] > 0 and d2["vol"] == 0:
            pct = "跌出TOP10"
        else:
            pct = "—"
        lines.append(f"| {z} | {v1} | {a1} | {v2} | {a2} | {pct} |")
    return "\n".join(lines)

def _get_sku_markdown_table(db, target_month=None, top_n=10):
    """生成SKU销量排名Markdown表格（DB兜底）"""
    from collections import defaultdict
    from ecommerce.analysis import _parse_sku_display
    rows = db.query(SalesData).filter(SalesData.user_id == _uid()).all()
    if not rows:
        return ""
    # 聚合：按 SKU 编码汇总
    sdata = defaultdict(lambda: {"vol": 0, "amt": 0, "name": ""})
    for r in rows:
        if target_month and r.data_month != target_month:
            continue
        sku = (r.sku_code or "").strip()
        if not sku:
            continue
        sdata[sku]["vol"] += int(r.sales_volume or 0)
        sdata[sku]["amt"] += float(r.sales_amount or 0)
        if not sdata[sku]["name"] and r.product_name:
            sdata[sku]["name"] = str(r.product_name).strip()
    # 排序取 TOP
    sorted_sku = sorted(sdata.items(), key=lambda x: -x[1]["vol"])[:top_n]
    if not sorted_sku:
        return ""
    lines = [""]
    month_label = f"{target_month}月 " if target_month else ""
    lines.append(f"| 排名 | SKU 编码 | 款号 | 商品名称 | {month_label}总销量 | {month_label}总销售额 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for i, (sku, d) in enumerate(sorted_sku, 1):
        display_code, model_code = _parse_sku_display(sku)
        name = d["name"] or "-"
        lines.append(f"| {i} | {display_code} | {model_code} | {name[:20]} | {d['vol']:,} 件 | {d['amt']:,.2f} 元 |")
    return "\n".join(lines)

def _get_comment_table(question: str, target_month=None):
    """从 assets/comments/ 加载评论数据并返回摘要"""
    import os, glob
    comment_dir = _user_comments_dir(_uid())
    if not os.path.isdir(comment_dir):
        return ""
    files = sorted(glob.glob(os.path.join(comment_dir, "*.csv")))
    if not files:
        return ""
    result_lines = []
    for fp in files:
        fname = os.path.basename(fp)
        # Extract season/month from filename
        import re
        m = re.search(r'(\d+)[月\.]', fname)
        month_num = int(m.group(1)) if m else 0
        if not m:
            continue  # 跳过文件名不含月份的文件（非评论文件）
        if target_month and month_num != target_month:
            continue
        try:
            import pandas as pd
            df = pd.read_csv(fp, encoding='utf-8-sig')
            # Find comment text columns - 优先匹配更精确的列名
            text_col = None
            # 优先级1: 精确匹配评价内容/评论内容/评语
            for c in df.columns:
                if c in ("评价内容", "评论内容", "评语", "评价", "评论"):
                    text_col = c
                    break
            # 优先级2: 包含"内容"或"评语"
            if text_col is None:
                for c in df.columns:
                    if "内容" in c or "评语" in c:
                        text_col = c
                        break
            # 优先级3: 包含"评价"或"评论"且不包含"时间"/"日期"
            if text_col is None:
                for c in df.columns:
                    if any(kw in c for kw in ["评价","评论"]) and not any(ex in c for ex in ["时间","日期","得分"]):
                        text_col = c
                        break
            # 优先级4: 包含"评价"或"评论"
            if text_col is None:
                for c in df.columns:
                    if any(kw in c for kw in ["评价","评论","review","text","content"]):
                        text_col = c
                        break
            # 兜底: 第一列
            if text_col is None:
                text_col = df.columns[0]
            comments = df[text_col].dropna().tolist()
            # Sample comments for LLM
            sample = comments[:100]
            result_lines.append(f"=== 商品评论{month_num}月 ===")
            result_lines.append(f"总评论数: {len(comments)}")
            result_lines.append(f"评论示例({min(5,len(sample))}条):")
            for c in sample[:5]:
                txt = str(c).strip()[:80]
                if txt:
                    result_lines.append(f"  - {txt}")
            result_lines.append("")
        except Exception as e:
            result_lines.append(f"=== {fname} 读取失败: {e} ===")
    if not result_lines:
        return ""
    return "\n".join(result_lines)


def _get_general_uploaded_data(question: str):
    """从 assets/uploads/ 读取所有上传文件，返回内容摘要供LLM分析"""
    import os, glob
    upload_dir = _user_uploads_dir(_uid())
    if not os.path.isdir(upload_dir):
        return ""
    files = sorted(glob.glob(os.path.join(upload_dir, "*")))
    if not files:
        return ""
    result_lines = ["【已上传文件数据】"]
    for fp in files:
        fname = os.path.basename(fp)
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        try:
            if ext == "csv":
                import pandas as pd
                df = pd.read_csv(fp, encoding='utf-8-sig', nrows=50)
                result_lines.append(f"\n📄 {fname} (CSV, {len(df)}行)")
                result_lines.append(f"   列: {', '.join(str(c) for c in df.columns)}")
                result_lines.append(f"   前3行: {df.head(3).to_string(index=False).replace(chr(10), chr(10)+'    ')}")
            elif ext in ("xlsx", "xls"):
                import pandas as pd
                df = pd.read_excel(fp, nrows=50)
                result_lines.append(f"\n📄 {fname} (Excel, {len(df)}行)")
                result_lines.append(f"   列: {', '.join(str(c) for c in df.columns)}")
                result_lines.append(f"   前3行: {df.head(3).to_string(index=False).replace(chr(10), chr(10)+'    ')}")
            else:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read(2000)
                result_lines.append(f"\n📄 {fname} (文本, {len(text)}字符)")
                result_lines.append(f"   内容预览: {text[:500]}")
        except Exception as e:
            result_lines.append(f"\n📄 {fname} 读取失败: {e}")
    return "\n".join(result_lines)


# ----- Session 文件追踪 (P0修复: 上传文件自动关联) -----
_session_latest_file: dict[str, str] = {}  # session_id → file_path


def _track_session_file(session_id: str, file_path: str, uid: int = 0):
    """记录 session 最近上传的文件（存入用户专属目录）"""
    import shutil as _shutil
    upload_dir = _user_uploads_dir(uid)
    dest = os.path.join(upload_dir, os.path.basename(file_path))
    if file_path != dest:
        try:
            _shutil.copy2(file_path, dest)
        except Exception:
            pass
    # LRU: 重新插入确保在末尾（最近使用），再淘汰最久未用的
    _session_latest_file.pop(session_id, None)
    _session_latest_file[session_id] = dest
    _MAX_SESSION_FILES = 500
    if len(_session_latest_file) > _MAX_SESSION_FILES:
        oldest_key = next(iter(_session_latest_file))
        oldest_path = _session_latest_file[oldest_key]
        del _session_latest_file[oldest_key]
        logger.debug(f"Session 文件追踪 LRU 淘汰: session={oldest_key}, file={oldest_path}")


def _deep_analyze_csv(file_path: str) -> str:
    """对单个 CSV 文件做全维度聚合分析（委托给 analysis 模块）。"""
    from ecommerce.analysis import analyze_csv_text
    return analyze_csv_text(file_path)


def _build_comparison_table(file_paths: list) -> str:
    """多文件对比：生成并排对比总览表（单表多列，而非每文件一个表）。"""
    from ecommerce.analysis import analyze_csv_structured

    if len(file_paths) < 2:
        return ""

    summaries = []
    for fp in file_paths:
        r = analyze_csv_structured(fp)
        if r.get("error"):
            continue
        ov = r["overview"]
        # 跳过空数据文件（如评论文件、非销售报表）
        if not ov or (ov.get("koutui_vol", 0) == 0 and ov.get("koutui_amt", 0) == 0):
            continue
        # 从文件名提取月份标签
        import re as _re2
        fname = os.path.basename(fp)
        m = _re2.search(r'(\d+)\s*月', fname)
        label = f"{m.group(1)}月" if m else fname.rsplit(".", 1)[0]
        summaries.append({"label": label, "ov": ov})

    if len(summaries) < 2:
        return ""

    # 计算环比变化
    def _pct_change(new, old):
        if old and old != 0:
            return f"+{((new - old) / abs(old)) * 100:.1f}%" if new >= old else f"{((new - old) / abs(old)) * 100:.1f}%"
        return "—"

    def _pp_change(new, old):
        if old is not None and old != 0:
            diff = new - old
            return f"+{diff:.1f}pp" if diff >= 0 else f"{diff:.1f}pp"
        return "—"

    def _fmt_amt(v):
        if abs(v) >= 10000:
            return f"¥{v/10000:.1f}万"
        return f"¥{v:,.0f}"

    # 构建表头
    headers = ["指标"] + [s["label"] for s in summaries]
    if len(summaries) == 2:
        headers.append("环比变化")
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "|" + "|".join([" --- " for _ in headers]) + "|"

    # 指标行 — 对齐单文件总览表的丰富度
    def _fmt_cost(ov):
        c = ov.get("cost", 0) or 0
        amt = ov.get("koutui_amt", 0) or 1
        pct = c / amt * 100 if amt > 0 else 0
        return f"{_fmt_amt(c)} ({pct:.1f}%)"

    def _fmt_refund(ov):
        qty = int(ov.get("refund_qty", 0) or 0)
        amt = ov.get("refund_amt", 0) or 0
        return f"{qty:,}件 / {_fmt_amt(amt)}"

    def _fmt_ship(ov):
        qty = int(ov.get("ship_qty", 0) or 0)
        amt = ov.get("ship_amt", 0) or 0
        return f"{qty:,}件 / {_fmt_amt(amt)}"

    def _fmt_fee(ov):
        fee = ov.get("total_fee", 0) or 0
        amt = ov.get("koutui_amt", 0) or 1
        pct = fee / amt * 100 if amt > 0 else 0
        return f"{_fmt_amt(fee)} ({pct:.1f}%)"

    # (显示名, overview_key, 格式化函数, 环比用 "pct" 还是 "num")
    metrics = [
        ("总行数",           "total_rows",   lambda ov: f"{int(ov.get('total_rows',0) or 0):,}条", "num"),
        ("销量(扣退)",       "koutui_vol",   lambda ov: f"{int(ov.get('koutui_vol',0) or 0):,}件", "num"),
        ("销售额(扣退)",     "koutui_amt",   lambda ov: _fmt_amt(ov.get("koutui_amt",0) or 0), "num"),
        ("商品成本",         "cost",         _fmt_cost, "num"),
        ("毛利率",           "gross_margin", lambda ov: f"{ov.get('gross_margin',0) or 0}%", "pct"),
        ("经营利润",         "oper_profit",  lambda ov: _fmt_amt(ov.get("oper_profit",0) or 0), "num"),
        ("经营利润率",       "oper_margin",  lambda ov: f"{ov.get('oper_margin',0) or 0}%", "pct"),
        ("费用合计",         "total_fee",    _fmt_fee, "num"),
        ("退款数量/金额",    "refund_qty",   _fmt_refund, "num"),
        ("退款率",           "refund_rate",  lambda ov: f"{ov.get('refund_rate',0) or 0}%", "pct"),
        ("实发数量/金额",    "ship_qty",     _fmt_ship, "num"),
        ("整体均价",         "avg_price",    lambda ov: f"¥{int(ov.get('avg_price',0) or 0)}", "num"),
    ]

    data_rows = []
    for name, key, fmt_fn, change_type in metrics:
        vals = [fmt_fn(s["ov"]) for s in summaries]

        row = [name] + vals
        if len(summaries) == 2:
            old_v = summaries[0]["ov"].get(key, 0) or 0
            new_v = summaries[1]["ov"].get(key, 0) or 0
            if change_type == "pct":
                row.append(_pp_change(new_v, old_v))
            else:
                row.append(_pct_change(new_v, old_v))
        data_rows.append("| " + " | ".join(row) + " |")

    return "\n".join([header_row, sep_row] + data_rows)


def _extract_display_table(llm_data: str, max_lines: int = 80) -> str:
    """从深度分析结果中提取适合前端展示的表格部分。

    保留：总览指标表 + TOP主播排名表 + 来源分布表
    去掉：SKU详细表、品牌分布、价格带等长表（LLM会在回答中按需引用）
    """
    if not llm_data:
        return ""
    lines = llm_data.split("\n")
    result = []
    line_count = 0

    for line in lines:
        # 检测章节切换 — 以下章节不展示（数据过长，留给 LLM 引用）
        if line.startswith("### 📦") or line.startswith("### 💰") or line.startswith("### 💵"):
            break  # SKU/费用/价格带
        if line.startswith("### 🏷️"):
            break  # 品牌分布

        result.append(line)
        line_count += 1
        if line_count >= max_lines:
            result.append("")
            result.append("*（表格较长，完整数据已在分析中引用）*")
            break

    return "\n".join(result) if result else ""


# ----- 认证 -----
@app.post("/api/register")
async def ec_register(req: RegisterRequest):
    """用户注册 — 独立于登录的注册入口"""
    db = _ec_db()
    try:
        existing = db.query(User).filter(User.username == req.username).first()
        if existing:
            raise HTTPException(status_code=409, detail="用户名已存在")
        user = User(
            username=req.username,
            password_hash=_make_pwd(req.password),
            display_name=req.display_name or req.username,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return {
            "success": True,
            "user_id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "avatar_url": user.avatar_url or "",
            "message": "注册成功",
        }
    finally:
        db.close()


@app.post("/api/login")
async def ec_login(req: LoginRequest):
    db = _ec_db()
    try:
        user = db.query(User).filter(User.username == req.username).first()
        if not user:
            raise HTTPException(status_code=401, detail="用户不存在，请先注册")
        if user.password_hash != _make_pwd(req.password):
            raise HTTPException(status_code=401, detail="密码错误")
        return {
            "success": True,
            "user_id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "avatar_url": user.avatar_url or "",
        }
    finally:
        db.close()

# ----- 用户信息 & 头像 -----
@app.get("/api/user/profile")
async def ec_user_profile():
    """获取当前用户信息"""
    db = _ec_db()
    uid = _uid()
    try:
        user = db.query(User).filter(User.id == uid).first()
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        return {
            "success": True,
            "user_id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "avatar_url": user.avatar_url or "",
        }
    finally:
        db.close()


@app.post("/api/user/avatar")
async def ec_upload_avatar(file: UploadFile = File(...)):
    """上传用户头像"""
    uid = _uid()
    if uid == 0:
        raise HTTPException(status_code=401, detail="请先登录")

    # 校验文件类型
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        raise HTTPException(status_code=400, detail="仅支持 PNG/JPG/GIF/WEBP 格式")

    # 限制大小 2MB
    content = await file.read()
    if len(content) > 2 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="头像大小不能超过 2MB")

    # 保存到 assets/avatars/{user_id}{ext}
    avatar_dir = os.path.join(PROJECT_ROOT, "assets", "avatars")
    os.makedirs(avatar_dir, exist_ok=True)
    # 删除旧头像文件
    for old_ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        old_path = os.path.join(avatar_dir, f"{uid}{old_ext}")
        if os.path.isfile(old_path):
            try:
                os.remove(old_path)
            except Exception:
                pass

    save_path = os.path.join(avatar_dir, f"{uid}{ext}")
    with open(save_path, "wb") as f:
        f.write(content)

    # 更新数据库
    avatar_url = f"/avatars/{uid}{ext}?t={int(time.time())}"  # 加时间戳防缓存
    db = _ec_db()
    try:
        user = db.query(User).filter(User.id == uid).first()
        if user:
            user.avatar_url = avatar_url
            db.commit()
    finally:
        db.close()

    return {"success": True, "avatar_url": avatar_url, "message": "头像已更新"}


# ============================================================
# 后台任务跟踪 (解决大文件上传上游超时)
# ============================================================
_upload_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()

def _new_task(file_name: str, kind: str = "data") -> str:
    """创建一个上传任务记录，返回 task_id"""
    task_id = uuid.uuid4().hex[:16]
    with _tasks_lock:
        _upload_tasks[task_id] = {
            "task_id": task_id,
            "file_name": file_name,
            "kind": kind,
            "status": "pending",
            "progress": "等待处理",
            "message": "",
            "row_count": 0,
            "error": None,
        }
    return task_id

def _update_task(task_id: str, **kw):
    with _tasks_lock:
        if task_id in _upload_tasks:
            _upload_tasks[task_id].update(kw)

def _get_task(task_id: str) -> dict | None:
    with _tasks_lock:
        return _upload_tasks.get(task_id, None)

# ----- 数据上传 -----
@app.post("/api/data/upload")
async def ec_upload(file: UploadFile = File(...), data_year: int = Form(date.today().year), data_month: int = Form(0)):
    uid = _uid()
    # 1. 保存文件到 /tmp
    import re
    tmp_dir = os.path.join(os.sep, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, file.filename)
    # 流式写入，不等待完整读取
    chunk_size = 64 * 1024  # 64KB
    with open(tmp_path, "wb") as f:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    logger.info(f"文件已保存到临时路径: {tmp_path} (大小: {os.path.getsize(tmp_path)})")
    
    # 2. 创建任务并立即返回
    task_id = _new_task(file.filename, "data")
    _update_task(task_id, status="processing", progress="准备处理文件...")
    
    # 3. 启动后台线程处理
    def _process():
        db = None
        try:
            db = _ec_db()
            _update_task(task_id, progress="正在解析文件...")
            fn = file.filename.lower()
            # 从文件名提取月份
            m = re.search(r'(?<!\d)(\d{1,2})\s*月', file.filename)
            file_month = int(m.group(1)) if m and 1 <= int(m.group(1)) <= 12 else (data_month or 1)

            # 读取文件
            if fn.endswith(".csv"):
                _encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312", "gb18030", "latin-1", "iso-8859-1"]
                df = None
                for _enc in _encodings:
                    try:
                        df = pd.read_csv(tmp_path, encoding=_enc)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                if df is None:
                    df = pd.read_csv(tmp_path, encoding="utf-8", errors="replace")
            elif fn.endswith((".xlsx", ".xls")):
                df = pd.read_excel(tmp_path)
            else:
                _update_task(task_id, status="completed", message="非CSV/Excel文件已保存", row_count=0)
                return

            df.columns = [c.strip().lower() for c in df.columns]
            is_comment = "评论" in fn or "评价" in fn or "comment" in fn.lower()

            if is_comment:
                comment_dir = _user_comments_dir(uid)
                save_name = fn.rsplit(".", 1)[0] + ".csv"
                csv_path = os.path.join(comment_dir, save_name)
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                csv_size = os.path.getsize(csv_path)
                _upsert_report_file(file_name=save_name, user_id=uid, file_size=csv_size, file_type="comment", data_year=data_year, report_period=f"{data_year}-{file_month:02d}", row_count=len(df))
                # 评论数据已保存为 CSV 在 assets/comments/，无需额外复制 XLSX 到 uploads
                _update_task(task_id, status="completed", message=f"成功导入评论数据 {len(df)} 条", row_count=len(df))
                return
            
            _update_task(task_id, progress=f"正在处理 {len(df)} 条销售数据...")
            cm = {"sku编码":"sku_code","sku_code":"sku_code","商品名称":"product_name","product_name":"product_name","类目":"category","版型":"style","颜色":"color","尺码":"size","销量":"sales_volume","sales_volume":"sales_volume","销售额":"sales_amount","成本":"cost","利润":"profit","利润率":"profit_margin","退货数":"return_count","退货率":"return_rate","库存":"inventory"}
            df.rename(columns=cm, inplace=True)
            df["data_month"] = file_month
            records = []
            for _, row in df.iterrows():
                try:
                    r = SalesData(user_id=uid, report_name=file.filename, data_year=data_year, data_month=file_month, sku_code=str(row.get("sku_code",""))[:100], product_name=str(row.get("product_name",""))[:300], category=str(row.get("category",""))[:100], style=str(row.get("style",""))[:100], color=str(row.get("color",""))[:50], size=str(row.get("size",""))[:50], sales_volume=int(float(row.get("sales_volume",0))), sales_amount=float(row.get("sales_amount",0)), cost=float(row.get("cost",0)), profit=float(row.get("profit",0)), profit_margin=float(row.get("profit_margin",0)), return_count=int(float(row.get("return_count",0))), return_rate=float(row.get("return_rate",0)), inventory=int(float(row.get("inventory",0))))
                    if r.profit == 0 and r.sales_amount > 0: r.profit = r.sales_amount - r.cost
                    if r.profit_margin == 0 and r.profit > 0 and r.sales_amount > 0: r.profit_margin = r.profit / r.sales_amount
                    records.append(r)
                except Exception as e:
                    logger.warning(f"跳过异常行: {e}")
            
            if records:
                _update_task(task_id, progress=f"正在写入数据库 ({len(records)} 条)...")
                db.add_all(records)
                db.commit()
            _upsert_report_file(file_name=file.filename, user_id=uid, file_size=os.path.getsize(tmp_path), file_type="csv" if fn.endswith(".csv") else "excel", data_year=data_year, report_period=f"{data_year}-{file_month:02d}", row_count=len(records))
            # 复制到 uploads/
            _copy_to_uploads(tmp_path, file.filename, uid)
            _track_session_file("default", tmp_path, uid)
            _update_task(task_id, status="completed", message=f"成功导入 {len(records)} 条", row_count=len(records))
        except Exception as e:
            logger.exception(f"后台处理失败 task={task_id}")
            if db: db.rollback()
            _update_task(task_id, status="failed", error=str(e))
        finally:
            if db: db.close()
            # 清理临时文件
            try:
                if os.path.isfile(tmp_path): os.remove(tmp_path)
            except: pass
    
    threading.Thread(target=_process, daemon=True).start()
    return {"success": True, "task_id": task_id, "message": "文件已接收，正在后台处理...", "file_name": file.filename}

def _copy_to_uploads(src_path: str, file_name: str, uid: int = 0):
    """复制处理后的文件到用户专属 uploads 目录"""
    upload_dir = _user_uploads_dir(uid)
    import shutil
    dest = os.path.join(upload_dir, file_name)
    try:
        shutil.copy2(src_path, dest)
    except Exception as e:
        logger.warning(f"复制到 uploads/ 失败: {e}")

# ----- AI助理文件上传 -----
@app.post("/api/ask/upload")
async def ec_ask_upload(file: UploadFile = File(...), session_id: str = Form("")):
    """电商小助手的文件上传接口"""
    # 1. 流式保存到 /tmp
    import re
    tmp_dir = os.path.join(os.sep, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, file.filename)
    with open(tmp_path, "wb") as f:
        while True:
            chunk = await file.read(64 * 1024)
            if not chunk: break
            f.write(chunk)
    logger.info(f"AI助理文件已保存: {tmp_path} ({os.path.getsize(tmp_path)} bytes)")
    
    # 2. 创建任务并立即返回
    task_id = _new_task(file.filename, "ask")
    _update_task(task_id, status="processing", progress="准备处理文件...")
    
    # 3. 后台处理
    def _process():
        db = None
        try:
            db = _ec_db()
            _update_task(task_id, progress="正在解析文件...")
            fn = file.filename.lower()
            m = re.search(r'(?<!\d)(\d{1,2})\s*月', file.filename)
            file_month = int(m.group(1)) if m else 1
            if file_month < 1 or file_month > 12: file_month = 1
            data_year = date.today().year
            
            if fn.endswith(".csv"):
                _encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312", "gb18030", "latin-1", "iso-8859-1"]
                df = None
                for _enc in _encodings:
                    try:
                        df = pd.read_csv(tmp_path, encoding=_enc)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                if df is None:
                    df = pd.read_csv(tmp_path, encoding="utf-8", errors="replace")
            elif fn.endswith((".xlsx", ".xls")):
                df = pd.read_excel(tmp_path)
            else:
                _copy_to_uploads(tmp_path, file.filename, uid)
                _track_session_file(session_id or "default", tmp_path)
                _update_task(task_id, status="completed", message=f"文件已保存", row_count=0)
                return
            
            df.columns = [c.strip().lower() for c in df.columns]
            is_comment = "评论" in fn or "评价" in fn or "comment" in fn.lower()
            
            if is_comment:
                comment_dir = _user_comments_dir(uid)
                save_name = fn.rsplit(".", 1)[0] + ".csv"
                csv_path = os.path.join(comment_dir, save_name)
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                csv_size = os.path.getsize(csv_path)
                _upsert_report_file(file_name=save_name, user_id=uid, file_size=csv_size, file_type="comment", data_year=data_year, report_period=f"{data_year}-{file_month:02d}", row_count=len(df))
                _copy_to_uploads(tmp_path, file.filename, uid)
                _track_session_file(session_id or "default", tmp_path)
                _update_task(task_id, status="completed", message=f"成功导入评论数据 {len(df)} 条", row_count=len(df))
                return
            
            _update_task(task_id, progress=f"正在处理 {len(df)} 条销售数据...")
            cm = {"sku编码":"sku_code","sku_code":"sku_code","商品名称":"product_name","product_name":"product_name","类目":"category","版型":"style","颜色":"color","尺码":"size","销量":"sales_volume","sales_volume":"sales_volume","销售额":"sales_amount","成本":"cost","利润":"profit","利润率":"profit_margin","退货数":"return_count","退货率":"return_rate","库存":"inventory"}
            df.rename(columns=cm, inplace=True)
            df["data_month"] = file_month
            records = []
            for _, row in df.iterrows():
                try:
                    r = SalesData(user_id=uid, report_name=file.filename, data_year=data_year, data_month=file_month, sku_code=str(row.get("sku_code",""))[:100], product_name=str(row.get("product_name",""))[:300], category=str(row.get("category",""))[:100], style=str(row.get("style",""))[:100], color=str(row.get("color",""))[:50], size=str(row.get("size",""))[:50], sales_volume=int(float(row.get("sales_volume",0))), sales_amount=float(row.get("sales_amount",0)), cost=float(row.get("cost",0)), profit=float(row.get("profit",0)), profit_margin=float(row.get("profit_margin",0)), return_count=int(float(row.get("return_count",0))), return_rate=float(row.get("return_rate",0)), inventory=int(float(row.get("inventory",0))))
                    if r.profit == 0 and r.sales_amount > 0: r.profit = r.sales_amount - r.cost
                    if r.profit_margin == 0 and r.profit > 0 and r.sales_amount > 0: r.profit_margin = r.profit / r.sales_amount
                    records.append(r)
                except Exception as e:
                    logger.warning(f"跳过异常: {e}")
            if records:
                _update_task(task_id, progress=f"正在写入数据库 ({len(records)} 条)...")
                db.add_all(records)
                db.commit()
            _upsert_report_file(file_name=file.filename, user_id=uid, file_size=os.path.getsize(tmp_path), file_type="csv" if fn.endswith(".csv") else "excel", data_year=data_year, report_period=f"{data_year}-{file_month:02d}", row_count=len(records))
            _copy_to_uploads(tmp_path, file.filename, uid)
            _track_session_file(session_id or "default", tmp_path)
            _update_task(task_id, status="completed", message=f"成功导入 {len(records)} 条", row_count=len(records))
        except Exception as e:
            logger.exception(f"AI助理后台处理失败 task={task_id}")
            if db: db.rollback()
            _update_task(task_id, status="failed", error=str(e))
        finally:
            if db: db.close()
            try:
                if os.path.isfile(tmp_path): os.remove(tmp_path)
            except: pass
    
    threading.Thread(target=_process, daemon=True).start()
    return {"success": True, "task_id": task_id, "message": "文件已接收，正在后台处理...", "file_name": file.filename}

@app.get("/api/data/task/{task_id}")
async def ec_task_status(task_id: str):
    """查询上传任务状态"""
    task = _get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task

def _upsert_report_file(file_name: str, user_id: int, file_size: int = 0, file_type: str = "csv", data_year: int = date.today().year, report_period: str = "", row_count: int = 0):
    """创建或更新报表记录（按 file_name + user_id 去重，防止重复上传产生冗余记录）"""
    db = _ec_db()
    try:
        existing = db.query(ReportFile).filter(
            ReportFile.file_name == file_name,
            ReportFile.user_id == user_id,
        ).first()
        if existing:
            existing.file_size = file_size or existing.file_size
            existing.file_type = file_type
            existing.data_year = data_year
            existing.report_period = report_period or f"{data_year}-{date.today().month:02d}"
            existing.row_count = row_count or existing.row_count
        else:
            db.add(ReportFile(
                user_id=user_id,
                file_name=file_name,
                file_size=file_size,
                file_type=file_type,
                data_year=data_year,
                report_period=report_period or f"{data_year}-{date.today().month:02d}",
                row_count=row_count,
            ))
        db.commit()
    finally:
        db.close()


def _create_report_record(year: int, month: int, report_name: str, row_count: int, file_type: str = "csv", user_id: int = 0):
    """创建或更新报表记录（按 file_name + user_id 去重）"""
    _upsert_report_file(
        file_name=report_name + ".csv",
        user_id=user_id,
        file_type=file_type,
        data_year=year,
        report_period=f"{year}-{month:02d}",
        row_count=row_count,
    )

# ----- 批量上传（客户端解析CSV分块上传，绕过代理超时）-----
_session_batches: dict[str, dict] = {}  # session_id -> {"total_rows": int, "received": int, "file_name": str, ...}

def _build_sales_record(row: list, headers: list, year: int, month: int, file_name: str, user_id: int = 0) -> dict | None:
    """从一行CSV数据构建SalesData记录字典"""
    def g(name: str, default="") -> str:
        for i, h in enumerate(headers):
            if h.strip().lower() == name.lower():
                return str(row[i]).strip() if i < len(row) else default
        return default
    def gi(name: str, default=0) -> int:
        try: return int(float(g(name, str(default))))
        except: return default
    def gf(name: str, default=0.0) -> float:
        try: return float(g(name, str(default)))
        except: return default
    
    vol = gi("销量", 0)
    amt = gf("销售额", 0)
    cost = gf("成本", 0)
    profit = gf("利润", 0)
    ret = gi("退货数", 0)
    
    if profit == 0 and amt > 0:
        profit = amt - cost
    margin = profit / amt if profit > 0 and amt > 0 else 0
    ret_rate = ret / vol if vol > 0 else 0
    
    return {
        "user_id": user_id,
        "report_name": file_name,
        "data_year": year,
        "data_month": month,
        "sku_code": g("sku编码", "")[:100],
        "product_name": g("商品名称", "")[:300],
        "category": g("类目", "")[:100],
        "style": g("版型", "")[:100],
        "color": g("颜色", "")[:50],
        "size": g("尺码", "")[:50],
        "sales_volume": vol,
        "sales_amount": amt,
        "cost": cost,
        "profit": profit,
        "profit_margin": margin,
        "return_count": ret,
        "return_rate": ret_rate,
        "inventory": gi("库存", 0),
    }

@app.post("/api/data/upload_batch")
async def ec_upload_batch(
    headers: list[str] = Body(...),
    rows: list[list] = Body(...),
    file_name: str = Body(...),
    batch_index: int = Body(...),
    total_batches: int = Body(...),
    data_year: int = Body(date.today().year),
    data_month: int = Body(0),
    user_session_id: str = Body(""),
):
    uid = _uid()
    from datetime import datetime as _dt
    session_id = f"batch_{file_name}_{data_year}_{data_month}"
    
    # 首次创建session
    if batch_index == 0:
        import csv as _csv
        _sess_tmp = os.path.join('/tmp', f"upload_{session_id}.csv")
        os.makedirs(os.path.join(WORKSPACE_DIR, "assets", "uploads"), exist_ok=True)
        with open(_sess_tmp, 'w', newline='', encoding='utf-8-sig') as _f:
            _csv.writer(_f).writerow(headers)
        _session_batches[session_id] = {
            "file_name": file_name,
            "year": data_year,
            "month": data_month,
            "total_batches": total_batches,
            "received": 0,
            "total_rows": 0,
            "tmp_path": _sess_tmp,
            "started_at": _dt.now().isoformat(),
        }
    
    # 构建记录并批量插入
    records = []
    for row in rows:
        rec = _build_sales_record(row, headers, data_year, data_month, file_name, user_id=uid)
        if rec:
            records.append(rec)
    
    if records:
        db = _ec_db()
        try:
            from sqlalchemy import insert as sa_insert
            stmt = sa_insert(SalesData).values(records)
            db.execute(stmt)
            db.commit()
        finally:
            db.close()
    
    # 更新session统计
    sess = _session_batches.get(session_id)
    if sess:
        sess["received"] += 1
        sess["total_rows"] += len(records)
    
    # 追加行数据到临时CSV
    if sess and sess.get("tmp_path"):
        try:
            import csv as _csv2
            with open(sess["tmp_path"], 'a', newline='', encoding='utf-8-sig') as _f:
                _csv2.writer(_f).writerows(rows)
        except Exception:
            pass
    
    is_last = batch_index == total_batches - 1
    result = {
        "success": True,
        "batch_index": batch_index,
        "total_batches": total_batches,
        "is_last": is_last,
        "inserted": len(records),
        "message": f"批次 {batch_index+1}/{total_batches} 已完成 ({len(records)}条数据)",
    }
    
    if is_last:
        # 保存CSV到uploads目录便于查看
        try:
            if sess and sess.get("tmp_path") and os.path.isfile(sess["tmp_path"]):
                uploads_dir = _user_uploads_dir(uid)
                import shutil
                dest_path = os.path.join(uploads_dir, file_name)
                shutil.copy2(sess["tmp_path"], dest_path)
                # 关联到电商小助手的 session，确保 /api/ask 能定位到刚上传的文件
                if user_session_id:
                    _track_session_file(user_session_id, dest_path)
        except Exception:
            pass
        # 创建汇总报表记录
        total = sess["total_rows"] if sess else len(records)
        report_name = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
        _create_report_record(data_year, data_month, report_name, total, user_id=uid)
        result["total_rows"] = total
        result["message"] = f"全部完成！共导入 {total} 条数据"
        # 清理session（延迟清理，让前端有机会查询）
        import threading as _th
        def _clean():
            import time as _t; _t.sleep(30)
            _session_batches.pop(session_id, None)
        _th.Thread(target=_clean, daemon=True).start()
    
    return result

@app.get("/api/data/files")
async def ec_data_files(year: int = Query(0)):
    db = _ec_db()
    uid = _uid()
    files_dict = {}
    try:
        db_files = db.query(ReportFile).filter(ReportFile.user_id == uid)
        if year > 0:
            db_files = db_files.filter(ReportFile.data_year == year)
        db_files = db_files.order_by(ReportFile.created_at.desc()).all()
        for f in db_files:
            files_dict[f.file_name] = {
                "id": f.id, "file_name": f.file_name, "file_size": f.file_size,
                "file_type": f.file_type, "data_year": f.data_year,
                "report_period": f.report_period, "row_count": f.row_count,
                "status": f.status,
                "created_at": f.created_at.isoformat() if f.created_at else ""
            }
    finally:
        db.close()
    for pf in _scan_user_files(uid):
        if pf["file_name"] not in files_dict:
            files_dict[pf["file_name"]] = pf
    files_list = sorted(files_dict.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    return {"success": True, "files": files_list}

@app.get("/api/data/list")
async def ec_data_list(year: int = Query(date.today().year), month: int = Query(0), report_name: str = Query(""), page: int = Query(1), page_size: int = Query(50)):
    db = _ec_db()
    uid = _uid()
    try:
        q = db.query(SalesData).filter(SalesData.user_id == uid, SalesData.data_year == year)
        if month > 0: q = q.filter(SalesData.data_month == month)
        if report_name: q = q.filter(SalesData.report_name == report_name)
        total = q.count()
        items = q.order_by(SalesData.data_month.desc(), SalesData.sales_volume.desc()).offset((page-1)*page_size).limit(page_size).all()
        result = [{"id":i.id,"report_name":i.report_name,"data_year":i.data_year,"data_month":i.data_month,"data_quarter":(i.data_month-1)//3+1,"sku_code":i.sku_code,"product_name":i.product_name,"category":i.category,"style":i.style,"color":i.color,"size":i.size,"sales_volume":i.sales_volume,"sales_amount":float(i.sales_amount),"cost":float(i.cost),"profit":float(i.profit),"profit_margin":float(i.profit_margin),"return_count":i.return_count,"return_rate":float(i.return_rate),"inventory":i.inventory} for i in items]
        return {"success": True, "total": total, "items": result, "page": page, "page_size": page_size}
    finally:
        db.close()

@app.delete("/api/data/{data_id}")
async def ec_data_delete(data_id: int):
    db = _ec_db()
    uid = _uid()
    try:
        item = db.query(SalesData).filter(SalesData.id == data_id, SalesData.user_id == uid).first()
        if not item: raise HTTPException(status_code=404, detail="不存在")
        db.delete(item); db.commit()
        return {"success": True, "message": "已删除"}
    finally:
        db.close()

@app.get("/api/data/file/{file_name:path}/raw")
async def ec_data_file_raw(file_name: str, page: int = Query(1, ge=1), page_size: int = Query(100, ge=10, le=1000)):
    """返回上传文件的原始内容"""
    import os, pandas as pd
    workspace = PROJECT_ROOT

    # 1) 精确匹配文件（用户目录优先，再查共享目录）
    uid = _uid()
    search_dirs = [_user_uploads_dir(uid), _user_comments_dir(uid)]
    for subdir in search_dirs:
        fp = os.path.join(subdir, file_name) if os.path.isabs(subdir) else os.path.join(workspace, subdir, file_name)
        if os.path.isfile(fp):
            return _read_raw_file(fp, file_name, page, page_size)

    # 2) 尝试自动补扩展名（.xlsx ↔ .csv）
    base, ext = os.path.splitext(file_name)
    for subdir in search_dirs:
        for try_ext in (".csv", ".xlsx", ".xls"):
            if ext.lower() == try_ext.lower():
                continue
            fp = os.path.join(workspace, subdir, base + try_ext)
            if os.path.isfile(fp):
                return _read_raw_file(fp, file_name, page, page_size)

    # 3) 仍未找到 → 从数据库重建（已解析的销售数据）
    try:
        db = _ec_db()
        row_count = db.query(ReportFile).filter(ReportFile.file_name == file_name).count()
        if row_count > 0:
            records = db.query(SalesData).filter(SalesData.report_name == file_name).order_by(SalesData.id).all()
            if records:
                total = len(records)
                start = (page - 1) * page_size
                end = min(start + page_size, total)
                columns = ["序号","主播","商品编码","销量(扣退)","销售额(扣退)","付款金额","实发件数","毛利额","实发金额"]
                rows = []
                for _i, r in enumerate(records[start:end]):
                    ed = r.extra_data or {}
                    name_val = _parse_kv_text(ed, ["主播","主播名称"])
                    sku_val = _parse_kv_text(ed, ["商品编码","商品编号"])
                    qty_val = _parse_kv(ed, ["商品销售数据-商品销售数量(扣退)","商品销售数据-商品销售数量","商品数据-实发数量"])
                    sale_val = _parse_kv(ed, ["商品销售数据-商品销售金额(扣退)","商品数据-付款金额","商品数据-实发金额"])
                    amt_val = _parse_kv(ed, ["商品数据-付款金额","商品销售数据-商品销售金额(扣退)","商品数据-实发金额"])
                    sent_val = _parse_kv(ed, ["商品数据-实发数量","商品销售数据-商品销售数量(扣退)"])
                    profit_val = _parse_kv(ed, ["利润-毛利额","毛利额","毛利"])
                    amt_sent_val = _parse_kv(ed, ["商品数据-实发金额","商品销售数据-商品销售金额(扣退)"])
                    rows.append([str(start + _i + 1),
                                 str(name_val)[:80], str(sku_val)[:80],
                                 str(qty_val), str(sale_val), str(amt_val),
                                 str(sent_val), str(profit_val), str(amt_sent_val)])
                db.close()
                return {"success": True, "file_name": file_name, "total": total,
                        "page": page, "page_size": page_size, "columns": columns, "rows": rows}
        db.close()
    except Exception:
        pass

    return {"success": False, "detail": f"文件 {file_name} 不存在（旧文件可重新上传）"}


def _read_raw_file(fp: str, display_name: str, page: int, page_size: int) -> dict:
    """读取物理文件内容（CSV/Excel/TXT），支持分页"""
    import pandas as pd
    import chardet
    ext = fp.rsplit(".", 1)[-1].lower() if "." in fp else ""
    try:
        if ext == "csv":
            # 用 chardet 准确检测编码
            with open(fp, "rb") as _f:
                _raw = _f.read(100000)
            _detect = chardet.detect(_raw)
            _enc = _detect.get("encoding", "utf-8") or "utf-8"
            _enc = _enc.lower().replace("-", "")
            _enc_map = {"utf8": "utf-8", "utf8sig": "utf-8-sig", "gb2312": "gbk",
                        "ascii": "utf-8", "gb18030": "gbk", "big5": "big5"}
            _enc = _enc_map.get(_enc, _enc)
            if _enc == "utf-8-sig":
                df = pd.read_csv(fp, encoding="utf-8-sig")
            else:
                try:
                    df = pd.read_csv(fp, encoding=_enc)
                except (UnicodeDecodeError, UnicodeError):
                    df = pd.read_csv(fp, encoding="utf-8", encoding_errors="replace")
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(fp)
        else:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return {"success": True, "file_name": display_name, "total": 1,
                    "page": 1, "page_size": 1, "columns": ["内容"], "rows": [[content[:5000]]]}
        total = len(df)
        start = (page - 1) * page_size
        end = min(start + page_size, total)
        page_df = df.iloc[start:end]
        columns = ["序号"] + [str(c) for c in df.columns]
        rows = []
        for idx, (_, row) in enumerate(page_df.iterrows()):
            rows.append([str(start + idx + 1)] + [str(v)[:200] if pd.notna(v) else "" for v in row])
        return {"success": True, "file_name": display_name, "total": total,
                "page": page, "page_size": page_size, "columns": columns, "rows": rows}
    except Exception as e:
        return {"success": False, "detail": f"读取失败: {e}"}


@app.delete("/api/data/file/{file_name:path}")
async def ec_data_file_delete(file_name: str):
    db = _ec_db()
    uid = _uid()
    try:
        # 删除文件记录
        files = db.query(ReportFile).filter(ReportFile.file_name == file_name, ReportFile.user_id == uid).all()
        for f in files: db.delete(f)
        # 删除该文件导入的所有数据
        rows = db.query(SalesData).filter(SalesData.report_name == file_name, SalesData.user_id == uid).all()
        for r in rows: db.delete(r)
        db.commit()
        # 删除物理文件
        _delete_uploaded_file(file_name, uid)
        return {"success": True, "message": f"已删除文件 {file_name}，共移除 {len(rows)} 条数据"}
    finally:
        db.close()

@app.post("/api/data/files/batch-delete")
async def ec_data_files_batch_delete(req: dict):
    """批量删除文件"""
    file_names = req.get("file_names", [])
    if not file_names:
        raise HTTPException(status_code=400, detail="请选择要删除的文件")
    db = _ec_db()
    uid = _uid()
    total_rows = 0
    try:
        for file_name in file_names:
            files = db.query(ReportFile).filter(ReportFile.file_name == file_name, ReportFile.user_id == uid).all()
            for f in files: db.delete(f)
            rows = db.query(SalesData).filter(SalesData.report_name == file_name, SalesData.user_id == uid).all()
            for r in rows: db.delete(r)
            total_rows += len(rows)
            _delete_uploaded_file(file_name, uid)
        db.commit()
        return {"success": True, "message": f"已删除 {len(file_names)} 个文件，共移除 {total_rows} 条数据"}
    finally:
        db.close()


# ============================================================
# 数据统计 & 分类清除
# ============================================================

@app.get("/api/data/stats")
async def ec_data_stats():
    """返回所有数据表的统计概览（仅当前用户）

    使用「合并视图」统计文件数：DB ReportFile + 物理磁盘文件 → 去重后即为实际文件数。
    这样无论文件是通过哪种方式上传的，统计结果都与文件管理页保持一致。
    """
    import os as _os
    db = _ec_db()
    uid = _uid()
    _valid_exts = {".csv", ".xlsx", ".xls"}
    try:
        sales_count = db.query(SalesData).filter(SalesData.user_id == uid).count()

        conv_count = db.query(Conversation).filter(Conversation.user_id == uid).count()
        session_count = db.query(Conversation.session_id).filter(Conversation.user_id == uid).distinct().count()
        image_count = db.query(GeneratedImage).filter(GeneratedImage.user_id == uid).count()

        # ── 文件统计：DB 记录 + 物理文件 → 合并去重 ──
        db_names = set()
        for (fn,) in db.query(ReportFile.file_name).filter(ReportFile.user_id == uid).distinct().all():
            db_names.add(fn)

        physical_files = 0
        physical_size = 0
        physical_names = set()
        for d in (_user_uploads_dir(uid), _user_comments_dir(uid)):
            if _os.path.isdir(d):
                for fn in _os.listdir(d):
                    if fn.startswith("."):
                        continue
                    ext = _os.path.splitext(fn)[1].lower()
                    if ext not in _valid_exts:
                        continue
                    fp = _os.path.join(d, fn)
                    if _os.path.isfile(fp):
                        physical_names.add(fn)
                        physical_files += 1
                        physical_size += _os.path.getsize(fp)

        # 合并视图：DB 有的 + 磁盘有的 = 所有文件
        all_file_names = db_names | physical_names
        total_file_count = len(all_file_names)

        return {
            "success": True,
            "stats": {
                "sales_rows": sales_count,
                "report_files": total_file_count,
                "conversations": conv_count,
                "sessions": session_count,
                "generated_images": image_count,
                "upload_files": total_file_count,
                "upload_size_mb": round(physical_size / 1024 / 1024, 2),
            }
        }
    finally:
        db.close()


class ClearRequest(BaseModel):
    category: str  # sales | conversations | images | uploads


@app.post("/api/data/clear")
async def ec_data_clear(req: ClearRequest):
    """按类别清除数据（仅清除当前用户的数据）"""
    import os as _os, glob as _glob, shutil as _shutil
    db = _ec_db()
    uid = _uid()
    deleted = 0
    try:
        if req.category == "sales":
            deleted = db.query(SalesData).filter(SalesData.user_id == uid).delete(synchronize_session="fetch")
            db.query(ReportFile).filter(ReportFile.user_id == uid).delete(synchronize_session="fetch")
            db.commit()
        elif req.category == "conversations":
            deleted = db.query(Conversation).filter(Conversation.user_id == uid).delete(synchronize_session="fetch")
            db.commit()
        elif req.category == "images":
            deleted = db.query(GeneratedImage).filter(GeneratedImage.user_id == uid).delete(synchronize_session="fetch")
            db.commit()
        elif req.category == "uploads":
            # 删除用户专属物理文件
            for d in (_user_uploads_dir(uid), _user_comments_dir(uid)):
                if _os.path.isdir(d):
                    for fn in _os.listdir(d):
                        fp = _os.path.join(d, fn)
                        if _os.path.isfile(fp) and not fn.startswith("."):
                            try:
                                _os.remove(fp)
                                deleted += 1
                            except Exception:
                                pass
            # 同步清空当前用户的 DB 记录
            db.query(SalesData).filter(SalesData.user_id == uid).delete(synchronize_session="fetch")
            db.query(ReportFile).filter(ReportFile.user_id == uid).delete(synchronize_session="fetch")
            db.commit()
        else:
            raise HTTPException(status_code=400, detail=f"未知类别: {req.category}，可选: sales / conversations / images / uploads")

        return {"success": True, "deleted": deleted, "message": f"已清除 {req.category} 数据"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"清除失败: {e}")
    finally:
        db.close()


def _delete_uploaded_file(file_name: str, uid: int = 0):
    """删除用户专属目录下的物理文件"""
    import os as _os, glob as _glob
    dirs = [_user_uploads_dir(uid), _user_comments_dir(uid)]
    for dir_path in dirs:
        if not os.path.isdir(dir_path):
            continue
        fp = os.path.join(dir_path, file_name)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except Exception:
                pass
        for found in _glob.glob(os.path.join(dir_path, f"*{os.path.splitext(file_name)[0]}*")):
            if found != fp and os.path.isfile(found):
                try:
                    os.remove(found)
                except Exception:
                    pass

# ----- 电商问答 -----
def _parse_question_zhubo(q: str):
    """从问题中解析出主播名称关键词
    如问"与辉同行的数据" → "与辉同行"
    如问"排名第一的主播" → 从数据中查找第1名主播
    """
    import re
    # 已知主播关键词列表
    known_zhubos = [
        "与辉同行", "小王家", "兰知春序", "柴碧云", "陈西贝",
        "小鱼故事studio", "小蛋黄omi", "李是时髦人", "雅君悦读",
        "王嘉绮", "东方甄选", "雅歌美学", "小柴有米盐",
        "淼淼", "一米五五胖欢", "珍妮冯冯", "草莓", "是苑苑",
        "海豚惊喜社", "是曼曼", "雪澜Yuki", "Eva形象美学",
        "Leo形象美学", "智勇别这样", "芋总", "莎莎"
    ]
    for z in known_zhubos:
        if z in q:
            return z
    # 通用匹配："主播"后的词
    m = re.search(r'主播[：:]\s*(\S+)', q)
    if m: return m.group(1)
    # 匹配"关于X的数据"、"X的销售额"等
    for prefix in ["关于", "查询", "查"]:
        for suffix in ["的数据", "的销售额", "的销量", "的成交"]:
            m = re.search(f'{prefix}(.+?){suffix}', q)
            if m: return m.group(1)

    # 排名匹配：如"排名第一的主播"、"第2名"、"榜首"等
    rank_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    rank_num = None
    # "排名第X" 或 "第X名" 或 "第X" 模式
    m = re.search(r'(?:排名)?第\s*([一二三四五六七八九十\d]+)\s*(?:名|位|个|的主播)?', q)
    if m:
        rank_str = m.group(1)
        if rank_str.isdigit():
            rank_num = int(rank_str)
        elif rank_str in rank_map:
            rank_num = rank_map[rank_str]
    # "榜首" → 第1名
    if '榜首' in q or '第一名' in q or 'TOP1' in q.upper():
        rank_num = 1
    if rank_num is not None and rank_num <= 20:
        # 从CSV数据中查找第N名主播
        import glob, os, pandas as pd
        upload_dir = _user_uploads_dir(_uid())
        files = sorted(glob.glob(os.path.join(upload_dir, "*.csv")))
        zhubo_cols = ["主播", "主播名称", "主播名", "达人", "达人名称"]
        sales_cols = ["商品销售数据-商品销售数量(扣退)", "扣退后销量(件)", "扣退后销量", "销量(扣退)", "销量", "销售数量"]
        from tools.data_query_tools import _read_file_to_df
        for fp in reversed(files):
            try:
                df = _read_file_to_df(fp)
                if df is None or df.empty: continue
                zhubo_col = next((c for c in zhubo_cols if c in df.columns), None)
                sales_col = next((c for c in sales_cols if c in df.columns), None)
                if zhubo_col is None: continue
                grouped = df.groupby(zhubo_col).agg({sales_col: 'sum'} if sales_col else {}).reset_index() if sales_col else df
                grouped = grouped.sort_values(by=sales_col, ascending=False) if sales_col else grouped
                ranked = grouped.reset_index(drop=True)
                if rank_num <= len(ranked):
                    name = str(ranked.iloc[rank_num-1][zhubo_col]).strip()
                    if name:
                        return name
            except:
                continue
    return None

def _get_sku_enhanced(top_n=20) -> str:
    """生成SKU的增强分析数据（占比、颜色分布等）"""
    import glob, os, pandas as pd
    upload_dir = _user_uploads_dir(_uid())
    files = sorted(glob.glob(os.path.join(upload_dir, "*.csv")))
    if not files:
        return ""
    from tools.data_query_tools import _read_file_to_df
    sku_cols = ["sku编码", "sku_code", "商品编码", "SKU编码"]
    for fp in reversed(files):
        try:
            df = _read_file_to_df(fp)
            sku_col = next((c for c in sku_cols if c in df.columns), None)
            qty_col = next((c for c in df.columns if "商品销售数据-商品销售数量(扣退)" in c or "销量" in c or "销售数量" in c), None)
            amt_col = next((c for c in df.columns if "商品销售数据-商品销售金额(扣退)" in c or "销售额" in c or "销售金额" in c), None)
            if sku_col and qty_col:
                sku_data = df.groupby(sku_col)[qty_col].sum()
                total_qty = sku_data.sum()
                top = sku_data.sort_values(ascending=False).head(top_n)
                top_total = top.sum()
                top_share = top_total / total_qty * 100 if total_qty > 0 else 0
                
                lines = [f"总SKU数: {len(sku_data)}, 总销量: {int(total_qty):,}件"]
                lines.append(f"TOP{top_n} SKU销量占比: {top_share:.1f}%（{int(top_total):,}/{int(total_qty):,}）")
                lines.append(f"TOP1单品占比: {top.iloc[0]/total_qty*100:.1f}%" if len(top) > 0 else "")
                
                # 颜色分析：从SKU编码提取颜色词
                color_map = {"丹宁": "丹宁色", "灰杏": "灰杏色", "复古浅蓝": "复古浅蓝", "米白": "米白色"}
                color_data = {}
                for sku, qty in sku_data.items():
                    sku_str = str(sku)
                    matched = False
                    for key, color in color_map.items():
                        if key in sku_str:
                            color_data[color] = color_data.get(color, 0) + qty
                            matched = True
                            break
                    if not matched:
                        color_data["其他"] = color_data.get("其他", 0) + qty
                
                if color_data:
                    color_sorted = sorted(color_data.items(), key=lambda x: -x[1])
                    lines.append("\n颜色分布（全量）:")
                    for color, qty in color_sorted:
                        pct = qty / total_qty * 100
                        lines.append(f"  {color}: {int(qty):,}件 ({pct:.1f}%)")
                
                # 价格带分析（如果有销售额列）
                if sku_col and amt_col and amt_col != qty_col:
                    amt_data = df.groupby(sku_col)[amt_col].sum()
                    top_amt = amt_data.sort_values(ascending=False).head(top_n)
                    total_amt = amt_data.sum()
                    lines.append(f"\n销售额TOP{top_n}占总销售额: {top_amt.sum()/total_amt*100:.1f}%")
                    avg_price = total_amt / total_qty if total_qty > 0 else 0
                    lines.append(f"整体平均客单价: ¥{avg_price:.2f}")
                
                return "\n".join(l for l in lines if l)
        except Exception:
            continue
    return ""

def _get_sku_from_csv(top_n=20) -> str:
    """从原始 CSV 文件读取 SKU 销售排行数据（取最新文件）"""
    import glob, os, pandas as pd
    upload_dir = _user_uploads_dir(_uid())
    files = sorted(glob.glob(os.path.join(upload_dir, "*.csv")))
    if not files:
        return ""
    return _get_sku_from_csv_for_file(files[-1], top_n=top_n)


def _get_sku_from_csv_for_file(file_path: str, top_n=20, label: str = "") -> str:
    """从指定 CSV 文件读取 SKU 销售排行数据。label 用于表头标注月份。"""
    import pandas as pd
    from tools.data_query_tools import _read_file_to_df
    from ecommerce.analysis import _parse_sku_display
    sku_cols = ["sku编码", "sku_code", "商品编码", "SKU编码"]
    try:
        df = _read_file_to_df(file_path)
        sku_col = next((c for c in sku_cols if c in df.columns), None)
        qty_col = next((c for c in df.columns if "商品销售数据-商品销售数量(扣退)" in c or "销量" in c or "销售数量" in c), None)
        amt_col = next((c for c in df.columns if "商品销售数据-商品销售金额(扣退)" in c or "销售额" in c or "销售金额" in c), None)
        name_col = next((c for c in df.columns if c in ("商品简称", "商品名称", "产品名称") or "商品简称" in c), None)
        if sku_col and qty_col:
            # 聚合：按 SKU 编码汇总 销量 + 销售额
            agg_map = {qty_col: "sum"}
            if amt_col and amt_col in df.columns:
                agg_map[amt_col] = "sum"
            grouped = df.groupby(sku_col).agg(agg_map).fillna(0)
            # 按销量降序
            sort_col = qty_col
            grouped = grouped.sort_values(sort_col, ascending=False).head(top_n)

            # 预建 SKU→商品名称 映射
            sku_name_map = {}
            if name_col and name_col in df.columns:
                for _, row in df[[sku_col, name_col]].dropna(subset=[name_col]).iterrows():
                    code = str(row[sku_col]).strip()
                    if code and code not in sku_name_map:
                        nm = str(row[name_col]).strip()
                        if nm and nm.lower() not in ("nan", "none", ""):
                            sku_name_map[code] = nm

            header = f"## {label} SKU TOP{top_n}" if label else f"## SKU TOP{top_n}"
            lines = [header, "", "| 排名 | SKU 编码 | 款号 | 商品名称 | 总销量 | 总销售额 |", "| --- | --- | --- | --- | --- | --- |"]
            for i, (sku, row) in enumerate(grouped.iterrows(), 1):
                code = str(sku).strip()
                display_code, model_code = _parse_sku_display(code)
                name = sku_name_map.get(code, "-")
                vol = int(row[qty_col])
                amt = float(row[amt_col]) if amt_col and amt_col in row.index else 0.0
                lines.append(f"| {i} | {display_code} | {model_code} | {name[:20]} | {vol:,} 件 | {amt:,.2f} 元 |")
            return "\n".join(lines) + f"\n\n共 {len(grouped)} 个 SKU"
    except Exception:
        pass
    return ""


def _file_month(file_path: str) -> int:
    """从文件名提取月份数字，如 '3月.csv' → 3"""
    import re, os
    m = re.search(r'(?<!\d)(\d{1,2})\s*月', os.path.basename(file_path))
    return int(m.group(1)) if m else 0


def _file_month_label(file_path: str) -> str:
    """从文件名提取月份标签，如 '3月.csv' → '3月'"""
    m = _file_month(file_path)
    return f"{m}月" if m else os.path.basename(file_path)[:20]


def _detect_question_context(question: str, session_id: str = "", previous_months: list = None) -> dict:
    """统一预处理：扫描数据中心、检测月份/对比意图/问题类型、定位目标文件。

    所有分析分支（SKU/评论/通用）共享此结果，确保 AI 不会「看不到」已上传的数据。
    previous_months: 从对话历史中提取的之前提到的月份，用于对比模式补全。
    """
    import re as _re, glob as _g, os as _os
    upload_dir = _user_uploads_dir(_uid())
    comment_dir = _user_comments_dir(_uid())

    ctx = {
        "question": question,
        "is_compare": False,
        "is_sku": False,
        "is_comment": False,
        "has_zhubo": False,
        "asked_months": [],
        "top_n": 10,
        "focus_zhubo": None,
        # 文件定位结果
        "sales_files": [],       # 匹配到的销售数据文件（按月份筛选后）
        "comment_files": [],     # 匹配到的评论文件
        "all_sales_files": [],   # uploads 下所有销售文件
        "all_comment_files": [], # comments 下所有评论文件
        "file_source": "",       # 数据来源标签
    }

    # ── 1. 扫描数据中心所有文件 ──
    for d, key in [(upload_dir, "all_sales_files"), (comment_dir, "all_comment_files")]:
        if _os.path.isdir(d):
            files = sorted(_g.glob(_os.path.join(d, "*.csv")) + _g.glob(_os.path.join(d, "*.xlsx")) + _g.glob(_os.path.join(d, "*.xls")))
            ctx[key] = files

    # ── 2. 问题类型检测 ──
    q = question
    ctx["is_sku"] = any(kw in q for kw in ["SKU", "sku", "商品编码", "款号", "货号", "单品"])
    ctx["is_comment"] = any(kw in q.lower() for kw in ["评论", "评价", "review", "好评", "差评", "中评"])
    ctx["has_zhubo"] = any(kw in q for kw in ["主播", "达人", "博主", "直播间", "排名"])
    ctx["is_compare"] = any(kw in q for kw in ["对比", "比较", "vs", "VS", "趋势", "变化", "差异", "区别"])

    # ── 3. 月份检测 ──
    ctx["asked_months"] = [int(m) for m in _re.findall(r'(\d+)\s*月', q) if 1 <= int(m) <= 12]
    # 中文数字月份（一~十二）
    _cn_num = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,"十一":11,"十二":12}
    for cn, num in _cn_num.items():
        if cn + "月" in q and num not in ctx["asked_months"]:
            ctx["asked_months"].append(num)
    # 自然语言月份
    if not ctx["asked_months"]:
        if any(kw in q for kw in ["本月", "这个月", "当月"]):
            ctx["asked_months"] = [date.today().month]
        elif "上个月" in q or "前一个月" in q:
            ctx["asked_months"] = [12 if date.today().month == 1 else date.today().month - 1]
        elif "下个月" in q:
            ctx["asked_months"] = [1 if date.today().month == 12 else date.today().month + 1]
        elif any(kw in q for kw in ["年底", "年末", "最后一个月", "末尾那个月", "最后那个月"]):
            ctx["asked_months"] = [12]
        elif any(kw in q for kw in ["年初", "第一个月", "最开始"]):
            ctx["asked_months"] = [1]
        elif "年中" in q:
            ctx["asked_months"] = [6]
        else:
            # "倒数第X个月" → 13-X
            m = _re.search(r'倒数第\s*(\S+)\s*个?月', q)
            if m:
                val = _cn_num.get(m.group(1)) or (int(m.group(1)) if m.group(1).isdigit() else None)
                if val and 1 <= val <= 11:
                    ctx["asked_months"] = [12 - val + 1]
            if not ctx["asked_months"]:
                # "倒数第二个月" → 11, etc
                for cn, num in _cn_num.items():
                    if f"倒数第{cn}个月" in q and num <= 11:
                        ctx["asked_months"] = [12 - num + 1]
                        break
    # 多月份自动启用对比模式（如 "7月和8月" 即使没说"对比"也按对比处理）
    if len(ctx["asked_months"]) >= 2:
        ctx["is_compare"] = True
    # 对比/追问模式只提到1个月 → 从对话历史补全另一个月
    # 如：先问"3月数据"再问"和2月对比" → 自动组合 [2, 3]
    if ctx["is_compare"] and len(ctx["asked_months"]) == 1 and previous_months:
        for pm in previous_months:
            if pm not in ctx["asked_months"]:
                ctx["asked_months"].append(pm)
                break

    # ── 4. TOP N 检测 ──
    top_match = _re.search(r'前\s*(\d+)', q)
    if top_match:
        ctx["top_n"] = min(int(top_match.group(1)), 50)
    else:
        # 支持中文数字：前十→10, 前二十→20, 前十五→15, 前三十→30 等
        _cn_num_map = {
            "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        }
        cn_top = _re.search(r'前\s*([一二两三四五六七八九十]+)', q)
        if cn_top:
            raw = cn_top.group(1)
            if raw == "十":
                ctx["top_n"] = 10
            elif raw.endswith("十"):
                # 二十/三十/四十... → 20/30/40...
                tens = _cn_num_map.get(raw[0], 0)
                ctx["top_n"] = min(tens * 10, 50) if tens > 0 else 10
            elif raw.startswith("十"):
                # 十一/十二/十三... → 11/12/13...
                ones = _cn_num_map.get(raw[1], 0) if len(raw) > 1 else 0
                ctx["top_n"] = min(10 + ones, 50)
            else:
                # 单字: 三→3, 五→5 等（"前三"这种说法）
                val = _cn_num_map.get(raw, 0)
                ctx["top_n"] = min(val, 50) if val > 0 else 10

    # ── 5. 文件定位：按月份匹配 ──
    def _is_comment_file(fp):
        fn = _os.path.basename(fp).lower()
        return any(kw in fn for kw in ["评论", "评价", "comment", "review"])

    # 分离销售文件和评论文件
    sales_all = [f for f in ctx["all_sales_files"] if not _is_comment_file(f)]
    comment_all = ctx["all_comment_files"] + [f for f in ctx["all_sales_files"] if _is_comment_file(f)]
    ctx["all_sales_files"] = sales_all
    ctx["all_comment_files"] = comment_all

    if ctx["asked_months"]:
        # 精确匹配指定月份的文件（不再做降级——没有就是没有）
        ctx["sales_files"] = [f for f in sales_all if _file_month(f) in ctx["asked_months"]]
        ctx["comment_files"] = [f for f in comment_all if _file_month(f) in ctx["asked_months"]]
        if ctx["sales_files"]:
            ctx["file_source"] = f"month:{','.join(str(m) for m in ctx['asked_months'])}"
        # 匹配不到时 sales_files 保持空，上层会给出明确提示
    elif ctx["is_compare"] and len(sales_all) >= 2:
        ctx["sales_files"] = sales_all  # 对比模式：用全部文件
        ctx["file_source"] = "all_compare"
    elif sales_all:
        # 1) 文件名关键词匹配：如问 "分析 11月.csv" → 直接定位该文件
        matched_file = None
        for f in sales_all:
            basename = _os.path.basename(f)
            name_no_ext = _os.path.splitext(basename)[0]
            if basename in q or name_no_ext in q:
                matched_file = f
                break
        if matched_file:
            ctx["sales_files"] = [matched_file]
            ctx["file_source"] = "filename_match"
        else:
            # 2) session 关联文件 → 追问上下文
            session_file = None
            if session_id:
                session_file = _session_latest_file.get(session_id)
            if not session_file:
                session_file = _session_latest_file.get("default")
            if session_file and _os.path.exists(session_file) and not _is_comment_file(session_file):
                ctx["sales_files"] = [session_file]
                ctx["file_source"] = "session"
            else:
                # 3) 没指定月份、没关联文件 → 不猜，提示用户指定
                ctx["sales_files"] = []
                ctx["file_source"] = "unspecified"
    if not ctx["comment_files"] and comment_all:
        ctx["comment_files"] = comment_all  # 评论兜底：全部评论文件

    # ── 6. 主播名检测 ──
    ctx["focus_zhubo"] = _parse_question_zhubo(q)

    return ctx


@app.post("/api/ask")
async def ec_ask(req: AskRequest):
    db = _ec_db()
    try:
        # ── 非分析类消息：直接闲聊，不加载数据 ──
        q = req.question.strip()
        _data_kw = ["数据", "销量", "销售", "利润", "分析", "对比", "排行", "多少",
                    "主播", "SKU", "sku", "商品", "排名", "占比", "趋势", "月", "年",
                    "报表", "统计", "费用", "成本", "退款", "退货", "库存", "价格",
                    "品牌", "来源", "渠道", "怎么样", "情况", "表现", "诊断",
                    "第", "最后", "上个", "上个", "这月", "上季度"]
        if not any(kw in q for kw in _data_kw):
            sp = "你是女装牛仔裤电商客服助手，风格亲切幽默。"
            answer = await asyncio.to_thread(_call_llm, prompt=q, system_prompt=sp, temperature=0.8, max_tokens=1024)
            if req.session_id:
                db.add(Conversation(user_id=_uid(), session_id=req.session_id, role="user", content=req.question, msg_type="chat"))
                db.add(Conversation(user_id=_uid(), session_id=req.session_id, role="assistant", content=answer, msg_type="chat"))
                db.commit()
            return {"success": True, "answer": answer, "table": ""}

        # 检测"本月"关键词，让 DB 兜底也按当前月份过滤
        period = "本月" if any(kw in req.question for kw in ["本月", "这个月", "当月"]) else ""
        summary = _get_sales_summary(db, period=period)

        # 从对话历史提取之前提到的月份，用于对比追问补全
        # 如：先问"3月数据"再问"和2月对比" → 自动组合 [2, 3] 做双月对比
        import re as _re_hist
        previous_months = []
        if req.session_id:
            recent = db.query(Conversation).filter(
                Conversation.session_id == req.session_id,
                Conversation.role == "user"
            ).order_by(Conversation.created_at.desc()).limit(3).all()
            for msg in recent:
                months = [int(m) for m in _re_hist.findall(r'(\d+)\s*月', msg.content) if 1 <= int(m) <= 12]
                previous_months.extend(months)
            # 去重并保持顺序
            seen = set()
            previous_months = [m for m in previous_months if not (m in seen or seen.add(m))]

        ctx = _detect_question_context(req.question, req.session_id, previous_months=previous_months)

        # ====== 评论分析 ======
        if ctx["is_comment"]:
            from ecommerce.analysis import analyze_comments_text
            comment_parts = []
            for fp in ctx["comment_files"][:3]:  # 最多3个评论文件
                text = analyze_comments_text(fp)
                if text and "[评论分析失败" not in text:
                    comment_parts.append(text)
            if not comment_parts:
                # 兜底：ctx 没匹配到评论文件时，扫描整个 comments 目录
                comment_parts.append(_get_comment_table(req.question, None))
            comment_table = "\n\n".join(comment_parts)

            sp = """# 角色
你是女装牛仔裤电商用户评论分析专家，擅长分析用户反馈、提取关键词感和发现产品问题。

# 规则
1. 所有分析基于提供的评论数据
2. 如果没有评论数据，明确告知用户尚未上传或导入评论文件
3. 结构化回答：结论→分析→建议
4. 按正向评价、负向评价、中性评价分类统计
5. 提取高频关键词（版型、面料、尺码、颜色、质量等维度）
6. 如果用户提到具体月份（如2月、3月），对比不同月份评论的差异
7. 格式整洁：段与段之间最多1个空行"""
            prompt = f"【评论数据】\n{comment_table}\n\n【用户问题】\n{req.question}"
            answer = await asyncio.to_thread(_call_llm, prompt=prompt, system_prompt=sp, temperature=0.3, max_tokens=32768)
            if req.session_id:
                db.add(Conversation(user_id=_uid(), session_id=req.session_id, role="user", content=req.question, msg_type="chat"))
                meta = {"table": comment_table} if comment_table else None
                db.add(Conversation(user_id=_uid(), session_id=req.session_id, role="assistant", content=answer, msg_type="chat", extra_meta=meta))
                db.commit()
            return {"success": True, "answer": answer, "table": comment_table}

        # ====== SKU / 商品分析 ======
        # 混合意图（SKU + 主播）路由到通用分析，避免丢失主播维度
        if ctx["is_sku"] and not ctx["has_zhubo"]:
            top_n = ctx["top_n"]
            has_compare = ctx["is_compare"] and len(ctx["sales_files"]) >= 2

            if has_compare:
                # 多月对比：每个匹配文件单独出 SKU 表
                sku_parts = []
                for fp in ctx["sales_files"]:
                    month_label = _file_month_label(fp)
                    table = _get_sku_from_csv_for_file(fp, top_n=top_n, label=month_label)
                    if table:
                        sku_parts.append(table)
                sku_table = "\n\n".join(sku_parts) if sku_parts else _get_sku_from_csv(top_n=top_n)
                sku_enhanced = ""
            elif ctx["sales_files"]:
                # 单月或全部：用匹配到的文件
                sku_table = _get_sku_from_csv_for_file(ctx["sales_files"][0], top_n=top_n)
                sku_enhanced = _get_sku_enhanced(top_n=top_n)
                if not sku_table:
                    target_month = _file_month(ctx["sales_files"][0]) if ctx["asked_months"] else None
                    sku_table = _get_sku_markdown_table(db, target_month=target_month, top_n=top_n)
                    sku_enhanced = ""
            else:
                sku_table = _get_sku_from_csv(top_n=top_n)
                sku_enhanced = _get_sku_enhanced(top_n=top_n)
                if not sku_table or len(sku_table.strip()) < 20:
                    sku_table = _get_sku_markdown_table(db, target_month=None, top_n=top_n)
                    sku_enhanced = ""

            sp = """# 角色定义
你是顶级电商数据分析师，精通女装牛仔裤品类的商品管理与销售分析。分析风格犀利、数据驱动、一针见血。

# ⚠️ 铁律：禁止输出表格
- 系统已将完整的数据表渲染在对话上方，用户已经看到了表格。
- **绝对禁止在你的回答中重复输出任何表格（包括简化的表格）。**
- 如果用户没有要求具体数据明细，直接从分析维度切入，不要列出数据行。
- 违反此规则会导致用户看到两份相同的表格，体验极差。

# 核心分析维度
1. **爆品特征**：哪些款式/颜色/尺码是爆款？共同特征？
2. **销售集中度**：TOP SKU销量占比，是否过度依赖某几个款？
3. **多月对比**（如有）：哪个月表现更好？爆款是否稳定？哪些SKU上升/下滑？
4. ** actionable 建议**：补货、营销、清仓等具体建议

# 回答规则
1. 所有分析基于提供的数据，**禁止编造**
2. 需要对比具体SKU时，用内联格式引用（如"522003C 丹宁色 28（515件，¥71,678.79）"），不要列成表格
3. 用数据说话，简洁有力，指出异常和风险"""
            data_parts = [f"SKU数据（{ctx['file_source']}）：\n{sku_table}"]
            if sku_enhanced:
                data_parts.append(f"增强分析：\n{sku_enhanced}")
            prompt = f"{chr(10).join(data_parts)}\n\n用户问题：{req.question}"
            answer = await asyncio.to_thread(_call_llm, prompt=prompt, system_prompt=sp, temperature=0.3, max_tokens=32768)
            if req.session_id:
                db.add(Conversation(user_id=_uid(), session_id=req.session_id, role="user", content=req.question, msg_type="chat"))
                meta = {"table": sku_table} if sku_table else None
                db.add(Conversation(user_id=_uid(), session_id=req.session_id, role="assistant", content=answer, msg_type="chat", extra_meta=meta))
                db.commit()
            return {"success": True, "answer": answer, "table": sku_table}

        # ====== 通用分析 ======
        import re as _re
        upload_dir = _user_uploads_dir(_uid())
        target_files = ctx["sales_files"]
        file_source = ctx["file_source"]

        # ---- 2. 对每个目标文件做深度聚合分析 ----
        zhubo_table = ""
        llm_data = ""
        source_hint = ""

        if target_files:
            deep_parts = []
            for fp in target_files:
                deep_result = _deep_analyze_csv(fp)
                if deep_result and "[CSV分析失败" not in deep_result:
                    deep_parts.append(deep_result)
            if deep_parts:
                llm_data = "\n\n---\n\n".join(deep_parts)
                # 多文件对比 → 生成并排对比表；单文件 → 提取概述表
                if len(target_files) >= 2:
                    zhubo_table = _build_comparison_table(target_files)
                if not zhubo_table:
                    zhubo_table = _extract_display_table(llm_data, max_lines=80)

        # ---- 2.5 检测问题是否针对特定主播 → 精简 LLM 上下文 ----
        focus_anchor = None
        if llm_data and not ctx["is_compare"]:
            # 从 deep analysis 的主播排名表中提取主播名列表（保持顺序 = 排名）
            # 实际表格格式: | 1 | 与辉同行 | 5,865件 | ¥889,345 | 8.5% | 72.3% |
            ranked_anchors = []
            in_zhubo_section = False
            for line in llm_data.split("\n"):
                if line.startswith("### 👤") or "主播排名" in line:
                    in_zhubo_section = True
                    continue
                if in_zhubo_section:
                    if line.startswith("###") or line.startswith("##"):
                        break  # 主播排名章节结束
                    m = _re.match(r'\|\s*(\d+)\s*\|\s*(.+?)\s*\|', line)
                    if m:
                        ranked_anchors.append(m.group(2).strip())

            # 方式0：调用 _parse_question_zhubo 从问题文本中检测主播名
            if not ranked_anchors:
                detected = _parse_question_zhubo(req.question)
                if detected:
                    focus_anchor = detected

            # 方式1：检测排名词（"第X名"、"榜首"、"TOP3"）
            rank_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
            rank_num = None
            m_rank = _re.search(r'第\s*([一二三四五六七八九十\d]+)\s*[名位个]', req.question)
            if m_rank:
                rs = m_rank.group(1)
                rank_num = int(rs) if rs.isdigit() else rank_map.get(rs)
            if not rank_num:
                if '榜首' in req.question or '第一名' in req.question:
                    rank_num = 1
                m_top = _re.search(r'(?:TOP|top)\s*(\d+)', req.question)
                if m_top:
                    rank_num = int(m_top.group(1))

            if rank_num and 1 <= rank_num <= len(ranked_anchors):
                focus_anchor = ranked_anchors[rank_num - 1]

            # 方式2：检查问题中是否包含某个主播全名
            if not focus_anchor:
                for anchor in ranked_anchors:
                    if len(anchor) >= 2 and anchor in req.question:
                        focus_anchor = anchor
                        break

            # 方式3：部分匹配（如"小王家"匹配"小王家女装旗舰店"）
            if not focus_anchor:
                for anchor in sorted(ranked_anchors, key=len, reverse=True):
                    if len(anchor) >= 3:
                        # 提取问题中的连续中文片段
                        cn_words = _re.findall(r'[一-鿿]{2,}', req.question)
                        for w in cn_words:
                            if len(w) >= 2 and w in anchor:
                                focus_anchor = anchor
                                break
                    if focus_anchor:
                        break

        if focus_anchor:
            # 保留排名表上下文让 AI 能做对比表格，同时提供该主播的精确数字
            overview_lines = []
            anchor_line = ""
            top_anchor_lines = []  # 保留 TOP 排名表供 AI 做对比
            in_overview = True
            in_anchor_table = False
            for line in llm_data.split("\n"):
                if in_overview:
                    overview_lines.append(line)
                    if line.strip() == "" and len(overview_lines) > 5:
                        in_overview = False
                # 捕获主播排名表（### 👤 之后的表格行）
                if line.startswith("### 👤"):
                    in_anchor_table = True
                    top_anchor_lines.append(line)
                    continue
                if in_anchor_table:
                    if line.startswith("###") or line.startswith("##"):
                        in_anchor_table = False
                    elif line.strip():
                        top_anchor_lines.append(line)
                if focus_anchor in line and "|" in line:
                    anchor_line = line
                # 在遇到 SKU 章节前保留排名表
                if line.startswith("### 📦"):
                    break

            # 解析该主播的精确数字
            anchor_parts = [p.strip() for p in anchor_line.split("|")] if anchor_line else []
            anchor_numbers = ""
            anchor_answers = ""  # 直接回答常见问题的答案
            # 格式: | 排名 | 主播 | 销量(扣退) | 销售额(扣退) | 利润率 | 退款率 |
            if len(anchor_parts) >= 7:
                _vol = anchor_parts[3]  # e.g. "3,733件"
                _amt = anchor_parts[4]  # e.g. "¥602,659.15"
                anchor_numbers = (
                    f"- 销量(扣退): {_vol}\n"
                    f"- 销售额(扣退): {_amt}\n"
                    f"- 经营利润率: {anchor_parts[5]}\n"
                    f"- 退款率: {anchor_parts[6]}\n"
                )
                anchor_answers = (
                    f"## 用户可能问的指标，直接用以下数字回答\n"
                    f"- 实际成交件数: {_vol}\n"
                    f"- 实际成交金额: {_amt}\n"
                    f"- 销售额: {_amt}\n"
                )

            # 从结构化数据补全实发件数/金额 & 退款数量/金额
            struct_extra = ""
            if target_files and focus_anchor:
                try:
                    from ecommerce.analysis import analyze_csv_structured
                    _s = analyze_csv_structured(target_files[0])
                    for _z in (_s.get("top_zhubo") or []):
                        if _z.get("zhubo") == focus_anchor:
                            _sq = _z.get("ship_qty", 0) or 0
                            _sa = _z.get("ship_amt", 0) or 0
                            _rq = _z.get("refund_qty", 0) or 0
                            _ra = _z.get("refund_amt", 0) or 0
                            if _sq > 0 or _sa > 0:
                                struct_extra = (
                                    f"- 实发件数: {_sq:,} 件\n"
                                    f"- 实发金额: {_sa:,.2f} 元\n"
                                    f"- 退款数量: {_rq:,} 件\n"
                                    f"- 退款金额: {_ra:,.2f} 元\n"
                                )
                            break
                except Exception:
                    pass
            if struct_extra:
                anchor_numbers += "\n" + struct_extra

            # 大盘对比 — 从 overview 提取
            from ecommerce.analysis import analyze_csv_structured
            overview_text = "\n".join(overview_lines[:12])
            benchmark = ""
            for line in overview_text.split("\n"):
                if "经营利润率" in line and "**" in line:
                    m = _re.search(r'\*\*(\d+\.?\d*)%\*\*', line)
                    if m:
                        benchmark += f"- 大盘经营利润率: {m.group(1)}%\n"
                if "退款率" in line and "**" in line:
                    m = _re.search(r'\*\*(\d+\.?\d*)%\*\*', line)
                    if m:
                        benchmark += f"- 大盘退款率: {m.group(1)}%\n"

            # 保留排名表（截取 TOP 15 行），AI 可用来做对比表格
            peer_context = ""
            if top_anchor_lines:
                # 取表头 + TOP 15 行数据
                peer_context = "\n".join(top_anchor_lines[:16])  # 1 header row + 15 data rows
                if len(top_anchor_lines) > 16:
                    peer_context += "\n（更多排名见上方完整数据表）"

            focused_data = (
                f"## {focus_anchor} — 精确数据（以下数字必须原样引用，不得改写）\n\n"
                f"⚠️ 术语说明（严格遵守）：\n"
                f"- 「实际成交」= 扣退后净数据（已扣除退货的最终确认数据）\n"
                f"- 「实际发货」= 仓库实际发出的数量和金额（含后续退货）\n\n"
                f"{anchor_answers}\n"
                f"## 完整数据\n{anchor_numbers}\n"
                f"## 实发与退款\n{struct_extra}\n"
                f"## 大盘对比\n{benchmark}\n"
            )
            if peer_context:
                focused_data += (
                    f"\n## 主播排名对比上下文（可用来做对比表格）\n"
                    f"（列说明：销量=扣退后净销量，销售额=扣退后净销售额，均已扣除退货）\n\n"
                    f"{peer_context}\n"
                )
            llm_data = focused_data
            source_hint = f"**数据来源：{os.path.basename(target_files[0])}** — 聚焦 {focus_anchor}"
            # 聚焦单一主播时不需要前端表格，LLM 回答已包含所有数据
            zhubo_table = ""

        # ---- 3. 无数据处理：用户没指定分析目标 ----
        if ctx["file_source"] == "unspecified":
            available = [os.path.basename(f) for f in ctx.get("all_sales_files", [])]
            if available:
                return {
                    "success": True,
                    "answer": f'请问要分析哪个月的数据？当前可用：{", ".join(available)}。例如输入「分析3月数据」。',
                    "table": "",
                }
            else:
                return {
                    "success": True,
                    "answer": "当前没有任何销售数据文件，请先在数据中心上传CSV报表。",
                    "table": "",
                }

        # ---- 4. 无数据处理：用户指定了月份但没匹配到文件 ----
        if not llm_data and ctx["asked_months"] and not target_files:
            month_list = "、".join(f"{m}月" for m in ctx["asked_months"])
            available = [os.path.basename(f) for f in ctx.get("all_sales_files", [])]
            available_hint = f"当前可用的数据：{', '.join(available)}" if available else "当前没有任何销售数据文件，请先上传。"
            return {
                "success": True,
                "answer": f"您没有{month_list}的数据。{available_hint}",
                "table": "",
            }

        # ---- 4. DB 兜底 ----
        if not llm_data:
            llm_data = summary

        # ---- 4. 构造 LLM prompt ----
        # 数据来源说明（如果 focus_anchor 已设置则不覆盖）
        if not source_hint:
            if file_source == "session":
                source_hint = f"**数据来源：你刚刚上传的文件**（{os.path.basename(target_files[0])}）"
            elif file_source.startswith("month:"):
                src_names = [os.path.basename(f) for f in target_files]
                source_hint = f"数据来源：{'、'.join(src_names)}"
            elif file_source == "all":
                src_names = [os.path.basename(f) for f in target_files]
                source_hint = f"数据来源：全部可用文件（{'、'.join(src_names)}）"
            elif file_source == "filename_match":
                source_hint = f"**数据来源：{os.path.basename(target_files[0])}**"

        sp = """# 角色定义
你是顶级电商数据分析师，专精女装牛仔裤品类。分析风格：犀利、数据驱动、直击要害。

# 🔴 铁律一：数字精确复制，严禁改写
**你引用的每一个数字都必须原样来自上方数据。** 这是硬性要求，没有任何例外。
- 如果数据里写的是「5,865件」，你必须写 5,865件，**不能**写成 6,500件、近6,000件、六千多件
- 如果数据里写的是「8.5%」，你必须写 8.5%，**不能**写成 8.3%、约8%、不到10%
- 如果数据里写的是「¥889,344.86」，你必须写 ¥889,345 或 ¥88.9万，**不能**写成 ¥98万
- 金额可以四舍五入到万位（¥88.9万），但数字必须来自原始数据
- **宁可少引用一个数字，也绝不编造一个数字**

# 🔴 铁律二：不要照抄原始数据块
不要把上方提供的数据原文照抄一遍。数据在你脑子里，直接给结论。
- ❌ 错误：先重复"📄 3月.csv — 8,957条记录...总销量25,440件..."再分析
- ✅ 正确：直接说"3月销售额¥403万，利润率22.4%，退款率71.3%——核心矛盾是..."

# 表格使用规则 — 在以下场景**应该使用 Markdown 表格**：
- **对比分析**（多月份/多主播/多SKU对比）：用表格并列展示关键指标，一目了然
- **排名展示**（TOP N）：用表格列出排名、名称、核心数字
- **维度拆解**（如按颜色/尺码/价格带分布）：表格比纯文字更清晰
- **单体分析**：可在末尾附一个精简的汇总表（3-5行核心指标），帮助用户快速回顾

表格示例格式：
| 月份 | 销量 | 销售额 | 利润率 | 退款率 |
|------|------|--------|--------|--------|
| 2月 | 12,340件 | ¥189万 | 21.3% | 68.5% |
| 3月 | 25,440件 | ¥403万 | 22.4% | 71.3% |

# 何时不用表格：
- 只有1-2个数字的简单问答
- 纯趋势描述、策略建议、定性分析
- 用户明确问"为什么"而非"是什么"

# 核心原则
1. 所有结论必须基于上方数据。没有的数据就说"数据未提供"，不要猜。
2. 表格服务于分析，不是为做而做——每个表下面给一句关键解读。

# 分析框架
- 整体分析 → 扫描所有维度，挑最关键的3-5个发现，附核心指标汇总表
- 单个主播 → **只讲这个主播**，用大盘均值做对比，可附该主播 vs 大盘对比表
- SKU/商品 → 只讲相关 SKU，TOP 排名用表格
- 对比/趋势 → 横向对比，必须用表格并列展示
- **来源/渠道** → 如果数据中有「成交来源分布」表，务必分析各渠道的销量、利润率差异，指出最赚钱的渠道和低效渠道

# 回答规则
1. **回答结构：先答数字，再给分析**。用户问某个主播的具体数据时，标准模板：
   第一句：「{主播} {月份}实际成交金额 ¥XXX，销售额 ¥XXX，实际成交 XXX件。」
   然后另起一段「**核心解读：**」从以下角度展开：
   · 排名定位 — 在全品类中的位置，和头部/同级主播的差距
   · 利润率 — vs 大盘均值、vs 排名相邻主播，判断赚钱效率
   · 退款表现 — vs 大盘，实发金额 vs 净销售额的差距，分析转化效率
   · 可执行建议 — 具体、可落地的优化方向
   ⚠️ 禁止只给数字不分析，也禁止只分析不给数字。两者缺一不可。
2. 用**加粗小标题**组织，段落间空一行
3. 数字用千分位（5,865件），金额用 ¥ 前缀
4. 表格在对比场景才用，单体分析用文字即可
5. **重要：所有的"销量""销售额"均已扣除退货（扣退后数据），不要再用退款率去折算这些数字**"""
        prompt = f"{source_hint}\n\n【全维度深度数据 — 仅供你分析，数字必须精确引用，不要在回答中重复原文】\n{llm_data}\n\n【用户问题】\n{req.question}"
        answer = _call_llm(prompt=prompt, system_prompt=sp, temperature=0.3, max_tokens=32768)
        # 压缩多余空行
        answer = _re.sub(r'\n{3,}', '\n\n', answer)
        if req.session_id:
            db.add(Conversation(user_id=_uid(), session_id=req.session_id, role="user", content=req.question, msg_type="chat"))
            meta = {"table": zhubo_table} if zhubo_table else None
            db.add(Conversation(user_id=_uid(), session_id=req.session_id, role="assistant", content=answer, msg_type="chat", extra_meta=meta))
            db.commit()
        return {"success": True, "answer": answer, "table": zhubo_table}
    except Exception as e:
        logger.exception("问答失败"); raise HTTPException(status_code=500, detail=f"问答失败: {e}")
    finally:
        db.close()

# ----- 运营文案 -----
@app.post("/api/copywriting")
async def ec_copywriting(req: CopywritingRequest):
    sp = """你是抖音女装牛仔裤顶级文案专家。你的文案风格：
- 超级口语化，像闺蜜聊天
- 每句话都有画面感，让用户"看到"自己穿上后的样子
- 精准戳中微胖/腿不直/胯宽/梨形身材女生的痛点
- 不喊口号，用细节说话

输出必须严格按以下格式，不要使用 Markdown 标记（不要用 # * - --- 等符号）：

【3个抖音标题】
① 标题一
② 标题二
③ 标题三

【3条FAB文案】

文案1（针对具体人群）
Feature（特性）：产品具体特性
Advantage（优势）：这个特性带来的好处
Benefit（利益）：用户穿上后的体感和效果

文案2（针对具体人群）
Feature（特性）：产品具体特性
Advantage（优势）：这个特性带来的好处
Benefit（利益）：用户穿上后的体感和效果

文案3（针对具体场景）
Feature（特性）：产品具体特性
Advantage（优势）：这个特性带来的好处
Benefit（利益）：用户穿上后的体感和效果

【完整口播】
写一段完整的口播文案"""
    prompt = f"""商品：{req.product_name}
版型：{req.style}
场景：{req.scene}
卖点：{req.key_selling_points}
风格：{req.tone}
人群：{req.target_audience}

请严格按照要求的格式输出。"""
    content = _call_llm(prompt=prompt, system_prompt=sp, temperature=0.8)
    return {"success": True, "content": content}

@app.post("/api/script")
async def ec_script(req: ScriptRequest):
    sp = "你是抖音女装牛仔裤视频脚本专家。输出分镜：时长/画面/台词/BGM/字幕。标注完播率/互动/转化节点。"
    prompt = f"生成{req.script_type}脚本\n商品：{req.product_name}\n时长：{req.duration}"
    content = _call_llm(prompt=prompt, system_prompt=sp, temperature=0.8)
    return {"success": True, "content": content}

def _search_web(query: str, max_results: int = 5) -> str:
    """用 DuckDuckGo 搜索，返回结果摘要。失败静默返回空字符串。"""
    try:
        import requests as _req
        from bs4 import BeautifulSoup
        url = "https://html.duckduckgo.com/html/"
        resp = _req.post(url, data={"q": query}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for r in soup.select(".result")[:max_results]:
            title = r.select_one(".result__title")
            snippet = r.select_one(".result__snippet")
            if title and snippet:
                results.append(f"- {title.get_text(strip=True)}：{snippet.get_text(strip=True)}")
        return "\n".join(results) if results else ""
    except Exception:
        return ""


@app.post("/api/hot-topics")
async def ec_hot_topics():
    try:
        # 多平台关键词搜索，采集真实热点数据
        queries = [
            "抖音 女装牛仔裤 2026年7月 最新热门趋势 爆款",
            "小红书 牛仔裤 穿搭 2026夏季 最新流行",
            "快手 女装牛仔裤 热卖 最新趋势 2026",
            "2026年夏季 牛仔裤 流行趋势 版型 颜色 最新",
        ]
        search_results = []
        for q in queries:
            logger.info(f"搜索: {q}")
            r = await asyncio.to_thread(_search_web, q)
            if r:
                search_results.append(f"【{q}】\n{r}")

        web_context = "\n\n---\n\n".join(search_results) if search_results else "（实时搜索未获取到结果，请基于训练数据作答）"

        sp = f"""你是抖音女装牛仔裤内容策略分析师，精通抖音、小红书、快手等平台的女装内容趋势。
当前时间是 2026年7月。你的分析必须基于最新的实时搜索结果和当前季节特点。
如果搜索结果不够新，请结合你的训练知识补充最新的2026年趋势。
风格：专业、具体、有数据感。用纯文字+序号呈现，禁止使用任何Markdown符号。"""

        prompt = f"""以下是各大平台的最新搜索结果（注意：当前是2026年7月夏季）：

{web_context}

基于以上搜索结果，按以下结构输出分析（务必针对2026年当前最新情况，不要用过时数据）：

一、当前热门方向（3-5个）
- 必须是2026年当下的最新热点
- 结合夏季特点分析

二、用户偏好分析
- 2026年最新版型偏好、身材适配、风格流量、颜色风向

三、季节性趋势
- 当前7月盛夏的牛仔裤穿搭策略
- 2026秋冬趋势预告

四、机会点与选题建议
- 2026年蓝海方向
- 内容形式建议
- 10个最新爆款选题方向

要求：所有分析都要标注时间感，优先引用最新的搜索结果，明确这是2026年当下的分析。"""
        content = await asyncio.to_thread(_call_llm, prompt=prompt, system_prompt=sp, temperature=0.7)
        return {"success": True, "content": content}
    except Exception as e:
        logger.error(f"热门选题分析失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"热门选题分析失败: {str(e)}")

@app.post("/api/strategy")
async def ec_strategy(req: StrategyRequest):
    db = _ec_db()
    try:
        summary = _get_sales_summary(db, period=req.period)
        sp = "你是女装牛仔裤电商运营策略顾问。基于数据给建议，要求具体可执行。"
        prompt = f"基于以下数据给{req.period}策略建议:\n{summary}\n\n分析:1)数据总览 2)主推款 3)定价 4)活动 5)预警"
        content = await asyncio.to_thread(_call_llm, prompt=prompt, system_prompt=sp, temperature=0.5)
        return {"success": True, "content": content}
    except Exception as e:
        logger.error(f"策略分析失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"策略分析失败: {str(e)}")
    finally:
        db.close()

# ----- AI生图 -----
@app.post("/api/generate-image")
async def ec_gen_image(req: ImageGenRequest):
    try:
        from utils.llm import generate_image
        urls = generate_image(prompt=req.prompt, size=req.size)
        if urls:
            db = _ec_db()
            try:
                for url in urls:
                    db.add(GeneratedImage(user_id=_uid(), prompt=req.prompt, image_url=url, style=req.style, size=req.size))
                db.commit()
            except Exception as e:
                logger.warning(f"保存记录失败: {e}")
            finally:
                db.close()
            return {"success": True, "image_urls": urls, "prompt": req.prompt}
        else:
            return {"success": False, "message": "当前 LLM 服务商不支持生图。请在 .env 中设置 IMAGE_GEN_BASE_URL / IMAGE_GEN_API_KEY / IMAGE_GEN_MODEL 指向支持生图的服务（如硅基流动 SiliconFlow、OpenAI 等）", "image_urls": []}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("生图失败"); raise HTTPException(status_code=500, detail=f"失败: {e}")

@app.get("/api/image-history")
async def ec_img_history(page: int = Query(1), page_size: int = Query(20)):
    db = _ec_db()
    uid = _uid()
    try:
        items = db.query(GeneratedImage).filter(GeneratedImage.user_id == uid).order_by(GeneratedImage.created_at.desc()).offset((page-1)*page_size).limit(page_size).all()
        total = db.query(GeneratedImage).filter(GeneratedImage.user_id == uid).count()
        return {"success": True, "items": [{"id":i.id,"prompt":i.prompt,"image_url":i.image_url,"style":i.style,"size":i.size,"created_at":i.created_at.isoformat() if i.created_at else ""} for i in items], "total": total, "page": page}
    finally:
        db.close()

# ============================================================
# 对话历史管理 (重写版 — 无截断、支持搜索/删除/导出)
# ============================================================

@app.post("/api/conversations")
async def ec_conversations(req: SessionHistoryRequest):
    """获取指定会话的全部消息（完整内容，不截断）"""
    db = _ec_db()
    uid = _uid()
    try:
        q = db.query(Conversation).filter(Conversation.user_id == uid, Conversation.session_id == req.session_id)
        if req.msg_type:
            q = q.filter(Conversation.msg_type == req.msg_type)
        items = q.order_by(Conversation.created_at).all()
        return {
            "success": True,
            "items": [
                {
                    "id": i.id,
                    "role": i.role,
                    "content": i.content,          # 完整内容，不截断
                    "msg_type": i.msg_type,
                    "extra_meta": i.extra_meta,
                    "created_at": i.created_at.isoformat() if i.created_at else "",
                }
                for i in items
            ],
            "session_id": req.session_id,
        }
    finally:
        db.close()


@app.get("/api/conversations/sessions")
async def ec_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=5, le=100),
    search: str = Query("", description="搜索关键词"),
):
    """列出当前用户的全部会话 — 单次查询避免 N+1，支持关键词搜索"""
    db = _ec_db()
    uid = _uid()
    try:
        # 子查询：每个 session 的聚合信息（仅当前用户）
        subq = (
            db.query(
                Conversation.session_id,
                func.max(Conversation.created_at).label("last_time"),
                func.min(Conversation.created_at).label("first_time"),
                func.count(Conversation.id).label("message_count"),
            )
            .filter(Conversation.user_id == uid)
            .group_by(Conversation.session_id)
            .subquery()
        )

        # 基础查询
        base_q = db.query(subq)

        # 关键词搜索：找到包含关键词的 session（仅当前用户）
        if search.strip():
            matched_sessions = (
                db.query(Conversation.session_id)
                .filter(Conversation.user_id == uid, Conversation.content.ilike(f"%{search.strip()}%"))
                .distinct()
                .subquery()
            )
            base_q = base_q.filter(subq.c.session_id.in_(
                db.query(matched_sessions.c.session_id)
            ))

        total = base_q.count()
        sessions = (
            base_q
            .order_by(subq.c.last_time.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

        if not sessions:
            return {"success": True, "items": [], "total": 0, "page": page, "page_size": page_size}

        # 批量取每个 session 的首条 user 消息（用作标题）
        session_ids = [s.session_id for s in sessions]
        first_msgs = (
            db.query(Conversation.session_id, Conversation.content)
            .filter(
                Conversation.user_id == uid,
                Conversation.session_id.in_(session_ids),
                Conversation.role == "user",
            )
            .order_by(Conversation.session_id, Conversation.created_at)
            .all()
        )
        # 每个 session 取第一条
        title_map = {}
        for sid, content in first_msgs:
            if sid not in title_map:
                title_map[sid] = (content[:120] + "…") if len(content) > 120 else content

        # 批量取每个 session 的 assistant 消息数（用于丰富信息）
        assistant_counts = dict(
            db.query(
                Conversation.session_id,
                func.count(Conversation.id),
            )
            .filter(
                Conversation.session_id.in_(session_ids),
                Conversation.role == "assistant",
            )
            .group_by(Conversation.session_id)
            .all()
        )

        result = []
        for s in sessions:
            title = title_map.get(s.session_id, "新对话")
            result.append({
                "session_id": s.session_id,
                "title": title,
                "message_count": s.message_count,
                "assistant_count": assistant_counts.get(s.session_id, 0),
                "created_at": s.first_time.isoformat() if s.first_time else "",
                "last_time": s.last_time.isoformat() if s.last_time else "",
            })

        return {
            "success": True,
            "items": result,
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    finally:
        db.close()


@app.delete("/api/conversations/sessions/{session_id}")
async def ec_delete_session(session_id: str):
    """删除整个会话的全部消息（仅限当前用户）"""
    db = _ec_db()
    uid = _uid()
    try:
        deleted = (
            db.query(Conversation)
            .filter(Conversation.user_id == uid, Conversation.session_id == session_id)
            .delete(synchronize_session="fetch")
        )
        db.commit()
        return {"success": True, "deleted": deleted, "message": f"已删除 {deleted} 条消息"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")
    finally:
        db.close()


@app.delete("/api/conversations/{msg_id}")
async def ec_delete_message(msg_id: int):
    """删除单条消息（仅限当前用户）"""
    db = _ec_db()
    uid = _uid()
    try:
        msg = db.query(Conversation).filter(Conversation.id == msg_id, Conversation.user_id == uid).first()
        if not msg:
            raise HTTPException(status_code=404, detail="消息不存在")
        db.delete(msg)
        db.commit()
        return {"success": True, "message": "已删除"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")
    finally:
        db.close()

# ----- 看板 -----
@app.get("/api/dashboard")
async def ec_dashboard(year: int = Query(0), month: int = Query(0), file_name: str = Query("")):
    """数据看板 — 优先从 CSV 计算（快速+完整），DB 兜底。"""
    from ecommerce.analysis import get_dashboard_data
    uid = _uid()

    # 1) 优先 CSV（比 DB 快 100x，且数据完整）
    csv_data = get_dashboard_data(file_name=file_name, month=month, user_id=uid)
    if csv_data:
        csv_data["year"] = year
        csv_data["month"] = month
        return csv_data

    # 2) DB 兜底（只有 CSV 不可用时才走这里）
    db = _ec_db()
    try:
        base_q = db.query(SalesData).filter(SalesData.user_id == uid)
        if year > 0:
            base_q = base_q.filter(SalesData.data_year == year)
        if month > 0:
            base_q = base_q.filter(SalesData.data_month == month)
        if file_name:
            base_q = base_q.filter(SalesData.report_name == file_name)
        rows = base_q.all()
        if not rows:
            return {"success": True, "overview": {"koutui_vol": 0, "koutui_amt": 0, "profit": 0, "profit_rate": 0, "refund_qty": 0, "refund_rate": 0, "sku_count": 0, "zhubo_count": 0}, "top_zhubo": [], "top_sku": [], "files": []}

        total_kv = total_ka = total_p = total_rq = 0.0
        sku_set, zhubo_map, sku_map = set(), {}, {}
        for r in rows:
            ed = r.extra_data or {}
            def ev(*ks, d=0.0):
                for k in ks:
                    v = ed.get(k)
                    if v is not None and v != "" and str(v).lower() != "nan":
                        try: return float(v)
                        except: pass
                return float(d)
            kv = ev("利润-销售数量(扣退)", "商品销售数据-商品销售数量(扣退)", d=float(r.sales_volume or 0))
            ka = ev("利润-销售金额(扣退)", "商品销售数据-商品销售金额(扣退)", d=float(r.sales_amount or 0))
            pv = ev("利润-经营利润", "利润-毛利额", d=float(r.profit or 0))
            rq = ev("售后合计-退款数量合计", "退款数量合计", d=float(r.return_count or 0))
            total_kv += kv; total_ka += ka; total_p += pv; total_rq += rq
            sku = (r.sku_code or "").strip()
            if sku:
                sku_set.add(sku)
                sku_map.setdefault(sku, {"n": r.product_name or sku, "v": 0, "a": 0, "p": 0})
                sku_map[sku]["v"] += kv; sku_map[sku]["a"] += ka; sku_map[sku]["p"] += pv
            zb = (ed.get("主播") or "").strip()
            if zb and zb.lower() != "nan":
                zhubo_map.setdefault(zb, {"v": 0, "a": 0, "p": 0})
                zhubo_map[zb]["v"] += kv; zhubo_map[zb]["a"] += ka; zhubo_map[zb]["p"] += pv
        pr = round(total_p / total_ka * 100, 2) if total_ka > 0 else 0
        rr = round(total_rq / (total_kv + total_rq) * 100, 2) if (total_kv + total_rq) > 0 else 0
        tz = sorted(zhubo_map.items(), key=lambda x: -x[1]["v"])[:20]
        ts = sorted(sku_map.items(), key=lambda x: -x[1]["v"])[:20]
        files_q = db.query(ReportFile).filter(ReportFile.user_id == uid).all()
        return {"success": True, "year": year, "month": month,
                "overview": {"koutui_vol": int(total_kv), "koutui_amt": round(total_ka, 2), "profit": round(total_p, 2), "profit_rate": pr, "refund_qty": int(total_rq), "refund_rate": rr, "sku_count": len(sku_set), "zhubo_count": len(zhubo_map)},
                "top_zhubo": [{"zhubo": z[0], "vol": int(z[1]["v"]), "amt": round(z[1]["a"], 2), "profit": round(z[1]["p"], 2)} for z in tz],
                "top_sku": [{"product_code": s[0], "product_name": s[1]["n"], "vol": int(s[1]["v"]), "amt": round(s[1]["a"], 2), "profit": round(s[1]["p"], 2)} for s in ts],
                "files": [{"file_name": f.file_name, "row_count": f.row_count} for f in files_q]}
    finally:
        db.close()

def _calc_dashboard_from_csv(year: int = 0, month: int = 0, file_name: str = "") -> Optional[Dict]:
    """从 CSV 文件构建 Dashboard（委托给 analysis 模块）。"""
    from ecommerce.analysis import get_dashboard_data
    return get_dashboard_data(file_name=file_name, month=month, user_id=_uid())


# ----- 首页 -----
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ec_index():
    idx = os.path.join(EC_STATIC_DIR, "index.html")
    if os.path.exists(idx):
        with open(idx, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>女装牛仔裤电商智能助手</h1><p>正在加载...</p>"


def parse_args():
    parser = argparse.ArgumentParser(description="Start FastAPI server")
    parser.add_argument("-m", type=str, choices=["http", "flow", "node", "agent"], default="http", help="运行模式")
    parser.add_argument("-p", type=int, default=8002, help="HTTP服务端口")
    parser.add_argument("-n", type=str, default=None, help="节点名称")
    parser.add_argument("-i", type=str, default=None, help="输入数据")
    return parser.parse_args()


def parse_input(input_str: str) -> Dict[str, Any]:
    """Parse input string, support both JSON string and plain text"""
    if not input_str:
        return {"text": "你好"}

    # Try to parse as JSON first
    try:
        return json.loads(input_str)
    except json.JSONDecodeError:
        # If not valid JSON, treat as plain text
        return {"text": input_str}

def start_http_server(port):
    workers = 1
    reload = False
    if _is_dev_env():
        reload = True

    logger.info(f"Start HTTP Server, Port: {port}, Workers: {workers}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload, workers=workers)

if __name__ == "__main__":
    args = parse_args()
    if args.m == "http":
        start_http_server(args.p)
    elif args.m == "flow":
        payload = parse_input(args.i)
        result = asyncio.run(service.run(payload))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.m == "node" and args.n:
        payload = parse_input(args.i)
        result = asyncio.run(service.run_node(args.n, payload))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.m == "agent":
        agent_ctx = new_context(method="agent")
        for chunk in service.stream(
                {
                    "type": "query",
                    "session_id": "1",
                    "message": "你好",
                    "content": {
                        "query": {
                            "prompt": [
                                {
                                    "type": "text",
                                    "content": {"text": "现在几点了？请调用工具获取当前时间"},
                                }
                            ]
                        }
                    },
                },
                run_config={"configurable": {"session_id": "1"}},
                ctx=agent_ctx,
        ):
            print(chunk)
