import argparse
import asyncio
import glob
import json
import os
import threading
import traceback
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterable, AsyncIterable, AsyncGenerator, Optional
import cozeloop
import uvicorn
import time
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from coze_coding_utils.runtime_ctx.context import new_context, Context
from coze_coding_utils.helper import graph_helper
from coze_coding_utils.log.node_log import LOG_FILE
from coze_coding_utils.log.write_log import setup_logging, request_context
from coze_coding_utils.log.config import LOG_LEVEL
from coze_coding_utils.error.classifier import ErrorClassifier, classify_error
from coze_coding_utils.helper.stream_runner import AgentStreamRunner, WorkflowStreamRunner,agent_stream_handler,workflow_stream_handler, RunOpt
from storage.database.db import get_session, get_engine
from storage.memory.memory_saver import get_memory_saver
from storage.database.shared.model import Base
from coze_coding_utils.async_tasks import (
    AsyncTaskRuntime,
    AsyncTaskStorageError,
    extract_biz_context,
    parse_deadline_sec,
)
from coze_coding_utils.async_tasks import config as async_task_config
from coze_coding_utils.async_tasks.headers import HEADER_X_RUN_ID as _ASYNC_HEADER_X_RUN_ID
from coze_coding_utils.runtime_ctx.context import new_context as _new_async_ctx
from sqlalchemy import event, func
from datetime import datetime, date

setup_logging(
    log_file=LOG_FILE,
    max_bytes=100 * 1024 * 1024, # 100MB
    backup_count=5,
    log_level=LOG_LEVEL,
    use_json_format=True,
    console_output=True
)

logger = logging.getLogger(__name__)
from coze_coding_utils.helper.agent_helper import to_stream_input

WORKSPACE_DIR = os.environ.get("COZE_WORKSPACE_PATH", os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from coze_coding_utils.openai.handler import OpenAIChatHandler
from coze_coding_utils.log.parser import LangGraphParser
from coze_coding_utils.log.err_trace import extract_core_stack
from coze_coding_utils.log.loop_trace import init_run_config, init_agent_config


# 超时配置常量
TIMEOUT_SECONDS = 900  # 15分钟

class GraphService:
    def __init__(self):
        # 用于跟踪正在运行的任务（使用asyncio.Task）
        self.running_tasks: Dict[str, asyncio.Task] = {}
        # 错误分类器
        self.error_classifier = ErrorClassifier()
        # stream runner
        self._agent_stream_runner = AgentStreamRunner()
        self._workflow_stream_runner = WorkflowStreamRunner()
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
            if graph_helper.is_agent_proj():
                self._graph = graph_helper.get_agent_instance("agents.agent", ctx)
            else:
                self._graph = graph_helper.get_graph_instance("graphs.graph")
            return self._graph

    @staticmethod
    def _sse_event(data: Any, event_id: Any = None) -> str:
        id_line = f"id: {event_id}\n" if event_id else ""
        return f"{id_line}event: message\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

    def _get_stream_runner(self):
        if graph_helper.is_agent_proj():
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
            run_config = init_run_config(graph, ctx)
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
                f"Traceback:\n{extract_core_stack()}"
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
        if graph_helper.is_agent_proj():
            run_config = init_agent_config(graph, ctx)
        else:
            run_config = init_run_config(graph, ctx)  # vibeflow

        is_workflow = not graph_helper.is_agent_proj()

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
            cozeloop.flush()

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
        node_func, input_cls, output_cls = graph_helper.get_graph_node_func_with_inout(_graph.get_graph(), node_id)
        if node_func is None or input_cls is None:
            raise KeyError(f"node_id '{node_id}' not found")

        parser = LangGraphParser(_graph)
        metadata = parser.get_node_metadata(node_id) or {}

        _g = StateGraph(input_cls, input_schema=input_cls, output_schema=output_cls)
        _g.add_node("sn", node_func, metadata=metadata)
        _g.set_entry_point("sn")
        _g.add_edge("sn", END)
        _graph = _g.compile()

        run_config = init_run_config(_graph, ctx)
        return await _graph.ainvoke(payload, config=run_config)

    def graph_inout_schema(self) -> Any:
        if graph_helper.is_agent_proj():
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

async_runtime: Optional[AsyncTaskRuntime] = None
async_graph: Optional[CompiledStateGraph] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_engine()
    @event.listens_for(engine, "connect")
    def _set_utc(dbapi_conn, _):
        with dbapi_conn.cursor() as cur:
            cur.execute("SET TIME ZONE 'UTC'")
    checkpointer = get_memory_saver()
    if graph_helper.is_agent_proj():
        base = graph_helper.get_agent_instance("agents.agent", None)
        sync_graph = base.builder.compile(checkpointer=checkpointer)
    else:
        base = graph_helper.get_graph_instance("graphs.graph")
        sync_graph = base.builder.compile()
    global async_graph, async_runtime
    async_graph = base.builder.compile(checkpointer=checkpointer)
    service.set_graph(sync_graph)
    async_runtime = AsyncTaskRuntime(
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

# OpenAI 兼容接口处理器
openai_handler = OpenAIChatHandler(service)


@app.post("/async_run")
async def http_async_run(request: Request) -> dict:
    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in http_async_run: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {extract_core_stack()}")
    try:
        deadline_sec = parse_deadline_sec(request.headers)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 一个 ID 走到底：task_id == run_id == thread_id == ctx.run_id == coze_run_id。
    # 优先用上游 x-run-id；没传就生成 UUID。
    run_id = request.headers.get(_ASYNC_HEADER_X_RUN_ID) or uuid.uuid4().hex

    # ctx 在 handler scope 构造，与同步 /run 路径一致；后面 new_context 默认会
    # 给 run_id 一个新 UUID，同步路径也是显式覆盖（main.py /run 处），这里同理。
    ctx = _new_async_ctx(method="async_run", headers=request.headers)
    ctx.run_id = run_id
    request_context.set(ctx)  # 与其他 HTTP endpoint 一致：让日志组件拿到 run_id 等信息
    run_config = init_run_config(async_graph, ctx)
    run_config["recursion_limit"] = async_task_config.RECURSION_LIMIT
    run_config.setdefault("configurable", {})["thread_id"] = run_id

    biz_context = extract_biz_context(request.headers) or {}
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
    except AsyncTaskStorageError as e:
        raise HTTPException(status_code=503,
                            detail=f"async-task storage unavailable: {e}")


@app.get("/task/{task_id}")
async def http_get_task(task_id: str) -> dict:
    try:
        row = await async_runtime.get(task_id)
    except AsyncTaskStorageError as e:
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
        raise HTTPException(status_code=400, detail=f"Invalid JSON format, {extract_core_stack()}")

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
                "stack_trace": extract_core_stack(),
            }
        )
    finally:
        cozeloop.flush()


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
                            detail=f"Invalid JSON format: {body_text}, traceback: {extract_core_stack()}, error: {e}")
    run_id = ctx.run_id
    is_agent = graph_helper.is_agent_proj()
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
        raise HTTPException(status_code=400, detail=f"Invalid JSON format:{extract_core_stack()}")

    if is_agent:
        stream_generator = agent_stream_handler(
            payload=payload,
            ctx=ctx,
            run_id=run_id,
            stream_sse_func=service.stream_sse,
            sse_event_func=service._sse_event,
            error_classifier=service.error_classifier,
            register_task_func=_register_task,
        )
    else:
        stream_generator = workflow_stream_handler(
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
        raise HTTPException(status_code=400, detail=f"Invalid JSON format:{extract_core_stack()}")
    try:
        return await service.run_node(node_id, payload, ctx)
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"node_id '{node_id}' not found or input miss required fields, traceback: {extract_core_stack()}")
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
                "stack_trace": extract_core_stack(),
            }
        )
    finally:
        cozeloop.flush()


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
        cozeloop.flush()


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
EC_STATIC_DIR = os.path.join(os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects"), "assets", "ecommerce")

@app.on_event("startup")
def create_ecommerce_tables():
    """创建电商数据表"""
    try:
        engine = get_engine()
        EcomBase.metadata.create_all(bind=engine)
        logger.info("✅ 电商智能助手数据表创建完成")
    except Exception as e:
        logger.warning(f"电商表创建(可能已存在): {e}")

def _ec_db():
    return get_session()

def _make_pwd(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _call_llm(prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
    """调用大模型"""
    from coze_coding_dev_sdk import LLMClient
    from coze_coding_utils.runtime_ctx.context import new_context
    from langchain_core.messages import SystemMessage, HumanMessage
    ctx = new_context(method="ec_llm")
    client = LLMClient(ctx=ctx)
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    resp = client.invoke(messages=messages, model="doubao-seed-2-0-lite-260215", temperature=temperature, max_completion_tokens=8192)
    if isinstance(resp.content, str):
        return resp.content
    elif isinstance(resp.content, list):
        parts = []
        for item in resp.content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    return str(resp.content)

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

def _get_sales_summary(db, year=None):
    """获取完整销售数据摘要（含extra_data解析和主播维度）"""
    rows = db.query(SalesData).filter(SalesData.data_year == year).all()
    # 解析所有行
    records = []
    for r in rows:
        ed = r.extra_data if isinstance(r.extra_data, dict) else {}
        vol = _parse_kv(ed, ["利润-销售数量(扣退)","销售数量(扣退)","sales_volume"]) or (float(r.sales_volume or 0))
        amt = _parse_kv(ed, ["利润-销售金额(扣退)","销售金额(扣退)","sales_amount"]) or (float(r.sales_amount or 0))
        profit = _parse_kv(ed, ["利润-毛利额","毛利额","profit"]) or (float(r.profit or 0))
        ship_qty = _parse_kv(ed, ["商品数据-实发数量","实发数量"])
        ship_amt = _parse_kv(ed, ["商品数据-实发金额","实发金额"])
        zhubo = (ed.get("主播") or "").strip()
        sku = (ed.get("商品编码") or r.sku_code or "").strip()
        pname = (ed.get("商品简称") or r.product_name or "").strip()
        cat = (ed.get("分类") or r.category or "").strip()
        month = r.data_month or 0
        records.append({"month":month,"zhubo":zhubo,"sku":sku,"pname":pname,"cat":cat,
                        "vol":vol,"amt":amt,"profit":profit,"ship_qty":ship_qty,"ship_amt":ship_amt})
    # 汇总
    total_vol = sum(r["vol"] for r in records)
    total_amt = sum(r["amt"] for r in records)
    total_profit = sum(r["profit"] for r in records)
    total_ship = sum(r["ship_qty"] for r in records)
    # 月度
    from collections import defaultdict
    mdata = defaultdict(lambda: {"vol":0,"amt":0,"profit":0})
    for r in records:
        m = r["month"]
        mdata[m]["vol"] += r["vol"]
        mdata[m]["amt"] += r["amt"]
        mdata[m]["profit"] += r["profit"]
    # 主播汇总
    zdata = defaultdict(lambda: {"vol":0,"amt":0,"profit":0,"ship":0,"ship_amt":0})
    for r in records:
        z = r["zhubo"]
        if z and z.lower() != "nan":
            zdata[z]["vol"] += r["vol"]
            zdata[z]["amt"] += r["amt"]
            zdata[z]["profit"] += r["profit"]
            zdata[z]["ship"] += r["ship_qty"]
            zdata[z]["ship_amt"] += r["ship_amt"]
    top_zhubo = sorted(zdata.items(), key=lambda x: -x[1]["vol"])[:20]
    # 各月主播汇总
    zmdata = defaultdict(lambda: defaultdict(lambda: {"vol": 0, "amt": 0, "profit": 0}))
    for r in records:
        m, z = r["month"], r["zhubo"]
        if z and z.lower() != "nan":
            zmdata[m][z]["vol"] += r["vol"]
            zmdata[m][z]["amt"] += r["amt"]
            zmdata[m][z]["profit"] += r["profit"]
    # SKU汇总
    sdata = defaultdict(lambda: {"vol":0,"amt":0,"profit":0,"pname":""})
    for r in records:
        k = r["sku"]
        if k:
            sdata[k]["vol"] += r["vol"]
            sdata[k]["amt"] += r["amt"]
            sdata[k]["profit"] += r["profit"]
            sdata[k]["pname"] = r["pname"] or sdata[k]["pname"]
    top_sku = sorted(sdata.items(), key=lambda x: -x[1]["vol"])[:20]
    # 分类汇总
    cdata = defaultdict(lambda: {"vol":0,"amt":0})
    for r in records:
        c = r["cat"]
        if c:
            cdata[c]["vol"] += r["vol"]
            cdata[c]["amt"] += r["amt"]
    # 构建文本
    lines = [f"=== {year}年 女装牛仔裤销售数据总览 ==="]
    lines.append(f"总扣退后销量: {total_vol:,.0f}")
    lines.append(f"总扣退后销售额: ¥{total_amt:,.2f}")
    lines.append(f"总毛利额: ¥{total_profit:,.2f}")
    lines.append(f"总实发数量: {total_ship:,.0f}")
    lines.append(f"参与主播数: {len(zdata)}")
    lines.append(f"SKU品类数: {len(cdata)}")
    lines.append("")
    lines.append("=== 月度销售数据 ===")
    for m in sorted(mdata.keys()):
        d = mdata[m]
        lines.append(f"  {year}年{m}月 - 扣退后销量:{d['vol']:,.0f}, 销售额:¥{d['amt']:,.2f}, 毛利额:¥{d['profit']:,.2f}")
    lines.append("")
    lines.append("")
    lines.append("=== 各月主播数据 ===")
    months = sorted(zmdata.keys())
    if len(months) >= 2:
        m1, m2 = months[0], months[1]
        m1_zhubos = {z: d for z, d in sorted(zmdata[m1].items(), key=lambda x: -x[1]["vol"])}
        m2_zhubos = {z: d for z, d in sorted(zmdata[m2].items(), key=lambda x: -x[1]["vol"])}
        all_names = set(list(m1_zhubos.keys())[:15] + list(m2_zhubos.keys())[:15])
        lines.append(f"对比月份: {m1}月 vs {m2}月")
        for z in sorted(all_names, key=lambda x: -(m2_zhubos.get(x,{})|m1_zhubos.get(x,{})).get("vol",0)):
            d1 = m1_zhubos.get(z, {"vol":0,"amt":0})
            d2 = m2_zhubos.get(z, {"vol":0,"amt":0})
            v1 = f"{d1['vol']:,.0f}" if d1["vol"] > 0 else "0"
            a1 = f"¥{d1['amt']:,.2f}" if d1["amt"] > 0 else "¥0"
            v2 = f"{d2['vol']:,.0f}" if d2["vol"] > 0 else "0"
            a2 = f"¥{d2['amt']:,.2f}" if d2["amt"] > 0 else "¥0"
            if d1["vol"] > 0 and d2["vol"] > 0:
                pct = f"+{(d2['vol']/d1['vol']-1)*100:.0f}%" if d2['vol'] > d1['vol'] else f"-{(1-d2['vol']/d1['vol'])*100:.0f}%"
            elif d1["vol"] == 0 and d2["vol"] > 0:
                pct = "新上榜"
            elif d1["vol"] > 0 and d2["vol"] == 0:
                pct = "跌出TOP10"
            else:
                pct = "—"
            lines.append(f"  {z[:30]}: {m1}月-销量{v1} 销售额{a1} | {m2}月-销量{v2} 销售额{a2} | 变化:{pct}")
    else:
        for m in months:
            lines.append(f"  {year}年{m}月 TOP主播:")
            top_mz = sorted(zmdata[m].items(), key=lambda x: -x[1]["vol"])[:10]
            for i, (z, d) in enumerate(top_mz):
                lines.append(f"    {i+1}. {z[:20]} - 销量:{d['vol']:,.0f}, 销售额:¥{d['amt']:,.2f}")
    lines.append("")
    lines.append("=== 分类销量排行 ===")
    for i,(c,d) in enumerate(sorted(cdata.items(), key=lambda x:-x[1]["vol"])):
        lines.append(f"  {i+1}. {c} - 销量:{d['vol']:,.0f}, 销售额:¥{d['amt']:,.2f}")
    lines.append("")
    lines.append("=== TOP 20 主播排行（按扣退后销量）===")
    for i,(z,d) in enumerate(top_zhubo):
        lines.append(f"  {i+1}. {z} - 销量:{d['vol']:,.0f}, 销售额:¥{d['amt']:,.2f}, 毛利:¥{d['profit']:,.2f}")
    lines.append("")
    lines.append("=== TOP 20 SKU销量排行 ===")
    for i,(k,d) in enumerate(top_sku):
        lines.append(f"  {i+1}. [{k}] {d['pname']} - 销量:{d['vol']:,.0f}, 销售额:¥{d['amt']:,.2f}")
    return "\n".join(lines)

def _get_zhubo_markdown_table(db, target_month=None, top_n=10, filter_zhubo=None):
    """生成主播对比/排行Markdown表格
    - target_month=None: 对比最近两个月TOP N
    - target_month=3: 仅显示该月销量TOP N
    - filter_zhubo="与辉同行": 单主播详情（多维度）
    """
    import logging
    logging.info(f"[_get_zhubo_markdown_table] filter_zhubo={filter_zhubo!r}, target_month={target_month}")
    sales = db.query(SalesData)
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
    """生成SKU销量排名Markdown表格"""
    rows = db.query(SalesData).all()
    if not rows: return ""
    year = rows[0].data_year
    # 从extra_data解析
    records = []
    for r in rows:
        if target_month and r.data_month != target_month: continue
        ed = r.extra_data or {}
        sku = str(ed.get("商品编码") or r.sku_code or "").strip()
        if not sku: continue
        vol = _parse_kv(ed, ["利润-销售数量(扣退)","销售数量(扣退)","sales_volume"]) or (float(r.sales_volume or 0))
        amt = _parse_kv(ed, ["利润-销售金额(扣退)","销售金额(扣退)","sales_amount"]) or (float(r.sales_amount or 0))
        profit = _parse_kv(ed, ["利润-毛利额","毛利额","profit"]) or (float(r.profit or 0))
        records.append({"sku": sku, "vol": vol, "amt": amt, "profit": profit})
    # 按SKU聚合
    from collections import defaultdict
    sdata = defaultdict(lambda: {"vol":0,"amt":0,"profit":0})
    for r in records:
        sdata[r["sku"]]["vol"] += r["vol"]
        sdata[r["sku"]]["amt"] += r["amt"]
        sdata[r["sku"]]["profit"] += r["profit"]
    # 排序取TOP
    sorted_sku = sorted(sdata.items(), key=lambda x: -x[1]["vol"])[:top_n]
    if not sorted_sku: return ""
    lines = [""]
    if target_month:
        lines.append(f"| SKU编码 | {target_month}月销量 | {target_month}月销售额 | 毛利额 |")
    else:
        lines.append("| SKU编码 | 总销量 | 总销售额 | 毛利额 |")
    lines.append("| --- | --- | --- | --- |")
    for sku, d in sorted_sku:
        vol_s = f"{d['vol']:,.0f}"
        amt_s = f"¥{d['amt']:,.2f}"
        profit_s = f"¥{d['profit']:,.2f}" if d['profit'] > 0 else "—"
        lines.append(f"| {sku} | {vol_s} | {amt_s} | {profit_s} |")
    return "\n".join(lines)

def _get_comment_table(question: str, target_month=None):
    """从 assets/comments/ 加载评论数据并返回摘要"""
    import os, glob
    comment_dir = os.path.join(os.getenv("COZE_WORKSPACE_PATH","/workspace/projects"), "assets", "comments")
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
    upload_dir = os.path.join(os.getenv("COZE_WORKSPACE_PATH","/workspace/projects"), "assets", "uploads")
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


# ----- 认证 -----
@app.post("/api/login")
async def ec_login(req: LoginRequest):
    db = _ec_db()
    try:
        user = db.query(User).filter(User.username == req.username).first()
        if not user:
            user = User(username=req.username, password_hash=_make_pwd(req.password), display_name=req.username)
            db.add(user); db.commit(); db.refresh(user)
            return {"success": True, "user_id": user.id, "username": user.username, "display_name": user.display_name, "is_new": True}
        if user.password_hash != _make_pwd(req.password):
            raise HTTPException(status_code=401, detail="密码错误")
        return {"success": True, "user_id": user.id, "username": user.username, "display_name": user.display_name}
    finally:
        db.close()

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
            m = re.search(r'(\d+)\s*月', file.filename)
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
                comment_dir = os.path.join(WORKSPACE_DIR, "assets", "comments")
                os.makedirs(comment_dir, exist_ok=True)
                save_name = fn.rsplit(".", 1)[0] + ".csv"
                csv_path = os.path.join(comment_dir, save_name)
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                csv_size = os.path.getsize(csv_path)
                db.add(ReportFile(file_name=save_name, file_size=csv_size, file_type="comment", data_year=data_year, report_period=f"{data_year}-{file_month:02d}", row_count=len(df)))
                db.commit()
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
                    r = SalesData(report_name=file.filename, data_year=data_year, data_month=file_month, sku_code=str(row.get("sku_code",""))[:100], product_name=str(row.get("product_name",""))[:300], category=str(row.get("category",""))[:100], style=str(row.get("style",""))[:100], color=str(row.get("color",""))[:50], size=str(row.get("size",""))[:50], sales_volume=int(float(row.get("sales_volume",0))), sales_amount=float(row.get("sales_amount",0)), cost=float(row.get("cost",0)), profit=float(row.get("profit",0)), profit_margin=float(row.get("profit_margin",0)), return_count=int(float(row.get("return_count",0))), return_rate=float(row.get("return_rate",0)), inventory=int(float(row.get("inventory",0))))
                    if r.profit == 0 and r.sales_amount > 0: r.profit = r.sales_amount - r.cost
                    if r.profit_margin == 0 and r.profit > 0 and r.sales_amount > 0: r.profit_margin = r.profit / r.sales_amount
                    records.append(r)
                except Exception as e:
                    logger.warning(f"跳过异常行: {e}")
            
            if records:
                _update_task(task_id, progress=f"正在写入数据库 ({len(records)} 条)...")
                db.add_all(records)
                db.commit()
            db.add(ReportFile(file_name=file.filename, file_size=os.path.getsize(tmp_path), file_type="csv" if fn.endswith(".csv") else "excel", data_year=data_year, report_period=f"{data_year}-{file_month:02d}", row_count=len(records)))
            db.commit()
            # 复制到 uploads/
            _copy_to_uploads(tmp_path, file.filename)
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

def _copy_to_uploads(src_path: str, file_name: str):
    """复制处理后的文件到 assets/uploads/ 供前端展示"""
    upload_dir = os.path.join(WORKSPACE_DIR, "assets", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
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
            m = re.search(r'(\d+)\s*月', file.filename)
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
                _copy_to_uploads(tmp_path, file.filename)
                _update_task(task_id, status="completed", message=f"文件已保存", row_count=0)
                return
            
            df.columns = [c.strip().lower() for c in df.columns]
            is_comment = "评论" in fn or "评价" in fn or "comment" in fn.lower()
            
            if is_comment:
                comment_dir = os.path.join(WORKSPACE_DIR, "assets", "comments")
                os.makedirs(comment_dir, exist_ok=True)
                save_name = fn.rsplit(".", 1)[0] + ".csv"
                csv_path = os.path.join(comment_dir, save_name)
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                csv_size = os.path.getsize(csv_path)
                db.add(ReportFile(file_name=save_name, file_size=csv_size, file_type="comment", data_year=data_year, report_period=f"{data_year}-{file_month:02d}", row_count=len(df)))
                db.commit()
                _copy_to_uploads(tmp_path, file.filename)
                _update_task(task_id, status="completed", message=f"成功导入评论数据 {len(df)} 条", row_count=len(df))
                return
            
            _update_task(task_id, progress=f"正在处理 {len(df)} 条销售数据...")
            cm = {"sku编码":"sku_code","sku_code":"sku_code","商品名称":"product_name","product_name":"product_name","类目":"category","版型":"style","颜色":"color","尺码":"size","销量":"sales_volume","sales_volume":"sales_volume","销售额":"sales_amount","成本":"cost","利润":"profit","利润率":"profit_margin","退货数":"return_count","退货率":"return_rate","库存":"inventory"}
            df.rename(columns=cm, inplace=True)
            df["data_month"] = file_month
            records = []
            for _, row in df.iterrows():
                try:
                    r = SalesData(report_name=file.filename, data_year=data_year, data_month=file_month, sku_code=str(row.get("sku_code",""))[:100], product_name=str(row.get("product_name",""))[:300], category=str(row.get("category",""))[:100], style=str(row.get("style",""))[:100], color=str(row.get("color",""))[:50], size=str(row.get("size",""))[:50], sales_volume=int(float(row.get("sales_volume",0))), sales_amount=float(row.get("sales_amount",0)), cost=float(row.get("cost",0)), profit=float(row.get("profit",0)), profit_margin=float(row.get("profit_margin",0)), return_count=int(float(row.get("return_count",0))), return_rate=float(row.get("return_rate",0)), inventory=int(float(row.get("inventory",0))))
                    if r.profit == 0 and r.sales_amount > 0: r.profit = r.sales_amount - r.cost
                    if r.profit_margin == 0 and r.profit > 0 and r.sales_amount > 0: r.profit_margin = r.profit / r.sales_amount
                    records.append(r)
                except Exception as e:
                    logger.warning(f"跳过异常: {e}")
            if records:
                _update_task(task_id, progress=f"正在写入数据库 ({len(records)} 条)...")
                db.add_all(records)
                db.commit()
            db.add(ReportFile(file_name=file.filename, file_size=os.path.getsize(tmp_path), file_type="csv" if fn.endswith(".csv") else "excel", data_year=data_year, report_period=f"{data_year}-{file_month:02d}", row_count=len(records)))
            db.commit()
            _copy_to_uploads(tmp_path, file.filename)
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

def _create_report_record(year: int, month: int, report_name: str, row_count: int, file_type: str = "csv"):
    """创建报表记录"""
    db = _ec_db()
    try:
        db.add(ReportFile(
            file_name=report_name + ".csv",
            file_size=0,
            file_type=file_type,
            data_year=year,
            report_period=f"{year}-{month:02d}",
            row_count=row_count,
        ))
        db.commit()
    finally:
        db.close()

# ----- 批量上传（客户端解析CSV分块上传，绕过代理超时）-----
_session_batches: dict[str, dict] = {}  # session_id -> {"total_rows": int, "received": int, "file_name": str, ...}

def _build_sales_record(row: list, headers: list, year: int, month: int, file_name: str) -> dict | None:
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
):
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
        rec = _build_sales_record(row, headers, data_year, data_month, file_name)
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
                uploads_dir = os.path.join(WORKSPACE_DIR, "assets", "uploads")
                os.makedirs(uploads_dir, exist_ok=True)
                import shutil
                shutil.copy2(sess["tmp_path"], os.path.join(uploads_dir, file_name))
        except Exception:
            pass
        # 创建汇总报表记录
        total = sess["total_rows"] if sess else len(records)
        report_name = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
        _create_report_record(data_year, data_month, report_name, total)
        result["total_rows"] = total
        result["message"] = f"全部完成！共导入 {total} 条数据"
        # 清理session（延迟清理，让前端有机会查询）
        import threading as _th
        def _clean():
            import time as _t; _t.sleep(30)
            _session_batches.pop(session_id, None)
        _th.Thread(target=_clean, daemon=True).start()
    
    return result

# ----- 数据列表（按文件）-----
def _scan_physical_files():
    """扫描物理文件目录，返回不在数据库中的文件列表"""
    dirs = [os.path.join(WORKSPACE_DIR, "assets", "uploads"),
            os.path.join(WORKSPACE_DIR, "assets", "comments")]
    files = []
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
            except:
                pass
    return files

@app.get("/api/data/files")
async def ec_data_files(year: int = Query(0)):
    db = _ec_db()
    files_dict = {}
    try:
        db_files = db.query(ReportFile)
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
    for pf in _scan_physical_files():
        if pf["file_name"] not in files_dict:
            files_dict[pf["file_name"]] = pf
    files_list = sorted(files_dict.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    return {"success": True, "files": files_list}

@app.get("/api/data/list")
async def ec_data_list(year: int = Query(date.today().year), month: int = Query(0), report_name: str = Query(""), page: int = Query(1), page_size: int = Query(50)):
    db = _ec_db()
    try:
        q = db.query(SalesData).filter(SalesData.data_year == year)
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
    try:
        item = db.query(SalesData).filter(SalesData.id == data_id).first()
        if not item: raise HTTPException(status_code=404, detail="不存在")
        db.delete(item); db.commit()
        return {"success": True, "message": "已删除"}
    finally:
        db.close()

@app.get("/api/data/file/{file_name:path}/raw")
async def ec_data_file_raw(file_name: str, page: int = Query(1, ge=1), page_size: int = Query(100, ge=10, le=1000)):
    """返回上传文件的原始内容"""
    import os, pandas as pd
    workspace = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")

    # 1) 精确匹配文件（禁止模糊匹配，防止错读）
    for subdir in ("assets/uploads", "assets/comments"):
        fp = os.path.join(workspace, subdir, file_name)
        if os.path.isfile(fp):
            return _read_raw_file(fp, file_name, page, page_size)

    # 2) 尝试自动补扩展名（.xlsx ↔ .csv）
    base, ext = os.path.splitext(file_name)
    for subdir in ("assets/uploads", "assets/comments"):
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
    try:
        # 删除文件记录
        files = db.query(ReportFile).filter(ReportFile.file_name == file_name).all()
        for f in files: db.delete(f)
        # 删除该文件导入的所有数据
        rows = db.query(SalesData).filter(SalesData.report_name == file_name).all()
        for r in rows: db.delete(r)
        db.commit()
        # 删除物理文件
        _delete_uploaded_file(file_name)
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
    total_rows = 0
    try:
        for file_name in file_names:
            files = db.query(ReportFile).filter(ReportFile.file_name == file_name).all()
            for f in files: db.delete(f)
            rows = db.query(SalesData).filter(SalesData.report_name == file_name).all()
            for r in rows: db.delete(r)
            total_rows += len(rows)
            _delete_uploaded_file(file_name)
        db.commit()
        return {"success": True, "message": f"已删除 {len(file_names)} 个文件，共移除 {total_rows} 条数据"}
    finally:
        db.close()


def _delete_uploaded_file(file_name: str):
    """删除 assets/uploads/ 和 assets/comments/ 下的物理文件"""
    import os, glob
    workspace = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    for subdir in ("assets/uploads", "assets/comments"):
        dir_path = os.path.join(workspace, subdir)
        if not os.path.isdir(dir_path):
            continue
        # 尝试精确文件名
        fp = os.path.join(dir_path, file_name)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except Exception:
                pass
        # 也尝试 glob 匹配（如文件名编码差异）
        for found in glob.glob(os.path.join(dir_path, f"*{os.path.splitext(file_name)[0]}*")):
            if found != fp and os.path.isfile(found):
                try:
                    os.remove(found)
                except Exception:
                    pass

# ----- 电商问答 -----
def _parse_question_month(q: str):
    """从问题中解析出目标月份
    - 返回 (target_month, is_comparison)
    - target_month=None 表示不限定单月
    - is_comparison=True 表示需要对比
    """
    import re
    months = re.findall(r'(\d+)\s*月', q)
    if not months:
        return None, False
    # 如果包含"对比"、"比较"、"vs" 或 提到两个不同月份 → 对比模式
    has_compare = any(kw in q for kw in ["对比", "比较", "vs", "VS", "V.S"])
    unique_months = list(set(int(m) for m in months))
    if has_compare or len(unique_months) >= 2:
        return None, True  # 对比模式
    return unique_months[0], False  # 单月模式

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
        upload_dir = os.path.join(os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects"), "assets", "uploads")
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
    upload_dir = os.path.join(os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects"), "assets", "uploads")
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
    """从原始 CSV 文件读取 SKU 销售排行数据"""
    import glob, os, pandas as pd
    upload_dir = os.path.join(os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects"), "assets", "uploads")
    files = sorted(glob.glob(os.path.join(upload_dir, "*.csv")))
    if not files:
        return ""
    from tools.data_query_tools import _read_file_to_df
    # 尝试所有文件，找有 SKU 列的文件
    sku_cols = ["sku编码", "sku_code", "商品编码", "SKU编码"]
    for fp in reversed(files):
        try:
            df = _read_file_to_df(fp)
            sku_col = next((c for c in sku_cols if c in df.columns), None)
            qty_col = next((c for c in df.columns if "商品销售数据-商品销售数量(扣退)" in c or "销量" in c or "销售数量" in c), None)
            if sku_col and qty_col:
                sku_data = df.groupby(sku_col)[qty_col].sum().sort_values(ascending=False).head(top_n)
                lines = ["| 排名 | SKU | 销量(扣退) |", "| --- | --- | --- |"]
                for i, (sku, qty) in enumerate(sku_data.items(), 1):
                    lines.append(f"| {i} | {sku} | {int(qty)} |")
                return "\n".join(lines) + f"\n\n共 {len(sku_data)} 个 SKU"
        except Exception:
            continue
    return ""

@app.post("/api/ask")
async def ec_ask(req: AskRequest):
    db = _ec_db()
    try:
        summary = _get_sales_summary(db)
        # 读取所有已上传文件数据，让LLM能分析任意文件
        uploaded_data = _get_general_uploaded_data(req.question)
        if uploaded_data:
            summary += "\n\n" + uploaded_data
        # 解析问题中的月份和主播名
        target_month, is_compare = _parse_question_month(req.question)
        filter_zhubo = _parse_question_zhubo(req.question)
        is_sku = any(kw in req.question for kw in ["SKU","sku","商品","款号","货号"])
        is_comment = any(kw in req.question for kw in ["评论","评价","review","好评","差评","中评","reviews"])
        print(f"[DEBUG] question={req.question!r}, filter_zhubo={filter_zhubo!r}, is_sku={is_sku}, is_comment={is_comment}")
        if is_comment:
            comment_table = _get_comment_table(req.question, target_month)
            sp = """# 角色
你是女装牛仔裤电商用户评论分析专家，擅长分析用户反馈、提取关键词感和发现产品问题。

# 规则
1. 所有分析基于提供的评论数据
2. 如果没有评论数据，明确告知用户尚未上传或导入评论文件
3. 结构化回答：结论→分析→建议
4. 按正向评价、负向评价、中性评价分类统计
5. 提取高频关键词（版型、面料、尺码、颜色、质量等维度）
6. 如果用户提到具体月份（如2月、3月），对比不同月份评论的差异
7. 格式整洁：段与段之间最多1个空行
8. 评论数据表会单独展示在你的回答上方，你不需要再生成表格"""
            prompt = f"【评论数据】\n{comment_table}\n\n【用户问题】\n{req.question}"
            answer = _call_llm(prompt=prompt, system_prompt=sp, temperature=0.3)
            if req.session_id:
                db.add(Conversation(session_id=req.session_id, role="user", content=req.question[:2000], msg_type="chat"))
                db.add(Conversation(session_id=req.session_id, role="assistant", content=answer[:5000], msg_type="chat"))
                db.commit()
            return {"success": True, "answer": answer, "table": comment_table}
        if is_sku:
            # 直接从原始 CSV 文件获取 SKU 数据（DB 可能为空）
            sku_table = _get_sku_from_csv(top_n=20)
            # 同时获取带分析维度的增强SKU数据
            sku_enhanced = _get_sku_enhanced(top_n=20)
            if not sku_table or len(sku_table.strip()) < 20:
                sku_table = _get_sku_markdown_table(db, target_month=target_month, top_n=10)
                sku_enhanced = ""
            sp = """# 角色定义
你是顶级电商数据分析师，精通女装牛仔裤品类的商品管理与销售分析。你的分析风格犀利、数据驱动、一针见血。

# 核心分析维度（必须覆盖以下角度）
1. **爆品特征分析**：哪些款式/颜色/尺码是爆款？它们有什么共同特征？
2. **销售集中度**：TOP SKU的销量占比，判断销售是否健康（是否过度依赖某几个SKU）
3. **品类结构**：不同颜色、不同尺码的分布是否合理？有无品类机会？
4. ** actionable 建议**：给出可执行的具体建议（补货、营销、淘汰）

# 回答规则
1. 所有分析必须基于"SKU数据表"，**禁止编造**数据
2. 结构化回答，但不要用"结论→分析→建议"这种模板，而是直接用自然段落呈现洞察
3. 用数据说话，但不要罗列数字——只说"最关键的发现"
4. 风格：专业、简洁、有价值。不写废话
5. **不要在回答中生成任何表格**——数据表已展示在回答上方
6. **如果用户问的是某个SKU的具体数值（如"卖了多少"、"多少钱"）**：直接回答数字，不要长篇分析"""
            summary = f"SKU数据表（来自原始CSV全量数据）：\n{sku_table}"
            if sku_enhanced:
                summary += f"\n\n【增强分析数据】\n{sku_enhanced}"
            answer = _call_llm(prompt=f"数据摘要：\n{summary}\n\n用户问题：{req.question}", system_prompt=sp, temperature=0.3)
            if req.session_id:
                db.add(Conversation(session_id=req.session_id, role="user", content=req.question[:2000], msg_type="chat"))
                db.add(Conversation(session_id=req.session_id, role="assistant", content=answer[:5000], msg_type="chat"))
                db.commit()
            return {"success": True, "answer": answer, "table": sku_table}
        # 优先从 DB 获取主播表格，若空则回退到从原始 CSV 文件分析
        zhubo_table = _get_zhubo_markdown_table(db, target_month=target_month, top_n=10, filter_zhubo=filter_zhubo)
        if not zhubo_table or len(zhubo_table.strip()) < 20:
            # DB 数据为空（导入时可能没存好），直接从原始 CSV 分析
            try:
                from tools.data_query_tools import analyze_file_data_direct
                import glob, os
                upload_dir = os.path.join(os.getenv("COZE_WORKSPACE_PATH","/workspace/projects"), "assets", "uploads")
                files = sorted(glob.glob(os.path.join(upload_dir, "*.csv")))
                if files:
                    candidate = files[-1]  # 最近上传的 CSV
                    fname = os.path.basename(candidate)
                    df_analysis = analyze_file_data_direct(candidate, group_by_columns="主播",
                        agg_columns="商品销售数据-商品销售数量(扣退),商品销售数据-商品销售金额(扣退),商品销售数据-商品销售成本(扣退),利润-毛利额,利润-经营利润")
                    if df_analysis is not None:
                        import pandas as pd
                        # 如果用户指定了某位主播（如"兰知春序"），只显示该主播数据
                        if filter_zhubo:
                            # 模糊匹配主播名
                            matched_rows = df_analysis[df_analysis.index.str.contains(filter_zhubo, case=False, na=False)]
                            if len(matched_rows) > 0:
                                zhubo_table_lines = [f"| 主播 | 销量(扣退) | 销售额(扣退) | 成本 | 毛利额 | 经营利润 | 利润率 |",
                                                     f"| --- | --- | --- | --- | --- | --- | --- |"]
                                for name, row in matched_rows.iterrows():
                                    sales = row.iloc[1]
                                    profit = row.iloc[4]
                                    profit_margin = profit / sales * 100 if sales > 0 else 0
                                    zhubo_table_lines.append(f"| {name} | {row.iloc[0]:,.0f} | ¥{sales:,.2f} | ¥{row.iloc[2]:,.2f} | ¥{row.iloc[3]:,.2f} | ¥{profit:,.2f} | {profit_margin:.1f}% |")
                                zhubo_table = f"📄 **{fname} — {filter_zhubo} 数据**\n\n" + "\n".join(zhubo_table_lines)
                            else:
                                zhubo_table = None  # 没找到匹配的主播，不要展示错误表格
                        else:
                            total_sales = df_analysis.iloc[:,1].sum()
                            total_profit = df_analysis.iloc[:,4].sum()
                            total_cost = df_analysis.iloc[:,2].sum()
                            
                            top10 = df_analysis.head(10)
                            lines = ["| 排名 | 主播 | 销量(扣退) | 销售额(扣退) | 成本 | 毛利额 | 经营利润 | 利润率 | 销售额占比 |",
                                     "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
                            for idx, (name, row) in enumerate(top10.iterrows(), 1):
                                sales = row.iloc[1]
                                profit = row.iloc[4]
                                profit_margin = profit / sales * 100 if sales > 0 else 0
                                sales_share = sales / total_sales * 100 if total_sales > 0 else 0
                                lines.append(f"| {idx} | {name} | {row.iloc[0]:,.0f} | ¥{sales:,.2f} | ¥{row.iloc[2]:,.2f} | ¥{row.iloc[3]:,.2f} | ¥{profit:,.2f} | {profit_margin:.1f}% | {sales_share:.1f}% |")
                            
                            top10_sales = top10.iloc[:,1].sum()
                            top10_share = top10_sales / total_sales * 100 if total_sales > 0 else 0
                            total_str = (f"\n\n**共 {len(df_analysis)} 位主播，"
                                        f"总销售额 ¥{total_sales:,.2f}，"
                                        f"总经营利润 ¥{total_profit:,.2f}，"
                                        f"整体利润率 {total_profit/total_sales*100:.1f}%**\n"
                                        f"**TOP10主播销售额占比: {top10_share:.1f}%**")
                            zhubo_table = f"📄 **{fname} 全量数据分析结果**\n\n" + "\n".join(lines) + total_str
            except Exception:
                pass
        # 使用 CSV 全量分析数据（DB 数据导入时可能存了零值）
        if zhubo_table and len(zhubo_table.strip()) > 20:
            # CSV 数据可用，完全替代 DB summary
            data_source = zhubo_table
        else:
            data_source = summary  # 回退到 DB
        sp = """# 角色定义
你是顶级电商数据分析师，专精女装牛仔裤品类。你的分析风格：犀利、数据驱动、直击要害，不写废话。

# 核心分析维度（必须覆盖以下角度）
1. **梯队分析**：主播按销售额分梯队（头部/腰部/尾部），判断销售健康度
2. **盈利效率**：不只谈销售额，更要关注利润率。谁在赚钱、谁在烧钱？
3. **集中度风险**：TOP主播占比是否过高？有没有过度依赖某个主播的风险？
4. **增长机会**：哪些腰部主播有潜力？哪些尾部主播该淘汰？
5. ** actionable 建议**：具体、可执行的运营建议（合作策略、资源分配、优化方向）

# 回答规则
1. 所有分析必须基于"真实数据"，**禁止编造数据**
2. 不要用"结论→分析→建议"三段模板，用自然段落呈现洞察
3. 用数据说话但不要罗列数字——只说最关键的那些数据
4. 风格：不要像AI写的，要像资深分析师写的。专业、简洁、有深度
5. **不要在回答中生成任何表格**——数据表已单独展示在回答上方
6. **如果用户问的是具体某位主播的数值（如"成交了多少"、"利润多少"）**：直接回答数字即可，不要长篇分析，不要解释数据含义（表头已标注），不要加"建议"
7. **如果用户问的是具体某位主播的表现**：专注分析该主播即可，不要列出其他主播对比

# 常见分析框架参考
- 销售额高≠利润高，要区分"规模主播"和"利润主播"
- 看利润率比看绝对值更有价值
- 关注头腰尾占比是否健康（一般头部30%、腰部50%、尾部20%为佳）"""
        prompt = f"【真实数据】\n{data_source}\n\n【表格说明】真实数据表已展示在你的回答上方，你不需要再生成表格，只需基于上方的数据进行文字分析。\n\n【问题】\n{req.question}"
        answer = _call_llm(prompt=prompt, system_prompt=sp, temperature=0.3)
        if req.session_id:
            db.add(Conversation(session_id=req.session_id, role="user", content=req.question[:2000], msg_type="chat"))
            db.add(Conversation(session_id=req.session_id, role="assistant", content=answer[:5000], msg_type="chat"))
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

@app.post("/api/hot-topics")
async def ec_hot_topics():
    sp = "你是抖音女装牛仔裤内容策略分析师。禁止使用#、---、*等Markdown标记，全部用纯文字+序号展示。"
    prompt = "分析当前抖音女装牛仔裤，按以下结构用纯文字+序号输出：一、当前热门方向（场景化细分/功能痛点/内容形式）二、用户偏好（版型/身材适配/风格）三、季节性趋势 四、机会点。禁止使用#、---、*等任何Markdown符号"
    content = _call_llm(prompt=prompt, system_prompt=sp, temperature=0.7)
    return {"success": True, "content": content}

@app.post("/api/strategy")
async def ec_strategy(req: StrategyRequest):
    db = _ec_db()
    try:
        summary = _get_sales_summary(db)
        sp = "你是女装牛仔裤电商运营策略顾问。基于数据给建议，要求具体可执行。"
        prompt = f"基于以下数据给{req.period}策略建议:\n{summary}\n\n分析:1)数据总览 2)主推款 3)定价 4)活动 5)预警"
        content = _call_llm(prompt=prompt, system_prompt=sp, temperature=0.5)
        return {"success": True, "content": content}
    finally:
        db.close()

# ----- AI生图 -----
@app.post("/api/generate-image")
async def ec_gen_image(req: ImageGenRequest):
    try:
        from coze_coding_dev_sdk import ImageGenerationClient
        from coze_coding_utils.runtime_ctx.context import new_context as new_ctx
        ctx = new_ctx(method="ec_img")
        client = ImageGenerationClient(ctx=ctx)
        prompt = f"女装牛仔裤, {req.style}风格, {req.prompt}, 高质量商品展示, 商业摄影"
        resp = client.generate(prompt=prompt, size=req.size, model="doubao-seedream-5-0-260128")
        if resp.success and resp.image_urls:
            urls = resp.image_urls
            db = _ec_db()
            try:
                for url in urls:
                    db.add(GeneratedImage(prompt=req.prompt, image_url=url, style=req.style, size=req.size))
                db.commit()
            except Exception as e:
                logger.warning(f"保存记录失败: {e}")
            finally:
                db.close()
            return {"success": True, "image_urls": urls, "prompt": req.prompt}
        else:
            raise HTTPException(status_code=500, detail=f"生图失败: {resp.error_messages}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("生图失败"); raise HTTPException(status_code=500, detail=f"失败: {e}")

@app.get("/api/image-history")
async def ec_img_history(page: int = Query(1), page_size: int = Query(20)):
    db = _ec_db()
    try:
        items = db.query(GeneratedImage).order_by(GeneratedImage.created_at.desc()).offset((page-1)*page_size).limit(page_size).all()
        total = db.query(GeneratedImage).count()
        return {"success": True, "items": [{"id":i.id,"prompt":i.prompt,"image_url":i.image_url,"style":i.style,"size":i.size,"created_at":i.created_at.isoformat() if i.created_at else ""} for i in items], "total": total, "page": page}
    finally:
        db.close()

# ----- 对话 -----
@app.post("/api/conversations")
async def ec_conversations(req: SessionHistoryRequest):
    db = _ec_db()
    try:
        q = db.query(Conversation).filter(Conversation.session_id == req.session_id)
        if req.msg_type: q = q.filter(Conversation.msg_type == req.msg_type)
        items = q.order_by(Conversation.created_at).all()
        return {"success": True, "items": [{"id":i.id,"role":i.role,"content":i.content,"msg_type":i.msg_type,"created_at":i.created_at.isoformat() if i.created_at else ""} for i in items], "session_id": req.session_id}
    finally:
        db.close()

@app.get("/api/conversations/sessions")
async def ec_sessions(page: int = Query(1), page_size: int = Query(20)):
    db = _ec_db()
    try:
        subq = db.query(Conversation.session_id, func.max(Conversation.created_at).label("last_time"), func.min(Conversation.created_at).label("first_time")).group_by(Conversation.session_id).subquery()
        total = db.query(subq).count()
        sessions = db.query(subq).order_by(subq.c.last_time.desc()).offset((page-1)*page_size).limit(page_size).all()
        result = []
        for s in sessions:
            first = db.query(Conversation).filter(Conversation.session_id == s.session_id, Conversation.role == "user").order_by(Conversation.created_at).first()
            cnt = db.query(Conversation).filter(Conversation.session_id == s.session_id).count()
            result.append({"session_id": s.session_id, "title": (first.content[:80]+"...") if first and len(first.content)>80 else (first.content if first else "新对话"), "message_count": cnt, "created_at": s.first_time.isoformat() if s.first_time else "", "last_time": s.last_time.isoformat() if s.last_time else ""})
        return {"success": True, "items": result, "total": total, "page": page}
    finally:
        db.close()

# ----- 看板 -----
@app.get("/api/dashboard")
async def ec_dashboard(year: int = Query(0), month: int = Query(0), file_name: str = Query("")):
    db = _ec_db()
    try:
        base_q = db.query(SalesData)
        if year > 0:
            base_q = base_q.filter(SalesData.data_year == year)
        if month > 0:
            base_q = base_q.filter(SalesData.data_month == month)
        if file_name:
            base_q = base_q.filter(SalesData.report_name == file_name)
        rows = base_q.all()
        if not rows:
            # DB无数据 → 从CSV文件计算仪表盘
            csv_data = _calc_dashboard_from_csv(year, month, file_name)
            if csv_data:
                return csv_data
            return {"success": True, "year": year, "month": month, "overview": {"koutui_vol":0,"koutui_amt":0,"ship_qty":0,"ship_amt":0,"profit":0,"profit_rate":0,"refund_qty":0,"refund_rate":0,"sku_count":0,"zhubo_count":0}, "monthly_trend": [], "top_zhubo": [], "top_sku": [], "files": []}
        total_kv, total_ka, total_sq, total_sa, total_p, total_rq = 0,0,0,0,0,0
        sku_set, zhubo_map, sku_map, monthly_agg = set(), {}, {}, {}
        for r in rows:
            ed = r.extra_data if isinstance(r.extra_data, dict) else {}
            def ev(*ks, d=0.0):
                for k in ks:
                    v = ed.get(k)
                    if v is not None and v != "" and str(v).lower() != "nan":
                        try: return float(v)
                        except: pass
                return float(d) if d else 0.0
            def es(*ks, d=""):
                for k in ks:
                    v = ed.get(k)
                    if v is not None and v != "" and str(v).lower() != "nan": return str(v).strip()
                return d
            sku = (r.sku_code or "").strip() or es("商品编码","商品编号","sku编码","SKU")
            pname = es("商品简称","商品名称","商品名") or (r.product_name or "").strip()
            if sku: sku_set.add(sku)
            kv = ev("利润-销售数量(扣退)","利润-销售数量（扣退）","扣退数量", d=float(r.sales_volume or 0))
            ka = ev("利润-销售金额(扣退)","利润-销售金额（扣退）","扣退金额", d=float(r.sales_amount or 0))
            sq = ev("商品数据-实发数量","商品数据-实发数量（含企业定制）","实发数量", d=float(r.sales_volume or 0))
            sa = ev("商品数据-实发金额","商品数据-实发金额（含企业定制）","实发金额", d=float(r.sales_amount or 0))
            p = ev("利润-毛利额","利润-毛利额（含企业定制）","毛利","利润", d=float(r.profit or 0))
            rq = ev("利润-退款数量","退款数量","退货数", d=float(r.return_count or 0))
            total_kv += kv; total_ka += ka; total_sq += sq; total_sa += sa; total_p += p; total_rq += rq
            if sku:
                if sku not in sku_map: sku_map[sku] = {"n":r.product_name or sku,"v":0,"a":0,"p":0}
                sku_map[sku]["v"] += kv; sku_map[sku]["a"] += ka; sku_map[sku]["p"] += p
            zb = es("主播","主播名字","主播名称")
            if zb and zb.lower() != "nan":
                if zb not in zhubo_map: zhubo_map[zb] = {"v":0,"a":0,"sq":0,"sa":0,"p":0}
                zhubo_map[zb]["v"] += kv; zhubo_map[zb]["a"] += ka; zhubo_map[zb]["sq"] += sq; zhubo_map[zb]["sa"] += sa; zhubo_map[zb]["p"] += p
            m = r.data_month or 1
            if m not in monthly_agg: monthly_agg[m] = {"sales_volume":0,"sales_amount":0,"profit":0}
            monthly_agg[m]["sales_volume"] += kv; monthly_agg[m]["sales_amount"] += ka; monthly_agg[m]["profit"] += p
        if total_kv == 0 and total_ka == 0:
            csv_data = _calc_dashboard_from_csv(year, month, file_name)
            if csv_data:
                return csv_data
        pr = round(total_p/total_ka*100,2) if total_ka > 0 else 0
        rr = round(total_rq/(total_kv+total_rq)*100,2) if total_kv+total_rq>0 else 0
        tz = sorted(zhubo_map.items(), key=lambda x:x[1]["v"], reverse=True)[:100]
        ts = sorted(sku_map.items(), key=lambda x:x[1]["v"], reverse=True)[:10]
        files_q = db.query(ReportFile)
        if year > 0:
            files_q = files_q.filter(ReportFile.data_year == year)
        if month > 0 and year > 0:
            files_q = files_q.filter(ReportFile.report_period.like(f"{year}-{month:02d}%"))
        elif month > 0:
            files_q = files_q.filter(ReportFile.report_period.like(f"%-{month:02d}%"))
        return {"success": True, "year": year, "month": month, "overview": {"koutui_vol":int(total_kv),"koutui_amt":round(total_ka,2),"ship_qty":int(total_sq),"ship_amt":round(total_sa,2),"profit":round(total_p,2),"profit_rate":pr,"refund_qty":int(total_rq),"refund_rate":rr,"sku_count":len(sku_set),"zhubo_count":len(zhubo_map)}, "monthly_trend": [{"month":m,**v} for m,v in sorted(monthly_agg.items())], "top_zhubo": [{"zhubo":z[0],"vol":z[1]["v"],"amt":z[1]["a"],"ship_qty":z[1]["sq"],"ship_amt":z[1]["sa"],"profit":z[1]["p"]} for z in tz], "top_sku": [{"product_code":s[0],"product_name":s[1]["n"],"vol":s[1]["v"],"amt":s[1]["a"],"profit":s[1]["p"]} for s in ts], "files": [{"file_name":f.file_name,"row_count":f.row_count} for f in files_q.all()]}
    finally:
        db.close()

def _calc_dashboard_from_csv(year: int = 0, month: int = 0, file_name: str = "") -> Optional[Dict]:
    """从CSV文件计算仪表盘数据（DB无数据时回退）"""
    base_dir = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    upload_dir = os.path.join(base_dir, "assets", "uploads")
    comment_dir = os.path.join(base_dir, "assets", "comments")
    files = []
    if os.path.isdir(upload_dir):
        files += sorted(glob.glob(os.path.join(upload_dir, "*.csv")))
    if os.path.isdir(comment_dir):
        files += sorted(glob.glob(os.path.join(comment_dir, "*.csv")))
    if file_name:
        files = [f for f in files if os.path.basename(f) == file_name]
    if not files:
        return None
    try:
        dfs = []
        for fp in files:
            df = pd.read_csv(fp, encoding="utf-8-sig", low_memory=False)
            # 月份过滤
            bn = os.path.basename(fp)
            import re
            m = re.search(r'(\d+)月', bn)
            if m:
                fm = int(m.group(1))
                if month > 0 and fm != month:
                    continue
            dfs.append(df)
        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True)

        # 检测文件类型: 评论文件 vs 销售数据文件
        # 如果用户指定了具体文件，根据文件名判断
        if file_name:
            is_comment = '评论' in file_name
        else:
            # 未指定文件时：默认按销售数据处理
            is_comment = False

        # 必要的列映射
        col_kv = "商品销售数据-商品销售数量(扣退)"
        col_ka = "商品销售数据-商品销售金额(扣退)"
        col_cost = "商品销售数据-商品销售成本(扣退)"
        col_sq = "商品数据-实发数量"
        col_sa = "商品数据-实发金额"
        col_profit = "利润-经营利润"
        col_rq = "售后合计-退款数量合计"
        col_zhubo = "主播"
        col_sku = "商品编码"
        col_sku_name = "商品名称"

        def safe_sum(s: pd.Series) -> float:
            return float(s.fillna(0).astype(str).replace(["nan",""],"0").astype(float).sum())

        def safe_col(df, col):
            return col if col in df.columns else None

        if is_comment:
            # ====== 评论文件 ======
            col_content = next((c for c in ["评论内容","评价内容","评论"] if c in df.columns), "评论内容")
            col_rating = next((c for c in ["商品评价得分","评分","星级"] if c in df.columns), None)
            col_time = next((c for c in ["评论时间","评价时间","创建时间"] if c in df.columns), None)
            col_product = next((c for c in ["商品名称","商品编码","商品ID","产品名称"] if c in df.columns), col_sku if col_sku in df.columns else None)
            col_sku_name_c = next((c for c in ["商品名称","商品标题","产品名称"] if c in df.columns), None)

            total_reviews = len(df)
            # 好评/中评/差评 判定
            if col_rating:
                try:
                    rating_vals = pd.to_numeric(df[col_rating].astype(str).str.replace(r'[^0-9.]','',regex=True), errors='coerce')
                except:
                    rating_vals = pd.Series([None]*len(df))
                positive = int((rating_vals >= 4).sum()) if rating_vals.notna().any() else 0
                neutral = int((rating_vals == 3).sum()) if rating_vals.notna().any() else 0
                negative = int((rating_vals <= 2).sum()) if rating_vals.notna().any() else 0
                avg_rating = round(float(rating_vals.mean()), 2) if rating_vals.notna().any() else 0
            else:
                positive = total_reviews; neutral = 0; negative = 0; avg_rating = 0

            positive_rate = round(positive/total_reviews*100, 2) if total_reviews > 0 else 0

            # 评论数最多的商品（含好评/差评/好评率）
            top_reviewed_products = []
            if col_product and col_product in df.columns:
                prod_gb = df[col_product].value_counts().head(20)
                prod_name_map = {}
                if col_sku_name_c and col_sku_name_c in df.columns:
                    prod_name_map = df.groupby(col_product)[col_sku_name_c].first().to_dict()
                for pname, cnt in prod_gb.items():
                    display = str(prod_name_map.get(pname, pname)).strip()
                    # 计算该商品的好评/差评
                    p_positive = 0; p_negative = 0; p_rate = 0
                    if col_rating:
                        mask = df[col_product] == pname
                        try:
                            p_vals = pd.to_numeric(df.loc[mask, col_rating].astype(str).str.replace(r'[^0-9.]','',regex=True), errors='coerce')
                            p_positive = int((p_vals >= 4).sum()) if p_vals.notna().any() else 0
                            p_negative = int((p_vals <= 2).sum()) if p_vals.notna().any() else 0
                            p_rate = round(p_positive/int(cnt)*100, 2) if int(cnt) > 0 else 0
                        except:
                            pass
                    top_reviewed_products.append({
                        "product": str(pname).strip(),
                        "product_name": display,
                        "total": int(cnt),
                        "review_count": int(cnt),
                        "positive": p_positive,
                        "negative": p_negative,
                        "positive_rate": p_rate,
                    })

            overview = {
                "total_reviews": total_reviews,
                "positive": positive, "neutral": neutral, "negative": negative,
                "positive_rate": positive_rate, "avg_rating": avg_rating,
                "product_count": len(top_reviewed_products),
                "has_rating": col_rating is not None,
            }
            top_zhubo = []
            top_sku = top_reviewed_products
            monthly_trend = [{"month": month if month else 0, "review_count": total_reviews}]
        else:
            # ====== 销售数据文件 ======
            total_kv = safe_sum(df[col_kv])
            total_ka = safe_sum(df[col_ka])
            total_cost = safe_sum(df[col_cost])
            total_sq = safe_sum(df[col_sq]) if col_sq in df else 0
            total_sa = safe_sum(df[col_sa]) if col_sa in df else total_ka
            total_p = safe_sum(df[col_profit])
            total_rq = safe_sum(df[col_rq]) if col_rq in df else 0
            sku_set = set(df[col_sku].dropna().unique()) if col_sku in df else set()
            zhubo_set = set(df[col_zhubo].dropna().unique()) if col_zhubo in df else set()

            pr = round(total_p/total_ka*100, 2) if total_ka > 0 else 0
            rr = round(total_rq/(total_kv+total_rq)*100, 2) if total_kv+total_rq>0 else 0

            # 主播TOP100
            top_zhubo = []
            if col_zhubo in df:
                zb_gb = df.groupby(col_zhubo).agg({col_kv:"sum", col_ka:"sum", col_profit:"sum"})
                if col_sq in df: zb_gb[col_sq] = df.groupby(col_zhubo)[col_sq].sum()
                if col_sa in df: zb_gb[col_sa] = df.groupby(col_zhubo)[col_sa].sum()
                for name, row in zb_gb.sort_values(col_kv, ascending=False).head(100).iterrows():
                    top_zhubo.append({
                        "zhubo": str(name).strip(),
                        "vol": float(row[col_kv]),
                        "amt": float(row[col_ka]),
                        "ship_qty": float(row.get(col_sq,0)),
                        "ship_amt": float(row.get(col_sa,0)),
                        "profit": float(row[col_profit]),
                    })

            # SKU TOP10
            top_sku = []
            if col_sku in df:
                sk_gb = df.groupby(col_sku).agg({col_kv:"sum", col_ka:"sum", col_profit:"sum"})
                sk_name = df.groupby(col_sku)[col_sku_name].first().to_dict() if col_sku_name in df else {}
                for code, row in sk_gb.sort_values(col_kv, ascending=False).head(10).iterrows():
                    top_sku.append({
                        "product_code": str(code).strip(),
                        "product_name": str(sk_name.get(code,code)).strip(),
                        "vol": float(row[col_kv]),
                        "amt": float(row[col_ka]),
                        "profit": float(row[col_profit]),
                    })

            overview = {
                "koutui_vol": int(total_kv), "koutui_amt": round(total_ka,2),
                "ship_qty": int(total_sq), "ship_amt": round(total_sa,2),
                "profit": round(total_p,2), "profit_rate": pr,
                "refund_qty": int(total_rq), "refund_rate": rr,
                "sku_count": len(sku_set), "zhubo_count": len(zhubo_set),
            }

            monthly_trend = [{"month": month if month else 3, "sales_volume": int(total_kv), "sales_amount": round(total_ka,2), "profit": round(total_p,2)}]

        all_csv_files = sorted(glob.glob(os.path.join(upload_dir, "*.csv")))
        if os.path.isdir(comment_dir):
            all_csv_files += sorted(glob.glob(os.path.join(comment_dir, "*.csv")))
        file_list = [{"file_name": os.path.basename(f)} for f in all_csv_files]

        return {"success": True, "year": year, "month": month,
                "dashboard_type": "comment" if is_comment else "sales",
                "overview": overview,
                "monthly_trend": monthly_trend, "top_zhubo": top_zhubo, "top_sku": top_sku,
                "top_products": top_reviewed_products if is_comment else [],
                "files": file_list}
    except Exception as e:
        logger.warning(f"CSV dashboard fallback error: {e}")
        return None


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
    parser.add_argument("-p", type=int, default=5000, help="HTTP服务端口")
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
    if graph_helper.is_dev_env():
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
