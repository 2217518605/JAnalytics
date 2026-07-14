"""女装牛仔裤电商智能助手 - Agent 核心模块"""
import os
import json
from typing import Annotated
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from storage.memory.memory_saver import get_memory_saver
from tools.data_query_tools import list_files, read_file_data, analyze_file_by_column, query_sales_data, analyze_data_deep
from utils.paths import PROJECT_ROOT

LLM_CONFIG = "config/agent_llm_config.json"

# 默认保留最近 20 轮对话 (40 条消息)
MAX_MESSAGES = 40


def _windowed_messages(old, new):
    """滑动窗口: 只保留最近 MAX_MESSAGES 条消息"""
    return add_messages(old, new)[-MAX_MESSAGES:]  # type: ignore


class AgentState(MessagesState):
    messages: Annotated[list[AnyMessage], _windowed_messages]


def build_agent(ctx=None):
    # 优先从环境变量读取，fallback 到配置文件
    config_path = os.path.join(PROJECT_ROOT, LLM_CONFIG)

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model = cfg.get("config", {}).get("model", os.getenv("OPENAI_MODEL", "gpt-4o"))
        temperature = cfg.get("config", {}).get("temperature", 0.7)
    else:
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        temperature = 0.7

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        streaming=True,
        timeout=600,
    )

    # 如果有 ctx，从中提取 headers 传递给 LLM
    extra_kwargs = {}
    if ctx and hasattr(ctx, "headers") and ctx.headers:
        extra_kwargs["default_headers"] = ctx.headers

    return create_agent(
        model=llm,
        system_prompt=cfg.get("sp") if os.path.exists(config_path) else "",
        tools=[list_files, read_file_data, analyze_file_by_column, query_sales_data, analyze_data_deep],
        checkpointer=get_memory_saver(),
        state_schema=AgentState,
    )