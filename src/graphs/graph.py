"""占位工作流图 — 本地开发用，实际工作流由 Coze 平台定义"""

from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph


class PlaceholderState(TypedDict):
    messages: list


def _noop(state: PlaceholderState) -> PlaceholderState:
    return state


builder = StateGraph(PlaceholderState)
builder.add_node("noop", _noop)
builder.set_entry_point("noop")
builder.add_edge("noop", END)

graph: CompiledStateGraph = builder.compile()
