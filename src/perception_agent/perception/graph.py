
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from perception_agent.perception.assistant import assistant_node
from perception_agent.perception.state import PerceptionState
from perception_agent.perception.tools import TOOLS

tool_node = ToolNode(TOOLS)

builder = StateGraph(PerceptionState)

builder.add_node("assistant", assistant_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "assistant")

builder.add_conditional_edges(
    "assistant",
    tools_condition,
)

builder.add_edge(
    "tools",
    "assistant",
)

graph = builder.compile()

