from langgraph.graph import START, END, StateGraph

from perception_agent.coordinator.state import ParentState
from perception_agent.perception.graph import graph as perception_graph


def after_perception(state: ParentState):
    print("\n[PARENT] Perception completed")
    return {}


builder = StateGraph(ParentState)

builder.add_node(
    "perception",
    perception_graph,
)

builder.add_node(
    "after_perception",
    after_perception,
)

builder.add_edge(
    START,
    "perception",
)

builder.add_edge(
    "perception",
    "after_perception",
)

builder.add_edge(
    "after_perception",
    END,
)

parent_graph = builder.compile()


# print(
#     parent_graph
#     .get_graph(xray=1)
#     .draw_mermaid()
# )