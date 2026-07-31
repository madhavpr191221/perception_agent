# perception_agent/assistant.py

from perception_agent.models import get_vlm
from perception_agent.state import PerceptionState
from perception_agent.tools import TOOLS


def assistant_node(state: PerceptionState):
    """
    Let the VLM reason over the current message history and decide
    whether additional perception tools are needed.
    """

    model = get_vlm().bind_tools(TOOLS)

    response = model.invoke(state["messages"])

    return {
        "messages": [response],
    }