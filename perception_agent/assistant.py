from langchain_core.messages import SystemMessage

from perception_agent.models import get_vlm
from perception_agent.prompts import PERCEPTION_AGENT_PROMPT
from perception_agent.state import PerceptionState
from perception_agent.tools import TOOLS


def assistant_node(state: PerceptionState):
    model = get_vlm().bind_tools(TOOLS)

    messages = [
        SystemMessage(content=PERCEPTION_AGENT_PROMPT),
        *state["messages"],
    ]

    response = model.invoke(messages)

    return {
        "messages": [response],
    }