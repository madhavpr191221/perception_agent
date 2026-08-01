from langchain_core.messages import SystemMessage
from perception_agent.perception.prompts import PERCEPTION_AGENT_PROMPT
from perception_agent.perception.state import PerceptionState
from perception_agent.perception.tools import TOOLS
from perception_agent.vision.models import get_vlm


def assistant_node(state: PerceptionState):
    model = get_vlm().bind_tools(TOOLS)

    messages = [
        SystemMessage(content=PERCEPTION_AGENT_PROMPT),
        *state["messages"],
    ]

    response = model.invoke(messages)

    update = {
        "messages": [response],
    }

    if not response.tool_calls:
        update["perception_report"] = response.content

    return update