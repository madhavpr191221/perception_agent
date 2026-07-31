# perception_agent/state.py
from typing import Annotated
from langgraph.graph import MessagesState
import operator

class PerceptionState(MessagesState):
    image_path: str

    detected_objects: list[dict]

    inspected_regions: Annotated[list[dict], operator.add]

    current_hypothesis: str | None