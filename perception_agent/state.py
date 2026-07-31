# perception_agent/state.py

from langgraph.graph import MessagesState


class PerceptionState(MessagesState):
    image_path: str

    detected_objects: list[dict]

    inspected_regions: list[dict]

    current_hypothesis: str | None