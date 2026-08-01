from langgraph.graph import MessagesState


class ParentState(MessagesState):
    image_path: str
    perception_report: str | None