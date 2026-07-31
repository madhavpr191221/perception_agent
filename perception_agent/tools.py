# perception_agent/tools.py

import json
from typing import Annotated

from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId

from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from perception_agent.models import (
    get_detector,
    get_detector_device,
)
from perception_agent.state import PerceptionState


@tool
def detect_objects(
    state: Annotated[PerceptionState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Detect objects in the current image using the object detector.

    The detections are stored in graph state under `detected_objects`.
    """

    image_path = state["image_path"]

    detector = get_detector()
    device = get_detector_device()

    results = detector.predict(
        source=image_path,
        device=device,
        verbose=False,
    )

    result = results[0]

    detections = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detection = {
            "label": result.names[class_id],
            "confidence": round(confidence, 4),
            "bbox": [
                round(x1),
                round(y1),
                round(x2),
                round(y2),
            ],
        }

        detections.append(detection)

    tool_summary = {
        "num_objects": len(detections),
        "detections": detections,
    }

    return Command(
        update={
            "detected_objects": detections,

            "messages": [
                ToolMessage(
                    content=json.dumps(
                        tool_summary,
                        indent=2,
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


TOOLS = [
    detect_objects,
]