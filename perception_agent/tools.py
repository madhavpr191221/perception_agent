import json
from pathlib import Path
from typing import Annotated

from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId

from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from perception_agent.models import (
    get_detector,
    get_detector_device,
    get_vlm,
)
from perception_agent.state import PerceptionState
from perception_agent.vision_utils import (
    encode_image_base64,
    save_crop,
)


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

@tool
def inspect_crop(
    bbox: list[int],
    state: Annotated[PerceptionState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Inspect a specific region of the current image more closely.

    Use bbox = [x1, y1, x2, y2].
    """

    image_path = state["image_path"]

    crop_path = Path("artifacts/crops") / f"{tool_call_id}.jpg"

    save_crop(
        image_path=image_path,
        bbox=bbox,
        output_path=crop_path,
    )

    image_base64, mime_type = encode_image_base64(crop_path)

    inspection_prompt = (
        "Inspect this cropped region carefully. "
        "Describe what is actually visible. "
        "Focus on objects, spatial relationships, occlusions, "
        "and anything genuinely unusual. "
        "Do not assume that an anomaly exists."
    )

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": inspection_prompt,
            },
            {
                "type": "image",
                "base64": image_base64,
                "mime_type": mime_type,
            },
        ]
    )

    response = get_vlm().invoke([message])

    inspection = {
        "bbox": bbox,
        "crop_path": str(crop_path),
        "analysis": response.content,
    }

    return Command(
        update={
            "inspected_regions": [inspection],
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        inspection,
                        indent=2,
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )

TOOLS = [
    detect_objects,
    inspect_crop,
]