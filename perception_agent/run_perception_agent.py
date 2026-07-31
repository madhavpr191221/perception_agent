# perception_agent/run_perception_agent.py

from langchain_core.messages import HumanMessage

from perception_agent.graph import graph
from perception_agent.vision_utils import encode_image_base64


def main():
    image_path = "images/street.jpg"

    user_request = (
        "Something looks strange in this image. "
        "Inspect it and use tools if needed."
    )

    image_base64, mime_type = encode_image_base64(image_path)

    human_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": user_request,
            },
            {
                "type": "image",
                "base64": image_base64,
                "mime_type": mime_type,
            },
        ]
    )

    initial_state = {
        "image_path": image_path,
        "detected_objects": [],
        "inspected_regions": [],
        "current_hypothesis": None,
        "messages": [human_message],
    }

    # result = graph.invoke(initial_state)


    # print("\nFinal answer:\n")
    # print(result["messages"][-1].content)

    # print("\nDetected objects:\n")
    # print(result["detected_objects"])

    for chunk in graph.stream(
    initial_state,
    stream_mode="updates"
            ):
        print("\n--- GRAPH UPDATE ---")
        print(chunk)

if __name__ == "__main__":
    main()