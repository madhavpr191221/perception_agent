from langchain_core.messages import HumanMessage

from perception_agent.coordinator.graph import parent_graph
from perception_agent.vision.utils import encode_image_base64


def print_graph_update(chunk):
    for node_name, update in chunk.items():
        if update is None:
            continue

        if node_name == "perception":
            print("\n[PERCEPTION SUBGRAPH COMPLETED]")

            for message in update.get("messages", []):
                if getattr(message, "tool_calls", None):
                    for call in message.tool_calls:
                        print(
                            f"Tool call: {call['name']} "
                            f"{call['args']}"
                        )

                elif type(message).__name__ == "ToolMessage":
                    if message.name == "detect_objects":
                        print("Tool result: object detection completed")

                    elif message.name == "inspect_crop":
                        print("Tool result: crop inspection completed")

                    else:
                        print(f"Tool result from: {message.name}")

        elif node_name == "after_perception":
            print("\n[PARENT CONTINUES]")


def main():
    image_path = "images/street.jpg"

    user_request = (
        "Something looks strange in this image. "
        "Inspect it and use tools if needed."
    )

    image_base64, mime_type = encode_image_base64(
        image_path
    )

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
        "perception_report": None,
        "messages": [human_message],
    }

    final_report = None

    for chunk in parent_graph.stream(
        initial_state,
        stream_mode="updates",
    ):
        print_graph_update(chunk)

        for _, update in chunk.items():
            if update and "perception_report" in update:
                final_report = update["perception_report"]

    print("\n=== FINAL PERCEPTION REPORT ===\n")
    print(final_report)


if __name__ == "__main__":
    main()