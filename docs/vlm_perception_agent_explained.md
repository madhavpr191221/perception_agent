# VLM Visual Investigation Agent with LangGraph

## Purpose

This document explains the current `perception_agent` project in detail: what the agent does, how the LangGraph control flow works, how state is structured, how tools are implemented, why the VLM and YOLO detector play different roles, how crop inspection works, and what subtle issues matter when building this kind of visual agent.

> **Documentation note**
>
> I attempted to verify the current LangGraph/LangChain APIs with Context7 as requested, but Context7 was unavailable in this session. The API explanations below therefore reflect the code we have already run successfully in this project and the observed runtime behavior. Where something is an architectural interpretation rather than a directly observed behavior, this document says so explicitly.

---

# 1. What the agent is trying to do

The user gives the system an image and a broad question such as:

> “Something looks strange in this image. Inspect it and use tools if needed.”

The system does **not** immediately run every vision model it has. Instead, the main VLM acts as an investigator.

Its rough loop is:

```text
observe full image
    ↓
form a tentative interpretation
    ↓
decide whether more evidence is needed
    ↓
choose a perception tool
    ↓
receive tool evidence
    ↓
reconsider the hypothesis
    ↓
stop or gather more evidence
```

The current capabilities are intentionally small:

1. **Direct visual inspection by the main VLM**
2. **YOLO object detection** through `detect_objects`
3. **Focused crop inspection** through `inspect_crop`

The project is therefore not yet a general-purpose visual agent. It is a compact foundation for one.

---

# 2. High-level architecture

The main graph is deliberately simple:

```text
START
  ↓
assistant
  ↓
tools_condition
  ├── no tool call → END
  └── tool call    → tools
                       ↓
                    assistant
```

The graph itself contains only two real nodes:

- `assistant`
- `tools`

The main reasoning loop is therefore:

```text
assistant → tools → assistant → tools → ... → END
```

The graph is small because the intelligence is split across:

- the **system prompt**,
- the **main VLM**,
- the **tool schemas**,
- the **tool implementations**,
- and the **structured graph state**.

This is a useful design choice. A complicated graph is not automatically a more intelligent agent.

---

# 3. Current project structure

The relevant files are:

```text
perception_agent/
├── assistant.py
├── graph.py
├── models.py
├── prompts.py
├── run_perception_agent.py
├── state.py
├── tools.py
└── vision_utils.py
```

Their responsibilities are intentionally separated.

| File | Responsibility |
|---|---|
| `state.py` | Defines the graph state schema |
| `models.py` | Provides the VLM and YOLO model instances |
| `vision_utils.py` | Deterministic image operations |
| `tools.py` | Agent-callable perception capabilities |
| `prompts.py` | Investigation policy for the main VLM |
| `assistant.py` | Main VLM reasoning node |
| `graph.py` | LangGraph wiring |
| `run_perception_agent.py` | Builds initial multimodal state and runs the graph |

The graph should remain relatively boring. That is a feature.

---

# 4. The state schema

Current `state.py`:

```python
from typing import Annotated
from langgraph.graph import MessagesState
import operator

class PerceptionState(MessagesState):
    image_path: str

    detected_objects: list[dict]

    inspected_regions: Annotated[list[dict], operator.add]

    debug_artifacts: Annotated[list[dict], operator.add]

    current_hypothesis: str | None
```

## 4.1 `MessagesState`

`PerceptionState` inherits from `MessagesState`.

That gives the graph a `messages` field whose job is to hold conversational history such as:

- `HumanMessage`
- `AIMessage`
- `ToolMessage`
- potentially `SystemMessage`

The important conceptual split is:

```text
messages
    = conversation + tool-call protocol history

other state fields
    = structured application data
```

So we do not try to encode everything as natural-language messages.

---

# 5. `Annotated[T, M]` and reducers

Two fields use:

```python
Annotated[list[dict], operator.add]
```

This is the same Python `Annotated` idea discussed earlier:

```text
Annotated[T, M]
```

means:

- underlying type = `T`
- metadata = `M`

LangGraph interprets `operator.add` as the reducer for that field.

So if the current state contains:

```python
inspected_regions = [A]
```

and a new node returns:

```python
{"inspected_regions": [B]}
```

LangGraph applies:

```python
operator.add([A], [B])
```

which gives:

```python
[A, B]
```

Without a reducer, later writes would normally replace the previous field value.

This means the current design intentionally distinguishes:

```text
detected_objects
    → overwrite with the latest detector result

inspected_regions
    → accumulate investigation evidence

debug_artifacts
    → accumulate generated artifacts
```

That is a sensible semantic choice for the current agent.

---

# 6. Meaning of each state field

## `image_path`

```python
image_path: str
```

The local image being investigated.

This is application/runtime state. The LLM does not need to invent the path.

---

## `detected_objects`

```python
detected_objects: list[dict]
```

Stores machine-readable YOLO detections:

```python
[
    {
        "label": "person",
        "confidence": 0.7885,
        "bbox": [198, 223, 219, 272],
    },
    ...
]
```

This is structured perception output.

---

## `inspected_regions`

```python
inspected_regions: Annotated[list[dict], operator.add]
```

Stores the semantic reports produced by focused crop inspection.

Example:

```python
{
    "requested_bbox": [405, 210, 485, 275],
    "inspected_bbox": [381, 190, 509, 294],
    "crop_path": "artifacts/crops/...jpg",
    "analysis": "Three people are standing near a taxi...",
}
```

This is accumulated evidence gathered during investigation.

---

## `debug_artifacts`

```python
debug_artifacts: Annotated[list[dict], operator.add]
```

This field records generated files that are useful for debugging or human inspection.

Examples:

```python
{
    "type": "detection_overlay",
    "path": "artifacts/debug/..._detections.jpg",
    "num_objects": 15,
}
```

and:

```python
{
    "type": "crop",
    "path": "artifacts/crops/...jpg",
    "requested_bbox": [...],
    "inspected_bbox": [...],
}
```

Important subtlety:

**The main VLM does not automatically see `state["debug_artifacts"]`.**

LangGraph state is not automatically dumped into the model prompt.

The main VLM only sees whatever `assistant.py` explicitly sends to `model.invoke(...)`.

So `debug_artifacts` is currently an observability field, not a reasoning input.

---

# 7. Multimodal input and the earlier bug

The runner creates the initial `HumanMessage` with both text and image content:

```python
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
```

This message is then placed directly into state:

```python
"messages": [human_message]
```

This matters because the first version temporarily created a multimodal message inside `assistant.py` but did not write it back into graph state.

That meant:

```text
first assistant call
    → saw image

later assistant call after tool use
    → saw only text + tool results
```

The fix was to put the multimodal `HumanMessage` into state from the beginning.

Now later assistant calls receive the original image again as part of the accumulated message history.

---

# 8. Base64 is transport, not visual reasoning

The image is encoded into base64 by:

```python
def encode_image_base64(...)
```

The VLM does not reason by reading characters from the base64 string.

Base64 is simply the transport representation used to send image bytes through the multimodal API format.

Conceptually:

```text
image file
    ↓
base64 transport encoding
    ↓
multimodal image content block
    ↓
VLM image input
```

---

# 9. The main assistant node

The main assistant node is conceptually simple:

```python
model = get_vlm().bind_tools(TOOLS)

messages = [
    SystemMessage(content=PERCEPTION_AGENT_PROMPT),
    *state["messages"],
]

response = model.invoke(messages)

return {
    "messages": [response],
}
```

Its responsibilities are:

1. provide the system policy,
2. provide the accumulated message history,
3. expose the available tool schemas,
4. ask the VLM to decide what to do next.

The assistant node does **not** itself run YOLO or crop images.

---

# 10. `bind_tools(TOOLS)`

The model is bound to:

```python
TOOLS = [
    detect_objects,
    inspect_crop,
]
```

Binding tools tells the model that these actions are available.

Conceptually, the model sees schemas resembling:

```text
detect_objects()
inspect_crop(bbox: list[int])
```

The injected parameters are not intended to be model-controlled arguments.

This is a major design principle:

```text
model-controlled arguments
    = decisions the LLM should make

injected arguments
    = runtime/framework context
```

For this agent:

```text
bbox
    → model decides

state
    → runtime injects

tool_call_id
    → runtime injects
```

---

# 11. `InjectedState`

The tool signature contains:

```python
state: Annotated[PerceptionState, InjectedState]
```

Underlying Python type:

```python
PerceptionState
```

Metadata:

```python
InjectedState
```

This tells the tool machinery to supply graph state automatically.

Without it, the tool interface could expose `state` as something the LLM is expected to provide.

That would be undesirable.

The model should decide:

```text
"Call detect_objects"
```

not:

```text
"Call detect_objects with this manually reconstructed graph state"
```

Inside the tool, however, using:

```python
image_path = state["image_path"]
```

is completely normal.

`InjectedState` answers the question:

> How did the tool receive `state` in the first place?

---

# 12. `InjectedToolCallId`

Each individual tool invocation has an identifier.

A model tool call might look conceptually like:

```python
{
    "id": "call_abc123",
    "name": "detect_objects",
    "args": {},
}
```

The corresponding tool result must identify the request it answers:

```python
ToolMessage(
    content="...",
    tool_call_id="call_abc123",
)
```

Therefore the tool signature contains:

```python
tool_call_id: Annotated[str, InjectedToolCallId]
```

The model does not supply this ID.

The runtime already knows which tool invocation is currently executing and injects the correct ID.

This solves the pairing requirement:

```text
AI tool request ID
       ↕
ToolMessage tool_call_id
```

---

# 13. Why the earlier ToolMessage error happened

An earlier version returned a `Command(update=...)` that changed structured state but failed to include a matching `ToolMessage`.

From the state-update perspective, the tool succeeded.

From the conversational tool protocol perspective, the tool call was left unanswered.

Conceptually the history looked like:

```text
AIMessage
  └── tool call id=call_abc123

??? no ToolMessage for call_abc123
```

That is why the framework rejected it.

The corrected pattern is:

```python
return Command(
    update={
        "detected_objects": detections,
        "messages": [
            ToolMessage(
                content=...,
                tool_call_id=tool_call_id,
            )
        ],
    }
)
```

So a tool can update both:

- structured graph state,
- and message/tool history.

---

# 14. `Command`

A plain node return such as:

```python
return {"x": 5}
```

is just a state patch.

A router returning:

```python
return "inspect"
```

is a control-flow decision.

`Command` is useful when a computation needs to carry structured update information and potentially control information together.

Conceptually:

```text
plain dict
    = state update

router return
    = control decision

Command
    = state update + optional control instruction
```

In the current tools, `Command` is used mainly to update state and messages. The explicit graph edge:

```python
builder.add_edge("tools", "assistant")
```

still handles the return to the assistant.

---

# 15. `ToolNode`

The graph creates:

```python
tool_node = ToolNode(TOOLS)
```

and registers it as:

```python
builder.add_node("tools", tool_node)
```

`ToolNode` is the executor for model-requested tools.

The model itself does not directly execute Python functions.

The actual chain is:

```text
main VLM
    ↓
AIMessage(tool_calls=[...])
    ↓
tools_condition
    ↓
ToolNode
    ↓
Python tool function
    ↓
ToolMessage / Command update
    ↓
assistant
```

---

# 16. `tools_condition`

The graph contains a conditional edge after the assistant:

```python
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
```

Conceptually, `tools_condition` behaves like:

```python
last = state["messages"][-1]

if last.tool_calls:
    return "tools"

return END
```

So:

```text
assistant produced tool call
    → tools

assistant produced normal final answer
    → END
```

This is the core inner loop of the agent.

---

# 17. `END` does not mean the conversation is permanently over

`END` means:

> the current graph invocation is complete.

It does not mean the agent can never be invoked again.

With checkpointing, a future human turn could invoke the same graph again under the same thread.

There are therefore two conceptual loops:

```text
inner loop:
assistant → tools → assistant → tools → ... → END

outer loop:
user turn → graph invocation → END
next user turn → graph invocation → END
```

The current perception runner is a single outer-turn experiment.

---

# 18. `detect_objects`: design and logic

The detector tool begins with:

```python
@tool
def detect_objects(
    state: Annotated[PerceptionState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
```

The model therefore does not need to supply any ordinary argument.

Conceptually it calls:

```text
detect_objects()
```

The tool then obtains the image path from state:

```python
image_path = state["image_path"]
```

This is a deliberate design choice.

The VLM should decide **whether detection is useful**, not reconstruct internal file-system paths.

---

# 19. YOLO inference

The detector is loaded through:

```python
detector = get_detector()
device = get_detector_device()
```

Then inference runs:

```python
results = detector.predict(
    source=image_path,
    device=device,
    verbose=False,
)
```

The first image result is:

```python
result = results[0]
```

Each box provides:

```python
class_id = int(box.cls.item())
confidence = float(box.conf.item())
x1, y1, x2, y2 = box.xyxy[0].tolist()
```

The result is normalized into a project-specific dictionary:

```python
{
    "label": result.names[class_id],
    "confidence": round(confidence, 4),
    "bbox": bbox,
}
```

This is a useful abstraction boundary.

The rest of the graph does not need to know the Ultralytics tensor representation.

---

# 20. Why clamp detector boxes

Raw detector coordinates are passed through:

```python
bbox = clamp_bbox(...)
```

`clamp_bbox` enforces:

- exactly four coordinates,
- coordinates inside image bounds,
- correct left/right ordering,
- correct top/bottom ordering,
- non-degenerate area.

This protects later image operations from malformed geometry.

The tool therefore converts:

```text
model output geometry
    ↓
validated application geometry
```

before writing it to state.

---

# 21. Detection overlay

After detection, the tool generates:

```python
overlay_path = Path("artifacts/debug") / f"{tool_call_id}_detections.jpg"
```

Then:

```python
save_detection_overlay(...)
```

draws boxes and labels over the original image.

This image is a **developer-facing diagnostic artifact**.

It helps answer questions such as:

- Are the boxes aligned correctly?
- Did coordinate conventions get mixed up?
- Are detections duplicated?
- Is a low-confidence detection plausible?

The path is then written into:

```python
state["debug_artifacts"]
```

and also included in the textual tool summary.

Important subtlety:

A path string such as:

```text
artifacts/debug/call_xyz_detections.jpg
```

does not magically give the VLM vision access to that file.

The VLM would need the actual image sent as multimodal input.

---

# 22. Detector tool output has two representations

The detector result is deliberately returned in two forms.

## Structured graph state

```python
"detected_objects": detections
```

This is for programmatic use.

## Tool message

```python
ToolMessage(
    content=json.dumps(tool_summary, indent=2),
    tool_call_id=tool_call_id,
)
```

This is for the main VLM.

That split is useful:

```text
structured state
    = machines / later nodes

ToolMessage
    = main VLM reasoning context
```

---

# 23. Why `bbox` is not injected in `inspect_crop`

The crop tool signature begins with:

```python
bbox: list[int]
```

Unlike `state` and `tool_call_id`, this argument is intentionally model-controlled.

Why?

Because state may contain many detections.

The reasoning question is:

> Which region should be investigated next?

The main VLM should make that decision.

So the responsibility split is:

```text
state
    → contains available evidence

main VLM
    → chooses region of interest

inspect_crop
    → performs the requested measurement
```

---

# 24. The VLM is not restricted to exact YOLO boxes

The tool accepts arbitrary coordinates:

```python
inspect_crop(bbox=[x1, y1, x2, y2])
```

Therefore the main VLM may choose:

- one exact detector box,
- a larger region around one object,
- a region containing several nearby detections,
- another visually relevant region.

This was observed in actual runs.

For example, YOLO produced several nearby person boxes on the right side, while the VLM requested a larger region containing the group.

That first region selection was model-generated.

The later `expand_bbox(...)` step was deterministic Python logic.

Those are different operations.

---

# 25. Prompt policy for spatial use of YOLO boxes

The current system prompt explicitly tells the agent:

```text
When object detection returns bounding boxes, use their coordinates as
spatial evidence. Decide how those boxes relate to what you see in the
image and to the user's question.
```

It further permits:

```text
- direct use of a detector box,
- enlargement when context matters,
- a region containing multiple nearby detections.
```

This is intentionally less rigid than hard-coded merging.

The goal is not:

```text
if distance(box_i, box_j) < threshold:
    merge
```

The goal is:

```text
use detector geometry as evidence in a reasoning problem
```

---

# 26. Minimal tool-use policy

The prompt also says:

```text
Prefer the smallest number of crop inspections needed to resolve the current uncertainty.
Do not inspect multiple regions merely for coverage.
```

This was added after a run in which the model requested three crop inspections in the same turn.

That behavior was not technically wrong, but it showed that the prompt was encouraging spatial coverage without enough cost discipline.

The refined objective is closer to:

```text
maximize useful information
subject to minimal unnecessary tool use
```

---

# 27. `inspect_crop`: deterministic part

The crop tool first loads the original image:

```python
image = load_image(image_path)
```

Then it expands the requested region:

```python
expanded_bbox = expand_bbox(
    bbox=bbox,
    image_size=image.size,
    scale=0.3,
)
```

Then it saves the crop:

```python
save_crop(
    image_path=image_path,
    bbox=expanded_bbox,
    output_path=crop_path,
)
```

This part is ordinary deterministic image processing.

---

# 28. Why expand the bbox

Detector boxes are optimized for localization, not scene reasoning.

A tight box may show:

```text
person
```

while an expanded crop may reveal:

```text
person + curb + taxi + pole + nearby people + doorway
```

The second view is often more useful when the question concerns:

- relationships,
- occlusion,
- interactions,
- scene plausibility,
- unusual context.

The current rule adds 30% of box width and height on each side, then clamps the result to image bounds.

---

# 29. Crop VLM vs main VLM

The project currently uses the same underlying VLM in two conceptual roles.

## Main VLM

Acts as the investigator:

```text
What do I know?
What am I uncertain about?
Which tool should I call?
Should I stop?
```

## Crop VLM

Acts as a specialist sensor:

```text
Here is one cropped region.
Describe what is actually visible here.
```

The crop tool sends the cropped image to the VLM using a dedicated prompt.

The main VLM does not directly receive that crop image in the next step.

Instead it receives the crop specialist's **textual report** through a `ToolMessage`.

---

# 30. Full crop flow

```text
main VLM
    ↓
chooses bbox
    ↓
inspect_crop(bbox)
    ↓
load original image
    ↓
expand bbox
    ↓
save cropped image
    ↓
encode crop as multimodal image input
    ↓
crop VLM analyzes crop
    ↓
text analysis
    ↓
ToolMessage
    ↓
main VLM sees new evidence
```

The crop report is also saved in:

```python
state["inspected_regions"]
```

---

# 31. Example of the evidence flow

After detection and crop inspection, the main VLM effectively sees a message history resembling:

```text
HumanMessage
    ├── user question
    └── original image

AIMessage
    └── tool call: detect_objects

ToolMessage
    └── YOLO detections

AIMessage
    └── tool call: inspect_crop(bbox=...)

ToolMessage
    └── crop VLM textual report

AIMessage
    └── final reasoning
```

This is the heart of the current agent.

---

# 32. Multiple tool calls in one assistant turn

The model is capable of returning several tool calls in one `AIMessage`.

For example, an observed run produced conceptually:

```python
[
    inspect_crop(bbox=A),
    inspect_crop(bbox=B),
    inspect_crop(bbox=C),
]
```

Each invocation gets a different `tool_call_id`.

This matters because the framework must pair each result with the correct request.

---

# 33. Parallel execution inside `ToolNode`

When one `AIMessage` contains several independent tool calls, `ToolNode` can execute those calls concurrently when supported.

The graph itself still looks like:

```text
assistant → tools → assistant
```

The parallelism happens inside the tools node:

```text
                    ┌─ inspect_crop(A)
assistant → ToolNode├─ inspect_crop(B) → assistant
                    └─ inspect_crop(C)
```

This is different from graph-level fan-out.

---

# 34. `ToolNode` parallelism vs `Send`

These are two different kinds of dynamic parallel work.

## ToolNode parallel calls

Who decides?

```text
LLM
```

What happens?

```text
one ToolNode executes several requested tool invocations
```

Graph structure still contains one `tools` node.

---

## `Send`

Who decides?

```text
graph/router code
```

What happens?

```text
LangGraph dynamically creates multiple executions of a graph node
```

Example conceptual form:

```python
return [
    Send("analyze_object", {"object": obj})
    for obj in state["objects"]
]
```

This creates dynamic graph-level fan-out.

So:

```text
ToolNode multiple calls
    = multiple tool invocations inside one graph node

Send
    = multiple graph-node executions
```

---

# 35. Can `Send` appear together with tool logic?

Conceptually, LangGraph control mechanisms such as `Command` and `Send` can be composed in more advanced designs.

However, this project currently keeps the architecture cleaner:

```text
tools
    = perform capabilities / measurements

graph/router
    = own graph scheduling
```

That separation makes control flow easier to reason about.

A future design might detect candidate regions with a tool, store them in state, and then use a graph router with `Send` to fan out a dedicated region-analysis node.

That would be a natural use of `Send`.

---

# 36. `vision_utils.py`: deterministic image layer

This file deliberately contains no LangGraph or agent reasoning.

Current responsibilities:

```text
load_image
encode_image_base64
crop_image
save_crop
clamp_bbox
expand_bbox
save_detection_overlay
```

This separation is important because deterministic geometry/image operations should not be mixed with LLM policy logic.

---

# 37. `clamp_bbox`

The current function:

```python
def clamp_bbox(
    bbox: list[int],
    image_size: tuple[int, int],
) -> list[int]:
```

handles several edge cases.

## Wrong coordinate count

```python
if len(bbox) != 4:
    raise ValueError(...)
```

## Out-of-bounds coordinates

Coordinates are constrained to image dimensions.

## Reversed corners

Using `min`/`max` guarantees:

```text
left ≤ right
top ≤ bottom
```

## Degenerate rectangles

Zero-width or zero-height boxes raise an error.

This is good defensive image geometry.

---

# 38. `expand_bbox`

The requested bbox is first clamped.

Then:

```python
box_width = x2 - x1
box_height = y2 - y1

dx = box_width * scale
dy = box_height * scale
```

The box is enlarged symmetrically:

```text
left   = x1 - dx
right  = x2 + dx
top    = y1 - dy
bottom = y2 + dy
```

Then the expanded result is clamped again.

That final clamp matters near image boundaries.

---

# 39. `save_detection_overlay`

This utility loads the source image and draws:

- bounding rectangles,
- labels,
- confidences.

It is currently intended for debugging, not model reasoning.

The explicit red/white drawing choices are presentation details; they do not affect graph semantics.

---

# 40. Prompt design

The current prompt is important enough to be treated as part of the architecture.

It contains several policies.

## Do not assume the user's suspicion is true

This guards against hypothesis pressure.

Without this instruction, a model asked:

> “Something looks strange. Figure it out.”

may feel pressured to invent an anomaly.

The prompt keeps open:

```text
H0: nothing unusual is present
```

---

## Reason from evidence

The VLM should combine:

```text
direct visual evidence
+
detector evidence
+
crop evidence
```

rather than treat tool outputs as unquestionable truth.

---

## Use tools when they reduce uncertainty

This is the most important agentic principle in the current prompt.

The agent should not call tools merely because they exist.

The intended decision is:

```text
Which observation would most help distinguish plausible explanations?
```

---

## Stop when evidence is sufficient

Without a stopping policy, an agent can endlessly gather redundant evidence.

The prompt explicitly discourages unnecessary calls.

---

# 41. Current system prompt

The current prompt is:

```text
You are a visual investigation agent.

Your task is to answer the user's question about the image by combining
direct visual inspection with perception tools when useful.

Do not assume the user's suspicion is correct.
Always keep open the possibility that nothing unusual is present.

Reason from evidence.

You may use object detection when identifying objects and their locations
would help.

You may use crop inspection when a particular region is:
- ambiguous,
- small,
- partially occluded,
- suspicious,
- or important for distinguishing between plausible explanations.

When object detection returns bounding boxes, use their coordinates as
spatial evidence. Decide how those boxes relate to what you see in the
image and to the user's question.

When choosing a region for crop inspection, you may:
- inspect a detector bounding box directly,
- enlarge a detector box when surrounding context matters,
- or choose a region that contains multiple nearby detections when their
  spatial relationship is relevant.

Do not treat detector bounding boxes as exact or infallible. Use them as
guides for deciding where additional visual inspection would be useful.
Prefer the smallest number of crop inspections needed to resolve the current uncertainty.
Do not inspect multiple regions merely for coverage.
Choose the region whose inspection is most likely to reduce uncertainty
about the user's question.

After every tool result:
1. reconsider the current evidence,
2. revise your hypothesis if necessary,
3. decide whether another observation is actually needed.

Do not keep calling tools when the evidence is already sufficient.

Your final answer should distinguish between:
- what is directly visible,
- what is supported by tool evidence,
- and what remains uncertain.
```

This is more than style. It is a lightweight decision policy.

---

# 42. Streaming the graph

The runner uses:

```python
for chunk in graph.stream(
    initial_state,
    stream_mode="updates",
):
    print("\n--- GRAPH UPDATE ---")
    print(chunk)
```

`updates` shows the partial state patch produced by each node execution.

Conceptually:

```text
node computes Δs
```

and the runtime merges it into graph state.

This is useful for observing:

```text
assistant → tool call
tools → detector output
assistant → crop call
tools → crop result
assistant → final response
```

---

# 43. Why raw stream output is noisy

The streamed objects contain much more than the logical action:

- token counts,
- model metadata,
- message IDs,
- tool call IDs,
- serialized JSON,
- response metadata.

The useful logical trace is much smaller.

For example:

```text
ASSISTANT → detect_objects
TOOL → 15 detections
ASSISTANT → inspect_crop([405,210,485,275])
TOOL → crop analysis: nothing anomalous
ASSISTANT → final answer
```

A future helper should convert raw LangGraph messages into such a concise trace.

---

# 44. What an observed run actually did

A representative run followed this control flow:

```text
1. Main VLM sees image + user question
2. Main VLM requests detect_objects()
3. YOLO finds 15 objects
4. Tool writes detections to state
5. Tool creates detector overlay
6. Tool returns ToolMessage
7. Main VLM sees original image + YOLO evidence
8. Main VLM selects one right-side region
9. inspect_crop expands the box
10. crop image is saved
11. crop VLM analyzes the crop
12. crop report is written to state
13. crop report is returned as ToolMessage
14. main VLM re-evaluates the scene
15. main VLM concludes no definite anomaly exists
16. no further tool call is emitted
17. tools_condition routes to END
```

That is a genuine reasoning/tool loop, even though the graph itself is small.

---

# 45. Direct visual evidence vs tool evidence

The final answer is instructed to separate three epistemic categories.

## Directly visible

What the main VLM believes it can see in the original image.

## Tool-supported

What is supported by YOLO or crop inspection.

## Uncertain

What remains ambiguous because of resolution, occlusion, detector uncertainty, or insufficient evidence.

This is a good pattern for avoiding the illusion that every statement has equal confidence.

---

# 46. Main VLM does not directly consume all graph state

This subtle point is important.

LangGraph state may contain:

```python
state["detected_objects"]
state["inspected_regions"]
state["debug_artifacts"]
state["current_hypothesis"]
```

But the model only sees what `assistant.py` passes into:

```python
model.invoke(messages)
```

Right now, tool results reach the model primarily because they are also represented as `ToolMessage`s.

A structured field sitting in state is not automatically visible to the LLM.

This separation is useful and should remain explicit.

---

# 47. Current role of `current_hypothesis`

The state declares:

```python
current_hypothesis: str | None
```

But the current implementation does not meaningfully use it yet.

The main VLM performs hypothesis revision implicitly in message history.

A future version could make hypothesis tracking explicit:

```python
current_hypothesis = "Right-side group may be anomalous due to overlap"
```

and then update it after crop inspection.

That would make the agent's epistemic state more structured and inspectable.

---

# 48. Current role of `debug_artifacts`

Likewise, `debug_artifacts` is currently useful for developers but not part of the main reasoning loop.

A future UI could render:

```text
original image
↓
detector overlay
↓
selected crop
↓
crop report
```

This would make the investigation trace visually auditable.

---

# 49. Important design choice: do not let the LLM invent paths

A weaker tool API would be:

```python
def detect_objects(image_path: str):
    ...
```

Then the model must provide a file path.

The current API is better:

```python
def detect_objects(
    state: Annotated[PerceptionState, InjectedState],
    ...
):
```

The model chooses the action.

The runtime supplies application context.

That reduces unnecessary model authority.

---

# 50. Important design choice: let the LLM choose bbox

The opposite decision was made for crop coordinates.

We intentionally expose:

```python
bbox: list[int]
```

because region choice is part of the investigation policy.

So the agent controls:

```text
where should I look next?
```

while the tool controls:

```text
how do I crop safely and consistently?
```

This is a good division between reasoning and deterministic execution.

---

# 51. Tool call ID identifies an invocation, not a tool type

If detection runs twice:

```text
call_A → detect_objects
call_B → detect_objects
```

those calls have different IDs.

So the identifier means:

```text
this particular invocation
```

not:

```text
this tool function
```

This becomes essential with parallel or repeated calls.

---

# 52. Session/thread ID is different from tool call ID

These are different identities.

```text
thread_id
    = persistent conversation / graph lineage

tool name
    = capability being requested

tool_call_id
    = one particular invocation of that capability
```

The current runner does not yet use a persistent checkpointer/thread.

---

# 53. Current graph does not use checkpointing

The project previously explored LangGraph checkpointing, but this perception-agent graph is currently compiled without a checkpointer.

That is deliberate for now.

Before adding persistence, the first goal was to prove a single visual investigation loop works correctly.

Checkpointing can later support multi-turn investigations such as:

```text
User: What's strange here?
Agent: I suspect region X.
User: Focus on the left side instead.
Agent: continues with prior state/history.
```

---

# 54. Current limitations

The agent works, but several limitations are important.

## 54.1 Crop VLM returns text, not structured perception

The crop specialist currently returns free-form text.

That makes downstream reasoning flexible but less machine-verifiable.

A future version could return structured fields such as:

```python
{
    "objects": [...],
    "occlusions": [...],
    "spatial_relations": [...],
    "anomaly_score": ...,
    "uncertainties": [...],
}
```

---

## 54.2 The agent may overuse tools

Prompting reduces this, but there is no hard budget yet.

A future state field might track:

```python
tool_calls_used
max_tool_calls
```

or define a cost-aware policy.

---

## 54.3 The agent can generate arbitrary bbox coordinates

This flexibility is useful but also risky.

The tool protects image operations with `clamp_bbox`, but the model could still request a semantically poor region.

That is an investigation-quality problem rather than a runtime safety problem.

---

## 54.4 YOLO detections are not ground truth

The prompt explicitly warns against treating them as infallible.

Low-confidence detections, duplicate boxes, missed objects, and class confusion are expected possibilities.

---

## 54.5 Crop enlargement is fixed at 30%

```python
scale=0.3
```

is a heuristic.

Different tasks may need different context windows.

A future tool could expose a context parameter or choose context adaptively.

---

## 54.6 Main VLM does not see the detector overlay

It receives detector coordinates through the `ToolMessage`, but the generated overlay is for debugging.

A future variant could optionally send the overlay back to the VLM as a multimodal artifact.

That would let the VLM visually inspect the detector's own geometric output.

---

## 54.7 No explicit uncertainty representation

The prompt talks about uncertainty, but state does not yet contain a formal uncertainty object.

A richer system might track:

```python
candidate_hypotheses
uncertainties
next_measurement_rationale
```

This would make the investigation loop easier to audit.

---

# 55. The deeper agentic idea

The current agent can be summarized as:

```text
perception is not one forward pass
```

Instead:

```text
perception
    = active evidence gathering under uncertainty
```

The main VLM is not merely classifying the image.

It can decide:

```text
I need object localization.
```

Then:

```text
I need a closer look at this region.
```

Then:

```text
The new evidence weakens my anomaly hypothesis, so I should stop.
```

That is the core of the project.

---

# 56. Control-flow authority

Several mechanisms participate in control flow.

## Fixed edges

Graph author decides:

```python
builder.add_edge("tools", "assistant")
```

## Conditional router

Runtime state determines route:

```text
tools_condition
```

## LLM tool call

The model decides which capability it wants.

## ToolNode

Executes the requested capabilities.

## Command

Lets tool/node return structured updates and potentially control instructions.

## Send

Supports dynamic graph-level fan-out.

These mechanisms are related but not interchangeable.

---

# 57. Reasoning policy vs scheduling mechanism

One of the most useful mental models is:

```text
reasoning policy
    ≠
graph scheduling mechanism
```

For example:

```text
main VLM decides:
"inspect these three regions"
```

That is a reasoning/action decision.

`ToolNode` then determines how those tool requests are actually executed within the graph runtime.

Likewise:

```text
router returns Send(...)
```

is a graph-scheduling decision, not a visual reasoning capability by itself.

---

# 58. Current end-to-end data flow

```text
USER
  │
  │ question + image
  ▼
HumanMessage
  │
  ▼
PerceptionState.messages
  │
  ▼
MAIN VLM
  │
  │ decides detect_objects
  ▼
AIMessage(tool_call)
  │
  ▼
tools_condition
  │
  ▼
ToolNode
  │
  ▼
detect_objects
  │
  ├── YOLO inference
  ├── bbox validation
  ├── structured detections
  ├── detection overlay
  ├── debug_artifacts update
  └── ToolMessage
  │
  ▼
MAIN VLM
  │
  │ original image + detector report
  │ chooses bbox
  ▼
AIMessage(tool_call=inspect_crop)
  │
  ▼
ToolNode
  │
  ▼
inspect_crop
  │
  ├── load original image
  ├── clamp/expand region
  ├── save crop
  ├── send crop to crop VLM
  ├── store inspected_regions
  ├── store debug_artifacts
  └── ToolMessage(crop analysis)
  │
  ▼
MAIN VLM
  │
  │ original image + detector evidence + crop evidence
  ▼
final AIMessage
  │
  ▼
tools_condition
  │
  └── no tool calls → END
```

---

# 59. Current runner

The runner builds the initial multimodal message, initializes state, and streams updates.

The important state initialization is:

```python
initial_state = {
    "image_path": image_path,
    "detected_objects": [],
    "inspected_regions": [],
    "debug_artifacts": [],
    "current_hypothesis": None,
    "messages": [human_message],
}
```

This establishes the initial conditions for the graph.

---

# 60. Why this architecture is useful for future perception work

The same control pattern can support additional capabilities without making the graph dramatically more complicated.

Potential future tools include:

```text
depth estimation
segmentation
OCR
camera calibration
geometric consistency checks
homography estimation
multi-view reasoning
optical diagnostics
retrieval of visually similar cases
```

The loop can remain:

```text
observe → decide → measure → update evidence → decide
```

The main design challenge becomes choosing the right measurement, not merely wiring more nodes.

---

# 61. A concise mental model

If only one picture of the architecture is remembered, use this one:

```text
                ┌─────────────────────┐
                │      MAIN VLM       │
                │ investigator/brain  │
                └─────────┬───────────┘
                          │
                 chooses a capability
                          │
                          ▼
                ┌─────────────────────┐
                │      ToolNode       │
                │ execution boundary  │
                └─────────┬───────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
      detect_objects              inspect_crop
             │                         │
             ▼                         ▼
           YOLO                  crop + crop VLM
             │                         │
             └────────────┬────────────┘
                          │
                          ▼
                     ToolMessage
                          │
                          ▼
                ┌─────────────────────┐
                │      MAIN VLM       │
                │ revise / act / stop │
                └─────────────────────┘
```

And the state underneath it contains four different kinds of information:

```text
messages
    = protocol/history

detected_objects
    = detector facts

inspected_regions
    = gathered semantic evidence

debug_artifacts
    = human-observable execution artifacts
```

---

# 62. Final perspective

The current system is small, but it already demonstrates several important agentic-AI ideas in a perception setting:

- an LLM does not need to own every computation;
- deterministic perception models can act as tools;
- structured state and conversational history serve different purposes;
- tool execution has a request/response protocol;
- injected runtime context should remain separate from model-chosen arguments;
- the model can use detector geometry as evidence rather than treating it as truth;
- focused inspection can be chosen dynamically;
- tool use can be parallel or sequential depending on the model's request;
- stopping is part of the reasoning policy;
- visual artifacts are valuable for observability even when they are not model inputs;
- a useful perception agent is fundamentally an **evidence-gathering loop**, not just a stack of vision models.

That is the current baseline of the project.
