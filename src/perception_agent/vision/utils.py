# perception_agent/vision_utils.py

import base64
import mimetypes
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_image(image_path: str | Path) -> Image.Image:
    """
    Load an image from disk as an RGB PIL image.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return Image.open(image_path).convert("RGB")


def encode_image_base64(image_path: str | Path) -> tuple[str, str]:
    """
    Return (base64_string, mime_type) for sending an image to a VLM.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    return image_base64, mime_type


def crop_image(
    image_path: str | Path,
    bbox: list[int],
) -> Image.Image:
    """
    Crop an image using bbox = [x1, y1, x2, y2].
    """
    image = load_image(image_path)

    x1, y1, x2, y2 = clamp_bbox(
        bbox=bbox,
        image_size=image.size,
    )

    return image.crop((x1, y1, x2, y2))


def save_crop(
    image_path: str | Path,
    bbox: list[int],
    output_path: str | Path,
) -> Path:
    """
    Crop an image and save the crop to disk.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop = crop_image(
        image_path=image_path,
        bbox=bbox,
    )

    crop.save(output_path)

    return output_path

def clamp_bbox(
    bbox: list[int],
    image_size: tuple[int, int],
) -> list[int]:
    """
    Clamp bbox = [x1, y1, x2, y2] to image bounds.
    """
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 coordinates, got: {bbox}")

    x1, y1, x2, y2 = bbox
    image_width, image_height = image_size

    x1 = max(0, min(image_width, round(x1)))
    y1 = max(0, min(image_height, round(y1)))
    x2 = max(0, min(image_width, round(x2)))
    y2 = max(0, min(image_height, round(y2)))

    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)

    if left == right or top == bottom:
        raise ValueError(f"Degenerate bbox after clamping: {bbox}")

    return [left, top, right, bottom]


def expand_bbox(
    bbox: list[int],
    image_size: tuple[int, int],
    scale: float = 0.3,
) -> list[int]:
    """
    Expand bbox = [x1, y1, x2, y2] by a fraction of its width/height.

    scale=0.3 means add 30% context on each side.
    """

    x1, y1, x2, y2 = clamp_bbox(
        bbox=bbox,
        image_size=image_size,
    )

    box_width = x2 - x1
    box_height = y2 - y1

    dx = box_width * scale
    dy = box_height * scale

    expanded = clamp_bbox(
        bbox=[
            round(x1 - dx),
            round(y1 - dy),
            round(x2 + dx),
            round(y2 + dy),
        ],
        image_size=image_size,
    )

    return expanded


def save_detection_overlay(
    image_path: str | Path,
    detections: list[dict],
    output_path: str | Path,
) -> Path:
    """
    Save a copy of the image with detector boxes drawn over it.
    """
    image = load_image(image_path)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    for detection in detections:
        bbox = clamp_bbox(
            bbox=detection["bbox"],
            image_size=image.size,
        )
        label = detection["label"]
        confidence = detection["confidence"]
        text = f"{label} {confidence:.2f}"

        draw.rectangle(bbox, outline="red", width=3)
        text_bbox = draw.textbbox((bbox[0], bbox[1]), text, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = max(0, bbox[1] - text_height - 4)
        label_box = [bbox[0], text_y, text_bbox[2] + 4, text_y + text_height + 4]
        draw.rectangle(label_box, fill="red")
        draw.text((bbox[0] + 2, text_y + 2), text, fill="white", font=font)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    image.save(output_path)

    return output_path


if __name__ == "__main__":
    image_path = "../images/street.jpg"

    image = load_image(image_path)
    print("Image size:", image.size)

    image_base64, mime_type = encode_image_base64(image_path)
    print("MIME type:", mime_type)
    print("Base64 length:", len(image_base64))

    output = save_crop(
        image_path=image_path,
        bbox=[180, 220, 430, 420],
        output_path="../artifacts/crops/test_crop.jpg",
    )

    print("Saved crop:", output)
