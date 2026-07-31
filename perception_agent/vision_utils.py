# perception_agent/vision_utils.py

import base64
import mimetypes
from pathlib import Path

from PIL import Image


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

    x1, y1, x2, y2 = bbox

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