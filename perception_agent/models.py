# perception_agent/models.py

from functools import cache

import torch
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from ultralytics import YOLO


load_dotenv(find_dotenv(), override=True)


VLM_MODEL = "openai:gpt-5.4"
DETECTOR_MODEL = "yolo11n.pt"


@cache
def get_vlm():
    """
    Return the multimodal language model used for visual reasoning.
    """
    return init_chat_model(VLM_MODEL)


@cache
def get_detector():
    """
    Return the object detector.

    The model is loaded only once per Python process.
    """
    return YOLO(DETECTOR_MODEL)


def get_detector_device():
    """
    Device passed to the detector during inference.
    """
    return 0 if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    detector = get_detector()
    print(f'Detector: {detector}')
    device = get_detector_device()
    print(f'Device found: {device}')
    