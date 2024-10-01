import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

from ccrestoration.utils.device import DEFAULT_DEVICE

ASSETS_PATH = Path(__file__).resolve().parent.parent.absolute() / "assets"
TEST_IMG_PATH = ASSETS_PATH / "test.jpg"


def get_device() -> torch.device:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return torch.device("cpu")
    return DEFAULT_DEVICE


def load_image() -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(TEST_IMG_PATH), dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def calculate_image_similarity(image1: np.ndarray, image2: np.ndarray) -> bool:
    """
    calculate image similarity, check SR is correct

    :param image1: original image
    :param image2: upscale image
    :return:
    """
    # Resize the two images to the same size
    height, width = image1.shape[:2]
    image2 = cv2.resize(image2, (width, height))
    # Convert the images to grayscale
    grayscale_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayscale_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Calculate the Structural Similarity Index (SSIM) between the two images
    (score, diff) = structural_similarity(grayscale_image1, grayscale_image2, full=True)
    print("SSIM: {}".format(score))
    return score > 0.8


def compare_image_size(image1: np.ndarray, image2: np.ndarray, scale: int) -> bool:
    """
    compare original image size and upscale image size, check targetscale is correct

    :param image1: original image
    :param image2: upscale image
    :param scale: upscale ratio
    :return:
    """
    target_size = (math.ceil(image1.shape[0] * scale), math.ceil(image1.shape[1] * scale))

    return image2.shape[0] == target_size[0] and image2.shape[1] == target_size[1]
