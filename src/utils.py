import os
from typing import Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image


def hash_image(image: np.ndarray, hash_size: int = 32) -> str:
    return imagehash.phash(Image.fromarray(image), hash_size=hash_size)


def load_image(filepath: str) -> np.ndarray:
    image = cv2.imread(filepath)
    assert image is not None, filepath

    return image


def get_meta(filepath: str, image: np.ndarray) -> dict:
    height, width, channels = image.shape

    return {
        "height": height,
        "width": width,
        "channels": channels,
        "filepath": filepath,
    }


def load_and_hash(filepath: str) -> Tuple[str, dict]:
    image = load_image(filepath)
    meta = get_meta(filepath, image)
    h = hash_image(cv2.resize(image, (64, 64)))

    return h, meta
