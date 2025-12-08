# app/utils/image_utils.py
import numpy as np
import cv2
from typing import Optional

def bytes_to_bgr_image(data: bytes) -> Optional[np.ndarray]:
    """
    Convert image bytes to an OpenCV BGR numpy image (no files written).
    Returns None if conversion fails.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img  # BGR (cv2 uses BGR)
