"""Image preprocessing utilities for the segmentation pipeline."""

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(file_bytes: bytes) -> np.ndarray:
    """Load an image from raw bytes.

    Args:
        file_bytes: Raw image file bytes.

    Returns:
        Image as numpy array in RGB format (H, W, 3).
    """
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from bytes")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_image(image: np.ndarray, target_size: int = 512) -> np.ndarray:
    """Preprocess image for model input.

    Args:
        image: Input image as numpy array (H, W, C).
        target_size: Target dimension for resizing.

    Returns:
        Preprocessed image array (target_size, target_size, 3).
    """
    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return resized


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Convert a binary mask to PNG bytes for API response.

    Args:
        mask: Binary mask array (H, W) with values in {0, 1}.

    Returns:
        PNG-encoded bytes.
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8, mode="L")
    import io

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to 3-channel RGB.

    Args:
        image: Grayscale image (H, W) or (H, W, 1).

    Returns:
        RGB image (H, W, 3).
    """
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    if image.ndim == 3 and image.shape[-1] == 1:
        return np.concatenate([image] * 3, axis=-1)
    return image
