"""Image loading, preprocessing, postprocessing, and encoding utilities."""

import base64
import io
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(source: bytes | str | np.ndarray) -> np.ndarray:
    """Load an image from bytes, file path, or numpy array.

    Args:
        source: Raw image bytes, a file path string, or a numpy array.

    Returns:
        Image as RGB numpy array (H, W, 3) with dtype uint8.

    Raises:
        ValueError: If the image cannot be decoded.
        FileNotFoundError: If the file path does not exist.
    """
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            return np.stack([source] * 3, axis=-1)
        return source

    if isinstance(source, str):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # bytes
    arr = np.frombuffer(source, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from bytes")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess(image: np.ndarray, target_size: int = 512) -> torch.Tensor:
    """Resize, normalize to [0,1], and convert to a batched tensor.

    Args:
        image: RGB image (H, W, 3) with dtype uint8.
        target_size: Target height and width.

    Returns:
        Float tensor of shape (1, 3, target_size, target_size).
    """
    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float() / 255.0  # (H, W, 3)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor


def postprocess(
    mask_tensor: torch.Tensor,
    original_size: tuple[int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert model output to a binary mask at the original resolution.

    Args:
        mask_tensor: Raw logits from the model, shape (1, 1, H, W) or (1, H, W).
        original_size: (height, width) of the original image.
        threshold: Binarization threshold applied after sigmoid.

    Returns:
        Binary mask as uint8 numpy array (H, W) with values 0 or 255.
    """
    probs = torch.sigmoid(mask_tensor).squeeze().cpu().numpy()  # (H, W)
    binary = (probs > threshold).astype(np.uint8) * 255
    if binary.shape[:2] != original_size:
        binary = cv2.resize(
            binary, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST
        )
    return binary


def mask_to_base64(mask: np.ndarray) -> str:
    """Encode a uint8 mask as a base64 PNG string.

    Args:
        mask: Mask array (H, W) with values 0-255.

    Returns:
        Base64-encoded PNG string.
    """
    img = Image.fromarray(mask, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_to_base64(image: np.ndarray) -> str:
    """Encode an RGB uint8 image as a base64 PNG string.

    Args:
        image: RGB image array (H, W, 3) with dtype uint8.

    Returns:
        Base64-encoded PNG string.
    """
    img = Image.fromarray(image, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.3,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Create a colored overlay with contour outline on the original image.

    Args:
        image: Original RGB image (H, W, 3).
        mask: Binary mask (H, W) with values 0 or 255.
        alpha: Overlay transparency (0 = invisible, 1 = opaque).
        color: RGB color for the mask overlay.

    Returns:
        RGB image (H, W, 3) with mask overlay and contour.
    """
    overlay = image.copy()
    mask_bool = mask > 127

    # Semi-transparent color fill on segmented region
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * np.array(color, dtype=np.float64)
    ).astype(np.uint8)

    # Draw contour outline for crisp boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness=2)

    return overlay
