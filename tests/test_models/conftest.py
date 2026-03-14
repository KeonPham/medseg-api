"""Shared fixtures for model tests."""

import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def white_png_bytes() -> bytes:
    """Generate a 64x64 white PNG image as bytes."""
    img = Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def gradient_png_bytes() -> bytes:
    """Generate a 128x96 gradient PNG image as bytes."""
    arr = np.zeros((96, 128, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 128, dtype=np.uint8)  # red gradient
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
