"""Tests for image utilities and the inference pipeline."""

import base64

import numpy as np
import pytest
import torch

from src.utils.image import (
    load_image,
    mask_to_base64,
    overlay_mask,
    postprocess,
    preprocess,
)


class TestLoadImage:
    """Tests for load_image from various sources."""

    def test_from_bytes(self, white_png_bytes: bytes) -> None:
        img = load_image(white_png_bytes)
        assert isinstance(img, np.ndarray)
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8

    def test_from_numpy(self) -> None:
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        img = load_image(arr)
        assert img is arr

    def test_from_grayscale_numpy(self) -> None:
        arr = np.zeros((32, 32), dtype=np.uint8)
        img = load_image(arr)
        assert img.shape == (32, 32, 3)

    def test_from_file(self, tmp_path, white_png_bytes: bytes) -> None:
        path = tmp_path / "test.png"
        path.write_bytes(white_png_bytes)
        img = load_image(str(path))
        assert img.shape == (64, 64, 3)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path.png")

    def test_invalid_bytes(self) -> None:
        with pytest.raises(ValueError, match="Failed to decode"):
            load_image(b"not an image")


class TestPreprocess:
    """Tests for preprocess tensor conversion."""

    def test_output_shape_default(self, white_png_bytes: bytes) -> None:
        img = load_image(white_png_bytes)
        tensor = preprocess(img, target_size=512)
        assert tensor.shape == (1, 3, 512, 512)

    def test_output_shape_custom(self, white_png_bytes: bytes) -> None:
        img = load_image(white_png_bytes)
        tensor = preprocess(img, target_size=256)
        assert tensor.shape == (1, 3, 256, 256)

    def test_output_dtype(self, white_png_bytes: bytes) -> None:
        img = load_image(white_png_bytes)
        tensor = preprocess(img)
        assert tensor.dtype == torch.float32

    def test_values_normalized(self, white_png_bytes: bytes) -> None:
        img = load_image(white_png_bytes)
        tensor = preprocess(img)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_non_square_input(self, gradient_png_bytes: bytes) -> None:
        img = load_image(gradient_png_bytes)
        assert img.shape[:2] == (96, 128)
        tensor = preprocess(img, target_size=512)
        assert tensor.shape == (1, 3, 512, 512)


class TestPostprocess:
    """Tests for postprocess mask conversion."""

    def test_binary_mask(self) -> None:
        logits = torch.randn(1, 1, 64, 64)
        mask = postprocess(logits, original_size=(64, 64))
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

    def test_resize_to_original(self) -> None:
        logits = torch.randn(1, 1, 64, 64)
        mask = postprocess(logits, original_size=(200, 300))
        assert mask.shape == (200, 300)

    def test_all_positive_logits(self) -> None:
        logits = torch.full((1, 1, 32, 32), 5.0)  # sigmoid -> ~0.993
        mask = postprocess(logits, original_size=(32, 32))
        assert np.all(mask == 255)

    def test_all_negative_logits(self) -> None:
        logits = torch.full((1, 1, 32, 32), -5.0)  # sigmoid -> ~0.007
        mask = postprocess(logits, original_size=(32, 32))
        assert np.all(mask == 0)


class TestMaskToBase64:
    """Tests for base64 mask encoding."""

    def test_returns_string(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = mask_to_base64(mask)
        assert isinstance(result, str)

    def test_valid_base64(self) -> None:
        mask = np.full((32, 32), 255, dtype=np.uint8)
        result = mask_to_base64(mask)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b"\x89PNG"  # PNG magic bytes

    def test_roundtrip(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        b64 = mask_to_base64(mask)
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0


class TestOverlayMask:
    """Tests for mask overlay rendering."""

    def test_output_shape(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        result = overlay_mask(image, mask)
        assert result.shape == (64, 64, 3)

    def test_no_mask_unchanged(self) -> None:
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        result = overlay_mask(image, mask)
        np.testing.assert_array_equal(result, image)

    def test_mask_modifies_region(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.full((32, 32), 255, dtype=np.uint8)
        result = overlay_mask(image, mask, alpha=0.5, color=(0, 255, 0))
        # Green channel should have non-zero values in masked region
        assert result[:, :, 1].sum() > 0

    def test_original_not_modified(self) -> None:
        image = np.full((32, 32, 3), 100, dtype=np.uint8)
        original_copy = image.copy()
        mask = np.full((32, 32), 255, dtype=np.uint8)
        overlay_mask(image, mask)
        np.testing.assert_array_equal(image, original_copy)
