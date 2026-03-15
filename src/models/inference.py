"""Inference pipeline for running segmentation predictions."""

import logging
import time
from datetime import datetime

import cv2
import numpy as np
import torch

from src.api.schemas.response import BatchResult, SegmentationResult
from src.models.registry import ModelRegistry
from src.utils.config import ModelConfig
from src.utils.image import (
    load_image,
    mask_to_base64,
    overlay_mask,
    postprocess,
    preprocess,
)

logger = logging.getLogger(__name__)


def _compute_metrics(probabilities: np.ndarray, binary_mask: np.ndarray) -> dict[str, float]:
    """Compute segmentation quality metrics from a prediction.

    Args:
        probabilities: Sigmoid probabilities (H, W), values in [0, 1].
        binary_mask: Thresholded mask (H, W), values 0 or 255.

    Returns:
        Dictionary with lung_coverage_pct, confidence_score, symmetry_ratio.
    """
    mask_bool = binary_mask > 127
    total_pixels = mask_bool.size
    lung_pixels = int(mask_bool.sum())

    lung_coverage_pct = round(lung_pixels / total_pixels * 100, 2) if total_pixels else 0.0

    confidence_score = round(float(probabilities[mask_bool].mean()), 4) if lung_pixels > 0 else 0.0

    h, w = mask_bool.shape
    mid = w // 2
    left_area = int(mask_bool[:, :mid].sum())
    right_area = int(mask_bool[:, mid:].sum())
    if max(left_area, right_area) > 0:
        symmetry_ratio = round(min(left_area, right_area) / max(left_area, right_area), 4)
    else:
        symmetry_ratio = 0.0

    return {
        "lung_coverage_pct": lung_coverage_pct,
        "confidence_score": confidence_score,
        "symmetry_ratio": symmetry_ratio,
    }


class InferencePipeline:
    """End-to-end inference pipeline: load, preprocess, infer, postprocess.

    Coordinates the model registry, image utilities, and metric
    computation to produce structured segmentation results.
    """

    def __init__(self, registry: ModelRegistry, config: ModelConfig) -> None:
        """Initialize the inference pipeline.

        Args:
            registry: Model registry for loading and caching models.
            config: Model serving configuration.
        """
        self.registry = registry
        self.config = config
        self.device = config.get_device()
        logger.info("InferencePipeline initialized on device: %s", self.device)

    async def predict_single(
        self,
        image_bytes: bytes,
        model_name: str = "hybrid",
        return_overlay: bool = False,
    ) -> SegmentationResult:
        """Run segmentation on a single image.

        Args:
            image_bytes: Raw image file bytes.
            model_name: Model variant to use.
            return_overlay: Whether to return an overlay image.

        Returns:
            SegmentationResult with mask, metrics, and timing.
        """
        start = time.perf_counter()

        # Load and record original size
        image = load_image(image_bytes)
        original_h, original_w = image.shape[:2]

        # Preprocess
        tensor = preprocess(image, target_size=self.config.image_size)
        tensor = tensor.to(self.device)

        # Get model and run inference
        model = self.registry.get_model(model_name, device=self.device)
        with torch.no_grad():
            logits = model(tensor)

        # Get probabilities for metrics before binarization
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Postprocess to original size
        mask = postprocess(logits, (original_h, original_w))

        # Resize probabilities to match the postprocessed mask dimensions
        if probabilities.shape != mask.shape:
            probabilities = cv2.resize(
                probabilities, (original_w, original_h), interpolation=cv2.INTER_LINEAR
            )

        # Compute metrics
        metrics = _compute_metrics(probabilities, mask)

        # Encode mask
        mask_b64 = mask_to_base64(mask)

        # Optional overlay
        overlay_b64 = None
        if return_overlay:
            overlay_img = overlay_mask(image, mask)
            overlay_b64 = mask_to_base64(overlay_img[:, :, 0])  # encode as grayscale

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Resolve version
        info = self.registry.get_model_info(model_name)
        version = info.version if info else "unknown"

        return SegmentationResult(
            model_name=model_name,
            model_version=version,
            inference_time_ms=round(elapsed_ms, 2),
            mask_base64=mask_b64,
            overlay_base64=overlay_b64,
            metrics=metrics,
            image_size={"width": original_w, "height": original_h},
            timestamp=datetime.now(),
        )

    async def predict_batch(
        self,
        images: list[bytes],
        model_name: str = "hybrid",
        return_overlay: bool = False,
    ) -> BatchResult:
        """Run segmentation on a batch of images.

        Args:
            images: List of raw image file bytes.
            model_name: Model variant to use.
            return_overlay: Whether to return overlay images.

        Returns:
            BatchResult containing individual results and total timing.
        """
        start = time.perf_counter()

        results = []
        for image_bytes in images:
            result = await self.predict_single(
                image_bytes, model_name=model_name, return_overlay=return_overlay
            )
            results.append(result)

        total_ms = (time.perf_counter() - start) * 1000

        return BatchResult(
            results=results,
            total_time_ms=round(total_ms, 2),
            count=len(results),
        )
