"""Model registry for managing segmentation model variants."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_VARIANTS = {
    "cnn": {
        "name": "CNN-only",
        "architecture": "ResNet-18 encoder + U-Net decoder",
        "parameters": "~15-18M",
        "class": "CNNLungSeg",
    },
    "vit": {
        "name": "ViT-only",
        "architecture": "DeiT-Tiny + progressive decoder",
        "parameters": "~8-10M",
        "class": "ViTLungSeg",
    },
    "hybrid": {
        "name": "Hybrid CNN-ViT",
        "architecture": "CNN + ViT + cross-attention fusion",
        "parameters": "~4.2M",
        "class": "HybridLungSeg",
    },
}


def list_available_models(model_dir: Path) -> list[dict]:
    """List models that have weights available on disk.

    Args:
        model_dir: Base directory containing model subdirectories.

    Returns:
        List of model metadata dicts for models with available weights.
    """
    available = []
    for variant_id, meta in MODEL_VARIANTS.items():
        variant_dir = model_dir / variant_id
        if variant_dir.exists() and any(variant_dir.glob("*.pth")):
            available.append({"model_id": variant_id, **meta})
            logger.info("Found model weights for %s in %s", variant_id, variant_dir)
    return available


def get_model_path(model_dir: Path, model_id: str) -> Path | None:
    """Get the path to a model's weight file.

    Args:
        model_dir: Base directory containing model subdirectories.
        model_id: Model variant identifier (cnn, vit, hybrid).

    Returns:
        Path to the weight file, or None if not found.
    """
    if model_id not in MODEL_VARIANTS:
        logger.error("Unknown model variant: %s", model_id)
        return None

    variant_dir = model_dir / model_id
    pth_files = sorted(variant_dir.glob("*.pth"))
    if not pth_files:
        logger.warning("No weights found for %s in %s", model_id, variant_dir)
        return None

    return pth_files[-1]
