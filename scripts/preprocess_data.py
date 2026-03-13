"""Preprocess raw datasets into standardized format for training."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def preprocess_dataset(raw_dir: Path, output_dir: Path, image_size: int = 512) -> None:
    """Preprocess a raw dataset: resize, normalize, and organize.

    Args:
        raw_dir: Directory containing raw images and masks.
        output_dir: Directory for processed output.
        image_size: Target image dimension.
    """
    raise NotImplementedError("Data preprocessing not yet implemented")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Preprocessing script — not yet implemented")
