"""Dataset classes for chest X-ray lung segmentation."""

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LungSegDataset(Dataset):
    """Dataset for chest X-ray lung segmentation.

    Supports Montgomery, Shenzhen, JSRT, and COVID-QU-Ex datasets.
    Expects image and mask pairs organized in directory structure.
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_size: int = 512,
        transform: object | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            image_dir: Directory containing chest X-ray images.
            mask_dir: Directory containing corresponding segmentation masks.
            image_size: Target size for image resizing.
            transform: Optional albumentations transform pipeline.
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.transform = transform
        self.image_paths = sorted(self.image_dir.glob("*.*"))
        logger.info("LungSegDataset: found %d images in %s", len(self.image_paths), image_dir)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single image-mask pair.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, mask_tensor).
        """
        raise NotImplementedError("Dataset loading not yet implemented")

    def get_mask_path(self, image_path: Path) -> Path:
        """Derive mask path from image path.

        Args:
            image_path: Path to the input image.

        Returns:
            Corresponding mask file path.
        """
        return self.mask_dir / image_path.name
