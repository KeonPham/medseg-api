"""Dataset classes for chest X-ray lung segmentation."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


class LungSegmentationDataset(Dataset):
    """Dataset for chest X-ray lung segmentation.

    Supports Montgomery, Shenzhen, JSRT, and COVID-QU-Ex datasets.
    Expects image and mask pairs organized in directory structure.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        transform: object | None = None,
        image_size: int = 512,
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
        self.image_paths = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        logger.info(
            "LungSegmentationDataset: %d images from %s",
            len(self.image_paths),
            image_dir,
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Get a single image-mask pair.

        Args:
            idx: Sample index.

        Returns:
            Dict with keys: image (C,H,W tensor), mask (1,H,W tensor), filename.
        """
        img_path = self.image_paths[idx]
        mask_path = self._get_mask_path(img_path)

        # Load image as RGB
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask as grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        # Resize if no transform will handle it
        if self.transform is None:
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
            mask = cv2.resize(
                mask,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )

        # Binarize mask
        mask = (mask > 127).astype(np.float32)

        # Apply augmentations
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image_tensor = transformed["image"]
            mask_tensor = torch.from_numpy(transformed["mask"]).float()
        else:
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            mask_tensor = torch.from_numpy(mask).float()

        # Ensure mask has channel dim (1, H, W)
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "filename": img_path.name,
        }

    def _get_mask_path(self, image_path: Path) -> Path:
        """Derive mask path from image path.

        Tries matching the exact filename first, then tries common
        mask naming conventions (stem-based matching across extensions).
        """
        # Exact name match
        exact = self.mask_dir / image_path.name
        if exact.exists():
            return exact

        # Try matching stem with different extensions
        for ext in IMAGE_EXTENSIONS:
            candidate = self.mask_dir / f"{image_path.stem}{ext}"
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"No mask found for {image_path.name} in {self.mask_dir}")


def create_dataloaders(
    image_dir: str | Path,
    mask_dir: str | Path,
    train_transform: object | None = None,
    val_transform: object | None = None,
    image_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test data loaders with a 70/15/15 split.

    Args:
        image_dir: Directory containing images.
        mask_dir: Directory containing masks.
        train_transform: Augmentation pipeline for training.
        val_transform: Augmentation pipeline for validation/test.
        image_size: Target image size.
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Build full dataset without transforms first (for splitting)
    full_dataset = LungSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=None,
        image_size=image_size,
    )

    n = len(full_dataset)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = random_split(
        range(n), [n_train, n_val, n_test], generator=generator
    )

    # Wrap subsets with appropriate transforms
    train_ds = _TransformedSubset(full_dataset, list(train_idx), train_transform)
    val_ds = _TransformedSubset(full_dataset, list(val_idx), val_transform)
    test_ds = _TransformedSubset(full_dataset, list(test_idx), val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "Dataloaders: train=%d, val=%d, test=%d (seed=%d)",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        seed,
    )
    return train_loader, val_loader, test_loader


class _TransformedSubset(Subset):
    """Subset wrapper that applies a transform to the parent dataset items."""

    def __init__(
        self,
        dataset: LungSegmentationDataset,
        indices: list[int],
        transform: object | None,
    ) -> None:
        super().__init__(dataset, indices)
        self.transform_override = transform

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        real_idx = self.indices[idx]
        item = self.dataset[real_idx]

        if self.transform_override is not None:
            # Re-load raw image/mask for proper augmentation
            img_path = self.dataset.image_paths[real_idx]
            mask_path = self.dataset._get_mask_path(img_path)

            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)

            transformed = self.transform_override(image=image, mask=mask)
            image_tensor = transformed["image"]
            mask_tensor = torch.from_numpy(transformed["mask"]).float()
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)

            return {
                "image": image_tensor,
                "mask": mask_tensor,
                "filename": img_path.name,
            }

        return item
