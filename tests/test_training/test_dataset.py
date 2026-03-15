"""Tests for dataset and data loading."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.training.dataset import LungSegmentationDataset, create_dataloaders


def _create_dummy_dataset(tmp_path: Path, n_images: int = 10, size: int = 64) -> tuple[Path, Path]:
    """Create a temp directory with dummy images and masks."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    for i in range(n_images):
        # Random RGB image
        img = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i:03d}.png"), img)

        # Random binary mask
        mask = np.random.choice([0, 255], (size, size)).astype(np.uint8)
        cv2.imwrite(str(mask_dir / f"img_{i:03d}.png"), mask)

    return img_dir, mask_dir


class TestLungSegmentationDataset:
    def test_length(self, tmp_path: Path) -> None:
        img_dir, mask_dir = _create_dummy_dataset(tmp_path, n_images=5)
        ds = LungSegmentationDataset(img_dir, mask_dir, image_size=32)
        assert len(ds) == 5

    def test_getitem_returns_dict(self, tmp_path: Path) -> None:
        img_dir, mask_dir = _create_dummy_dataset(tmp_path, n_images=3)
        ds = LungSegmentationDataset(img_dir, mask_dir, image_size=32)
        item = ds[0]
        assert isinstance(item, dict)
        assert "image" in item
        assert "mask" in item
        assert "filename" in item

    def test_image_shape(self, tmp_path: Path) -> None:
        img_dir, mask_dir = _create_dummy_dataset(tmp_path, n_images=2)
        ds = LungSegmentationDataset(img_dir, mask_dir, image_size=64)
        item = ds[0]
        assert item["image"].shape == (3, 64, 64)
        assert item["mask"].shape == (1, 64, 64)

    def test_image_dtype(self, tmp_path: Path) -> None:
        img_dir, mask_dir = _create_dummy_dataset(tmp_path, n_images=2)
        ds = LungSegmentationDataset(img_dir, mask_dir, image_size=32)
        item = ds[0]
        assert item["image"].dtype == torch.float32
        assert item["mask"].dtype == torch.float32

    def test_mask_is_binary(self, tmp_path: Path) -> None:
        img_dir, mask_dir = _create_dummy_dataset(tmp_path, n_images=2)
        ds = LungSegmentationDataset(img_dir, mask_dir, image_size=32)
        item = ds[0]
        unique = torch.unique(item["mask"])
        assert all(v in (0.0, 1.0) for v in unique.tolist())

    def test_filters_non_image_files(self, tmp_path: Path) -> None:
        img_dir, mask_dir = _create_dummy_dataset(tmp_path, n_images=3)
        # Add a non-image file
        (img_dir / "readme.txt").write_text("not an image")
        ds = LungSegmentationDataset(img_dir, mask_dir, image_size=32)
        assert len(ds) == 3

    def test_missing_mask_raises(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "orphan.png"), img)
        ds = LungSegmentationDataset(img_dir, mask_dir, image_size=32)
        with pytest.raises(FileNotFoundError):
            ds[0]


class TestCreateDataloaders:
    def test_split_sizes(self, tmp_path: Path) -> None:
        img_dir, mask_dir = _create_dummy_dataset(tmp_path, n_images=20)
        train_dl, val_dl, test_dl = create_dataloaders(
            img_dir,
            mask_dir,
            batch_size=4,
            num_workers=0,
        )
        total = len(train_dl.dataset) + len(val_dl.dataset) + len(test_dl.dataset)
        assert total == 20
        assert len(train_dl.dataset) == 14  # 70% of 20
        assert len(val_dl.dataset) == 3  # 15% of 20
        assert len(test_dl.dataset) == 3  # remainder

    def test_batch_shapes(self, tmp_path: Path) -> None:
        img_dir, mask_dir = _create_dummy_dataset(tmp_path, n_images=10)
        train_dl, _, _ = create_dataloaders(
            img_dir,
            mask_dir,
            batch_size=2,
            num_workers=0,
            image_size=32,
        )
        batch = next(iter(train_dl))
        assert batch["image"].shape[0] <= 2
        assert batch["image"].shape[1] == 3
        assert batch["mask"].shape[0] <= 2

    def test_reproducible_split(self, tmp_path: Path) -> None:
        img_dir, mask_dir = _create_dummy_dataset(tmp_path, n_images=10)
        train1, _, _ = create_dataloaders(
            img_dir,
            mask_dir,
            batch_size=2,
            num_workers=0,
            seed=42,
        )
        train2, _, _ = create_dataloaders(
            img_dir,
            mask_dir,
            batch_size=2,
            num_workers=0,
            seed=42,
        )
        assert list(train1.dataset.indices) == list(train2.dataset.indices)
