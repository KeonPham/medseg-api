"""Data augmentation pipelines for training and inference."""

import logging

import albumentations as alb
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


def get_train_transforms(image_size: int = 512) -> alb.Compose:
    """Get augmentation pipeline for training.

    Args:
        image_size: Target image size.

    Returns:
        Albumentations Compose pipeline.
    """
    return alb.Compose(
        [
            alb.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=1.0,
            ),
            alb.HorizontalFlip(p=0.5),
            alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            alb.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            alb.ElasticTransform(alpha=50, sigma=5, p=0.2),
            alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 512) -> alb.Compose:
    """Get augmentation pipeline for validation/inference.

    Args:
        image_size: Target image size.

    Returns:
        Albumentations Compose pipeline with only resize and normalize.
    """
    return alb.Compose(
        [
            alb.Resize(image_size, image_size),
            alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
