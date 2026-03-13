"""Data augmentation pipelines for training and inference."""

import logging

import albumentations as alb

logger = logging.getLogger(__name__)


def get_training_augmentations(image_size: int = 512) -> alb.Compose:
    """Get augmentation pipeline for training.

    Args:
        image_size: Target image size.

    Returns:
        Albumentations Compose pipeline.
    """
    return alb.Compose(
        [
            alb.Resize(image_size, image_size),
            alb.HorizontalFlip(p=0.5),
            alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def get_validation_augmentations(image_size: int = 512) -> alb.Compose:
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
        ]
    )
