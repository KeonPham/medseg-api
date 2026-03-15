"""Training entry point for lung segmentation models."""

import argparse
import logging

import torch

from src.models.architectures import get_model
from src.training.augmentations import get_train_transforms, get_val_transforms
from src.training.dataset import create_dataloaders
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lung segmentation model")
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        choices=["cnn", "vit", "hybrid"],
        help="Model architecture",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to dataset directory (must contain images/ and masks/ subdirs)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="lung_segmentation",
        help="MLflow experiment name",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    """Run model training with CLI arguments."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args = parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    # Create model
    model = get_model(args.model, pretrained=True)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s (%s params)", args.model, f"{param_count:,}")

    # Create data loaders
    image_dir = f"{args.dataset_dir}/images"
    mask_dir = f"{args.dataset_dir}/masks"

    train_loader, val_loader, _ = create_dataloaders(
        image_dir=image_dir,
        mask_dir=mask_dir,
        train_transform=get_train_transforms(args.image_size),
        val_transform=get_val_transforms(args.image_size),
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    # Build config dict
    config = {
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "experiment_name": args.experiment_name,
        "checkpoint_dir": f"checkpoints/{args.model}",
    }

    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    history = trainer.fit()

    best_dice = max(history["val_dice"]) if history["val_dice"] else 0.0
    logger.info("Training complete. Best val_dice: %.4f", best_dice)


if __name__ == "__main__":
    main()
