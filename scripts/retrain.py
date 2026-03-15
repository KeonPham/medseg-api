"""Continuous retraining pipeline.

Checks for new data, trains a model, evaluates against production,
and promotes if the new model meets the improvement threshold.

Usage:
    python scripts/retrain.py --model hybrid
    python scripts/retrain.py --model cnn --epochs 30 --batch-size 4
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import yaml

from src.models.architectures import get_model
from src.training.augmentations import get_train_transforms, get_val_transforms
from src.training.dataset import IMAGE_EXTENSIONS, create_dataloaders
from src.training.evaluation import compare_models, evaluate_model
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INCOMING_DIR = PROJECT_ROOT / "data" / "raw" / "incoming"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = PROJECT_ROOT / "configs" / "model_registry.yaml"

DICE_THRESHOLD = 0.005

# Map CLI model names to registry keys
MODEL_REGISTRY_MAP = {
    "cnn": "cnn_only",
    "vit": "vit_only",
    "hybrid": "hybrid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain a lung segmentation model")
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        choices=["cnn", "vit", "hybrid"],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def check_incoming_data() -> tuple[Path, Path] | None:
    """Check for new data in the incoming directory.

    Returns:
        Tuple of (image_dir, mask_dir) if data exists, else None.
    """
    img_dir = INCOMING_DIR / "images"
    mask_dir = INCOMING_DIR / "masks"

    if not img_dir.exists() or not mask_dir.exists():
        return None

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        return None

    logger.info("Found %d incoming images", len(images))
    return img_dir, mask_dir


def merge_data(incoming_img: Path, incoming_mask: Path) -> tuple[Path, Path]:
    """Merge incoming data with existing processed data.

    Copies incoming files into the processed directory and returns
    the merged image/mask directories.
    """
    processed_img = PROCESSED_DIR / "images"
    processed_mask = PROCESSED_DIR / "masks"
    processed_img.mkdir(parents=True, exist_ok=True)
    processed_mask.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in incoming_img.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = incoming_mask / img_path.name
        if not mask_path.exists():
            logger.warning("Skipping %s — no matching mask", img_path.name)
            continue

        shutil.copy2(img_path, processed_img / img_path.name)
        shutil.copy2(mask_path, processed_mask / img_path.name)
        count += 1

    logger.info("Merged %d new pairs into %s", count, PROCESSED_DIR)
    return processed_img, processed_mask


def get_production_metrics(model_name: str) -> dict[str, float]:
    """Load the current production model metrics from the registry YAML."""
    registry_key = MODEL_REGISTRY_MAP.get(model_name, model_name)

    if not REGISTRY_PATH.exists():
        return {}

    with open(REGISTRY_PATH) as f:
        config = yaml.safe_load(f) or {}

    model_def = config.get("models", {}).get(registry_key, {})
    versions = model_def.get("versions", {})
    if not versions:
        return {}

    latest = list(versions.values())[-1]
    return latest.get("metrics", {})


def save_new_version(
    model: torch.nn.Module,
    model_name: str,
    metrics: dict[str, float],
) -> Path:
    """Save a new model version and update the registry YAML."""
    registry_key = MODEL_REGISTRY_MAP.get(model_name, model_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_tag = f"v_{timestamp}"

    model_dir = MODELS_DIR / registry_key
    model_dir.mkdir(parents=True, exist_ok=True)
    weight_path = model_dir / f"lung_seg_{model_name}_{version_tag}.pth"
    torch.save(model.state_dict(), weight_path)
    logger.info("Saved weights: %s", weight_path)

    # Update registry YAML
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {"models": {}, "default_model": "hybrid"}

    if registry_key not in config.get("models", {}):
        config.setdefault("models", {})[registry_key] = {
            "architecture": model_name,
            "versions": {},
        }

    config["models"][registry_key]["versions"][version_tag] = {
        "path": str(weight_path.relative_to(PROJECT_ROOT)),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
    }

    with open(REGISTRY_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("Registry updated with version %s", version_tag)
    return weight_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args = parse_args()

    # Step 1: Check for new data
    logger.info("Step 1: Checking for incoming data...")
    incoming = check_incoming_data()
    if incoming is None:
        logger.info("No new data in %s. Nothing to retrain.", INCOMING_DIR)
        sys.exit(0)

    incoming_img, incoming_mask = incoming

    # Step 2: Merge with existing data
    logger.info("Step 2: Merging data...")
    img_dir, mask_dir = merge_data(incoming_img, incoming_mask)

    # Step 3: Create data loaders
    logger.info("Step 3: Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        image_dir=img_dir,
        mask_dir=mask_dir,
        train_transform=get_train_transforms(args.image_size),
        val_transform=get_val_transforms(args.image_size),
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    # Step 4: Resolve device and create model
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    model = get_model(args.model, pretrained=True)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s (%s params)", args.model, f"{param_count:,}")

    # Step 5: Train
    logger.info("Step 4: Training...")
    config = {
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": 10,
        "experiment_name": f"retrain_{args.model}",
        "checkpoint_dir": f"checkpoints/retrain_{args.model}",
    }

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    trainer.fit()

    # Load best checkpoint for evaluation
    best_ckpt = Path(config["checkpoint_dir"]) / "best.pth"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded best checkpoint from epoch %d", ckpt["epoch"])

    # Step 6: Evaluate on test set
    logger.info("Step 5: Evaluating on test set...")
    new_metrics = evaluate_model(model, test_loader, device=device)

    # Step 7: Compare with production
    logger.info("Step 6: Comparing with production model...")
    old_metrics = get_production_metrics(args.model)
    comparison = compare_models(old_metrics, new_metrics, dice_threshold=DICE_THRESHOLD)

    # Print comparison table
    print(f"\n{'=' * 60}")
    print(f"  Retraining Results: {args.model}")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<15} {'Old':>10} {'New':>10} {'Delta':>10}")
    print(f"  {'-' * 45}")
    for key in ["dice", "iou", "hd95", "sensitivity", "specificity"]:
        old_val = old_metrics.get(key, 0.0)
        new_val = new_metrics.get(key, 0.0)
        delta = comparison["deltas"].get(key, 0.0)
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<15} {old_val:>10.4f} {new_val:>10.4f} {sign}{delta:>9.4f}")

    print(f"\n  {comparison['recommendation']}")

    # Step 8: Promote or keep
    if comparison["improved"]:
        logger.info("Step 7: Promoting new model to staging...")
        weight_path = save_new_version(model, args.model, new_metrics)

        # Log promotion in MLflow
        mlflow.set_experiment(f"retrain_{args.model}")
        with mlflow.start_run(run_name=f"promotion_{args.model}"):
            mlflow.log_metrics(
                {f"new_{k}": v for k, v in new_metrics.items()},
            )
            mlflow.log_metrics(
                {f"old_{k}": v for k, v in old_metrics.items()},
            )
            mlflow.set_tag("stage", "staging")
            mlflow.set_tag("promoted", "true")
            mlflow.log_param("weight_path", str(weight_path))

        print(f"\n  Model promoted to staging: {weight_path}")

        # Clean up incoming data after successful retrain
        shutil.rmtree(INCOMING_DIR)
        logger.info("Cleaned incoming directory")
    else:
        logger.info("Step 7: Keeping current production model.")
        print("\n  Current production model retained.")


if __name__ == "__main__":
    main()
