"""Training loop for lung segmentation models."""

import logging
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from src.training.callbacks import EarlyStopping
from src.training.metrics import MetricsTracker, dice_coefficient, iou_score

logger = logging.getLogger(__name__)


class Trainer:
    """Manages the model training lifecycle.

    Handles training loop, validation, checkpointing,
    and integration with MLflow for experiment tracking.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
    ) -> None:
        """Initialize the trainer.

        Args:
            model: The segmentation model to train.
            config: Training configuration dict with keys:
                lr, epochs, patience, checkpoint_dir, experiment_name.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            device: Device string ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 5e-4))
        epochs = config.get("epochs", 50)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        from src.training.losses import CombinedLoss

        self.criterion = CombinedLoss(
            focal_alpha=config.get("focal_alpha", 0.25),
            focal_gamma=config.get("focal_gamma", 2.0),
            boundary_weight=config.get("boundary_weight", 0.3),
        )

        self.early_stopping = EarlyStopping(patience=config.get("patience", 10))

        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_dice = 0.0
        logger.info("Trainer initialized on %s", device)

    def train_epoch(self) -> dict[str, float]:
        """Run a single training epoch.

        Returns:
            Dict with 'loss' key containing the average training loss.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return {"loss": avg_loss}

    def validate_epoch(self) -> dict[str, float]:
        """Run validation and compute metrics.

        Returns:
            Dict with val_loss, val_dice, val_iou.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        tracker = MetricsTracker()

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                n_batches += 1

                # Compute per-sample metrics
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                targets = masks.cpu().numpy()

                for i in range(preds.shape[0]):
                    p = preds[i].squeeze()
                    t = targets[i].squeeze()
                    tracker.update(
                        {
                            "dice": dice_coefficient(p, t),
                            "iou": iou_score(p, t),
                        }
                    )

        avg_loss = total_loss / max(n_batches, 1)
        avg_metrics = tracker.compute()
        return {
            "val_loss": avg_loss,
            "val_dice": avg_metrics.get("dice", 0.0),
            "val_iou": avg_metrics.get("iou", 0.0),
        }

    def fit(self, epochs: int | None = None) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            epochs: Number of epochs. Defaults to config value.

        Returns:
            History dict mapping metric names to lists of per-epoch values.
        """
        epochs = epochs or self.config.get("epochs", 50)
        history: dict[str, list[float]] = {
            "loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "lr": [],
        }

        experiment = self.config.get("experiment_name", "lung_segmentation")
        mlflow.set_experiment(experiment)

        with mlflow.start_run():
            mlflow.log_params(
                {
                    "lr": self.config.get("lr", 5e-4),
                    "epochs": epochs,
                    "patience": self.config.get("patience", 10),
                    "device": self.device,
                }
            )

            for epoch in range(1, epochs + 1):
                train_metrics = self.train_epoch()
                val_metrics = self.validate_epoch()
                self.scheduler.step()

                lr = self.optimizer.param_groups[0]["lr"]

                history["loss"].append(train_metrics["loss"])
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_dice"].append(val_metrics["val_dice"])
                history["val_iou"].append(val_metrics["val_iou"])
                history["lr"].append(lr)

                # MLflow logging
                mlflow.log_metrics(
                    {
                        "train_loss": train_metrics["loss"],
                        "val_loss": val_metrics["val_loss"],
                        "val_dice": val_metrics["val_dice"],
                        "val_iou": val_metrics["val_iou"],
                        "lr": lr,
                    },
                    step=epoch,
                )

                logger.info(
                    "Epoch %d/%d — loss=%.4f val_dice=%.4f val_iou=%.4f lr=%.6f",
                    epoch,
                    epochs,
                    train_metrics["loss"],
                    val_metrics["val_dice"],
                    val_metrics["val_iou"],
                    lr,
                )

                # Checkpoint best model
                if val_metrics["val_dice"] > self.best_dice:
                    self.best_dice = val_metrics["val_dice"]
                    self._save_checkpoint("best.pth", epoch, val_metrics)

                # Always save latest
                self._save_checkpoint("latest.pth", epoch, val_metrics)

                # Early stopping
                if self.early_stopping(val_metrics["val_dice"]):
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        return history

    def _save_checkpoint(self, filename: str, epoch: int, metrics: dict[str, float]) -> None:
        """Save a training checkpoint.

        Args:
            filename: Checkpoint file name.
            epoch: Current epoch number.
            metrics: Current validation metrics.
        """
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics": metrics,
                "best_dice": self.best_dice,
            },
            path,
        )
        logger.info("Checkpoint saved: %s (epoch %d)", path, epoch)
