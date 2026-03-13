"""Training callbacks for early stopping and checkpointing."""

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to halt training when validation metric stops improving.

    Default patience is 10 epochs, matching thesis training configuration.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current validation metric (higher is better).

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                "EarlyStopping: no improvement for %d/%d epochs", self.counter, self.patience
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("EarlyStopping triggered after %d epochs", self.patience)

        return self.should_stop
