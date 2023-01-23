"""
Early stopping of training when the loss does not improve after certain epochs.

Adapted from here: https://bit.ly/3tTnyLU.
"""


import logging
import logging.config
import math

from otc.utils.colors import Colors

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Implementation of early stopping.

    For early stopping see: https://en.wikipedia.org/wiki/Early_stopping.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0) -> None:
        """
        Implement early stopping.

        Args:
            patience (int, optional): number of epochs to wait. Defaults to 5.
            min_delta (float, optional): minimum difference between old and new loss.
            Defaults to 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("nan")
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        """
        Tracks, whether training should be aborted.

        Args:
            val_loss (float): validation loss of current epoch.
        """
        if math.isnan(self.best_loss):
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter, if validation loss improves
            self.counter = 0
        # increase counter, if loss doesn't improve or exploded
        elif self.best_loss - val_loss <= self.min_delta or math.isnan(val_loss):
            self.counter += 1
            logger.info(
                f"{Colors.OKBLUE}[early stopping {self.counter}/"
                f"{self.patience}]{Colors.ENDC}"
                f" Stopping in {self.patience - self.counter} epochs @ {abs(val_loss)}."
            )

            if self.counter >= self.patience:
                logger.info(
                    f"{Colors.OKBLUE}[early stopping {self.counter}/"
                    f"{self.patience}]{Colors.ENDC}"
                    f" Stopped early."
                )
                self.early_stop = True
