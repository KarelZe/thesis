"""Learnin rate scheduler with linear warmup phase and cosine decay."""

from typing import List

import numpy as np
from torch import optim


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Cosine learning rate scheduler with linear warmup.

    Args:
    ----
        optim (optim): learning rate scheduler
    """

    def __init__(self, optimizer: optim.Optimizer, warmup: int, max_iters: int):
        """Cosine learning rate scheduler with linear warmup.

        Args:
            optimizer (optim.Optimizer): _description_
            warmup (int): number of warmup iterations
            max_iters (int): maximum number of iterations
        """
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Get the learning rate.

        Returns:
            List[float]: List of learning rates.
        """
        lr_factor = self.get_lr_factor(iteration=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, iteration: int) -> float:
        """Get the learning rate factor for the given epoch.

        Args:
            iteration (int): epoch number

        Returns:
            float: learning rate factor
        """
        lr_factor = 0.5 * (1 + np.cos(np.pi * iteration / self.max_num_iters))
        if iteration <= self.warmup:
            lr_factor *= iteration * 1.0 / self.warmup
        return lr_factor
