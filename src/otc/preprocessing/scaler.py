"""
PyTorch implementation of z-standardization.

See: https://en.wikipedia.org/w/index.php?title=Feature_scaling
"""

import torch


class TorchStandardScaler:
    """
    Performs z-scaling.

    Fit on training set. Transorm training set, validation set, and test
    set with training set mean and std deviation.
    """

    def __init__(self) -> None:
        """
        z-scaler.

        See: https://en.wikipedia.org/w/index.php?title=Feature_scaling
        """
        self._mean: torch.Tensor
        self._std: torch.Tensor
        self._threshold = 1e-7

    def fit(self, x: torch.Tensor) -> None:
        """
        Calculate mean and std deviation of input tensor.

        Args:
            x (torch.Tensor): input tensor.
        """
        self._mean = x.mean(0, keepdim=True)
        self._std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply z-scaling on input tensor.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: z-standardized tensor.
        """
        x -= self._mean
        # avoid division by zero through small const
        # scikit-learn does it differently by detecting near
        # constant features. See: https://bit.ly/3tYVWnW
        x /= self._std + self._threshold
        return x

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse z-scaling.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: unscaled output tensor.
        """
        x *= self._std + self._threshold
        x += self._mean
        return x
