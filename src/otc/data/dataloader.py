"""
A fast dataloader-like object to load batches of tabular data sets.

Adapted from here:
https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
"""
from __future__ import annotations

from typing import Any

import torch


class TabDataLoader:
    """
    PyTorch Implementation of a dataloader for tabular data.

    Due to a chunk-wise reading or several rows at once it is preferred
    over the standard dataloader that reads row-wise.
    """

    def __init__(
        self,
        *tensors: torch.Tensor | None,
        batch_size: int = 32,
        shuffle: bool = False,
        device: str = "cpu",
        **kwargs: Any,
    ):
        """
        TabDataLoader.

        Args:
            batch_size (int, optional): size of batch. Defaults to 32.
            shuffle (bool, optional): shuffle data. Defaults to False.
        """
        self.device = device
        # check if any tensor is None e. g., when no categorical features are present.
        self.has_none_tensor = None in tensors
        # filter if for not none tensors
        self.tensors = tuple(t for t in tensors if t is not None)

        # check if all tensors have same length
        assert all(t.shape[0] == self.tensors[0].shape[0] for t in self.tensors)

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    # TODO: improve type hint with Self (?)
    def __iter__(self) -> TabDataLoader:
        """
        Return itself.

        Returns:
            TabDataLoader: TabDataLoader
        """
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = tuple(t[r] for t in self.tensors)
        self.i = 0
        return self

    def __next__(self) -> tuple[torch.Tensor | None, ...]:
        """
        Generate next batch with size of 'batch_size'.

        Batches can be underful.
        Raises:
            StopIteration: stopping criterion.
        Returns:
            Tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]: (X_cat), X_cont, y
        """
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size

        # move tensors device
        batch = tuple(t.to(self.device) for t in batch)

        # return none + batch if input tensors included non  e.g., when no
        # categorical features are present
        return batch if self.has_none_tensor else (None,) + batch

    def __len__(self) -> int:
        """
        Get number of full and partial batches in data set.

        Returns:
            int: number of batches.
        """
        return self.n_batches
