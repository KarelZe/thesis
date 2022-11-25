"""
A fast dataloader-like object to load batches of tabular data sets.

Adapted from here:
https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
"""


from typing import Any, Tuple

import torch


class TabDataLoader:
    """
    PyTorch Implementation of a dataloader for tabular data.

    Due to a chunk-wise reading or several rows at once it is preferred
    over the standard dataloader that reads row-wise.
    """

    def __init__(
        self,
        *tensors: torch.Tensor,
        batch_size: int = 32,
        shuffle: bool = False,
        **kwargs: Any
    ):
        """
        TabDataLoader.

        Args:
            batch_size (int, optional): size of batch. Defaults to 32.
            shuffle (bool, optional): shuffle data. Defaults to False.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.i = 0

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    # TODO: improve type hint with Self (?)
    def __iter__(self) -> "TabDataLoader":
        """
        Return itself.

        Returns:
            TabDataLoader: TabDataLoader
        """
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = tuple(t[r] for t in self.tensors)
        return self

    def __next__(self) -> Tuple[torch.Tensor, ...]:
        """
        Generate next batch with size of 'batch_size'.

        Batches can be underful.

        Raises:
            StopIteration: stopping criterion.

        Returns:
            Tuple[torch.Tensor, ...]: batch with tensors.
        """
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self) -> int:
        """
        Get number of full and partial batches in data set.

        Returns:
            int: number of batches.
        """
        return self.n_batches
