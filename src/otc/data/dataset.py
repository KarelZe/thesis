"""
Implementation of a dataset for tabular data.

Supports both categorical and continous data.
"""

from __future__ import annotations

import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import Dataset


class TabDataset(Dataset):
    """PyTorch Dataset for tabular data.

    Args:
        Dataset (Dataset): dataset
    """

    def __init__(
        self,
        x: pd.DataFrame | npt.ndarray,
        y: pd.Series | npt.ndarray,
        weight: pd.Series | npt.ndarray | None = None,
        feature_names: list[str] | None = None,
        cat_features: list[str] | None = None,
        cat_unique_counts: tuple[int, ...] | None = None,
    ):
        """
        Tabular data set holding data for the model.

        Args:
            x (pd.DataFrame | npt.ndarray): feature matrix
            y (pd.Series | npt.ndarray): target
            weight (pd.Series | npt.ndarray | None, optional): weights of samples. If
            not provided all samples are given a weight of 1. Defaults to None.
            feature_names (list[str] | None, optional): name of features. Defaults to
            None.
            cat_features (list[str] | None, optional): List with categorical columns.
            Defaults to None.
            cat_unique_counts (tuple[int, ...] | None, optional): Number of categories
            per categorical feature. Defaults to None.
        """
        self._cat_unique_counts = () if not cat_unique_counts else cat_unique_counts

        # calculate cat indices
        feature_names = [] if not feature_names else feature_names
        cat_features = [] if not cat_features else cat_features
        self._cat_idx = [
            feature_names.index(i) for i in cat_features if i in feature_names
        ]

        # calculate cont indices
        cont_features = [f for f in feature_names if f not in cat_features]
        self._cont_idx = [
            feature_names.index(i) for i in cont_features if i in feature_names
        ]

        # pd 2 np
        x = x.values if isinstance(x, pd.DataFrame) else x
        y = y.values if isinstance(y, pd.Series) else y
        weight = weight.values if isinstance(weight, pd.Series) else weight

        assert (
            x.shape[0] == y.shape[0]
        ), "Length of feature matrix must match length of target."
        assert len(cat_features) == len(
            self._cat_unique_counts
        ), "For all categorical features the number of unique entries must be provided."

        # adjust target to be either 0 or 1
        self.y = torch.tensor(y).float()
        self.y[self.y < 0] = 0

        # cut into continous and categorical tensor
        self.x_cat: torch.Tensor | None = None
        if len(self._cat_idx) > 0:
            self.x_cat = torch.tensor(x[:, self._cat_idx]).int()
        self.x_cont = torch.tensor(x[:, self._cont_idx]).float()

        # set weights
        weight = (
            torch.tensor(weight).float()
            if weight is not None
            else torch.ones(len(self.y)).float()
        )
        assert (
            y.shape[0] == weight.shape[0]
        ), "Length of traget must match length of weight."
        self.weight = weight

    def __len__(self) -> int:
        """
        Length of dataset.

        Returns:
            int: length
        """
        return len(self.x_cont)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get sample for model.

        Args:
            idx (int): index of item.

        Returns:
            Tuple[torch.Tensor | None, torch.Tensor, torch.Tensor torch.Tensor]:
            x_cat (if present if present otherwise None), x_cont, weight and y.
        """
        return (
            self.x_cat[idx] if self.x_cat else None,
            self.x_cont[idx],
            self.weight[idx],
            self.y[idx],
        )
