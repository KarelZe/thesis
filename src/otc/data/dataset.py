"""
Implementation of a dataset for tabular data.

Supports both categorical and continous data.
"""

from __future__ import annotations

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
        x: pd.DataFrame,
        y: pd.Series,
        cat_features: list[str] | None = None,
        cat_unique_counts: tuple[int, ...] | None = None,
    ):
        """
        Tabular data set holding data for the model.

        Args:
            x (pd.DataFrame): feature matrix.
            y (pd.Series): target.
            cat_features (Optional[List[str]], optional): List with categorical columns.
            Defaults to None.
            cat_unique_counts (Optional[Tuple[int]], optional): Number of categories per
            categorical feature. Defaults to None.
        """
        self._cat_unique_counts = () if not cat_unique_counts else cat_unique_counts

        # calculate cat indices
        features = x.columns.tolist()
        cat_features = [] if not cat_features else cat_features
        self._cat_idx = [features.index(i) for i in cat_features if i in features]

        # calculate cont indices
        cont_features = [f for f in features if f not in cat_features]
        self._cont_idx = [features.index(i) for i in cont_features if i in features]

        assert (
            x.shape[0] == y.shape[0]
        ), "Length of feature matrix must match length of target."
        assert len(cat_features) == len(
            self._cat_unique_counts
        ), "For all categorical features the number of unique entries must be provided."

        # adjust target to be either 0 or 1
        self._y = torch.tensor(y.values).float()
        self._y[self._y < 0] = 0

        # cut into continous and categorical tensor
        self._x_cat: torch.Tensor | None = None
        if len(self._cat_idx) > 0:
            self._x_cat = torch.tensor(x.iloc[:, self._cat_idx].values).int()
        self._x_cont = torch.tensor(x.iloc[:, self._cont_idx].values).float()

    def __len__(self) -> int:
        """
        Length of dataset.

        Returns:
            int: length
        """
        return len(self._x_cont)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """
        Get sample for model.

        Args:
            idx (int): _description_

        Returns:
            Tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
            x_cat (if present if present otherwise None), x_cont and y.
        """
        return (
            self._x_cat[idx] if self._x_cat else None,
            self._x_cont[idx],
            self._y[idx],
        )
