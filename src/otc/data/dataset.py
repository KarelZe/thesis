"""Implementation of a dataset for tabular data.

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
    ----
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
        """Tabular data set holding data for the model.

        Data set is inspired by CatBoost's Pool class:
        https://catboost.ai/en/docs/concepts/python-reference_pool

        Args:
            x (pd.DataFrame | npt.ndarray): feature matrix
            y (pd.Series | npt.ndarray): target
            weight (pd.Series | npt.ndarray | None, optional): weights of samples. If
            not provided all samples are given a weight of 1. Defaults to None.
            feature_names (list[str] | None, optional): list with name of features and
            length of `X.shape[1]`. Needed for npt.ndarrays. Optional for pd.DataFrame.
            If no feature names are provided for pd.DataFrames, names are taken from
            `X.columns`. Defaults to None.
            cat_features (list[str] | None, optional): List with categorical columns.
            Defaults to None.
            cat_unique_counts (tuple[int, ...] | None, optional): Number of categories
            per categorical feature. Defaults to None.
        """
        self._cat_unique_counts = cat_unique_counts if cat_unique_counts else ()
        feature_names = [] if feature_names is None else feature_names
        # infer feature names from dataframe.
        if isinstance(x, pd.DataFrame):
            feature_names = x.columns.tolist()
        assert len(feature_names) == x.shape[1], (
            "`len('feature_names)` must match `X.shape[1]`"
        )

        # calculate cat indices
        cat_features = cat_features if cat_features else []
        assert set(cat_features).issubset(feature_names), (
            "Categorical features must be a subset of feature names."
        )

        self._cat_idx = [
            feature_names.index(i) for i in cat_features if i in feature_names
        ]

        # calculate cont indices
        cont_features = [f for f in feature_names if f not in cat_features]
        self._cont_idx = [
            feature_names.index(i) for i in cont_features if i in feature_names
        ]

        # pd 2 np
        x = x.to_numpy() if isinstance(x, pd.DataFrame) else x
        y = y.to_numpy() if isinstance(y, pd.Series) else y
        weight = weight.to_numpy() if isinstance(weight, pd.Series) else weight

        assert x.shape[0] == y.shape[0], (
            "Length of feature matrix must match length of target."
        )
        assert len(cat_features) == len(self._cat_unique_counts), (
            "For all categorical features the number of unique entries must be provided."
        )

        # adjust target to be either 0 or 1
        self.y = torch.tensor(y).float()
        self.y[self.y < 0] = 0

        # cut into continous and categorical tensor
        self.x_cat: torch.Tensor | None = None
        if len(self._cat_idx) > 0:
            self.x_cat = torch.tensor(x[:, self._cat_idx]).int()
        self.x_cont = torch.tensor(x[:, self._cont_idx]).float()

        # remove extrem outliers / cause for exploding gradients
        min = self.x_cont.quantile(q=0.025, dim=1, keepdim=True)
        max = self.x_cont.quantile(q=0.975, dim=1, keepdim=True)
        self.x_cont = self.x_cont.clamp(min=min, max=max)

        # set weights, no gradient calculation
        weight = (
            torch.tensor(weight, requires_grad=False).float()
            if weight is not None
            else torch.ones(len(self.y), requires_grad=False).float()
        )
        assert y.shape[0] == weight.shape[0], (
            "Length of label must match length of weight."
        )
        self.weight = weight

    def __len__(self) -> int:
        """Length of dataset.

        Returns:
            int: length
        """
        return len(self.x_cont)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get sample for model.

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
