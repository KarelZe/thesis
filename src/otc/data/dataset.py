"""
Loads data for neural networks.

Slices features and target dataFrame into sequences.
"""


from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """PyTorch Dataset for fitting timeseries models.

    Args:
        Dataset (Dataset): dataset
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        mean: pd.Series,
        std: pd.Series,
        cat_features: Optional[List[str]] = None,
        cat_unique_counts: Optional[List[int]] = None,
        threshold: float = 1e-7,
    ):
        """
        Tabular data set holding data for the model.

        Args:
            X (pd.DataFrame): feature matrix.
            y (pd.Series): target.
            mean (pd.Series): column-wise means of continous features on training set.
            std (pd.Series): column-wise means of continous features on training set.
            cat_features (Optional[List[str]], optional): List with categorical columns.
            Defaults to None.
            cat_unique_counts (Optional[List[int]], optional): Number of categories per
            categorical feature. Defaults to None.
            threshold (float, optional): threshold for z-standardization.
            Defaults to 1e-7.
        """
        self._cat_unique_counts: Union[
            Optional[List[int]], Tuple[()]
        ] = cat_unique_counts

        # calculate cat indices
        features = X.columns.tolist()
        cat_features = [] if not cat_features else cat_features
        self._cat_idx = [features.index(i) for i in cat_features if i in features]

        # calculate cont indices
        cont_features = [x for x in features if x not in cat_features]
        self._cont_idx = [features.index(i) for i in cont_features if i in features]

        if not self._cat_unique_counts:
            self._cat_unique_counts = ()

        assert (
            X.shape[0] == y.shape[0]
        ), "Length of feature matrix must match length of target."
        assert len(cont_features) == len(
            mean
        ), "Number of continous must match length of means."
        assert len(cont_features) == len(
            mean
        ), "Number of continous features must match length of stds."
        assert len(cat_features) == len(
            self._cat_unique_counts
        ), "For all categorical features the number of unique entries must be provided."

        self._mean = torch.tensor(mean.values).float()
        self._std = torch.tensor(std.values).float()

        # adjust target to be either 0 or 1
        self._y = torch.tensor(y.values).float()
        self._y[self._y < 0] = 0

        # cut into continous and categorical tensor
        self._X_cat = torch.tensor(X.loc[:, self._cat_idx].values)
        self._X_cont = torch.tensor(X.loc[:, self._cont_idx].values)

        # z-standardize continous
        self._X_cont -= self._mean
        # avoid division by zero through small const.
        # scikit-learn does it differently. See https://bit.ly/3tYVWnW.
        self._X_cont /= self._std + threshold

    def __len__(self) -> int:
        """
        Length of dataset.

        Returns:
            int: length
        """
        return len(self._X_cont)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get sample for model.

        Args:
            idx (int): index of prediction (between ``0`` and ``len(dataset) - 1``)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: X_cat, X_cont and y.
        """
        return self._X_cat[idx], self._X_cont[idx], self._y[idx]
