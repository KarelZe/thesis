from typing import List, Optional, Tuple

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
        X: pd.DataFrame,
        y: pd.Series,
        cat_features: Optional[List[str]] = None,
        cat_unique_counts: Optional[Tuple[int]] = None,
        device: str = "cpu",
    ):
        """
        Tabular data set holding data for the model.
        Args:
            X (pd.DataFrame): feature matrix.
            y (pd.Series): target.
            cat_features (Optional[List[str]], optional): List with categorical columns.
            Defaults to None.
            cat_unique_counts (Optional[Tuple[int]], optional): Number of categories per
            categorical feature. Defaults to None.
            device (str, optional): Device for pre-fetching. Defaults to "cpu".
        """
        self._cat_unique_counts = () if not cat_unique_counts else cat_unique_counts

        # calculate cat indices
        features = X.columns.tolist()
        cat_features = [] if not cat_features else cat_features
        self._cat_idx = [features.index(i) for i in cat_features if i in features]

        # calculate cont indices
        cont_features = [x for x in features if x not in cat_features]
        self._cont_idx = [features.index(i) for i in cont_features if i in features]

        assert (
            X.shape[0] == y.shape[0]
        ), "Length of feature matrix must match length of target."
        assert len(cat_features) == len(
            self._cat_unique_counts
        ), "For all categorical features the number of unique entries must be provided."

        # adjust target to be either 0 or 1
        self._y = torch.tensor(y.values).float()
        self._y[self._y < 0] = 0

        # cut into continous and categorical tensor
        self._X_cat = torch.tensor(X.iloc[:, self._cat_idx].values).int()
        self._X_cont = torch.tensor(X.iloc[:, self._cont_idx].values).float()

        # pre-fetch dataset to device
        # https://discuss.pytorch.org/t/how-to-load-all-data-into-gpu-for-training/27609/15
        self._y = self._y.to(device)
        self._X_cat = self._X_cat.to(device)
        self._X_cont = self._X_cont.to(device)

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
