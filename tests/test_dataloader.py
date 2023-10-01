"""Perform automated tests.

Includes tests for data sets with categorical and without
categorical data.
"""

import numpy as np
import pandas as pd
import torch

from otc.data.dataloader import TabDataLoader
from otc.data.dataset import TabDataset


class TestDataLoader:
    """Perform automated tests.

    Args:
    ----
        unittest (_type_): testcase
    """

    def test_len(self) -> None:
        """Test, if length returned by data loader is correct.

        Lenth is simply the number of partial or full batches.
        """
        length = 10
        batch_size = 3
        x = pd.DataFrame(
            np.arange(length * 3).reshape(length, 3), columns=["a", "b", "c"]
        )
        y = pd.Series(np.arange(length))

        training_data = TabDataset(
            x=x,
            y=y,
            feature_names=["a", "b", "c"],
            cat_features=None,
            cat_unique_counts=None,
        )
        data_loader = TabDataLoader(
            training_data.x_cat,
            training_data.x_cont,
            training_data.y,
            batch_size=batch_size,
            shuffle=False,
        )

        # https://stackoverflow.com/a/23590097/5755604
        assert len(data_loader) == length // batch_size + (length % batch_size > 0)

    def test_with_cat_features(self) -> None:
        """Test, if data loader can be created with categorical features.

        If the data set contains categorical features, the data loader
        should return a tensor with the categorical features.
        """
        length = 3
        x = pd.DataFrame(
            np.arange(length * 3).reshape(length, 3), columns=["a", "b", "c"]
        )
        y = pd.Series(np.arange(length))

        # column a is categorical
        training_data = TabDataset(
            x=x,
            y=y,
            cat_features=["a"],
            feature_names=["a", "b", "c"],
            cat_unique_counts=(100,),
        )
        train_loader = TabDataLoader(
            training_data.x_cat,
            training_data.x_cont,
            training_data.y,
            batch_size=2,
            shuffle=False,
        )
        cat_features, _, _ = next(iter(train_loader))

        assert torch.tensor([[0], [3]]).equal(cat_features)  # type: ignore

    def test_no_cat_features(self) -> None:
        """Test, if data loader can be created without categorical features.

        If data set doesn't contain categorical features, the data loader
        should return None.
        """
        length = 3
        x = pd.DataFrame(
            np.arange(length * 3).reshape(length, 3), columns=["a", "b", "c"]
        )
        y = pd.Series(np.arange(length))

        # no categorical features
        training_data = TabDataset(
            x=x,
            y=y,
            feature_names=["a", "b", "c"],
            cat_features=None,
            cat_unique_counts=None,
        )
        train_loader = TabDataLoader(
            training_data.x_cat,
            training_data.x_cont,
            training_data.y,
            batch_size=2,
            shuffle=False,
        )
        cat_features, _, _ = next(iter(train_loader))

        assert cat_features is None
