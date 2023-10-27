"""Perform automated tests.

Includes tests for data sets with categorical and without
categorical data.
"""


import numpy as np
import pandas as pd
import pytest
import torch

from otc.data.dataset import TabDataset


class TestDataLoader:
    """Perform automated tests.

    Args:
    ----
        unittest (_type_): testcase
    """

    def test_len(self) -> None:
        """Test, if length returned by data set is correct.

        Lenth is simply the number of rows of input data frame.
        """
        length = 10
        x = pd.DataFrame(np.arange(30).reshape(length, 3))
        y = pd.Series(np.arange(length))

        training_data = TabDataset(x=x, y=y)
        assert len(training_data) == length

    def test_invalid_len(self) -> None:
        """Test, if an error is raised if length of x and y do not match.

        If input data is inconsistent, it should not be further processed.
        """
        length = 10
        x = pd.DataFrame(np.arange(30).reshape(length, 3))
        # make y one element longer
        y = pd.Series(np.arange(length + 1))

        with pytest.raises(AssertionError):
            TabDataset(x=x, y=y)

    def test_invalid_weight(self) -> None:
        """Test, if an error is raised if length of weight and y do not match."""
        length = 10
        x = pd.DataFrame(np.arange(30).reshape(length, 3))
        y = pd.Series(np.arange(length))
        # make weight one element longer
        weight = np.ones(length + 1)

        with pytest.raises(AssertionError):
            TabDataset(x=x, y=y, weight=weight)

    def test_invalid_feature_names_len(self) -> None:
        """Test, if fewer feature_names are provided than columns in the np.array, an assertation should be raised."""
        length = 10
        x = np.arange(30).reshape(length, 3)
        y = np.arange(length)

        # data set has 3 columns, but only 2 feature names are provided
        with pytest.raises(AssertionError):
            TabDataset(
                x=x,
                y=y,
                feature_names=["A", "F"],
            )

    def test_invalid_unique_count(self) -> None:
        """Test, if an error is raised if length of 'cat_features' and 'cat_unique_counts' do not match.

        Models like TabTransformer require to know the number of unique values for each
        categorical feature, as it used in the embedding layer.
        """
        length = 10
        x = pd.DataFrame(np.arange(30).reshape(length, 3), columns=["a", "b", "c"])
        y = pd.Series(np.arange(length))

        with pytest.raises(AssertionError):
            TabDataset(
                x=x,
                y=y,
                cat_features=["a", "b"],
                cat_unique_counts=(20,),
            )

    def test_with_cat_features(self) -> None:
        """Test, if data loader can be created with categorical features.

        If the data set contains categorical features, the data set should
        return a tensor for the attribute `_x_cat`.
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
            cat_unique_counts=(100,),
        )

        true_x_cat = torch.tensor([[0], [3], [6]])

        assert true_x_cat.equal(training_data.x_cat)

    def test_no_cat_features(self) -> None:
        """Test, if data loader can be created without categorical features.

        If data set doesn't contain categorical features, the data set should
        assign 'None' to the attribute `_x_cat`.
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
            cat_features=None,
            cat_unique_counts=None,
        )

        assert training_data.x_cat is None

    def test_weight(self) -> None:
        """Test, if weight is correctly assigned to data set."""
        length = 10
        x = pd.DataFrame(np.arange(30).reshape(length, 3))
        y = pd.Series(np.arange(length))
        weight = np.geomspace(0.001, 1, num=len(y))
        data = TabDataset(x=x, y=y, weight=weight)
        assert data.weight.equal(torch.tensor(weight).float())

    def test_no_weight(self) -> None:
        """Test, if no weight is provided, every sample should get weight 1."""
        length = 10
        x = pd.DataFrame(np.arange(30).reshape(length, 3))
        y = pd.Series(np.arange(length))
        data = TabDataset(x=x, y=y)
        assert data.weight.equal(torch.ones(length))

    def test_no_feature_names(self) -> None:
        """Test, if no feature names are provided, feature names are inferred from the `pd.DataFrame`."""
        length = 10
        x = pd.DataFrame(np.arange(30).reshape(length, 3), columns=["A", "B", "C"])
        y = pd.Series(np.arange(length))
        data = TabDataset(x=x, y=y, cat_features=["C"], cat_unique_counts=(1,))
        assert data.x_cont.shape == (length, 2)
        assert data.x_cat.shape == (length, 1)

    def test_feature_names(self) -> None:
        """Test, column selection wiht feature_names for `np.array`s."""
        length = 10
        x = np.arange(30).reshape(length, 3)
        y = np.arange(length)
        data = TabDataset(
            x=x,
            y=y,
            feature_names=["F", "G", "H"],
            cat_features=["H"],
            cat_unique_counts=(1,),
        )
        assert data.x_cont.shape == (length, 2)
        assert data.x_cat.shape == (length, 1)

    def test_empty_feature_names(self) -> None:
        """Test, if assertation is raised if resulting feature_names are empty.

        Might be the case if `X` is a numpy array and `feature_names` is not provided.
        """
        length = 10
        x = np.arange(30).reshape(length, 3)
        y = np.arange(length)
        with pytest.raises(AssertionError):
            TabDataset(
                x=x,
                y=y,
                feature_names=None,
            )

    def test_overlong_cat_features(self) -> None:
        """Test, if assertation is raised if `cat_feature` is provided, that is not in `feature_names`. Might be a typo."""
        length = 10
        x = pd.DataFrame(np.arange(30).reshape(length, 3), columns=["A", "B", "C"])
        y = pd.Series(np.arange(length))
        with pytest.raises(AssertionError):
            TabDataset(
                x=x,
                y=y,
                cat_features=["E"],
                cat_unique_counts=(1,),
            )
