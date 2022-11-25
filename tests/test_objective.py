"""
Tests for Objectives.

TODO: Update doc strings
"""
import datetime as dt
import os
import unittest

import numpy as np
import optuna
import pandas as pd

from otc.models.objective import GradientBoostingObjective


class TestObjectives(unittest.TestCase):
    """
    Perform automated tests for objectives.

    Args:
        metaclass (_type_, optional): parent. Defaults to abc.ABCMeta.
    """

    def setUp(self) -> None:
        """
        Set up basic data.

        Construct feature matrix and target.
        """
        self._old_cwd = os.getcwd()
        start = dt.datetime(2020, 1, 1).replace(tzinfo=dt.timezone.utc)
        end = dt.datetime(2020, 12, 31).replace(tzinfo=dt.timezone.utc)
        index = pd.date_range(start=start, end=end, freq="15min")

        # make 1 const feature and 1 non-const feature, as catboost requires non-const
        self._x_train = pd.DataFrame(data={"feature_1": 1}, index=index)
        self._x_train["feature_2"] = np.random.randint(1, 6, self._x_train.shape[0])
        self._y_train = self._x_train["feature_2"]
        self._x_val = self._x_train.copy()
        self._y_val = self._y_train.copy()

    def test_gradient_boosting_objective(self) -> None:
        """
        Test if gradient boosting objective returns a valid value.

        Value obtained is the accuracy. Should lie in [0,1].
        Value may not be NaN.
        """
        params = {
            "iterations": 1,
            "depth": 1,
            "grow_policy": "SymmetricTree",
            "learning_rate": 0.05,
            "cat_features": None,
        }

        study = optuna.create_study(direction="maximize")
        objective = GradientBoostingObjective(
            self._x_train, self._y_train, self._x_val, self._y_val
        )

        study.enqueue_trial(params)
        study.optimize(objective, n_trials=1)

        # check if accuracy is >= 0 and <=1.0
        self.assertTrue(0.0 <= study.best_value <= 1.0)
