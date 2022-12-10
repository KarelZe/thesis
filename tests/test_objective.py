"""
Tests for Objectives.
"""
import datetime as dt
import os
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd

from otc.models.objective import (
    ClassicalObjective,
    FTTransformerObjective,
    GradientBoostingObjective,
    TabNetObjective,
    TabTransformerObjective,
)


class TestObjectives:
    """
    Perform automated tests for objectives.

    Args:
        metaclass (_type_, optional): parent. Defaults to abc.ABCMeta.
    """

    def setup(self) -> None:
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

    def test_classical_objective(self) -> None:
        """
        Test if classical objective returns a valid value.

        Value obtained is the accuracy. Should lie in [0,1].
        Value may not be NaN.
        """
        params = {
            "layer_1": "nan_ex",
            "layer_2": "nan_ex",
            "layer_3": "nan_ex",
            "layer_4": "nan_ex",
            "layer_5": "nan_ex",
        }

        study = optuna.create_study(direction="maximize")
        objective = ClassicalObjective(
            self._x_train, self._y_train, self._x_val, self._y_val
        )

        study.enqueue_trial(params)
        study.optimize(objective, n_trials=1)

        # check if accuracy is >= 0 and <=1.0
        assert 0.0 <= study.best_value <= 1.0

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
        assert 0.0 <= study.best_value <= 1.0

    def test_fttransformer_objective(self) -> None:
        """
        Test if FTTransformer objective returns a valid value.

        Value obtained is the accuracy. Should lie in [0,1].
        Value may not be NaN.
        """
        params = {
            "n_blocks": 1,
            "d_token": 96,
            "attention_dropout": 0.1,
            "ffn_dropout": 0.1,
            "weight_decay": 0.01,
            "learning_rate": 3e-4,
            "batch_size": 8192,
        }

        study = optuna.create_study(direction="maximize")
        objective = FTTransformerObjective(
            self._x_train,
            self._y_train,
            self._x_val,
            self._y_val,
            cat_features=[],
            cat_cardinalities=[],
        )

        with patch.object(objective, "epochs", 1):
            study.enqueue_trial(params)
            study.optimize(objective, n_trials=1)

        # check if accuracy is >= 0 and <=1.0
        assert 0.0 <= study.best_value <= 1.0

    def test_tabtransformer_objective(self) -> None:
        """
        Test if TabTransformer objective returns a valid value.

        Value obtained is the accuracy. Should lie in [0,1].
        Value may not be NaN.
        """
        params = {
            "dim": 32,
            "depth": 1,
            "heads": 8,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "learning_rate": 3e-4,
            "batch_size": 8192,
        }

        study = optuna.create_study(direction="maximize")
        objective = TabTransformerObjective(
            self._x_train,
            self._y_train,
            self._x_val,
            self._y_val,
            cat_features=[],
            cat_cardinalities=[],
        )

        with patch.object(objective, "epochs", 1):
            study.enqueue_trial(params)
            study.optimize(objective, n_trials=1)

        # check if accuracy is >= 0 and <=1.0
        assert 0.0 <= study.best_value <= 1.0

    def test_tabnet_objective(self) -> None:
        """
        Test if TabNet objective returns a valid value.

        Value obtained is the accuracy. Should lie in [0,1].
        Value may not be NaN.
        """
        params = {
            "n_d": 8,
            "n_steps": 3,
            "gamma": 1,
            "cat_emb_dim": 0.01,
            "n_independent": 3e-4,
            "n_shared": 1,
            "momentum": 0.001,
            "mask_type": "entmax",
            "weight_decay": 0.01,
            "learning_rate": 3e-4,
            "batch_size": 8192,
        }

        study = optuna.create_study(direction="maximize")
        objective = TabNetObjective(
            self._x_train,
            self._y_train,
            self._x_val,
            self._y_val,
            cat_features=[],
            cat_cardinalities=[],
        )

        with patch.object(objective, "epochs", 1):
            study.enqueue_trial(params)
            study.optimize(objective, n_trials=1)

        # check if accuracy is >= 0 and <=1.0
        assert 0.0 <= study.best_value <= 1.0
