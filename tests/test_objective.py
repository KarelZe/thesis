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
    TabTransformerObjective,
)
from otc.models.transformer_classifier import TransformerClassifier


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
        end = dt.datetime(2021, 12, 31).replace(tzinfo=dt.timezone.utc)
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
            "depth": 1,
            "learning_rate": 0.1,
            "cat_features": None,
            "l2_leaf_reg": 3,
            "random_strength": 1e-08,
            "bagging_temperature": 0.1,
            "grow_policy": "SymmetricTree",
        }

        study = optuna.create_study(direction="maximize")
        objective = GradientBoostingObjective(
            self._x_train, self._y_train, self._x_val, self._y_val
        )

        study.enqueue_trial(params)
        study.optimize(objective, n_trials=1)

        # check if accuracy is >= 0 and <=1.0
        assert 0.0 <= study.best_value <= 1.0

    def test_gradient_boosting_pretraining_objective(self) -> None:
        """
        Test if gradient boosting objective returns a valid value.

        Pretraining is activated.

        Value obtained is the accuracy. Should lie in [0,1].
        Value may not be NaN.
        """
        params = {
            "depth": 1,
            "learning_rate": 0.1,
            "cat_features": None,
            "l2_leaf_reg": 3,
            "random_strength": 1e-08,
            "bagging_temperature": 0.1,
            "grow_policy": "SymmetricTree",
        }

        # labelled (-1,1) and unlabelled (0) instances
        # train set with -1, 1, and 0
        self._y_train = pd.Series(np.random.randint(-1, 2, self._y_train.shape[0]))
        # val set with 1
        self._y_val = np.random.randint(1, 2, self._y_train.shape[0])

        study = optuna.create_study(direction="maximize")
        objective = GradientBoostingObjective(
            self._x_train, self._y_train, self._x_val, self._y_val, pretrain=True
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
            "batch_size": 16192,
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

        with patch.object(TransformerClassifier, "epochs", 1):
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
            "batch_size": 16192,
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

        with patch.object(TransformerClassifier, "epochs", 1):
            study.enqueue_trial(params)
            study.optimize(objective, n_trials=1)

        # check if accuracy is >= 0 and <=1.0
        assert 0.0 <= study.best_value <= 1.0
