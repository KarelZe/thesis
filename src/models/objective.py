"""
Provides objectives for optimizations.

Adds support for classical rules, GBTs and transformer-based architectures.
"""
from abc import ABC, abstractmethod

import os
import random

from catboost import CatBoostClassifier
import optuna
from optuna.integration import CatBoostPruningCallback
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


from src.models.classical_classifier import ClassicalClassifier

from typing import List, Optional


def set_seed(seed_val: int = 42) -> int:
    """
    Seeds basic parameters for reproducibility of results

    Args:
        seed_val (int, optional): random seed used in rngs. Defaults to 42.

    Returns:
        int: seed
    """
    os.environ["PYTHONHASHSEED"] = str(seed_val)
    random.seed(seed_val)
    # pandas and numpy as discussed here: https://stackoverflow.com/a/52375474/5755604
    np.random.seed(seed_val)
    return seed_val


class Objective(ABC):
    """
    Generic implementation of objective.

    Args:
        ABC (abstract): abstract class
    """

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        name: str = "default",
    ):
        """
        Initialize objective.

        Args:
            x_train (pd.DataFrame): feature matrix (train)
            y_train (pd.Series): ground truth (train)
            x_val (pd.DataFrame): feature matrix (val)
            y_val (pd.Series): ground truth (val)
            name (str, optional): Name of objective. Defaults to "default".
        """

        self.x_train, self.y_train, self.x_val, self.y_val, = (
            x_train,
            y_train,
            x_val,
            y_val,
        )
        self.name = name
        self._clf = None

    @abstractmethod
    def save_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Callback to save models.

        Args:
            study (optuna.Study): current study.
            trial (optuna.Trial): current trial.
        """


class ClassicalObjective(Objective):
    """
    Implements an optuna optimization objective.

    See here: https://optuna.readthedocs.io/en/stable/
    Args:
        Objective (Objective): objective
    """
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Perform a new search trial in Bayesian search.
        Hyperarameters are suggested, unless they are fixed.


        Args:
            trial (optuna.Trial): current trial.
        Returns:
            float: accuracy of trial on validation set.
        """
        # see https://github.com/optuna/optuna/issues/3093#issuecomment-968075749
        values = [
            ("tick", "all"),
            ("tick", "ex"),
            ("rev_tick", "all"),
            ("rev_tick", "ex"),
            ("quote", "best"),
            ("quote", "ex"),
            ("lr", "best"),
            ("lr", "ex"),
            ("rev_lr", "best"),
            ("rev_lr", "ex"),
            ("emo", "best"),
            ("emo", "ex"),
            ("rev_emo", "best"),
            ("rev_emo", "ex"),
            ("trade_size", "ex"),
            ("depth", "ex"),
            ("nan", "ex"),
        ]

        indices = [
            "tick_all",
            "tick_ex",
            "rev_tick_all",
            "rev_tick_ex",
            "quote_best",
            "quote_ex",
            "lr_best",
            "lr_ex",
            "rev_lr_best",
            "rev_lr_ex",
            "emo_best",
            "emo_ex",
            "rev_emo_best",
            "rev_emo_ex",
            "trade_size_ex",
            "depth_ex",
            "nan_ex",
        ]
        mapping = dict(zip(indices, values))

        index_layer_1 = trial.suggest_categorical("layer_1", indices)
        layer_1 = mapping[index_layer_1]
        index_layer_2 = trial.suggest_categorical("layer_2", indices)
        layer_2 = mapping[index_layer_2]
        index_layer_3 = trial.suggest_categorical("layer_3", indices)
        layer_3 = mapping[index_layer_3]
        index_layer_4 = trial.suggest_categorical("layer_4", indices)
        layer_4 = mapping[index_layer_4]

        layers = [layer_1, layer_2, layer_3, layer_4]
        self._clf = ClassicalClassifier(
            layers=layers,
            random_state=set_seed(),
        )
        self._clf.fit(X=self.x_train, y=self.y_train)
        pred = self._clf.predict(self.x_val)
        return accuracy_score(self.y_val, pred)

    def save_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Callback to save models.

        Args:
            study (optuna.Study): current study.
            trial (optuna.Trial): current trial.
        """


class GradientBoostingObjective(Objective):
    """
    Implements an optuna optimization objective.

    See here: https://optuna.readthedocs.io/en/stable/
    Args:
        Objective (Objective): objective
    """

    def __call__(
        self,
        trial: optuna.Trial,
        features: List[str],
        cat_features: Optional[List[str]] = None,
    ) -> float:
        ignored_features = [
            x for x in self.x_train.columns.tolist() if x not in features
        ]

        iterations = trial.suggest_int("iterations", 100, 1500)
        learning_rate = trial.suggest_float("learning_rate", 0.005, 1, log=True)
        depth = trial.suggest_int("depth", 1, 8)
        grow_policy = trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Depthwise"]
        )
        params = {
            "iterations": iterations,
            "depth": depth,
            "grow_policy": grow_policy,
            "learning_rate": learning_rate,
            "od_type": "Iter",
            "logging_level": "Silent",
            "task_type": "GPU",
            "cat_features": cat_features,
            "ignored_features": ignored_features,
            "random_seed": set_seed(),
        }

        pruning_callback = CatBoostPruningCallback(trial, "Accuracy")

        self._clf = CatBoostClassifier(**params)
        self._clf.fit(
            self.x_train,
            self.y_train,
            callbacks=[pruning_callback],
        )

        pruning_callback.check_pruned()

        pred = self._clf.predict(self.x_val, prediction_type="Class")
        return accuracy_score(self.y_val, pred)

    def save_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Callback to save models.

        Args:
            study (optuna.Study): current study.
            trial (optuna.Trial): current trial.
        """
        # FIXME: delete all others from same study first.
        # FIXME: Save in wandb
        if study.best_trial == trial:
            self._clf.save_model(
                f"gcs://models/{study.study_name}_{self._clf.__class__.__name__}"
                f"_{self.name}_trial_{trial.number}.cbm",
                format="cbm",
            )
            # model_artifact = wandb.Artifact('cnn', type='model')
            # model_artifact.add_reference('s3://my-bucket/models/cnn/')
            # run.log_artifact(model_artifact)