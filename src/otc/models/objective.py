"""
Provides objectives for optimizations.

Adds support for classical rules, GBTs and transformer-based architectures.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from catboost.utils import get_gpu_device_count
from sklearn.base import BaseEstimator
from torch import nn

from otc.models.activation import ReGLU
from otc.models.callback import CallbackContainer, PrintCallback, SaveCallback
from otc.models.classical_classifier import ClassicalClassifier
from otc.models.fttransformer import FeatureTokenizer, FTTransformer, Transformer
from otc.models.tabtransformer import TabTransformer
from otc.models.transformer_classifier import TransformerClassifier


def set_seed(seed_val: int = 42) -> int:
    """
    Seeds basic parameters for reproducibility of results.

    Args:
        seed_val (int, optional): random seed used in rngs. Defaults to 42.

    Returns:
        int: seed
    """
    # python
    # see https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
    os.environ["PYTHONHASHSEED"] = str(seed_val)

    # pandas and numpy
    #  https://stackoverflow.com/a/52375474/5755604
    np.random.seed(seed_val)

    # python random module
    random.seed(seed_val)

    # torch
    # see https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed_val


class Objective:
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
        **kwargs: Any,
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
        self._clf: BaseEstimator | CatBoostClassifier
        self._callbacks: CallbackContainer

    def objective_callback(
        self, study: optuna.Study, trial: optuna.trial.Trial | optuna.trial.FrozenTrial
    ) -> None:
        """
        Perform operations at the end of trail.

        Args:
            study (optuna.Study): current study.
            trial (optuna.trial.Trial | optuna.trial.FrozenTrial): current trial.
        """
        self._callbacks.on_train_end(study, trial, self._clf, self.name)


class TabTransformerObjective(Objective):
    """
    Implements an optuna optimization objective.

    See here: https://optuna.readthedocs.io/en/stable/
    Args:
        Objective (Objective): objective
    """

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        cat_features: list[str] | None,
        cat_cardinalities: list[int] | None,
        name: str = "default",
    ):
        """
        Initialize objective.

        Args:
            x_train (pd.DataFrame): feature matrix (train)
            y_train (pd.Series): ground truth (train)
            x_val (pd.DataFrame): feature matrix (val)
            y_val (pd.Series): ground truth (val)
            cat_features (List[str] | None, optional): List of
            categorical features. Defaults to None.
            cat_cardinalities (List[int] | None, optional): Unique counts
            of categorical features.
            Defaults to None.
            name (str, optional): Name of objective. Defaults to "default".
        """
        self._cat_features = [] if not cat_features else cat_features
        self._cat_cardinalities = (
            () if not cat_cardinalities else tuple(cat_cardinalities)
        )
        self._cont_features: list[int] = [
            x for x in x_train.columns.tolist() if x not in self._cat_features
        ]

        self._callbacks = CallbackContainer([SaveCallback(), PrintCallback()])
        self._clf: BaseEstimator

        super().__init__(x_train, y_train, x_val, y_val, name)

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Perform a new search trial in Bayesian search.

        Hyperarameters are suggested, unless they are fixed.
        Args:
            trial (optuna.Trial): current trial.
        Returns:
            float: accuracy of trial on validation set.
        """
        # searchable params
        dim: int = trial.suggest_categorical("dim", [32, 64, 128, 256])  # type: ignore

        # done similar to borisov
        depth: int = trial.suggest_categorical("depth", [1, 2, 3, 6, 12])  # type: ignore # noqa: E501
        heads: int = trial.suggest_categorical("heads", [2, 4, 8])  # type: ignore
        weight_decay: float = trial.suggest_float("weight_decay", 1e-6, 1e-1)
        dropout = trial.suggest_float("dropout", 0, 0.5, step=0.1)
        lr = trial.suggest_float("lr", 1e-6, 4e-3, log=False)
        batch_size: int = trial.suggest_categorical("batch_size", [4096, 8192, 16192])  # type: ignore # noqa: E501

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        no_devices = torch.cuda.device_count()

        torch.cuda.empty_cache()

        dl_params: dict[str, Any] = {
            "batch_size": batch_size
            * max(no_devices, 1),  # dataprallel splits batches across devices
            "shuffle": False,
            "device": device,
        }

        module_params = {
            "depth": depth,
            "heads": heads,
            "dim": dim,
            "dim_out": 1,
            "mlp_act": nn.ReLU,
            "mlp_hidden_mults": (4, 2),
            "attn_dropout": dropout,
            "ff_dropout": dropout,
            "cat_features": self._cat_features,
            "cat_cardinalities": self._cat_cardinalities,
            "num_continuous": len(self._cont_features),
        }

        optim_params = {"lr": lr, "weight_decay": weight_decay}

        self._clf = TransformerClassifier(
            module=TabTransformer,  # type: ignore
            module_params=module_params,
            optim_params=optim_params,
            dl_params=dl_params,
            callbacks=self._callbacks,
        )

        self._clf.fit(
            self.x_train,
            self.y_train,
            eval_set=(self.x_val, self.y_val),
        )

        return self._clf.score(self.x_val, self.y_val)  # type: ignore


class FTTransformerObjective(Objective):
    """
    Implements an optuna optimization objective.

    See here: https://optuna.readthedocs.io/en/stable/
    Args:
        Objective (Objective): objective
    """

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        cat_features: list[str] | None,
        cat_cardinalities: list[int] | None,
        name: str = "default",
    ):
        """
        Initialize objective.

        Args:
            x_train (pd.DataFrame): feature matrix (train)
            y_train (pd.Series): ground truth (train)
            x_val (pd.DataFrame): feature matrix (val)
            y_val (pd.Series): ground truth (val)
            cat_features (List[str] | None, optional): List of
            categorical features. Defaults to None.
            cat_cardinalities (List[int] | None, optional): Unique counts
            of categorical features.
            Defaults to None.
            name (str, optional): Name of objective. Defaults to "default".
        """
        self._cat_features = [] if not cat_features else cat_features
        self._cat_cardinalities = (
            () if not cat_cardinalities else tuple(cat_cardinalities)
        )
        self._cont_features: list[int] = [
            x for x in x_train.columns.tolist() if x not in self._cat_features
        ]

        self._clf: BaseEstimator
        self._callbacks = CallbackContainer([SaveCallback(), PrintCallback()])

        super().__init__(x_train, y_train, x_val, y_val, name)

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Perform a new search trial in Bayesian search.

        Hyperarameters are suggested, unless they are fixed.
        Args:
            trial (optuna.Trial): current trial.
        Returns:
            float: accuracy of trial on validation set.
        """
        # searchable params done similar to fttransformer paper
        n_blocks: int = trial.suggest_int("n_blocks", 1, 6)
        d_tokens = [96, 128, 192, 256, 320, 384]
        d_token: int = trial.suggest_categorical("d_token", d_tokens)  # type: ignore
        attention_dropouts = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        attention_dropout: int = trial.suggest_categorical("attention_dropout", attention_dropouts)  # type: ignore # noqa: E501
        ffn_dropouts = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
        ffn_dropout: int = trial.suggest_categorical("ffn_dropout", ffn_dropouts)  # type: ignore # noqa: E501

        weight_decay: float = trial.suggest_float("weight_decay", 1e-6, 1e-1)
        lr = trial.suggest_float("lr", 1e-6, 4e-3, log=False)
        batch_size: int = trial.suggest_categorical("batch_size", [4096, 8192, 16192])  # type: ignore # noqa: E501

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        no_devices = torch.cuda.device_count()

        torch.cuda.empty_cache()

        feature_tokenizer_kwargs = {
            "num_continous": len(self._cont_features),
            "cat_cardinalities": self._cat_cardinalities,
            "d_token": d_token,
        }

        transformer_kwargs = {
            "d_token": d_token,
            "n_blocks": n_blocks,
            "attention_n_heads": 8,
            "attention_initialization": "kaiming",
            "ffn_activation": ReGLU,
            "attention_normalization": nn.LayerNorm,
            "ffn_normalization": nn.LayerNorm,
            "ffn_dropout": ffn_dropout,
            # minor simplification from original paper, fix at 4/3, as activation
            # is static with ReGLU / GeGLU
            "ffn_d_hidden": int(d_token * (4 / 3)),
            "attention_dropout": attention_dropout,
            "residual_dropout": 0.0,
            "prenormalization": True,
            "first_prenormalization": False,
            "last_layer_query_idx": None,
            "n_tokens": None,
            "kv_compression_ratio": None,
            "kv_compression_sharing": None,
            "head_activation": nn.ReLU,
            "head_normalization": nn.LayerNorm,
            "d_out": 1,  # fix at 1, due to binary classification
        }

        dl_params: dict[str, Any] = {
            "batch_size": batch_size
            * max(no_devices, 1),  # dataprallel splits batches across devices
            "shuffle": False,
            "device": device,
        }

        module_params = {
            "transformer": Transformer(**transformer_kwargs),  # type: ignore
            "feature_tokenizer": FeatureTokenizer(**feature_tokenizer_kwargs),  # type: ignore # noqa: E501
            "cat_features": self._cat_features,
            "cat_cardinalities": self._cat_cardinalities,
        }

        optim_params = {"lr": lr, "weight_decay": weight_decay}

        self._clf = TransformerClassifier(
            module=FTTransformer,
            module_params=module_params,
            optim_params=optim_params,
            dl_params=dl_params,
            callbacks=self._callbacks,  # type: ignore # noqa: E501
        )

        self._clf.fit(
            self.x_train,
            self.y_train,
            eval_set=(self.x_val, self.y_val),
        )

        return self._clf.score(self.x_val, self.y_val)  # type: ignore


class ClassicalObjective(Objective):
    """
    Implements an optuna optimization objective.

    See here: https://optuna.readthedocs.io/en/stable/
    Args:
        Objective (Objective): objective
    """

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        name: str = "default",
        **kwargs: Any,
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
        self._callbacks = CallbackContainer([SaveCallback()])
        super().__init__(x_train, y_train, x_val, y_val, name)

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
            ("clnv", "best"),
            ("clnv", "ex"),
            ("rev_clnv", "best"),
            ("rev_clnv", "ex"),
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
            "clnv_best",
            "clnv_ex",
            "rev_clnv_best",
            "rev_clnv_ex",
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
        index_layer_5 = trial.suggest_categorical("layer_5", indices)
        layer_5 = mapping[index_layer_5]

        layers = [layer_1, layer_2, layer_3, layer_4, layer_5]
        self._clf = ClassicalClassifier(
            layers=layers,
            random_state=set_seed(),
            features=self.x_train.columns.tolist(),
        )
        self._clf.fit(X=self.x_train, y=self.y_train)
        return self._clf.score(self.x_val, self.y_val)


class GradientBoostingObjective(Objective):
    """
    Implements an optuna optimization objective.

    See here: https://optuna.readthedocs.io/en/stable/
    Args:
        Objective (Objective): objective
    """

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        cat_features: list[str] | None = None,
        name: str = "default",
        **kwargs: Any,
    ):
        """
        Initialize objective.

        Args:
            x_train (pd.DataFrame): feature matrix (train)
            y_train (pd.Series): ground truth (train)
            x_val (pd.DataFrame): feature matrix (val)
            y_val (pd.Series): ground truth (val)
            cat_features (List[str] | None, optional): List of categorical features.
            Defaults to None.
            name (str, optional): Name of objective. Defaults to "default".
        """
        # decay weight of very old observations in training set. See eda notebook.
        weight = np.geomspace(0.001, 1, num=len(y_train))
        # keep ordering of data
        timestamp = np.linspace(0, 1, len(y_train))

        # save to pool for faster memory access
        self._train_pool = Pool(
            data=x_train,
            label=y_train,
            cat_features=cat_features,
            weight=weight,
            timestamp=timestamp,
        )
        self._val_pool = Pool(data=x_val, label=y_val, cat_features=cat_features)
        self.name = name
        self._callbacks = CallbackContainer([SaveCallback()])

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Perform a new search trial in Bayesian search.

        Hyperarameters are suggested, unless they are fixed.
        Args:
            trial (optuna.Trial): current trial.
        Returns:
            float: accuracy of trial on validation set.
        """
        # https://catboost.ai/en/docs/features/training-on-gpu
        gpu_count = get_gpu_device_count()
        task_type = "GPU" if gpu_count > 0 else "CPU"
        devices = f"0-{gpu_count-1}"

        # kaggle book + https://catboost.ai/en/docs/concepts/parameter-tuning
        # friedman paper
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.125, log=True)
        depth = trial.suggest_int("depth", 1, 12)
        l2_leaf_reg = trial.suggest_int("l2_leaf_reg", 2, 30)
        random_strength = trial.suggest_float("random_strength", 1e-9, 10.0, log=True)
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0)
        kwargs_cat = {
            "iterations": 2000,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "random_strength": random_strength,
            "bagging_temperature": bagging_temperature,
            "grow_policy": "Lossguide",
            "border_count": 254,
            "logging_level": "Silent",
            "task_type": task_type,
            "devices": devices,
            "random_seed": set_seed(),
            "eval_metric": "Accuracy",
            "early_stopping_rounds": 100,
        }

        # callback only works for CPU, thus removed. See: https://bit.ly/3FjiuFx
        self._clf = CatBoostClassifier(**kwargs_cat)  # type: ignore
        self._clf.fit(
            self._train_pool,
            eval_set=self._val_pool,
            callbacks=None,
        )

        # calculate accuracy
        return self._clf.score(self._val_pool)  # type: ignore
