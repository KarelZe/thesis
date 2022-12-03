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
from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from torch import nn, optim

from otc.data.dataloader import TabDataLoader
from otc.data.dataset import TabDataset
from otc.models.callback import CallbackContainer, PrintCallback, SaveCallback
from otc.models.classical_classifier import ClassicalClassifier
from otc.models.tabtransformer import TabTransformer
from otc.optim.early_stopping import EarlyStopping


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
        self._clf: BaseEstimator | nn.Module
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
        cat_unique: list[int] | None,
        name: str = "default",
    ):
        """
        Initialize objective.

        Args:
            x_train (pd.DataFrame): feature matrix (train)
            y_train (pd.Series): ground truth (train)
            x_val (pd.DataFrame): feature matrix (val)
            y_val (pd.Series): ground truth (val)
            cat_features (Optional[List[str]], optional): List of
            categorical features. Defaults to None.
            cat_unique (Optional[List[int]], optional): Unique counts
            of categorical features.
            Defaults to None.
            name (str, optional): Name of objective. Defaults to "default".
        """
        self._cat_features = [] if not cat_features else cat_features
        self._cat_unique = () if not cat_unique else tuple(cat_unique)
        self._cont_features: list[int] = [
            x for x in x_train.columns.tolist() if x not in self._cat_features
        ]

        self._clf: nn.Module
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
        # static params
        epochs = 1024

        # searchable params
        dim: int = trial.suggest_categorical("dim", [32, 64, 128, 256])  # type: ignore

        # done similar to borisov
        depths = [1, 2, 3, 6, 12]
        depth: int = trial.suggest_categorical("depth", depths)  # type: ignore
        heads: int = trial.suggest_categorical("heads", [2, 4, 8])  # type: ignore
        weight_decay: float = trial.suggest_float("weight_decay", 1e-6, 1e-1)
        lr = trial.suggest_float("lr", 1e-6, 4e-3, log=False)
        dropout = trial.suggest_float("dropout", 0, 0.5, step=0.1)
        bs = [8192, 16384, 32768]
        batch_size: int = trial.suggest_categorical("batch_size", bs)  # type: ignore

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        training_data = TabDataset(
            self.x_train, self.y_train, self._cat_features, self._cat_unique
        )
        val_data = TabDataset(
            self.x_val, self.y_val, self._cat_features, self._cat_unique
        )

        dl_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": False,
            "device": device,
        }

        # differentiate between continous features only and mixed.
        train_loader = TabDataLoader(
            training_data.x_cat, training_data.x_cont, training_data.y, **dl_kwargs
        )
        val_loader = TabDataLoader(
            val_data.x_cat, val_data.x_cont, val_data.y, **dl_kwargs
        )

        self._clf = TabTransformer(
            categories=self._cat_unique,
            num_continuous=len(self._cont_features),
            dim_out=1,
            mlp_act=nn.ReLU(),
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=dropout,
            ff_dropout=dropout,
            mlp_hidden_mults=(4, 2),
        ).to(device)

        # half precision, see https://pytorch.org/docs/stable/amp.html
        scaler = torch.cuda.amp.GradScaler()
        # Generate the optimizers
        optimizer = optim.AdamW(
            self._clf.parameters(), lr=lr, weight_decay=weight_decay
        )

        # see https://stackoverflow.com/a/53628783/5755604
        # no sigmoid required; numerically more stable
        criterion = nn.BCEWithLogitsLoss()

        # keep track of val loss and do early stopping
        early_stopping = EarlyStopping(patience=5)

        for epoch in range(epochs):

            # perform training
            loss_in_epoch_train = 0

            self._clf.train()

            for x_cat, x_cont, targets in train_loader:

                # reset the gradients back to zero
                optimizer.zero_grad()

                outputs = self._clf(x_cat, x_cont)
                outputs = outputs.flatten()
                with torch.cuda.amp.autocast():
                    train_loss = criterion(outputs, targets)

                # compute accumulated gradients
                scaler.scale(train_loss).backward()

                # perform parameter update based on current gradients
                scaler.step(optimizer)
                scaler.update()

                # add the mini-batch training loss to epoch loss
                loss_in_epoch_train += train_loss.item()

            self._clf.eval()

            loss_in_epoch_val = 0.0

            with torch.no_grad():
                for x_cat, x_cont, targets in val_loader:
                    outputs = self._clf(x_cat, x_cont)
                    outputs = outputs.flatten()

                    val_loss = criterion(outputs, targets)
                    loss_in_epoch_val += val_loss.item()

            train_loss = loss_in_epoch_train / len(train_loader)
            val_loss = loss_in_epoch_val / len(val_loader)

            self._callbacks.on_epoch_end(epoch, epochs, train_loss, val_loss)

            # return early if val loss doesn't decrease for several iterations
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break

        # make predictions with final model
        y_pred, y_true = [], []

        self._clf.eval()

        for x_cat, x_cont, targets in val_loader:
            output = self._clf(x_cat, x_cont)

            # map between zero and one, sigmoid is otherwise included in loss already
            # https://stackoverflow.com/a/66910866/5755604
            output = torch.sigmoid(output.squeeze())
            y_pred.append(output.detach().cpu().numpy())
            y_true.append(targets.detach().cpu().numpy())  # type: ignore

        # round prediction to nearest int
        y_pred = np.rint(np.concatenate(y_pred))
        y_true = np.concatenate(y_true)

        return accuracy_score(y_true, y_pred)  # type: ignore


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
        self._callbacks = CallbackContainer([])
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
        )
        self._clf.fit(X=self.x_train, y=self.y_train)
        pred = self._clf.predict(self.x_val)
        return accuracy_score(self.y_val, pred)


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
    ):
        """
        Initialize objective.

        Args:
            x_train (pd.DataFrame): feature matrix (train)
            y_train (pd.Series): ground truth (train)
            x_val (pd.DataFrame): feature matrix (val)
            y_val (pd.Series): ground truth (val)
            cat_features (Optional[List[str]], optional): List of categorical features.
            Defaults to None.
            name (str, optional): Name of objective. Defaults to "default".
        """
        self._cat_features = cat_features
        super().__init__(x_train, y_train, x_val, y_val, name)
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
        task_type = "GPU" if get_gpu_device_count() > 0 else "CPU"

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
            "task_type": task_type,
            "cat_features": self._cat_features,
            "random_seed": set_seed(),
        }

        self._clf = CatBoostClassifier(**params)
        self._clf.fit(
            self.x_train,
            self.y_train,
            eval_set=(self.x_val, self.y_val),
        )

        pred = self._clf.predict(self.x_val, prediction_type="Class")
        return accuracy_score(self.y_val, pred)
