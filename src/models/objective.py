"""
Provides objectives for optimizations.

Adds support for classical rules, GBTs and transformer-based architectures.
"""
import glob
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import optuna
import pandas as pd
import torch
from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
from sklearn.base import BaseEstimator

# from optuna.integration import CatBoostPruningCallback
from sklearn.metrics import accuracy_score

from torch import nn, optim, tensor
from torch.utils.data import DataLoader, TensorDataset

from data.fs import fs
from models.classical_classifier import ClassicalClassifier
from models.tabtransformer import TabTransformer
from optim.early_stopping import EarlyStopping
from utils.config import Settings


def set_seed(seed_val: int = 42) -> int:
    """
    Seeds basic parameters for reproducibility of results.

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
        self._clf: Union[BaseEstimator, nn.Module]

    @abstractmethod
    def save_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Save model using callback.

        Args:
            study (optuna.Study): current study.
            trial (optuna.Trial): current trial.
        """


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
        cat_features: Optional[List[str]] = None,
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
        features = x_train.columns.tolist()
        cat_features = [] if not cat_features else cat_features
        self._cat_idx = [features.index(i) for i in cat_features if i in features]
        print(cat_features)
        print(self._cat_idx)
        # FIXME: think about cat features not in training set, otherwise make external
        self._cat_unique = x_train[cat_features].nunique().to_list()
        if not self._cat_unique:
            self._cat_unique = ()

        print(self._cat_unique)
        # assume columns are duplicate free, which is standard in pandas
        cont_features = [x for x in x_train.columns.tolist() if x not in cat_features]
        self._cont_idx = [features.index(i) for i in cont_features if i in features]
        print(self._cont_idx)
        self._clf: nn.Module
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
        # done differently in borisov; this should be clearer, as search is not changed
        dim = trial.suggest_categorical("dim", [4,8,12])

        # done similar to borisov
        depth = trial.suggest_categorical("depth", [1, 2, 3, 6, 12])
        heads = trial.suggest_categorical("heads", [2, 4, 8])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1)
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0, 0.5, step=0.1)
        # done differntly to borisov; suggest batches
        batch_size = trial.suggest_categorical("batch_size", [2, 4])

        # FIXME: fix embedding lookup
        # see https://discuss.pytorch.org/t/embedding-error-index-out-of-range-in-self/81550/2
        # convert to tensor
        x_train = tensor(self.x_train.values).float()
        y_train = tensor(self.y_train.values).float()

        x_val = tensor(self.x_val.values).float()
        y_val = tensor(self.y_val.values).float()

        # create training and val set
        training_data = TensorDataset(x_train, y_train)
        val_data = TensorDataset(x_val, y_val)

        train_loader = DataLoader(
            training_data, batch_size=batch_size, shuffle=False, num_workers=8
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False, num_workers=8
        )

        #  use gpu if available
        device = "cpu" #  torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._clf = TabTransformer(
            categories=self._cat_unique,
            num_continuous=len(self._cont_idx),  # number of continuous values
            dim_out=1,
            mlp_act=nn.ReLU(),  # activation for final mlp (here relu)
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=dropout,
            ff_dropout=dropout,
            mlp_hidden_mults=(4, 2),
        ).to(device)

        # Generate the optimizers
        optimizer = optim.AdamW(
            self._clf.parameters(), lr=lr, weight_decay=weight_decay
        )

        # see https://stackoverflow.com/a/53628783/5755604
        criterion = nn.BCEWithLogitsLoss()

        # keep track of val loss and do early stopping
        early_stopping = EarlyStopping(patience=5)

        train_history, val_history = [], []

        for epoch in range(epochs):

            # perform training
            loss_in_epoch_train = 0

            self._clf.train()

            for inputs, targets in train_loader:

                # FIXME: refactor to custom data loader
                x_cat = (
                    inputs[:, self._cat_idx].int().to(device) if self._cat_idx else None
                )
                print(x_cat)

                x_cont = inputs[:, self._cont_idx].to(device)
                targets = targets.to(device)
                print(x_cont)
                # reset the gradients back to zero
                optimizer.zero_grad()

                outputs = self._clf(x_cat, x_cont)
                outputs = outputs.squeeze()

                train_loss = criterion(outputs, targets)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss_in_epoch_train += train_loss.item()

            self._clf.eval()

            loss_in_epoch_val = 0.0

            with torch.no_grad():
                for inputs, targets in val_loader:

                    x_cat = (
                        inputs[:, self._cat_idx].int().to(device)
                        if self._cat_idx
                        else None
                    )
                    x_cont = inputs[:, self._cont_idx].to(device)
                    targets = targets.to(device)

                    outputs = self._clf(x_cont, x_cat)
                    outputs = outputs.squeeze()

                    val_loss = criterion(outputs, targets)

                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = self._clf(x_cont, x_cat)

                    val_loss = criterion(outputs, targets)
                    loss_in_epoch_val += val_loss.item()

            train_loss = loss_in_epoch_train / len(train_loader)
            val_loss = loss_in_epoch_val / len(val_loader)

            train_history.append(train_loss)
            val_history.append(val_loss)

            print(f"epoch : {epoch + 1}/{epochs},", end=" ")
            print(f"loss (train) = {train_loss:.8f}, loss (val) = {val_loss:.8f}")

            # return early if val loss doesn't decrease for several iterations
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break

            trial.report(val_loss, epoch)

        # make predictions with final model
        y_pred, y_true = [], []

        self._clf.eval()

        for inputs, targets in val_loader:
            x_cat = inputs[:, self._cat_idx].int().to(device) if self._cat_idx else None
            x_cont = inputs[:, self._cont_idx].to(device)
            targets = targets.to(device)
            prediction = self._clf(x_cont, x_cat)
            y_pred.append(prediction.detach().cpu().numpy().flatten())
            y_true.append(targets.detach().cpu().numpy().flatten())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)

        return accuracy_score(y_true, y_pred)  # type: ignore

    def save_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Save model with callback.

        Args:
            study (optuna.Study): current study.
            trial (optuna.Trial): current trial.
        """
        # FIXME: Implement later


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

    def save_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Save model with callback.

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

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        cat_features: Optional[List[str]] = None,
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

        # pruning_callback = CatBoostPruningCallback(trial, "Accuracy")

        self._clf = CatBoostClassifier(**params)
        self._clf.fit(
            self.x_train,
            self.y_train,
            eval_set=(self.x_val, self.y_val),
            # callbacks=[pruning_callback],
        )

        # pruning_callback.check_pruned()

        pred = self._clf.predict(self.x_val, prediction_type="Class")
        return accuracy_score(self.y_val, pred)

    def save_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Save model with callback.

        Args:
            study (optuna.Study): current study.
            trial (optuna.Trial): current trial.
        """
        if study.best_trial == trial:

            settings = Settings()

            # e. g. dnurtlqv_CatBoostClassifier_default_trial_
            prefix_file = (
                f"{study.study_name}_"
                f"{self._clf.__class__.__name__}_{self.name}_trial_"
            )

            # FIXME: Replace with cloud path. One could directly upload to gcloud
            # without storing locally.
            # https://pypi.org/project/cloudpathlib/

            # remove old files on remote first
            outdated_files_remote = fs.glob(
                "gs://"
                + Path(
                    settings.GCS_BUCKET, settings.MODEL_DIR_REMOTE, prefix_file + "*"
                ).as_posix()
            )

            if len(outdated_files_remote) > 0:
                fs.rm(outdated_files_remote)

            # remove local files next
            outdated_files_local = glob.glob(
                Path(settings.MODEL_DIR_LOCAL, prefix_file + "*").as_posix()
            )
            if len(outdated_files_local) > 0:
                os.remove(*outdated_files_local)

            # save current best locally
            new_file = prefix_file + f"{trial.number}.cbm"
            loc_path = Path(settings.MODEL_DIR_LOCAL, new_file).as_posix()

            remote_path = (
                "gs://"
                + Path(
                    settings.GCS_BUCKET, settings.MODEL_DIR_REMOTE, new_file
                ).as_posix()
            )
            self._clf.save_model(loc_path, format="cbm")
            # save current best remotely
            fs.put(loc_path, remote_path)
