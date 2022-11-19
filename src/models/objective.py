"""
Provides objectives for optimizations.

Adds support for classical rules, GBTs and transformer-based architectures.
"""
import os
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.tensor as tensor
from sklearn.base import BaseEstimator

# from optuna.integration import CatBoostPruningCallback
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.tabtransformer import TabTransformer
from src.optim.early_stopping import EarlyStopping


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
        # FIXME: Update basetype
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
        cat_features: Optional[List[str]],
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
        # FIXME: think about cat features not in training set, otherwise make external
        self._cat_unique = x_train[cat_features].nunique().values
        if not self._cat_unique:
            self._cat_unique = ()

        # assume columns are duplicate free, which is standard in pandas
        cont_features = [x for x in x_train.columns.tolist() if x not in cat_features]
        self._cont_idx = [features.index(i) for i in cont_features if i in features]

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
        dim = trial.suggest_categorical("dim", [32, 64, 128, 256])

        # done similar to borisov
        depth = (trial.suggest_categorical("depth", [1, 2, 3, 6, 12]),)
        heads = (trial.suggest_categorical("heads", [2, 4, 8]),)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1)
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0, 0.5, step=0.1)
        # done differntly to borisov; suggest batches
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

        # convert to tensor
        x_train = tensor(self.x_train).float()
        y_train = tensor(self.y_train).float()

        x_val = tensor(self.x_val).float()
        y_val = tensor(self.y_val).float()

        # create training and val set
        training_data = TensorDataset(x_train, y_train)
        val_data = TensorDataset(x_val, y_val)

        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        #  use gpu if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                x_cont = inputs[:, self._cont_idx].to(device)
                targets = targets.to(device)

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

        return accuracy_score(y_true, y_pred) # type: ignore
