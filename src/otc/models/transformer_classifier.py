"""
Sklearn-like wrapper around pytorch transformer models.

Can be used as a consistent interface for evaluation and tuning.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn, optim

from otc.data.dataloader import TabDataLoader
from otc.data.dataset import TabDataset
from otc.optim.early_stopping import EarlyStopping


class TransformerClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn wrapper around transformer models.

    Args:
        BaseEstimator (_type_): base estimator
        ClassifierMixin (_type_): mixin

    Returns:
        _type_: classifier
    """

    epochs = 1024

    def __init__(
        self,
        module: nn.Module,
        module_params: dict[str, Any],
        optim_params: dict[str, Any],
        dl_params: dict[str, Any],
        features: list[str],
        callbacks: Any,
    ) -> None:
        """
        Initialize the model.

        Args:
            module (nn.Module): module to instantiate
            module_params (dict[str, Any]): params for module
            optim_params (dict[str, Any]): params for optimizer
            dl_params (dict[str, Any]): params for dataloader
            features (list[str]): list of features
            callbacks (CallbackContainer): Container with callbacks
        """
        self.module = module

        super().__init__()

        self._clf: nn.Module
        self._module_params = module_params
        self._optim_params = optim_params
        self._dl_params = dl_params
        self._features = features
        self._callbacks = callbacks

    def array_to_dataloader(
        self, X: npt.NDArray | pd.DataFrame, y: npt.NDArray | pd.Series
    ) -> TabDataLoader:
        """
        Convert array like to dataloader.

        Args:
            X (npt.NDArray | pd.DataFrame): feature matrix
            y (npt.NDArray | pd.Series): target vector

        Returns:
            TabDataLoader: data loader with X_cat, X_cont, y
        """
        data = TabDataset(
            X,
            y,
            features=self._features,
            cat_features=self._module_params["cat_features"],
            cat_unique_counts=self._module_params["cat_cardinalities"],
        )

        return TabDataLoader(data.x_cat, data.x_cont, data.y, **self._dl_params)

    def fit(
        self,
        X: npt.NDArray | pd.DataFrame,
        y: npt.NDArray | pd.Series,
        eval_set: tuple[npt.NDArray, npt.NDArray]
        | tuple[pd.DataFrame, pd.Series]
        | None,
    ) -> TransformerClassifier:
        """
        Fit the model.

        Args:
            X (npt.NDArray | pd.DataFrame): feature matrix
            y (npt.NDArray | pd.Series): target
            eval_set (tuple[npt.NDArray, npt.NDArray] |
            tuple[pd.DataFrame, pd.Series] | None): eval set.
            If no eval set is passed, the training set is used.

        Returns:
            TransformerClassifier: self
        """
        # use insample instead of validation set, if None is passed
        if eval_set:
            X_val, y_val = eval_set
        else:
            X_val, y_val = X, y

        train_loader = self.array_to_dataloader(X, y)
        val_loader = self.array_to_dataloader(X_val, y_val)

        self._clf = self.module(**self._module_params)

        # use multiple gpus, if available
        self._clf = nn.DataParallel(self._clf).to(self._dl_params["device"])

        # half precision, see https://pytorch.org/docs/stable/amp.html
        scaler = torch.cuda.amp.GradScaler()
        # Generate the optimizers
        optimizer = optim.AdamW(
            self._clf.parameters(),
            lr=self._optim_params["lr"],
            weight_decay=self._optim_params["weight_decay"],
        )

        # see https://stackoverflow.com/a/53628783/5755604
        # no sigmoid required; numerically more stable
        criterion = nn.BCEWithLogitsLoss()

        # keep track of val loss and do early stopping
        early_stopping = EarlyStopping(patience=5)

        for epoch in range(self.epochs):

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
            # correct = 0

            with torch.no_grad():
                for x_cat, x_cont, targets in val_loader:
                    outputs = self._clf(x_cat, x_cont)
                    outputs = outputs.flatten()

                    val_loss = criterion(outputs, targets)
                    loss_in_epoch_val += val_loss.item()

                    # # convert to propability, then round to nearest integer
                    # outputs = torch.sigmoid(outputs).round()
                    # correct += (outputs == targets).sum().item()

            train_loss = loss_in_epoch_train / len(train_loader)
            val_loss = loss_in_epoch_val / len(val_loader)

            self._callbacks.on_epoch_end(epoch, self.epochs, train_loss, val_loss)

            # return early if val loss doesn't decrease for several iterations
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break

        # is fitted flag
        self.is_fitted_ = True
        return self

    def predict(self, X: npt.NDArray | pd.DataFrame) -> npt.NDArray:
        """
        Predict class labels for X.

        Args:
            X (npt.NDArray | pd.DataFrame): feature matrix

        Returns:
            npt.NDArray: labels
        """
        probs = self.predict_proba(X)
        # convert probs to classes
        return np.where(probs > 0.5, 1, -1)

    def predict_proba(self, X: npt.NDArray | pd.DataFrame) -> npt.NDArray:
        """
        Predict class probabilities for X.

        Args:
            X (npt.NDArray | pd.DataFrame): feature matrix

        Returns:
            npt.NDArray: probabilities
        """
        # check if there are attributes with trailing _
        check_is_fitted(self)
        test_loader = self.array_to_dataloader(X, pd.Series(np.zeros(len(X))))

        self._clf.eval()

        probabilites = []
        with torch.no_grad():
            for x_cat, x_cont, _ in test_loader:
                probability = self._clf(x_cat, x_cont)
                probability = probability.flatten()
                probability = torch.sigmoid(probability)
                probabilites.append(probability.detach().cpu().numpy())

        return np.concatenate(probabilites)
