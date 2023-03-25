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
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
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

    epochs = 128

    def __init__(
        self,
        module: nn.Module,
        module_params: dict[str, Any],
        optim_params: dict[str, Any],
        dl_params: dict[str, Any],
        callbacks: Any,
        features: list[str] | None = None,
    ) -> None:
        """
        Initialize the model.

        Args:
            module (nn.Module): module to instantiate
            module_params (dict[str, Any]): params for module
            optim_params (dict[str, Any]): params for optimizer
            dl_params (dict[str, Any]): params for dataloader
            callbacks (CallbackContainer): Container with callbacks
            features (list[str] | None, optional): List of feature names in order of
            columns. Required to match columns in feature matrix with label. If no
            feature names are provided for pd.DataFrames, names are taken from
            `X.columns` in `fit()`. Defaults to None.
        """
        self.module = module

        super().__init__()

        self.clf: nn.Module
        self.module_params = module_params
        self.optim_params = optim_params
        self.dl_params = dl_params
        self.features = features
        self.callbacks = callbacks

    def _more_tags(self) -> dict[str, bool]:
        """
        Set tags for sklearn.

        See: https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        """
        return {
            "binary_only": True,
            "_skip_test": True,
            "poor_score": True,
        }

    def array_to_dataloader(
        self,
        X: npt.NDArray | pd.DataFrame,
        y: npt.NDArray | pd.Series,
        weight: npt.NDArray | None = None,
    ) -> TabDataLoader:
        """
        Convert array like to dataloader.

        Args:
            X (npt.NDArray | pd.DataFrame): feature matrix
            y (npt.NDArray | pd.Series): target vector
            weight (npt.NDArray | None, optional): weights for each sample.
            Defaults to None.
        Returns:
            TabDataLoader: data loader.
        """
        data = TabDataset(
            X,
            y,
            feature_names=self.features,
            weight=weight,
            cat_features=self.module_params["cat_features"],
            cat_unique_counts=self.module_params["cat_cardinalities"],
        )

        return TabDataLoader(
            data.x_cat, data.x_cont, data.weight, data.y, **self.dl_params
        )

    def fit(
        self,
        X: npt.NDArray | pd.DataFrame,
        y: npt.NDArray | pd.Series,
        eval_set: tuple[npt.NDArray, npt.NDArray]
        | tuple[pd.DataFrame, pd.Series]
        | None = None,
    ) -> TransformerClassifier:
        """
        Fit the model.

        Args:
            X (npt.NDArray | pd.DataFrame): feature matrix
            y (npt.NDArray | pd.Series): target
            eval_set (tuple[npt.NDArray, npt.NDArray] |
            tuple[pd.DataFrame, pd.Series] | None): eval set. Defaults to None.
            If no eval set is passed, the training set is used.
        Returns:
            TransformerClassifier: self
        """
        # get features from pd.DataFrame, if not provided
        if isinstance(X, pd.DataFrame) and self.features is None:
            self.features = X.columns.tolist()

        check_classification_targets(y)

        X, y = check_X_y(X, y, multi_output=False, accept_sparse=False)

        # if no features are provided or inferred, use default
        if not self.features:
            self.features = [str(i) for i in range(X.shape[1])]

        # use insample instead of validation set, if None is passed
        if eval_set:
            X_val, y_val = eval_set
            X_val, y_val = check_X_y(
                X_val, y_val, multi_output=False, accept_sparse=False
            )
        else:
            X_val, y_val = X, y

        self.classes_ = np.array([-1, 1])

        # decay weight of very old observations in training set. See eda notebook.
        weight = np.geomspace(0.001, 1, num=len(y))
        train_loader = self.array_to_dataloader(X, y, weight)
        # no weight for validation set / every sample with weight = 1
        val_loader = self.array_to_dataloader(X_val, y_val)

        self.clf = self.module(**self.module_params)

        # use multiple gpus, if available
        self.clf = nn.DataParallel(self.clf).to(self.dl_params["device"])

        # half precision, see https://pytorch.org/docs/stable/amp.html
        scaler = torch.cuda.amp.GradScaler()
        # Generate the optimizers
        optimizer = optim.AdamW(
            self.clf.parameters(),
            lr=self.optim_params["lr"],
            weight_decay=self.optim_params["weight_decay"],
        )

        # see https://stackoverflow.com/a/53628783/5755604
        # no sigmoid required; numerically more stable
        # do not reduce, calculate mean after multiplication with weight
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        # keep track of val loss and do early stopping
        early_stopping = EarlyStopping(patience=15)

        for epoch in range(self.epochs):

            # perform training
            loss_in_epoch_train = 0

            self.clf.train()

            for x_cat, x_cont, weights, targets in train_loader:

                # reset the gradients back to zero
                optimizer.zero_grad()

                # compute the model output and train loss
                with torch.cuda.amp.autocast():
                    logits = self.clf(x_cat, x_cont).flatten()
                    intermediate_loss = criterion(logits, targets)
                    # weight train loss with (decaying) weights
                    train_loss = torch.mean(weights * intermediate_loss)
                # compute accumulated gradients
                scaler.scale(train_loss).backward()

                # perform parameter update based on current gradients
                scaler.step(optimizer)
                scaler.update()

                # add the mini-batch training loss to epoch loss
                loss_in_epoch_train += train_loss.item()

            self.clf.eval()

            loss_in_epoch_val = 0.0
            correct = 0

            with torch.no_grad():
                for x_cat, x_cont, weights, targets in val_loader:
                    logits = self.clf(x_cat, x_cont)
                    logits = logits.flatten()

                    # get probabilities and round to nearest integer
                    preds = torch.sigmoid(logits).round()
                    correct += (preds == targets).sum().item()

                    # loss calculation.
                    # Criterion contains softmax already.
                    # Weight sample loss with (equal) weights
                    intermediate_loss = criterion(logits, targets)
                    val_loss = torch.mean(weights * intermediate_loss)
                    loss_in_epoch_val += val_loss.item()
            # loss average over all batches
            train_loss = loss_in_epoch_train / len(train_loader)
            val_loss = loss_in_epoch_val / len(val_loader)
            # correct samples / no samples
            val_accuracy = correct / len(X_val)

            self.callbacks.on_epoch_end(epoch, self.epochs, train_loss, val_loss)

            # return early if val accuracy doesn't improve. Minus to minimize.
            early_stopping(-val_accuracy)
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
        probability = self.predict_proba(X)

        # convert probs to classes by checking which class has highest probability.
        # Then assign -1 if first probability is >= 0.5 and 1 otherwise.
        return self.classes_[np.argmax(probability, axis=1)]

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

        X = check_array(X, accept_sparse=False)
        y = np.zeros(len(X))

        test_loader = self.array_to_dataloader(X, y)

        self.clf.eval()

        # calculate probability and counter-probability
        probabilites = []
        with torch.no_grad():
            for x_cat, x_cont, _, _ in test_loader:
                logits = self.clf(x_cat, x_cont)
                logits = logits.flatten()
                probability = torch.sigmoid(logits)
                probabilites.append(probability.detach().cpu().numpy())

        probabilites = np.concatenate(probabilites)
        return np.column_stack((1 - probabilites, probabilites))  # type: ignore
