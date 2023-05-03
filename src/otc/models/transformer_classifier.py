"""
Sklearn-like wrapper around pytorch transformer models.

Can be used as a consistent interface for evaluation and tuning.
"""
from __future__ import annotations

import gc
import glob
import os
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
from otc.optim.scheduler import CosineWarmupScheduler


class TransformerClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn wrapper around transformer models.

    Args:
        BaseEstimator (_type_): base estimator
        ClassifierMixin (_type_): mixin
    Returns:
        _type_: classifier
    """

    epochs = 50

    def __init__(
        self,
        module: nn.Module,
        module_params: dict[str, Any],
        optim_params: dict[str, Any],
        dl_params: dict[str, Any],
        callbacks: Any,
        features: list[str] | None = None,
        pretrain: bool = False,
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
            pretrain (bool, optional): Whether to pretrain the model. Defaults to False.
        """
        self.module = module

        super().__init__()

        self.clf: nn.Module
        self.module_params = module_params
        self.optim_params = optim_params
        self.dl_params = dl_params
        self.features = features
        self.callbacks = callbacks
        self.pretrain = pretrain

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

    def _checkpoint_write(self) -> None:
        """
        Write weights and biases to checkpoint.
        """
        # remove old files
        print("deleting old checkpoints.")
        for filename in glob.glob("checkpoints/tf_clf*"):
            os.remove(filename)

        # create_dir
        dir_checkpoints = "checkpoints/"
        os.makedirs(dir_checkpoints, exist_ok=True)

        # save new file
        print("saving new checkpoint.")
        torch.save(self.clf.state_dict(), os.path.join(dir_checkpoints, "tf_clf.ptx"))

    def _checkpoint_restore(self) -> None:
        """
        Restore weights and biases from checkpoint.
        """
        print("restore from checkpoint.")
        cp = glob.glob("checkpoints/tf_clf*")
        self.clf.load_state_dict(torch.load(cp[0]))

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

        tab_dl = TabDataLoader(
            data.x_cat, data.x_cont, data.weight, data.y, **self.dl_params
        )
        del data
        return tab_dl

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

        # isolate unlabelled data
        if self.pretrain:
            X_unlabelled = X[y == 0]
            y_unlabelled = y[y == 0]
            X_unlabelled = check_array(X_unlabelled, accept_sparse=False)
            y_unlabelled = check_array(y_unlabelled, accept_sparse=False)
            
            # remove unlabelled
            X = X[y != 0]
            y = y[y != 0]

        check_classification_targets(y)
        X, y = check_X_y(X, y, multi_output=False, accept_sparse=False)

        # if no features are provided or inferred, use default
        if not self.features:
            self.features = [str(i) for i in range(X.shape[1])]

        # use in-sample instead of validation set, if None is provided
        if eval_set:
            X_val, y_val = eval_set
            X_val, y_val = check_X_y(
                X_val, y_val, multi_output=False, accept_sparse=False
            )
        else:
            X_val, y_val = X, y

        # save for accuracy calculation
        len_x_val = len(X_val)

        self.classes_ = np.array([-1, 1])

        train_loader = self.array_to_dataloader(X, y)
        # no weight for validation set / every sample with weight = 1
        val_loader = self.array_to_dataloader(X_val, y_val)

        # free up memory
        del X, y, X_val, y_val
        gc.collect()
        torch.cuda.empty_cache()

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

        max_steps = self.epochs * len(train_loader)
        warmup_steps = int(0.05 * max_steps) + 1  # 5% of max steps
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer, warmup=warmup_steps, max_iters=max_steps
        )

        # see https://stackoverflow.com/a/53628783/5755604
        # no sigmoid required; numerically more stable
        # do not reduce, calculate mean after multiplication with weight
        criterion = nn.BCEWithLogitsLoss(reduction="mean")

        # keep track of val loss and do early stopping
        early_stopping = EarlyStopping(patience=10)

        step = 0
        best_accuracy = -1

        # save stats in classifier
        self._stats_step = []
        self._stats_epoch = []

        for epoch in range(self.epochs):

            # perform training
            loss_in_epoch_train = 0
            train_batch = 0

            self.clf.train()

            for x_cat, x_cont, _, targets in train_loader:

                # reset the gradients back to zero
                self.clf.train()
                optimizer.zero_grad()

                # compute the model output and train loss
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.clf(x_cat, x_cont).flatten()
                    train_loss = criterion(logits, targets)

                # https://pytorch.org/docs/stable/amp.html
                # https://discuss.huggingface.co/t/why-is-grad-norm-clipping-done-during-training-by-default/1866
                scaler.scale(train_loss).backward()
                # scaler.unscale_(optimizer)
                # nn.utils.clip_grad_norm_(
                #     self.clf.parameters(), 5, error_if_nonfinite=False
                # )
                scaler.step(optimizer)
                scaler.update()

                # apply lr scheduler per step
                scheduler.step()

                # add the mini-batch training loss to epoch loss
                loss_in_epoch_train += train_loss.item()

                self._stats_step.append({"train_loss": train_loss.item(), "step": step})

                train_batch += 1
                step += 1

            self.clf.eval()

            loss_in_epoch_val = 0.0
            correct = 0

            val_batch = 0
            with torch.no_grad():
                for x_cat, x_cont, _, targets in val_loader:
                    logits = self.clf(x_cat, x_cont)
                    logits = logits.flatten()

                    # get probabilities and round to nearest integer
                    preds = torch.sigmoid(logits).round()
                    correct += (preds == targets).sum().item()

                    # loss calculation.
                    # Criterion contains softmax already.
                    val_loss = criterion(logits, targets)
                    loss_in_epoch_val += val_loss.item()

                    # print(f"[{epoch}-{val_batch}] val loss: {val_loss}")
                    val_batch += 1
            # loss average over all batches
            train_loss_all = loss_in_epoch_train / len(train_loader)
            val_loss_all = loss_in_epoch_val / len(val_loader)
            # correct samples / no samples
            val_accuracy = correct / len_x_val
            print(f"train loss: {train_loss_all}")
            print(f"val loss: {val_loss_all}")
            print(f"val accuracy: {val_accuracy}")

            self._stats_epoch.append(
                {
                    "train_loss": train_loss_all,
                    "val_loss": val_loss_all,
                    "val_accuracy": val_accuracy,
                    "step": step,
                    "epoch": epoch,
                }
            )

            if best_accuracy < val_accuracy:
                self._checkpoint_write()
                best_accuracy = val_accuracy

            self.callbacks.on_epoch_end(
                epoch, self.epochs, train_loss_all, val_loss_all
            )

            # return early if val accuracy doesn't improve. Minus to minimize.
            early_stopping(-val_accuracy)
            if (
                early_stopping.early_stop
                or np.isnan(train_loss_all)
                or np.isnan(val_loss_all)
            ):
                break

        # restore best from checkpoint
        self._checkpoint_restore()

        # https://discuss.huggingface.co/t/clear-gpu-memory-of-transformers-pipeline/18310/2
        del train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

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
