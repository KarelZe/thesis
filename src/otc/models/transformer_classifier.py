"""Sklearn-like wrapper around pytorch transformer models.

Can be used as a consistent interface for evaluation and tuning.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from torch import nn, optim

from otc.data.dataloader import TabDataLoader
from otc.data.dataset import TabDataset
from otc.models.fttransformer import CLSHead
from otc.optim.early_stopping import EarlyStopping
from otc.optim.scheduler import CosineWarmupScheduler


class TransformerClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn wrapper around transformer models.

    Args:
    ----
        BaseEstimator (_type_): base estimator
        ClassifierMixin (_type_): mixin
    Returns:
        _type_: classifier
    """

    epochs_pretrain = 20
    epochs_finetune = 20

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
        """Initialize the model.

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
        self.classes_ = np.array([-1, 1])

    def _more_tags(self) -> dict[str, bool]:
        """Set tags for sklearn.

        See: https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        """
        return {
            "binary_only": True,
            "_skip_test": True,
            "poor_score": True,
        }

    def _checkpoint_write(self) -> None:
        """Write weights and biases to checkpoint."""
        # remove old files
        print("deleting old checkpoints.")
        for filename in Path("checkpoints/").glob("tf_clf*"):
            Path.unlink(filename)

        # create_dir
        dir_checkpoints = Path("checkpoints/")
        dir_checkpoints.mkdir(exist_ok=True, parents=True)

        # save new file
        print("saving new checkpoint.")
        torch.save(self.clf.state_dict(), dir_checkpoints / "tf_clf.ptx")

    def _checkpoint_restore(self) -> None:
        """Restore weights and biases from checkpoint."""
        print("restore from checkpoint.")
        cp = Path("checkpoints/").glob("tf_clf*")
        self.clf.load_state_dict(torch.load(next(cp)))

    def array_to_dataloader_finetune(
        self,
        X: npt.NDArray | pd.DataFrame,
        y: npt.NDArray | pd.Series,
        weight: npt.NDArray | None = None,
    ) -> TabDataLoader:
        """Convert array like to dataloader.

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

    def _gen_perm(self, X: torch.Tensor) -> torch.Tensor:
        """Generate index permutation."""
        if X is None:
            return None
        return torch.randint_like(X, X.shape[0], dtype=torch.long)

    def _gen_masks(
        self, X: torch.Tensor, perm: torch.Tensor, corrupt_probability: float = 0.15
    ) -> torch.Tensor:
        """Generate binary mask for detection."""
        masks = torch.empty_like(X).bernoulli(p=corrupt_probability).bool()
        return masks & (X[perm, torch.arange(X.shape[1], device=X.device)] != X)

    def array_to_dataloader_pretrain(
        self,
        X: npt.NDArray | pd.DataFrame,
        y: npt.NDArray | pd.Series,
    ) -> TabDataLoader:
        """Generate dataloader for pretraining."""
        data = TabDataset(
            X,
            y,
            cat_features=self.module_params["cat_features"],
            cat_unique_counts=self.module_params["cat_cardinalities"],
        )

        # split the continuous and categorical features
        x_cont = data.x_cont
        x_cat = data.x_cat

        # generate permutations and masks for the features
        x_cont_perm = self._gen_perm(x_cont)
        x_cat_perm = self._gen_perm(x_cat) if x_cat is not None else None
        x_cont_mask = self._gen_masks(x_cont, x_cont_perm)
        x_cat_mask = (
            self._gen_masks(x_cat, x_cat_perm) if x_cat_perm is not None else None
        )

        # apply the permutations and masks to the features
        x_cont_permuted = torch.gather(x_cont, 0, x_cont_perm)
        x_cont[x_cont_mask] = x_cont_permuted[x_cont_mask]
        if x_cat is not None:
            x_cat_permuted = torch.gather(x_cat, 0, x_cat_perm)
            x_cat[x_cat_mask] = x_cat_permuted[x_cat_mask]

        # concatenate the masks and create a TabDataLoader
        masks = (
            torch.cat([x_cont_mask, x_cat_mask], dim=1)
            if x_cat_mask is not None
            else x_cont_mask
        )
        tab_dl = TabDataLoader(x_cat, x_cont, masks, **self.dl_params)

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
        """Fit the model.

        Args:
            X (npt.NDArray | pd.DataFrame): feature matrix
            y (npt.NDArray | pd.Series): target
            eval_set (tuple[npt.NDArray, npt.NDArray] | tuple[pd.DataFrame, pd.Series] | None): eval set. Defaults to None.
            If no eval set is passed, the training set is used.

        Returns:
            TransformerClassifier: self
        """
        # get features from pd.DataFrame, if not provided
        if isinstance(X, pd.DataFrame) and self.features is None:
            self.features = X.columns.tolist()

        # if no features are provided or inferred, use default
        if not self.features:
            self.features = [str(i) for i in range(X.shape[1])]

        self.clf = self.module(**self.module_params)
        target_head = self.clf.transformer.head

        self._stats_pretrain_step = []
        self._stats_pretrain_epoch = []

        if self.pretrain:
            print("start pre-training...")
            mask = y == 0

            # isolate unlabelled
            X_unlabelled = X[mask]
            y_unlabelled = y[mask]

            # remove unlabelled
            X = X[~mask]
            y = y[~mask]

            # convert to dataloader
            train_loader_pretrain = self.array_to_dataloader_pretrain(
                X_unlabelled, y_unlabelled
            )

            # use in-sample instead of validation set, if None is provided
            X_val, y_val = (
                eval_set if eval_set is not None else (X_unlabelled, y_unlabelled)
            )

            val_loader_pretrain = self.array_to_dataloader_pretrain(X_val, y_val)

            # free up memory
            del X_unlabelled, y_unlabelled
            gc.collect()
            torch.cuda.empty_cache()

            # replace head with classification head capacity similar to rubachev
            self.clf.transformer.head = CLSHead(
                d_in=self.module_params["d_token"], d_hidden=512
            )
            self.clf.to(self.dl_params["device"])

            # half precision, see https://pytorch.org/docs/stable/amp.html
            scaler = torch.cuda.amp.GradScaler()

            # Specify parameters for which weight decay should be disabled
            no_decay = ["tokenizer", ".norm", ".bias"]

            # Create a list of parameter groups
            param_groups = [
                {
                    "params": [
                        p
                        for n, p in self.clf.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.optim_params["weight_decay"],
                },
                {
                    "params": [
                        p
                        for n, p in self.clf.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            # Generate the optimizers
            optimizer = optim.AdamW(
                param_groups,
                lr=self.optim_params["lr"],
            )

            # set up cosine lr scheduler
            max_steps = self.epochs_pretrain * len(train_loader_pretrain)
            warmup_steps = int(0.05 * max_steps) + 1  # 5 % of max steps
            scheduler = CosineWarmupScheduler(
                optimizer=optimizer, warmup=warmup_steps, max_iters=max_steps
            )

            # keep track of val loss and do early stopping
            early_stopping = EarlyStopping(patience=10)

            # mean bce with logits loss
            criterion = nn.BCEWithLogitsLoss(reduction="mean")

            step = 0
            best_accuracy = -1.0

            for epoch in range(self.epochs_pretrain):
                # perform training
                loss_in_epoch_train = 0

                batch = 0

                for x_cat, x_cont, mask in train_loader_pretrain:
                    self.clf.train()
                    optimizer.zero_grad()

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self.clf(x_cat, x_cont)
                        train_loss = criterion(logits, mask.float())

                    scaler.scale(train_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    # add the mini-batch training loss to epoch loss
                    loss_in_epoch_train += train_loss.item()

                    self._stats_pretrain_step.append(
                        {"train_loss": train_loss.item(), "step": step}
                    )

                    batch += 1
                    step += 1

                self.clf.eval()
                loss_in_epoch_val = 0.0
                correct = 0

                with torch.no_grad():
                    for x_cat, x_cont, mask in val_loader_pretrain:
                        # for my implementation
                        logits = self.clf(x_cat, x_cont)
                        val_loss = criterion(logits, mask.float())
                        loss_in_epoch_val += val_loss.item()

                        # accuracy
                        # adapted from here, but over columns + rows https://github.com/puhsu/tabular-dl-pretrain-objectives/blob/3f503d197867c341b4133efcafd3243eb5bb93de/bin/mask.py#L440
                        hard_predictions = torch.zeros_like(logits, dtype=torch.long)
                        hard_predictions[logits > 0] = 1
                        # sum columns and rows
                        correct += (hard_predictions.bool() == mask).sum()

                        batch += 1

                # loss average over all batches
                train_loss_all = loss_in_epoch_train / len(train_loader_pretrain)
                val_loss_all = loss_in_epoch_val / len(val_loader_pretrain)
                # correct / (rows * columns)
                val_accuracy = correct / (X_val.shape[0] * X_val.shape[1])

                print(f"train loss: {train_loss}")
                print(f"val loss: {val_loss}")
                print(f"val accuracy: {val_accuracy}")

                self._stats_pretrain_epoch.append(
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

            # https://discuss.huggingface.co/t/clear-gpu-memory-of-transformers-pipeline/18310/2
            del train_loader_pretrain, val_loader_pretrain
            gc.collect()
            torch.cuda.empty_cache()

        # set target head, if not set
        self.clf.transformer.head = target_head
        self.clf.to(self.dl_params["device"])

        # start finetuning beneath
        print("start finetuning...")

        # use in-sample instead of validation set, if None is provided
        X_val, y_val = eval_set if eval_set is not None else (X, y)

        # save for accuracy calculation
        len_x_val = len(X_val)
        weight = np.geomspace(0.001, 1, num=len(y))

        train_loader_finetune = self.array_to_dataloader_finetune(X, y, weight)
        # no weight for validation set / every sample with weight = 1
        val_loader_finetune = self.array_to_dataloader_finetune(X_val, y_val)

        # free up memory
        del X, y, X_val, y_val
        gc.collect()
        torch.cuda.empty_cache()

        # half precision, see https://pytorch.org/docs/stable/amp.html
        scaler = torch.cuda.amp.GradScaler()

        # Specify parameters for which weight decay should be disabled
        # https://github.com/Yura52/tabular-dl-revisiting-models/blob/main/bin/ft_transformer.py
        no_decay = ["tokenizer", ".norm", ".bias"]

        # Create a list of parameter groups
        param_groups = [
            {
                "params": [
                    p
                    for n, p in self.clf.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.optim_params["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.clf.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # Generate the optimizers
        optimizer = optim.AdamW(
            param_groups,
            lr=self.optim_params["lr"],
        )

        max_steps = self.epochs_finetune * len(train_loader_finetune)
        warmup_steps = int(0.05 * max_steps) + 1  # 5% of max steps
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer, warmup=warmup_steps, max_iters=max_steps
        )

        # see https://stackoverflow.com/a/53628783/5755604
        # no sigmoid required; numerically more stable
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        # keep track of val loss and do early stopping
        early_stopping = EarlyStopping(patience=10)

        step = 0
        best_accuracy = -1.0

        # save stats in classifier
        self._stats_step = []
        self._stats_epoch = []

        for epoch in range(self.epochs_finetune):
            # perform training
            loss_in_epoch_train = 0

            self.clf.train()

            for x_cat, x_cont, weights, targets in train_loader_finetune:
                # reset the gradients back to zero
                self.clf.train()
                optimizer.zero_grad()

                # compute the model output and train loss
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.clf(x_cat, x_cont).flatten()
                    intermediate_loss = criterion(logits, targets)
                    train_loss = torch.sum(weights * intermediate_loss) / torch.sum(
                        weights
                    )

                # https://pytorch.org/docs/stable/amp.html
                # https://discuss.huggingface.co/t/why-is-grad-norm-clipping-done-during-training-by-default/1866
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # apply lr scheduler per step
                scheduler.step()

                # add the mini-batch training loss to epoch loss
                loss_in_epoch_train += train_loss.item()

                self._stats_step.append({"train_loss": train_loss.item(), "step": step})

                step += 1

            self.clf.eval()

            loss_in_epoch_val = 0.0
            correct = 0

            with torch.no_grad():
                for x_cat, x_cont, weights, targets in val_loader_finetune:
                    logits = self.clf(x_cat, x_cont)
                    logits = logits.flatten()

                    # get probabilities and round to nearest integer
                    preds = torch.sigmoid(logits).round()
                    correct += (preds == targets).sum().item()

                    # loss calculation.
                    # Criterion contains softmax already.
                    intermediate_loss = criterion(logits, targets)
                    val_loss = torch.sum(weights * intermediate_loss) / torch.sum(
                        weights
                    )
                    loss_in_epoch_val += val_loss.item()

            # loss average over all batches
            train_loss_all = loss_in_epoch_train / len(train_loader_finetune)
            val_loss_all = loss_in_epoch_val / len(val_loader_finetune)
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
                epoch, self.epochs_finetune, train_loss_all, val_loss_all
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
        del train_loader_finetune, val_loader_finetune
        gc.collect()
        torch.cuda.empty_cache()

        # is fitted flag
        self.is_fitted_ = True

        # disable random shuffle once fitted
        self.dl_params.update({"shuffle": False})

        return self

    def predict(self, X: npt.NDArray | pd.DataFrame) -> npt.NDArray:
        """Predict class labels for X.

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
        """Predict class probabilities for X.

        Args:
            X (npt.NDArray | pd.DataFrame): feature matrix
        Returns:
            npt.NDArray: probabilities
        """
        # check if there are attributes with trailing _
        check_is_fitted(self)

        X = check_array(X, accept_sparse=False)
        y = np.zeros(len(X))

        test_loader = self.array_to_dataloader_finetune(X, y)

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
        return np.column_stack((1 - probabilites, probabilites))
