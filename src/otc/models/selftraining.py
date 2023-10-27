"""Implements self-training classifier with a sklearn-like interface.

Based on sklearn implementation.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from catboost import Pool
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils import safe_mask
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted


def _estimator_has(attr: str) -> Callable[[Any], bool]:
    """Check if `self.base_estimator_ `or `self.base_estimator_` has `attr`.

    Args:
    ----
        attr (str): attribute.

    Returns:
    -------
        bool: boolean.
    """
    return lambda self: (
        hasattr(self.base_estimator_, attr)
        if hasattr(self, "base_estimator_")
        else hasattr(self.base_estimator, attr)
    )


class SelfTrainingClassifier(MetaEstimatorMixin, BaseEstimator):
    """Self-training classifier.

    Based on http://dx.doi.org/10.3115/981658.981684.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        base_estimator: BaseEstimator,
        threshold: float = 0.75,
        criterion: Literal["threshold"] = "threshold",
        k_best: int = 10,
        max_iter: int = 10,
        verbose: bool = False,
    ):
        """Initialize a SelfTrainingClassifier.

        Args:
            base_estimator (BaseEstimator): An estimator object implementing
            fit and predict_proba. Invoking the fit method will fit a clone of
            the passed estimator, which will be stored in the base_estimator_
            attribute.
            threshold (float, optional): The decision threshold for use with
            criterion='threshold'. Should be in [0, 1). When using the
            'threshold' criterion, a well calibrated classifier should be used.
            Defaults to 0.75.
            criterion (Literal, optional): The selection criterion used to
            select which labels to add to the training set. If 'threshold',
            pseudo-labels with prediction probabilities above threshold are
            added to the dataset. If 'k_best', the k_best pseudo-labels
            with highest prediction probabilities are added to the dataset.
            When using the `threshold` criterion, a well calibrated classifier
            should be used. Defaults to "threshold".
            k_best (int, optional): The amount of samples to add in each
            iteration. Only used when criterion='k_best'. Defaults to 10
            max_iter (int, optional): Maximum number of iterations allowed.
            Should be greater than or equal to 0. If it is None, the classified
            will continue to predict labels until no new pseudo-labels are
            added, or all unlabeled samples have been labeled. Defaults to 10.
            verbose (bool, optional): Enable verbose output. Defaults to False.
        """
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.criterion = criterion
        self.k_best = k_best
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(
        self, train_set: dict, eval_set: Pool, **kwargs: Any
    ) -> SelfTrainingClassifier:
        """Fit self-training classifier using `X`, `y` as training data.

        Args:
            train_set (dict) dict with training data
            eval_set (Pool): pool of validation data
            **kwargs: keyword arguments

        Raises:
            ValueError: warning for wrong datatype

        Returns:
            SelfTrainingClassifier: self
        """
        # get features, labels etc from trian set
        X = train_set["data"]
        y = train_set["label"].to_numpy()
        weight = train_set["weight"]
        cat_features = train_set["cat_features"]

        self.base_estimator_ = clone(self.base_estimator)

        if y.dtype.kind in ["U", "S"]:
            raise ValueError(
                "y has dtype string. If you wish to predict on "
                "string targets, use dtype object, and use 0"
                " as the label for unlabeled samples."
            )

        has_label = y != 0

        if np.all(has_label):
            warnings.warn("y contains no unlabeled samples", UserWarning)

        if self.criterion == "k_best" and (
            self.k_best > X.shape[0] - np.sum(has_label)
        ):
            warnings.warn(
                "k_best is larger than the amount of unlabeled "
                "samples. All unlabeled samples will be labeled in "
                "the first iteration",
                UserWarning,
            )

        self.transduction_ = np.copy(y)
        self.labeled_iter_ = np.full_like(y, 0)
        self.labeled_iter_[has_label] = 0

        self.n_iter_ = 0

        # pbar = tqdm(total=self.max_iter)

        while not np.all(has_label) and (
            self.max_iter is None or self.n_iter_ < self.max_iter
        ):
            self.n_iter_ += 1

            train_pool = Pool(
                data=X[safe_mask(X, has_label)],
                label=self.transduction_[has_label],
                weight=weight[has_label],
                cat_features=cat_features,
            )

            self.base_estimator_.fit(train_pool, eval_set=eval_set)

            # Predict on the unlabeled samples
            prob = self.base_estimator_.predict_proba(X[safe_mask(X, ~has_label)])
            pred = self.base_estimator_.classes_[np.argmax(prob, axis=1)]
            max_proba = np.max(prob, axis=1)

            # Select new labeled samples
            if self.criterion == "threshold":
                selected = max_proba > self.threshold
            else:
                n_to_select = min(self.k_best, max_proba.shape[0])
                if n_to_select == max_proba.shape[0]:
                    selected = np.ones_like(max_proba, dtype=bool)
                else:
                    # NB these are indices, not a mask
                    selected = np.argpartition(-max_proba, n_to_select)[:n_to_select]

            # Map selected indices into original array
            selected_full = np.nonzero(~has_label)[0][selected]

            # Add newly labeled confident predictions to the dataset
            self.transduction_[selected_full] = pred[selected]
            has_label[selected_full] = True
            self.labeled_iter_[selected_full] = self.n_iter_

            if selected_full.shape[0] == 0:
                # no changed labels
                self.termination_condition_ = "no_change"
                break

            if self.verbose:
                print(
                    f"End of iteration {self.n_iter_},"
                    f" added {selected_full.shape[0]} new labels."
                )
            # pbar.update(1)

        if self.n_iter_ == self.max_iter:
            self.termination_condition_ = "max_iter"
        if np.all(has_label):
            self.termination_condition_ = "all_labeled"

        train_pool = Pool(
            data=X[safe_mask(X, has_label)],
            label=self.transduction_[has_label],
            weight=weight[has_label],
            cat_features=cat_features,
        )

        self.base_estimator_.fit(train_pool, eval_set=eval_set)

        self.classes_ = self.base_estimator_.classes_

        # pbar.close()

        return self

    @available_if(_estimator_has("predict"))
    def predict(self, X: npt.NDArray | pd.DataFrame) -> npt.NDArray:
        """Perform classification on test vectors `X`.

        Args:
            X (npt.NDArray | pd.DataFrame): feature matrix.

        Returns:
            npt.NDArray: Predicted traget values for X.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X,
            accept_sparse=True,
            force_all_finite=False,
            reset=False,
        )
        return self.base_estimator_.predict(X)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X: npt.NDArray | pd.DataFrame) -> npt.NDArra:
        """Predict class probabilities for X.

        Args:
            X (npt.NDArray | pd.DataFrame): feature matrix

        Returns:
            npt.NDArray: probabilities
        """
        check_is_fitted(self)
        X = self._validate_data(
            X,
            accept_sparse=True,
            force_all_finite=False,
            reset=False,
        )
        return self.base_estimator_.predict_proba(X)
