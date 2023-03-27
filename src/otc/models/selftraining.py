"""
Implements self-training classifier with a sklearn-like interface.

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
    """
    Check if `self.base_estimator_ `or `self.base_estimator_` has `attr`.

    Args:
        attr (str): attribute.

    Returns:
        bool: boolean.
    """
    return lambda self: (
        hasattr(self.base_estimator_, attr)
        if hasattr(self, "base_estimator_")
        else hasattr(self.base_estimator, attr)
    )


class SelfTrainingClassifier(MetaEstimatorMixin, BaseEstimator):
    """Self-training classifier.

    This :term:`metaestimator` allows a given supervised classifier to function as a
    semi-supervised classifier, allowing it to learn from unlabeled data. It
    does this by iteratively predicting pseudo-labels for the unlabeled data
    and adding them to the training set.

    The classifier will continue iterating until either max_iter is reached, or
    no pseudo-labels were added to the training set in the previous iteration.

    Read more in the :ref:`User Guide <self_training>`.

    Parameters
    ----------
    base_estimator : estimator object
        An estimator object implementing `fit` and `predict_proba`.
        Invoking the `fit` method will fit a clone of the passed estimator,
        which will be stored in the `base_estimator_` attribute.

    threshold : float, default=0.75
        The decision threshold for use with `criterion='threshold'`.
        Should be in [0, 1). When using the `'threshold'` criterion, a
        :ref:`well calibrated classifier <calibration>` should be used.

    criterion : {'threshold', 'k_best'}, default='threshold'
        The selection criterion used to select which labels to add to the
        training set. If `'threshold'`, pseudo-labels with prediction
        probabilities above `threshold` are added to the dataset. If `'k_best'`,
        the `k_best` pseudo-labels with highest prediction probabilities are
        added to the dataset. When using the 'threshold' criterion, a
        :ref:`well calibrated classifier <calibration>` should be used.

    k_best : int, default=10
        The amount of samples to add in each iteration. Only used when
        `criterion='k_best'`.

    max_iter : int or None, default=10
        Maximum number of iterations allowed. Should be greater than or equal
        to 0. If it is `None`, the classifier will continue to predict labels
        until no new pseudo-labels are added, or all unlabeled samples have
        been labeled.

    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    base_estimator_ : estimator object
        The fitted estimator.

    classes_ : ndarray or list of ndarray of shape (n_classes,)
        Class labels for each output. (Taken from the trained
        `base_estimator_`).

    transduction_ : ndarray of shape (n_samples,)
        The labels used for the final fit of the classifier, including
        pseudo-labels added during fit.

    labeled_iter_ : ndarray of shape (n_samples,)
        The iteration in which each sample was labeled. When a sample has
        iteration 0, the sample was already labeled in the original dataset.
        When a sample has iteration -1, the sample was not labeled in any
        iteration.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        The number of rounds of self-training, that is the number of times the
        base estimator is fitted on relabeled variants of the training set.

    termination_condition_ : {'max_iter', 'no_change', 'all_labeled'}
        The reason that fitting was stopped.

        - `'max_iter'`: `n_iter_` reached `max_iter`.
        - `'no_change'`: no new labels were predicted.
        - `'all_labeled'`: all unlabeled samples were labeled before `max_iter`
          was reached.

    See Also
    --------
    LabelPropagation : Label propagation classifier.
    LabelSpreading : Label spreading model for semi-supervised learning.

    References
    ----------
    :doi:`David Yarowsky. 1995. Unsupervised word sense disambiguation rivaling
    supervised methods. In Proceedings of the 33rd annual meeting on
    Association for Computational Linguistics (ACL '95). Association for
    Computational Linguistics, Stroudsburg, PA, USA, 189-196.
    <10.3115/981658.981684>`
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
        """
        Initialize a SelfTrainingClassifier.

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
            When using the ‘threshold’ criterion, a well calibrated classifier
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

    def fit(  # noqa: C901
        self, pool: Pool, eval_set: Pool, **kwargs: Any
    ) -> SelfTrainingClassifier:
        """
        Fit self-training classifier using `X`, `y` as training data.

        Args:
            pool (Pool): pool of training data
            eval_set (Pool): pool of validation data

        Raises:
            ValueError: warning for wrong datatype

        Returns:
            SelfTrainingClassifier: self
        """
        # get features, labels etc. from train pool
        X = pool.get_features()
        y = pool.get_label()
        weights = np.array(pool.get_weight())
        cat_features = pool.get_cat_feature_indices()

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
                weight=weights[has_label],
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
            weight=weights[has_label],
            cat_features=cat_features,
        )

        self.base_estimator_.fit(train_pool, eval_set=eval_set)

        self.classes_ = self.base_estimator_.classes_

        # pbar.close()

        return self

    @available_if(_estimator_has("predict"))
    def predict(self, X: npt.NDArray | pd.DataFrame) -> npt.NDArray:
        """
        Perform classification on test vectors `X`.

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
        """
        Predict class probabilities for X.

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
