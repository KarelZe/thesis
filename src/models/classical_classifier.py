"""
Implements classical trade classification rules.

Both simple rules like quote rule or tick test or hybrids are included.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    _check_sample_weight,
    check_consistent_length,
    check_is_fitted,
)

allowed_func_str = (
    "tick",
    "rev_tick",
    "quote",
    "lr",
    "rev_lr",
    "emo",
    "rev_emo",
    "trade_size",
    "depth",
    "nan",
)

allowed_subsets = ("all", "ex", "best")


class ClassicalClassifier(ClassifierMixin, BaseEstimator):
    """
    ClassicalClassifier implements several trade classification rules.

    Including:
    * Tick test
    * Reverse tick test
    * Quote rule
    * LR algorithm
    * LR algorithm with reverse tick test
    * EMO algorithm
    * EMO algorithm with reverse tick test
    * Trade size rule
    * Depth rule
    * nan

    Args:
        ClassifierMixin (_type_): ClassifierMixin
        BaseEstimator (_type_): Baseestimator
    """

    def __init__(
        self,
        *,
        layers: List[
            Tuple[
                str,
                str,
            ]
        ],
        random_state: Optional[float],
    ):
        """
        Initialize a ClassicalClassifier.

        Args:
            layers (List[Tuple[str, str]]): Layers of classical rule. Up to 4 possible.
            If fewer layers are needed use ("nan","ex").
            random_state (float, optional): random seed. Defaults to None.
        """
        self.layers = layers
        self.random_state = random_state

    def _tick(self, subset: Literal["all", "ex"]) -> npt.NDArray:
        """
        Classify a trade as a buy (sell) if its trade price is above (below)\
        the closest different price of a previous trade.

        Args:
            subset (Literal[&quot;all&quot;, &quot;ex&quot;]): subset i. e.,
            'all' or 'ex'.

        Returns:
            npt.NDArray: result of tick rule. Can be np.NaN.
        """
        return np.where(
            self.X_["TRADE_PRICE"] > self.X_[f"price_{subset}_lag"],
            1,
            np.where(
                self.X_["TRADE_PRICE"] < self.X_[f"price_{subset}_lag"], -1, np.nan
            ),
        )

    def _rev_tick(self, subset: Literal["all", "ex"]) -> npt.NDArray:
        """
        Classify a trade as a sell (buy) if its trade price is below (above)\
        the closest different price of a subsequent trade.

        Args:
            subset (Literal[&quot;all&quot;, &quot;ex&quot;]): subset i. e.,
            'all' or 'ex'.

        Returns:
            npt.NDArray: result of reverse tick rule. Can be np.NaN.
        """
        return np.where(
            self.X_[f"price_{subset}_lead"] > self.X_["TRADE_PRICE"],
            -1,
            np.where(
                self.X_[f"price_{subset}_lead"] < self.X_["TRADE_PRICE"], 1, np.nan
            ),
        )

    def _quote(self, subset: Literal["best", "ex"]) -> npt.NDArray:
        """
        Classify a trade as a buy (sell) if its trade price is above (below)\
        the midpoint of the bid and ask spread. Trades executed at the\
        midspread are not classified.

        Args:
            subset (Literal[&quot;ex&quot;, &quot;best&quot;]): subset i. e.,
            'ex' or 'best'.

        Returns:
            npt.NDArray: result of quote rule. Can be np.NaN.
        """
        mid = 0.5 * (self.X_[f"ask_{subset}"] + self.X_[f"bid_{subset}"])
        return np.where(
            self.X_["TRADE_PRICE"] > mid,
            1,
            np.where(self.X_["TRADE_PRICE"] < mid, -1, np.nan),
        )

    def _lr(self, subset: Literal["best", "ex"]) -> npt.NDArray:
        """
        Classify a trade as a buy (sell) if its price is above (below) the\
        midpoint (quote rule), and use the tick test to classify midspread\
        trades.

        Adapted from Lee and Ready (1991).
        Args:
            subset (Literal[&quot;ex&quot;, &quot;best&quot;]): subset i. e.,
            'ex' or 'best'.

        Returns:
            npt.ndarray: result of the lee and ready algorithm with tick rule.
            Can be np.NaN.
        """
        q_r = self._quote(subset)
        return np.where(~np.isnan(q_r), q_r, self._tick("ex"))

    def _rev_lr(self, subset: Literal["best", "ex"]) -> npt.NDArray:
        """
        Classify a trade as a buy (sell) if its price is above (below) the\
        midpoint (quote rule), and use the reverse tick test to classify\
        midspread trades.

        Adapted from Lee and Ready (1991).
        Args:
            subset (Literal[&quot;ex&quot;, &quot;best&quot;]): subset i. e.,
            'ex' or 'best'.

        Returns:
            npt.NDArray: result of the lee and ready algorithm with reverse tick
            rule. Can be np.NaN.
        """
        q_r = self._quote(subset)
        return np.where(~np.isnan(q_r), q_r, self._rev_tick("ex"))

    def _emo(self, subset: Literal["best", "ex"]) -> npt.NDArray:
        """
        Classify a trade as a buy (sell) if the trade takes place at the ask\
        (bid) quote, and use the tick test to classify all other trades.

        Adapted from Ellis et al. (2000).
        Args:
            subset (Literal[&quot;ex&quot;, &quot;best&quot;]): subset i. e.,
            'ex' or 'best'.

        Returns:
            npt.NDArray: result of the emo algorithm with tick rule. Can be
            np.NaN.
        """
        at_ask = self.X_["TRADE_PRICE"] == self.X_[f"ask_{subset}"]
        at_bid = self.X_["TRADE_PRICE"] == self.X_[f"bid_{subset}"]
        at_ask_or_bid = at_ask ^ at_bid
        return np.where(at_ask_or_bid, self._quote(subset), self._tick("ex"))

    def _rev_emo(self, subset: Literal["best", "ex"]) -> npt.NDArray:
        """
        Classify a trade as a buy (sell) if the trade takes place at the ask\
        (bid) quote, and use the reverse tick test to classify all other\
        trades.

        Adapted from Ellis et al. (2000).
        Args:
            subset (Literal[&quot;ex&quot;, &quot;best&quot;]): subset
            i. e., 'ex' or 'best'.

        Returns:
            npt.NDArray: result of the emo algorithm with reverse tick rule.
            Can be np.NaN.
        """
        at_ask = self.X_["TRADE_PRICE"] == self.X_[f"ask_{subset}"]
        at_bid = self.X_["TRADE_PRICE"] == self.X_[f"bid_{subset}"]
        at_ask_or_bid = at_ask ^ at_bid
        return np.where(at_ask_or_bid, self._quote(subset), self._rev_tick("ex"))

    # pylint: disable=W0613
    def _trade_size(self, *args: Any) -> npt.NDArray:
        """
        Classify a trade as a buy (sell) the trade size matches exactly either\
        the bid (ask) quote size.

        Adapted from Grauer et al. (2022).

        Returns:
            npt.NDArray: result of the trade size rule. Can be np.NaN.
        """
        bid_eq_ask = self.X_["ask_size_ex"] == self.X_["bid_size_ex"]

        ts_eq_bid = (self.X_["TRADE_SIZE"] == self.X_["bid_size_ex"]) & -bid_eq_ask
        ts_eq_ask = (self.X_["TRADE_SIZE"] == self.X_["ask_size_ex"]) & -bid_eq_ask

        return np.where(ts_eq_bid, 1, np.where(ts_eq_ask, -1, np.nan))

    # pylint: disable=W0613
    def _depth(self, *args: Any) -> npt.NDArray:
        """
        Classify midspread trades as buy (sell), if the ask size (bid size)\
        exceeds the bid size (ask size).

        Adapted from (Grauer et al., 2022).

        Returns:
            npt.NDArray: result of the trade size rule. Can be np.NaN.
        """
        return np.where(
            self.X_["ask_size_ex"] > self.X_["bid_size_ex"],
            1,
            np.where(
                self.X_["ask_size_ex"] < self.X_["bid_size_ex"],
                -1,
                np.nan,
            ),
        )

    # pylint: disable=W0613
    def _nan(self, *args: Any) -> npt.NDArray:
        """
        Classify nothing. Fast forward results from previous classifier.

        Returns:
            npt.NDArray: result of the trade size rule. Can be np.NaN.
        """
        return np.full(shape=(self.X_.shape[0],), fill_value=np.nan)

    # pylint: disable=C0103
    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: npt.NDArray = None
    ) -> ClassicalClassifier:
        """
        Fit the classifier.

        Args:
            X (pd.DataFrame): features
            y (pd.Series): ground truth (ignored)
            sample_weight (npt.NDArray, optional): Sample weights. Defaults to None.

        Raises:
            ValueError: Unknown subset e. g., 'ise'
            ValueError: Unknown function string e. g., 'lee-ready'
            ValueError: Multi output is not supported.

        Returns:
            TCRClassifier: Instance itself.
        """
        _check_sample_weight(sample_weight, X)

        funcs = (  # type: ignore
            self._tick,
            self._rev_tick,
            self._quote,
            self._lr,
            self._rev_lr,
            self._emo,
            self._rev_emo,
            self._trade_size,
            self._depth,
            self._nan,
        )
        # pylint: disable=W0201
        self.func_mapping_ = dict(zip(allowed_func_str, funcs))

        for func_str, subset in self.layers:
            if subset not in allowed_subsets:
                raise ValueError(
                    f"Unknown subset: {subset}, expected one of {allowed_subsets}."
                )
            if func_str not in allowed_func_str:
                raise ValueError(
                    (
                        f"Unknown function string: {func_str},"
                        f"expected one of {allowed_func_str}."
                    )
                )
        # pylint: disable=W0201, C0103
        self.layers_ = self.layers

        # pylint: disable=W0201, C0103
        self.sparse_output_ = sp.issparse(y)

        if not self.sparse_output_:
            y = np.asarray(y)
            y = np.atleast_1d(y)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        # pylint: disable=W0201, C0103
        self.n_outputs_ = y.shape[1]
        if self.n_outputs_ > 1:
            raise ValueError("Multi output not supported.")

        check_consistent_length(X, y)

        return self

    # pylint: disable=C0103
    def predict(self, X: pd.DataFrame) -> npt.NDArray:
        """
        Perform classification on test vectors X.

        Args:
            X (pd.DataFrame): Feature matrix.

        Raises:
            ValueError: X must be pd.DataFrame, as labels are required.

        Returns:
            npt.NDArray: Predicted traget values for X.
        """
        # check if there are attributes with trailing _
        check_is_fitted(self)

        rs = check_random_state(self.random_state)

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be pd.DataFrame, as labels are required.")
        mapping_cols = {"BEST_ASK": "ask_best", "BEST_BID": "bid_best"}
        # pylint: disable=W0201, C0103
        self.X_ = X.rename(columns=mapping_cols)

        pred = np.full(shape=(X.shape[0],), fill_value=np.nan)

        for func_str, subset in self.layers_:
            func = self.func_mapping_[func_str]
            pred = np.where(
                np.isnan(pred),
                func(subset),  # type: ignore
                pred,
            )

        if self.n_outputs_ == 1:
            pred = np.ravel(pred)

        # fill NaNs randomly with -1 and 1
        mask = np.isnan(pred)
        pred[mask] = rs.choice([-1, 1], pred.shape)[mask]
        return pred