"""
Sklearn implementation of effective spread.

See: https://hagstromer.org/2020/11/23/overestimated-effective-spreads/ for explanation.
"""
import numpy as np
import numpy.typing as npt
from sklearn.utils import check_consistent_length

from typing import Literal


def effective_spread(
    y_pred: npt.NDArray,
    trade_price: npt.NDArray,
    fundamental_value: npt.NDArray,
    mode: Literal["nominal", "relative"] = "nominal",
) -> np.float64:
    """
    Depending on mode, calculate the nominal effective spread given by:
    $$
    S_{i,t} = 2 (P_{i,t} - V_{i,t}) D_{i,t}
    $$

    Calculate the relative effective spread given by:
    $$
    {PS}_{i,t} = S_{i,t} / V_{i,t}.
    $$

    Args:
        y_pred (npt.NDArray): indicator if the trade is a buy or sell
        trade_price (npt.NDArray): trade price
        fundamental_value (npt.NDArray): fundamental value e. g., bid-ask midpoint.
        mode (Literal["nominal", "relative"], optional): "nominal" or "relative". Defaults to "nominal".
    Returns:
        float: average effective spread
    """
    check_consistent_length(y_pred, trade_price, fundamental_value)
    s = 2 * (trade_price - fundamental_value) * y_pred
    if mode == "nominal":
        return np.mean(s)
    else:
        # nan when div by zero https://stackoverflow.com/a/54364060/5755604
        ps = np.empty(y_pred.shape)
        np.divide(s, fundamental_value, out=ps, where=fundamental_value != 0)
        return np.mean(ps)
