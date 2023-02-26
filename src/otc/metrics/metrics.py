"""
Sklearn implementation of effective spread.

See: https://hagstromer.org/2020/11/23/overestimated-effective-spreads/ for explanation.
"""
import numpy as np
import numpy.typing as npt
from sklearn.utils import check_consistent_length


def effective_spread(
    y_pred: npt.NDArray, trade_price: npt.NDArray, fundamental_value: npt.NDArray
) -> np.float64:
    """
    Calculate the effective spread given by:
    $$
    S_{i,t} = 2 (P_{i,t} - V_{i,t}) D_{i,t}
    $$

    Args:
        y_pred (npt.NDArray): indicator if the trade is a buy or sell
        trade_price (npt.NDArray): trade price
        fundamental_value (npt.NDArray): fundamental value e. g., bid-ask midpoint.
    Returns:
        float: average effective spread
    """
    check_consistent_length(y_pred, trade_price, fundamental_value)
    return np.mean(2 * (trade_price - fundamental_value) * y_pred)
