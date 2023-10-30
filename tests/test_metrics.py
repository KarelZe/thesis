"""Tests for Metrics."""

import numpy as np

from otc.metrics.metrics import effective_spread


class TestMetrics:
    """Perform automated tests for objectives.

    Args:
    ----
        metaclass (_type_, optional): parent. Defaults to abc.ABCMeta.
    """

    def test_effective_spread(self) -> None:
        """Test if effective spread returns a valid value.

        Value may not be NaN.
        """
        rng = np.random.default_rng(seed=7)
        y_pred = rng.choice([-1, 1], size=(10))
        trade_price = rng.random(10) * 100
        fundamental_value = rng.random(10) * 100

        e_s = effective_spread(y_pred, trade_price, fundamental_value)

        assert np.isclose(e_s, 0.86, atol=1e-02, equal_nan=False)
