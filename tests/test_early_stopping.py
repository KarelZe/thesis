"""
Tests for the early stopping implementation.
For early stopping see: https://en.wikipedia.org/wiki/Early_stopping.
"""

import unittest

from src.optim.early_stopping import EarlyStopping


class TestEarlyStopping(unittest.TestCase):
    """
    Perform automated tests for early stopping.
    Args:
        unittest (_type_): testcase
    """

    def test_do_stop_early(self) -> None:
        """
        Tests, if early stopping applies for increasing loss.
        Based on: https://stackoverflow.com/a/71999355/5755604.
        """
        early_stopping = EarlyStopping(patience=2, min_delta=5)

        test_loss = [
            509.13619995,
            497.3125,
            506.17315674,
            497.68960571,
            505.69918823,
            459.78610229,
            480.25592041,
            418.08630371,
            446.42675781,
            372.09902954,
        ]

        for _, val in enumerate(test_loss):
            early_stopping(val)

        self.assertTrue(early_stopping.early_stop)

    def test_do_not_stop_early_decreasing(self) -> None:
        """
        Tests, if early stopping is ommited for decreasing loss.
        As long as loss decreases, training should continue.
        """
        early_stopping = EarlyStopping(patience=2, min_delta=5)

        test_loss = range(650, 600, 1)

        for _, val in enumerate(test_loss):
            early_stopping(val)

        self.assertFalse(early_stopping.early_stop)

    def test_best_loss_below_delta(self) -> None:
        """
        Tests, if best loss is kept for changes below the threshold min_delta.
        Best loss is used for comparsion.
        """
        early_stopping = EarlyStopping(patience=3, min_delta=100)

        test_loss = range(1, 100, 10)

        for _, val in enumerate(test_loss):
            early_stopping(val)

        self.assertEqual(early_stopping.best_loss, 1)

    def test_best_loss_above_delta(self) -> None:
        """
        Tests, if best loss is updated for changes above min_delta.
        Best loss is used for comparsion.
        Example:
        ```
        min_delta = 5
        patience=3
        loss = [100, 110, 120, 130, 90, 150, 160, 170, 70, 190]
        ```
        Best loss is initially 100, but improves to 90,
        and later to 70, as both are above min_delta.
        """
        early_stopping = EarlyStopping(patience=3, min_delta=5)

        test_loss = [100, 110, 120, 130, 90, 150, 160, 170, 70, 190]

        for _, val in enumerate(test_loss):
            early_stopping(val)

        self.assertEqual(early_stopping.best_loss, 70)