"""
Perform automated tests for ReGLU and GeGLU activation functions.

Tests adapted from:
https://github.com/Yura52/rtdl
"""

import torch

from otc.models.activation import GeGLU, ReGLU


class TestActivation:
    """
    Perform automated tests.

    Args:
        unittest (_type_): testcase
    """

    def test_geglu(self) -> None:
        module = GeGLU()
        x = torch.randn(3, 4)
        assert module(x).shape == (3, 2)

    def test_reglu(self) -> None:
        module = ReGLU()
        x = torch.randn(3, 4)
        assert module(x).shape == (3, 2)
