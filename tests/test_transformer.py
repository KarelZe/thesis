"""
Perform automated tests for transformer-based neural networks.

Partly inspired by:
https://github.com/tilman151/unittest_dl/blob/master/tests/test_model.py
"""

import unittest

import torch
import torch.nn as nn

from otc.models.objective import set_seed
from otc.models.tabtransformer import TabTransformer
from tests import templates


class TestTabTransformer(unittest.TestCase, templates.NeuralNetTestsMixin):
    """
    Perform tests specified in `NeuralNetTestsMixin` for\
    `TabTransformer` model.

    Args:
        TestCase (test case): test class
        NeuralNetTestsMixin (neural net mixin): mixin
    """

    def setUp(self) -> None:
        """
        Set up basic network and data.

        Prepares inputs and expected outputs for testing.
        """
        self.num_features_cont = 5
        self.num_features_cat = 1
        self.num_unique_cat = tuple([2])
        self.batch_size = 64

        set_seed()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.x_cat = torch.randint(0, 1, (self.batch_size, self.num_features_cat)).to(
            device
        )
        self.x_cont = (
            torch.randn(self.batch_size, self.num_features_cont).float().to(device)
        )
        self.expected_outputs = (
            torch.randint(0, 1, (self.batch_size, 1)).float().to(device)
        )

        self.net = TabTransformer(
            categories=self.num_unique_cat,
            num_continuous=self.num_features_cont,
            dim_out=1,
            mlp_act=nn.ReLU(),
            dim=32,
            depth=2,
            heads=6,
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4, 2),
        ).to(device)
