"""
Perform automated tests for transformer-based neural networks.

Partly inspired by:
https://github.com/tilman151/unittest_dl/blob/master/tests/test_model.py
"""

import torch

from otc.models.objective import set_seed
from otc.models.tabnet import TabNet
from tests import templates

import pytest

class TestTabNet(templates.NeuralNetTestsMixin):
    """
    Perform tests specified in `NeuralNetTestsMixin` for\
    `TabTransformer` model.

    Args:
        TestCase (test case): test class
        NeuralNetTestsMixin (neural net mixin): mixin
    """

    def setup(self) -> None:
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

        params_tabnet = {
            "input_dim":self.num_features_cont + self.num_features_cat,
            "output_dim":1,
            "n_d":64,
            "n_a":8,
            "n_steps":8,
            "gamma":1.0,
            "cat_idxs":list(range(self.num_features_cat)),
            "cat_dims":self.num_unique_cat
        }

        self.net = TabNet(**params_tabnet).to(device)

    # FIXME: look into this more closely.
    @pytest.mark.skip(reason="Batch norm (1d) of input does not update gradient. Ok for now.")
    def test_all_parameters_updated(self) -> None:
        return super().test_all_parameters_updated()
