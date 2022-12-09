"""
Perform automated tests for transformer-based neural networks.

Partly inspired by:
https://github.com/tilman151/unittest_dl/blob/master/tests/test_model.py
"""

import torch
import torch.nn as nn

from otc.models.objective import set_seed
from otc.models.tabtransformer import TabTransformer
from otc.models.fttransformer import FTTransformer, Transformer, FeatureTokenizer
from tests import templates


class TestTabTransformer(templates.NeuralNetTestsMixin):
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


class TestFTTransformer(templates.NeuralNetTestsMixin):
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
        self.num_unique_cat = [2]
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

        # https://github.com/Yura52/rtdl/blob/main/rtdl/modules.py

        params_feature_tokenizer = {
            "n_num_features": self.num_features_cont,
            "cat_cardinalities": self.num_unique_cat,
            "d_token": 96,
        }
        feature_tokenizer = FeatureTokenizer(**params_feature_tokenizer)
        params_transformer = {
            "d_token": 96,
            "n_blocks": 3,
            "attention_n_heads": 8,
            "attention_initialization": "kaiming",
            "ffn_activation": "ReGLU",
            "attention_normalization": "LayerNorm",
            "ffn_normalization": "LayerNorm",
            "ffn_dropout": 0.1,
            "ffn_d_hidden": 96 * 2,
            "attention_dropout": 0.1,
            'residual_dropout': 0.1,
            "prenormalization": True,
            "first_prenormalization": False,
            "last_layer_query_idx": None,
            "n_tokens": None,
            "kv_compression_ratio": None,
            "kv_compression_sharing": None,
            "head_activation": "ReLU",
            "head_normalization": "LayerNorm",
            "d_out":1
        }


        transformer = Transformer(**params_transformer)

        self.net = FTTransformer(feature_tokenizer, transformer).to(device)
