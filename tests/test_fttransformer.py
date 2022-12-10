"""
Perform automated tests for transformer-based neural networks.

Partly inspired by:
https://github.com/tilman151/unittest_dl/blob/master/tests/test_model.py
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from otc.models.activation import ReGLU
from otc.models.fttransformer import (
    CategoricalFeatureTokenizer,
    CLSToken,
    FeatureTokenizer,
    FTTransformer,
    MultiheadAttention,
    NumericalFeatureTokenizer,
    Transformer,
)
from otc.models.objective import set_seed
from tests import templates


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
        self.cat_cardinalities = [2]
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

        params_feature_tokenizer: Dict[str, Any] = {
            "n_num_features": self.num_features_cont,
            "cat_cardinalities": self.cat_cardinalities,
            "d_token": 96,
        }
        feature_tokenizer = FeatureTokenizer(**params_feature_tokenizer)
        params_transformer = {
            "d_token": 96,
            "n_blocks": 3,
            "attention_n_heads": 8,
            "attention_initialization": "kaiming",
            "ffn_activation": ReGLU,
            "attention_normalization": nn.LayerNorm,
            "ffn_normalization": nn.LayerNorm,
            "ffn_dropout": 0.1,
            "ffn_d_hidden": 96 * 2,
            "attention_dropout": 0.1,
            "residual_dropout": 0.1,
            "prenormalization": True,
            "first_prenormalization": False,
            "last_layer_query_idx": None,
            "n_tokens": None,
            "kv_compression_ratio": None,
            "kv_compression_sharing": None,
            "head_activation": nn.ReLU,
            "head_normalization": nn.LayerNorm,
            "d_out": 1,
        }

        transformer = Transformer(**params_transformer)

        self.net = FTTransformer(feature_tokenizer, transformer).to(device)

    def test_numerical_feature_tokenizer(self) -> None:
        """
        Test numerical feature tokenizer.

        Adapted from: https://github.com/Yura52/rtdl/.
        """
        x = torch.randn(4, 2)
        n_objects, n_features = x.shape
        d_token = 3
        tokenizer = NumericalFeatureTokenizer(n_features, d_token, True, "uniform")
        tokens = tokenizer(x)
        assert tokens.shape == (n_objects, n_features, d_token)

    def test_categorical_feature_tokenizer(self) -> None:
        """
        Test categorical feature tokenizer.

        Adapted from: https://github.com/Yura52/rtdl/.
        """
        # the input must contain integers. For example, if the first feature can
        # take 3 distinct values, then its cardinality is 3 and the first column
        # must contain values from the range `[0, 1, 2]`.
        cardinalities = [3, 10]
        x = torch.tensor([[0, 5], [1, 7], [0, 2], [2, 4]])
        n_objects, n_features = x.shape
        d_token = 3
        tokenizer = CategoricalFeatureTokenizer(cardinalities, d_token, True, "uniform")
        tokens = tokenizer(x)
        assert tokens.shape == (n_objects, n_features, d_token)

    def test_feature_tokenizer(self) -> None:
        """
        Test feature tokenizer.

        Adapted from: https://github.com/Yura52/rtdl/.
        """
        n_objects = 4
        num_continous = 3
        num_categorical = 2
        d_token = 7
        x_num = torch.randn(n_objects, num_continous)
        x_cat = torch.tensor([[0, 1], [1, 0], [0, 2], [1, 1]])
        # [2, 3] reflects cardinalities fr
        tokenizer = FeatureTokenizer(num_continous, [2, 3], d_token)
        tokens = tokenizer(x_num, x_cat)
        assert tokens.shape == (n_objects, num_continous + num_categorical, d_token)

    def test_cls_token(self) -> None:
        """
        Test [CLS] token.

        Adapted from: https://github.com/Yura52/rtdl/.
        """
        batch_size = 2
        n_tokens = 3
        d_token = 4
        cls_token = CLSToken(d_token, "uniform")
        x = torch.randn(batch_size, n_tokens, d_token)
        x = cls_token(x)
        assert x.shape == (batch_size, n_tokens + 1, d_token)
        assert (x[:, -1, :] == cls_token.expand(len(x))).all()

    def test_multihead_attention(self) -> None:
        """
        Test multi-headed attention.

        Adapted from: https://github.com/Yura52/rtdl/.
        """
        n_objects, n_tokens, d_token = 2, 3, 12
        n_heads = 6
        a = torch.randn(n_objects, n_tokens, d_token)
        b = torch.randn(n_objects, n_tokens * 2, d_token)
        module = MultiheadAttention(
            d_token=d_token,
            n_heads=n_heads,
            dropout=0.2,
            bias=True,
            initialization="kaiming",
        )
        # self-attention
        x, attention_stats = module(a, a, None, None)
        assert x.shape == a.shape
        assert attention_stats["attention_probs"].shape == (
            n_objects * n_heads,
            n_tokens,
            n_tokens,
        )
        assert attention_stats["attention_logits"].shape == (
            n_objects * n_heads,
            n_tokens,
            n_tokens,
        )
        # cross-attention
        assert module(a, b, None, None)
        # Linformer self-attention with the 'headwise' sharing policy
        k_compression = torch.nn.Linear(n_tokens, n_tokens // 4)
        v_compression = torch.nn.Linear(n_tokens, n_tokens // 4)
        assert module(a, a, k_compression, v_compression)
        # Linformer self-attention with the 'key-value' sharing policy
        kv_compression = torch.nn.Linear(n_tokens, n_tokens // 4)
        assert module(a, a, kv_compression, kv_compression)
