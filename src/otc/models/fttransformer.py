"""
Implementation of FT-Transformer model.

Adapted from:
https://github.com/Yura52/rtdl/
"""
from __future__ import annotations

import enum
import math
from typing import Any, Callable, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from otc.models.activation import GeGLU, ReGLU

_INTERNAL_ERROR_MESSAGE = "Internal error. Please, open an issue."


def _is_glu_activation(activation: Callable[..., nn.Module]) -> bool:
    """
    Check if the activation is a GLU variant i. e., ReGLU and GeGLU.

    See: https://arxiv.org/abs/2002.05202 for details.

    Args:
        activation (Callable[..., nn.Module]): activation function

    Returns:
        bool: truth value.
    """
    return (
        isinstance(activation, str)
        and activation.endswith("GLU")
        or activation in [ReGLU, GeGLU]
    )


def _all_or_none(values: list[Any]) -> bool:
    """
    Check if all values are None or all values are not None.

    Args:
        values (List[Any]): List with values

    Returns:
        bool: truth value.
    """
    return all(x is None for x in values) or all(x is not None for x in values)


class CLSHead(nn.Module):
    """
    2 Layer MLP projection head.
    """

    def __init__(self, *, d_in: int, d_hidden: int):
        """
        Initialize the module.
        """
        super().__init__()
        self.first = nn.Linear(d_in, d_hidden)
        self.out = nn.Linear(d_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        x = x[:, 1:]
        x = self.out(F.relu(self.first(x))).squeeze(2)
        return x


class _TokenInitialization(enum.Enum):
    """
    Implementation of TokenInitialization scheme.
    """

    UNIFORM = "uniform"
    NORMAL = "normal"

    @classmethod
    def from_str(cls, initialization: str) -> _TokenInitialization:
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f"initialization must be one of {valid_values}")

    def apply(self, x: torch.Tensor, d: int) -> None:
        """
        Initiliaze the tensor with specific initialization scheme.

        Args:
            x (torch.Tensor): input tensor
            d (int): degree of quare root
        """
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)


class NumericalFeatureTokenizer(nn.Module):
    """
    Transforms continuous features to tokens (embeddings).

    For one feature, the transformation consists of two steps:
    * the feature is multiplied by a trainable vector
    * another trainable vector is added

    Note that each feature has its separate pair of trainable vectors, i.e. the vectors
    are not shared between features.
    """

    def __init__(
        self,
        n_features: int,
        d_token: int,
        bias: bool,
        initialization: str,
    ) -> None:
        """
        Initialize the module.

        Args:
            n_features (int): number of continuous (scalar) features
            d_token (int): size of one token
            bias (bool): if `False`, then the transformation will include only
            multiplication.
            **Warning**: `bias=False` leads to significantly worse results for
            Transformer-like (token-based) architectures.
            initialization (str): initialization policy for parameters. Must be one of
            `['uniform', 'normal']`. Let `s = d ** -0.5`. Then, the corresponding
            distributions are `Uniform(-s, s)` and `Normal(0, s)`. In the FTTransformer
            paper, the 'uniform' initialization was used.
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(torch.Tensor(n_features, d_token))
        self.bias = nn.Parameter(torch.Tensor(n_features, d_token)) if bias else None
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        """
        Calculate the number of tokens.

        Returns:
            int: no. of tokens.
        """
        return len(self.weight)

    @property
    def d_token(self) -> int:
        """
        Calculate the dimension of the token.

        Returns:
            int: dimension of token.
        """
        return self.weight.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass.

        Multiply the input tensor by the weight and add the bias.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class CategoricalFeatureTokenizer(nn.Module):
    """
    Transforms categorical features to tokens (embeddings).

    The module efficiently implements a collection of `torch.nn.Embedding` (with
    optional biases).
    """

    category_offsets: torch.Tensor

    def __init__(
        self,
        cardinalities: list[int],
        d_token: int,
        bias: bool,
        initialization: str,
    ) -> None:
        """
        Initialize the module.

        Args:
            cardinalities (list[int]): the number of distinct values for each feature.
            For example, `cardinalities=[3, 4]` describes two features: the first one
            can take values in the range `[0, 1, 2]` and the second one can take values
            in the range `[0, 1, 2, 3]`.
            d_token (int): the size of one token.
            bias (bool): if `True`, for each feature, a trainable vector is added to the
            embedding regardless of feature value. The bias vectors are not shared
            between features.
            initialization (str): initialization policy for parameters. Must be one of
            `['uniform', 'normal']`. Let `s = d ** -0.5`. Then, the corresponding
            distributions are `Uniform(-s, s)` and `Normal(0, s)`. In the FTTransformer
            paper, the 'uniform' initialization was used.
        """
        super().__init__()
        assert cardinalities, "cardinalities must be non-empty"
        assert d_token > 0, "d_token must be positive"
        initialization_ = _TokenInitialization.from_str(initialization)

        category_offsets = torch.tensor([0] + cardinalities[:-1]).cumsum(0)
        self.register_buffer("category_offsets", category_offsets, persistent=False)
        self.embeddings = nn.Embedding(sum(cardinalities), d_token)
        self.bias = (
            nn.Parameter(torch.Tensor(len(cardinalities), d_token)) if bias else None
        )

        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        """
        Calculate the number of tokens.

        Returns:
            int: number of tokens.
        """
        return len(self.category_offsets)

    @property
    def d_token(self) -> int:
        """
        Calculate the dimension of the token.

        Returns:
            int: dimension of token.
        """
        return self.embeddings.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass.

        Calculate embedding from input vector and category offset and add bias.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        x = self.embeddings(x + self.category_offsets[None])
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class FeatureTokenizer(nn.Module):
    """
    Combines `NumericalFeatureTokenizer` and `CategoricalFeatureTokenizer`.

    The "Feature Tokenizer" module from FTTransformer paper. The module transforms
    continuous and categorical features to tokens (embeddings).
    """

    def __init__(
        self,
        num_continous: int,
        cat_cardinalities: list[int],
        d_token: int,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the module.

        Args:
            num_continous (int): number of continuous features. Pass `0` if there
                are no numerical features.
            cat_cardinalities (list[int]): number of unique values for each feature. See
                `CategoricalFeatureTokenizer` for details. Pass an empty list if there
                are no categorical features.
            d_token (int): size of one token.
        """
        super().__init__()
        assert num_continous >= 0, "n_num_features must be non-negative"
        assert (
            num_continous or cat_cardinalities
        ), "at least one of n_num_features or cat_cardinalities must be positive"
        "and non-empty"
        self.initialization = "uniform"
        self.num_tokenizer = (
            NumericalFeatureTokenizer(
                n_features=num_continous,
                d_token=d_token,
                bias=True,
                initialization=self.initialization,
            )
            if num_continous
            else None
        )
        self.cat_tokenizer = (
            CategoricalFeatureTokenizer(
                cat_cardinalities, d_token, True, self.initialization
            )
            if cat_cardinalities
            else None
        )

    @property
    def n_tokens(self) -> int:
        """
        Calculate the number of tokens.

        Returns:
            int: number of tokens.
        """
        return sum(
            x.n_tokens
            for x in [self.num_tokenizer, self.cat_tokenizer]
            if x is not None
        )

    @property
    def d_token(self) -> int:
        """
        Calculate the dimension of the token.

        Returns:
            int: dimension of token.
        """
        return (
            self.cat_tokenizer.d_token  # type: ignore
            if self.num_tokenizer is None
            else self.num_tokenizer.d_token
        )

    def forward(
        self, x_num: torch.Tensor | None, x_cat: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Perform the forward pass.

        Args:
            x_num (torch.Tensor | None): continuous features. Must be presented
            if `n_num_features > 0` was passed to the constructor.
            x_cat (torch.Tensor | None): categorical features
            (see `CategoricalFeatureTokenizer.forward` for details). Must be presented
            if non-empty `cat_cardinalities` was passed to the constructor.
        Returns:
            torch.Tensor: tokens.
        """
        assert (
            x_num is not None or x_cat is not None
        ), "At least one of x_num and x_cat must be presented"
        assert _all_or_none(
            [self.num_tokenizer, x_num]
        ), "If self.num_tokenizer is (not) None, then x_num must (not) be None"
        assert _all_or_none(
            [self.cat_tokenizer, x_cat]
        ), "If self.cat_tokenizer is (not) None, then x_cat must (not) be None"
        x = []
        if self.num_tokenizer is not None:
            x.append(self.num_tokenizer(x_num))
        if self.cat_tokenizer is not None:
            x.append(self.cat_tokenizer(x_cat))
        return x[0] if len(x) == 1 else torch.cat(x, dim=1)


class CLSToken(nn.Module):
    """
    [CLS]-token for BERT-like inference.

    To learn about the [CLS]-based inference, see [devlin2018bert]. When used as a
    module, the [CLS]-token is appended **to the end** of each item in the batch.

    References:
        * [devlin2018bert] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
         "BERT: Pre-training of Deep Bidirectional Transformers for Language
         Understanding" 2018
    """

    def __init__(self, d_token: int, initialization: str) -> None:
        """
        Initialize the module.

        Args:
            d_token (int): size of token.
            initialization (str): initialization policy for parameters. Must be one of
            `['uniform', 'normal']`. Let `s = d ** -0.5`. Then, the corresponding
            distributions are `Uniform(-s, s)` and `Normal(0, s)`. In the FTTransformer
            paper, the 'uniform' initialization was used.
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(torch.Tensor(d_token))
        initialization_.apply(self.weight, d_token)

    def expand(self, *leading_dimensions: int) -> torch.Tensor:
        """
        Expand (repeat) the underlying [CLS]-token to a tensor with the given\
        leading dimensions.

        A possible use case is building a batch of [CLS]-tokens. See `CLSToken` for
        examples of usage.
        Note:
            Under the hood, the `torch.torch.Tensor.expand` method is applied to the
            underlying `weight` parameter, so gradients will be propagated as
            expected.
        Args:
            leading_dimensions: the additional new dimensions
        Returns:
            torch.Tensor: tensor with shape [*leading_dimensions, len(self.weight)]
        """
        if not leading_dimensions:
            return self.weight
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Append self **to the end** of each item in the batch (see `CLSToken`).

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        return torch.cat([x, self.expand(len(x), 1)], dim=1)


class MultiheadAttention(nn.Module):
    """
    Multihead Attention (self-/cross-) with optional 'linear' attention.

    To learn more about Multihead Attention, see [devlin2018bert].

    See the implementation  of `Transformer` and the examples below to learn how to use
    the compression technique from [wang2020linformer] to speed up the module when the
    number of tokens is large.
    References:
        * [devlin2018bert] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
        "BERT: Pre-training
        of Deep Bidirectional Transformers for Language Understanding" 2018
        * [wang2020linformer] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao
        Ma "Linformer:
        Self-Attention with Linear Complexity", 2020
    """

    def __init__(
        self,
        *,
        d_token: int,
        n_heads: int,
        dropout: float,
        bias: bool,
        initialization: str,
    ) -> None:
        """
                Initialize the module.

        Args:
            d_token (int): token size. Must be a multiple of `n_heads`.
            n_heads (int): the number of heads. If greater than 1, then the module will
            have an addition output layer (so called "mixing" layer).
            dropout (float): dropout rate for the attention map. The dropout is applied
            to *probabilities* and do not affect logits.
            bias (bool): if `True`, then input (and output, if presented) layers also
            have bias.
            initialization (str): initialization for input projection layers. Must be
            one of `['kaiming', 'xavier']`. `kaiming` is a reasonable default choice.
        """
        super().__init__()
        if n_heads > 1:
            assert d_token % n_heads == 0, "d_token must be a multiple of n_heads"
        assert initialization in ["kaiming", "xavier"]

        self.W_q = nn.Linear(d_token, d_token, bias)
        self.W_k = nn.Linear(d_token, d_token, bias)
        self.W_v = nn.Linear(d_token, d_token, bias)
        self.W_out = nn.Linear(d_token, d_token, bias) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        self.attention_probs = None
        self.attention_probs_grad = None
        self.attn_gradients = None
        self.attn = None

        for m in [self.W_q, self.W_k, self.W_v]:
            # the "xavier" branch tries to follow torch.nn.MultiheadAttention;
            # the second condition checks if W_v plays the role of W_out; the latter one
            # is initialized with Kaiming in torch
            if initialization == "xavier" and (
                m is not self.W_v or self.W_out is not None
            ):
                # gain is needed since W_qkv is represented with 3 separate layers (it
                # implies different fan_out)
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def save_attn(self, attn: torch) -> None:
        """
        Save attention probabilities tensor.

        Args:
            attn (torch): attention probabilities.
        """
        self.attn = attn

    def get_attn(self) -> torch.Tensor:
        """
        Get attention probabilites tensor.
        """
        return self.attn

    def save_attn_gradients(self, attn_gradients: torch.Tensor) -> None:
        """
        Save attention gradients tensor.

        Args:
            attn_gradients (torch.Tensor): attention gradients.
        """
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self) -> torch.Tensor:
        """
        Get attention gradients tensor.
        """
        return self.attn_gradients

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape the input tensor to the shape [].

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        key_compression: nn.Linear | None,
        value_compression: nn.Linear | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Perform the forward pass.

        Args:
            x_q (torch.Tensor): query tokens
            x_kv (torch.Tensor): key-value tokens
            key_compression (nn.Linear | None): Linformer-style compression for keys
            value_compression (nn.Linear | None): Linformer-style compression for
            values

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Tuple with tokens and
            attention_stats
        """
        assert _all_or_none(
            [key_compression, value_compression]
        ), "If key_compression is (not) None, then value_compression must (not) be None"
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0, _INTERNAL_ERROR_MESSAGE
        if key_compression is not None:
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)  # type: ignore

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)

        self.save_attn(attention_probs)
        if attention_probs.requires_grad:
            attention_probs.register_hook(self.save_attn_gradients)

        x = attention_probs @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)

        return x, {
            "attention_logits": attention_logits,
            "attention_probs": attention_probs,
        }


class Transformer(nn.Module):
    """
    Transformer with extra features.

    This module is the backbone of `FTTransformer`.
    """

    class FFN(nn.Module):
        """
        The Feed-Forward Network module used in every `Transformer` block.
        """

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: Callable[..., nn.Module],
        ) -> None:
            """
            Initialize the module.

            Args:
                d_token (int): dimensionality of token.
                d_hidden (int): dimensionality of hidden layers.
                bias_first (bool): flag indicating whether to use bias in the first
                bias_second (bool): flag indicating whether to use bias in the second
                dropout (float): degree of dropout
                activation (Callable[..., nn.Module]): activation function.
            """
            super().__init__()
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation(activation) else 1),
                bias_first,
            )
            self.activation = activation()
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Perform the forward pass.

            Args:
                x (torch.Tensor): input tensor.

            Returns:
                torch.Tensor: output tensor.
            """
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x

    class Head(nn.Module):
        """
        The final module of the `Transformer` that performs BERT-like inference.
        """

        def __init__(
            self,
            *,
            d_in: int,
            bias: bool,
            activation: Callable[..., nn.Module],
            normalization: Callable[..., nn.Module],
            d_out: int,
        ):
            """
            Initialize the module.

            Args:
                d_in (int): dimension of the input
                bias (bool): add bias to the linear layer
                activation (Callable[..., nn.Module]): activation function
                normalization (Callable[..., nn.Module]): normalization layer
                d_out (int): dimension of the output
            """
            super().__init__()
            self.normalization = normalization(d_in)
            self.activation = activation()
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Perform the forward pass.

            Args:
                x (torch.Tensor): input tensor.

            Returns:
                torch.Tensor: output tensor.
            """
            x = x[:, -1]
            x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_token: int,
        n_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: nn.Module,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: nn.Module,
        ffn_normalization: nn.Module,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        last_layer_query_idx: None | list[int] | slice,
        n_tokens: int | None,
        kv_compression_ratio: float | None,
        kv_compression_sharing: str | None,
        head_activation: Callable[..., nn.Module],
        head_normalization: Callable[..., nn.Module],
        d_out: int,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the module.

        Args:
            d_token (int): dimensionality of token.
            n_blocks (int): number of blocks.
            attention_n_heads (int): number of attention heads.
            attention_dropout (float): degree of attention dropout.
            attention_initialization (str): initialization strategy for attention
            weights.
            attention_normalization (nn.Module): attention normalization layer.
            ffn_d_hidden (int): capacity of the hidden layers in the FFN.
            ffn_dropout (float): dropout in the FFN.
            ffn_activation (nn.Module): activation function in the FFN.
            ffn_normalization (nn.Module): normalization layer in the FFN.
            residual_dropout (float): degree of residual dropout.
            prenormalization (bool): flag to use prenormalization.
            first_prenormalization (bool): flag to use prenormalization in the first
            layer.
            last_layer_query_idx (None | list[int] | slice): query index for the
            last layer.
            n_tokens (int | None): number of tokens.
            kv_compression_ratio (float | None): compression ratio for the key and
            values.
            kv_compression_sharing (str | None): strategy for sharing the key and
            values of compression.
            head_activation (Callable[..., nn.Module]): activation function in the
            attention head.
            d_out (int): dimensionality of the output

        Raises:
            ValueError: value error

        Returns:
            None: None
        """
        super().__init__()
        if isinstance(last_layer_query_idx, int):
            raise ValueError(
                "last_layer_query_idx must be None, list[int] or slice. "
                f"Do you mean last_layer_query_idx=[{last_layer_query_idx}] ?"
            )
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization`"
            "must be False"
        assert _all_or_none([n_tokens, kv_compression_ratio, kv_compression_sharing]), (
            "If any of the following arguments is (not) None, then all of them must "
            "(not) be None: n_tokens, kv_compression_ratio, kv_compression_sharing"
        )
        assert kv_compression_sharing in [None, "headwise", "key-value", "layerwise"]

        def make_kv_compression() -> nn.Module:
            assert (
                n_tokens and kv_compression_ratio
            ), _INTERNAL_ERROR_MESSAGE  # for mypy
            # https://bit.ly/3h8RdO5
            return nn.Linear(n_tokens, int(n_tokens * kv_compression_ratio), bias=False)

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression_ratio and kv_compression_sharing == "layerwise"
            else None
        )

        self.prenormalization = prenormalization
        self.last_layer_query_idx = last_layer_query_idx

        self.blocks = nn.ModuleList([])
        for layer_idx in range(n_blocks):
            layer = nn.ModuleDict(
                {
                    "attention": MultiheadAttention(
                        d_token=d_token,
                        n_heads=attention_n_heads,
                        dropout=attention_dropout,
                        bias=True,
                        initialization=attention_initialization,
                    ),
                    "ffn": Transformer.FFN(
                        d_token=d_token,
                        d_hidden=ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=ffn_dropout,
                        activation=ffn_activation,
                    ),
                    "attention_residual_dropout": nn.Dropout(residual_dropout),
                    "ffn_residual_dropout": nn.Dropout(residual_dropout),
                    "output": nn.Identity(),  # for hooks-based introspection
                }
            )
            if layer_idx or not prenormalization or first_prenormalization:
                layer["attention_normalization"] = attention_normalization(d_token)
            layer["ffn_normalization"] = ffn_normalization(d_token)
            if kv_compression_ratio and self.shared_kv_compression is None:
                layer["key_compression"] = make_kv_compression()
                if kv_compression_sharing == "headwise":
                    layer["value_compression"] = make_kv_compression()
                else:
                    assert (
                        kv_compression_sharing == "key-value"
                    ), _INTERNAL_ERROR_MESSAGE
            self.blocks.append(layer)

        self.head = Transformer.Head(
            d_in=d_token,
            d_out=d_out,
            bias=True,
            activation=head_activation,  # type: ignore
            normalization=head_normalization if prenormalization else nn.Identity(),
        )

    def _get_kv_compressions(
        self, layer: dict[str, Any]
    ) -> tuple[nn.Module | None, nn.Module | None]:
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer["key_compression"], layer["value_compression"])
            if "key_compression" in layer and "value_compression" in layer
            else (layer["key_compression"], layer["key_compression"])
            if "key_compression" in layer
            else (None, None)
        )

    def _start_residual(
        self, layer: torch.ModuleDict, stage: str, x: torch.Tensor
    ) -> torch.Tensor:
        assert stage in ["attention", "ffn"], _INTERNAL_ERROR_MESSAGE
        x_residual = x
        if self.prenormalization:
            norm_key = f"{stage}_normalization"
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(
        self,
        layer: torch.ModuleDict,
        stage: str,
        x: torch.Tensor,
        x_residual: torch.Tensor,
    ) -> torch.Tensor:
        assert stage in ["attention", "ffn"], _INTERNAL_ERROR_MESSAGE
        x_residual = layer[f"{stage}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{stage}_normalization"](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        assert (
            x.ndim == 3
        ), "The input must have 3 dimensions: (n_objects, n_tokens, d_token)"
        for layer_idx, layer in enumerate(self.blocks):
            layer = cast(nn.ModuleDict, layer)

            query_idx = (
                self.last_layer_query_idx if layer_idx + 1 == len(self.blocks) else None
            )
            x_residual = self._start_residual(layer, "attention", x)
            x_residual, _ = layer["attention"](
                x_residual if query_idx is None else x_residual[:, query_idx],
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if query_idx is not None:
                x = x[:, query_idx]
            x = self._end_residual(layer, "attention", x, x_residual)

            x_residual = self._start_residual(layer, "ffn", x)
            x_residual = layer["ffn"](x_residual)
            x = self._end_residual(layer, "ffn", x, x_residual)
            x = layer["output"](x)

        x = self.head(x)
        return x


class FTTransformer(nn.Module):
    """
    Implementation of `FTTransformer`.

    @inproceedings{gorishniyRevisitingDeepLearning2021,
        title = {Revisiting {{Deep Learning Models}} for {{Tabular Data}}},
        booktitle = {Advances in {{Neural Information Processing Systems}}},
        author = {Gorishniy, Yury and Rubachev, Ivan and Khrulkov, Valentin and Babenko,
        Artem},
        year = {2021},
        volume = {34},
        pages = {18932--18943},
        publisher = {{Curran Associates, Inc.}},
        address = {{Red Hook, NY}},
    }

    Args:
        nn (module): module
    """

    def __init__(
        self,
        feature_tokenizer: FeatureTokenizer,
        transformer: Transformer,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the module.

        Args:
            feature_tokenizer (FeatureTokenizer): feature tokenizer.
            transformer (Transformer): transformer.
        """
        super().__init__()
        self.feature_tokenizer = feature_tokenizer
        self.cls_token = CLSToken(
            feature_tokenizer.d_token, feature_tokenizer.initialization
        )
        self.transformer = transformer

    def forward(
        self, x_cat: torch.Tensor | None, x_cont: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Perform forward pass.

        Args:
            x_cat (torch.Tensor | None): tensor with categorical data.
            x_cont (torch.Tensor | None): tensor with continous data.

        Returns:
            torch.Tensor: predictions
        """
        x = self.feature_tokenizer(x_cont, x_cat)
        x = self.cls_token(x)
        x = self.transformer(x)
        return x
