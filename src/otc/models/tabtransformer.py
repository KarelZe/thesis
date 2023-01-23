"""
Implementation of a TabTransformer.

Based on paper:
https://arxiv.org/abs/2012.06678
"""
from __future__ import annotations

from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MLP(nn.Module):
    """
    Pytorch model of a vanilla multi-layer perceptron.

    Args:
        nn (nn.Module): module with implementation of MLP.
    """

    def __init__(self, dims: list[int], act: Callable[..., nn.Module]):
        """
        Multilayer perceptron.

        Depth of network is given by `len(dims)`. Capacity is given by entries
        of `dim`. Activation function is used after each linear layer. There is
        no activation function for the final linear layer, as it is sometimes part
        of the loss function already e. g., `nn.BCEWithLogitsLoss()`.
        Args:
            dims (List[int]): List with dimensions of layers.
            act (Callable[..., nn.Module]): Activation function of each linear
            layer.
        """
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for dim_in, dim_out in dims_pairs:
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)
            layers.append(act())

        # drop last layer, as a sigmoid layer is included from BCELogitLoss
        del layers[-1]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagate tensor through MLP.

        Args:
            x (torch.Tensor): input tensor.
        Returns:
            torch.Tensor: output tensor.
        """
        return self.mlp(x)


class ColumnEmbedding(nn.Module):
    """
    Column-embedding from TabTransformer paper (https://arxiv.org/abs/2012.06678).

    Creates:
    1. a individual embedding for each category in each categorical column.
    2. a shared embedding for each categorical column, if `mode` is "add".

    If `mode` is "add", the individual embeddings and shared embeddings are added
    element-wisely.
    Args:
        nn (nn.Module): module
    """

    def __init__(
        self,
        cat_cardinalities: tuple[int, ...] | tuple[()],
        dim: int,
        mode: Literal["concat", "add"] | None = "add",
        dropout: float = 0.0,
        bias: bool = False,
    ):
        """
        Initialize column embedding.

        Args:
            cat_cardinalities (tuple[int, ...] | tuple[()]): cardinalities of
            categorical columns
            dim (int): dimensionality of embedding
            mode (Literal[&quot;concat&quot;, &quot;add&quot;] | None, optional):
            mode for shared embedding. Defaults to "add". `None` means no shared
            embedding.
            dropout (float, optional): degree of dropout. Defaults to 0.0.
            bias (bool, optional): add bias term. Defaults to False.
        """
        super().__init__()

        assert dim > 0, "dim must be positive"

        self.shared_embedding = mode
        self.dropout = nn.Dropout(p=dropout)
      
        if type(cat_cardinalities) is tuple:
            cat_cardinalities = list(cat_cardinalities)
        
        # embeddings for every class in every column
        category_offsets = torch.tensor([0] + cat_cardinalities[:-1]).cumsum(0)
        self.register_buffer("category_offsets", category_offsets, persistent=False)
        self.indiv_embed = nn.Embedding(sum(cat_cardinalities), dim)

        # embeddings for entire column
        if mode == "concat":
            NotImplementedError("Concatenation not implemented")
        elif mode == "add":
            self.shared_embed = nn.Parameter(
                torch.empty(len(cat_cardinalities), dim).uniform_(-1, 1)
            )

        # bias term
        self.bias = (
            nn.Parameter(torch.empty(len(cat_cardinalities), dim).zero_())
            if bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for column embedding.

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: tensor with embeddings
        """
        x = self.indiv_embed(x + self.category_offsets[None])
        if self.shared_embedding == "add":
            x = x + self.shared_embed[None]
        # add bias term, not part of paper but works in Gorishnyy et al.
        if self.bias is not None:
            x = x + self.bias[None]
        # add dropout, not part of paper, but could work
        return self.dropout(x)


class Transformer(nn.Module):
    """
    Pytorch model of Transformer encoder.

    Args:
        nn (nn.Module): module
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_feedforward: int,
        depth: int,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
        dropout: float = 0.5,
        norm_first: bool = False,
    ):
        """
        Pytorch model of Transformer encoder.

        Args:
            dim (int): dimensionality of model
            heads (int): number of attention heads
            dim_feedforward (int): dimensionality of point-wise feedforward network
            depth (int): no of transformer blocks
            activation (str | Callable[[Tensor], Tensor], optional): activation.
            Defaults to "gelu".
            dropout (float, optional): dropout. Defaults to 0.5.
            norm_first (bool, optional): placement of layer norm. Defaults to False.
        """
        super().__init__()
        self.model_type = "Transformer"
        encoder_layers = TransformerEncoderLayer(
            dim,
            heads,
            dim_feedforward,
            dropout,
            activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, depth)

    def forward(self, x_embed: Tensor) -> Tensor:
        """
        Forward pass of Transformer.

        Args:
            x_embed (Tensor): tensor with embedded features

        Returns:
            Tensor: output tensor
        """
        output = self.transformer_encoder(x_embed)
        return output


class TabTransformer(nn.Module):
    """
    PyTorch model of TabTransformer.

    Based on paper:
    https://arxiv.org/abs/2012.06678

    Args:
        nn (nn.Module): Module with implementation of TabTransformer.
    """

    def __init__(
        self,
        *,
        cat_cardinalities: tuple[int, ...] | tuple[()],
        num_continuous: int,
        dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        transformer_act: Callable[[Tensor], Tensor] = F.gelu,
        transformer_norm_first: bool = False,
        transformer_dropout: float = 0.0,
        mlp_hidden_mults: tuple[(int, int)] = (4, 2),
        mlp_act: Callable[..., nn.Module] = nn.ReLU,
        dim_out: int = 1,
        **kwargs: Any,
    ):
        """
        TabTransformer.

        Originally introduced in https://arxiv.org/abs/2012.06678.
        Args:
            cat_cardinalities (tuple[int, ...] | tuple[()]): list with number of
            categories for each categorical feature.
            num_continuous (int): number of continuous features.
            dim (int, optional): dimenssionality of model. Defaults to 32.
            depth (int, optional): No. of layers of the encoder. Defaults to 4.
            heads (int, optional): No. of attention heads. Defaults to 8.
            transformer_act (Callable[[Tensor], Tensor], optional): activation function
            in Transformer block. Defaults to F.gelu.
            transformer_norm_first (bool, optional): Flag for post-norm or pre-norm
            Transformer (see http://arxiv.org/abs/2002.04745). Defaults to False.
            transformer_dropout (float, optional): degree of dropout. Defaults to 0.0.
            mlp_hidden_mults (tuple[, optional): multipliers for hidden dim.
            Dim = multiplier * l, with l=no.of inputs. Defaults to (4, 2).
            mlp_act (Callable[..., nn.Module], optional): activation function in MLP.
            Defaults to nn.ReLU.
            dim_out (int, optional): dimensionality of output layer. Defaults to 1.
        """
        super().__init__()
        assert all(
            map(lambda n: n > 0, cat_cardinalities)
        ), "number of each category must be positive"

        # transformer
        self.col_embed = ColumnEmbedding(cat_cardinalities, dim)
        self.transformer = Transformer(
            dim,
            heads,
            4 * dim,
            depth,
            transformer_act,
            transformer_dropout,
            transformer_norm_first,
        )

        # mlp
        self.num_continuous = num_continuous
        self.num_cat = len(cat_cardinalities)
        self.layer_norm = nn.LayerNorm(num_continuous)
        # dim * m + c (p. 3)
        input_size = (dim * len(cat_cardinalities)) + num_continuous
        hidden_dimensions = list(map(lambda t: input_size * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        self.mlp = MLP(all_dimensions, act=mlp_act)

    def forward(self, x_cat: torch.Tensor | None, x_cont: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TabTransformer.

        Args:
            x_cat (torch.Tensor | None): tensor with categorical data.
            x_cont (torch.Tensor): tensor with continous data.

        Returns:
            torch.Tensor: probabilities
        """
        flat_x_cat: torch.Tensor | None = None

        if x_cat is not None:
            assert x_cat.shape[-1] == self.num_cat, (
                f"you must pass in {self.num_cat} " f"values for your categories input"
            )
            x_cat_embed = self.col_embed(x_cat)
            x_cat_contex = self.transformer(x_cat_embed)
            flat_x_cat = x_cat_contex.flatten(1)

        assert x_cont.shape[1] == self.num_continuous, (
            f"you must pass in {self.num_continuous} "
            f"values for your continuous input"
        )

        x_cont = self.layer_norm(x_cont)

        # Adaptation to work without categorical data
        x = (
            torch.cat((flat_x_cat, x_cont), dim=-1)
            if flat_x_cat is not None
            else x_cont
        )

        return self.mlp(x)
