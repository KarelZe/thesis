"""
Implementation of a Tab Transformer.

Based on paper:
https://arxiv.org/abs/2012.06678

Implementation adapted from: https://github.com/lucidrains/tab-transformer-pytorch
and https://github.com/kathrinse/TabSurvey/blob/main/models/tabtransformer.py
"""
import torch

import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn

from typing import Tuple


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):
    r"""
    Implementation of the GeGLU activation function.

    Given by:
    $\operatorname{GeGLU}(x, W, V, b, c)=\operatorname{GELU}(x W+b) \otimes(x V+c)$

    Proposed in https://arxiv.org/pdf/2002.05202v1.pdf.

    Args:
        nn (torch.Tensor): _description_
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout
    ):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)  # (Embed the categorical features.)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim,
                                    heads=heads,
                                    dim_head=dim_head,
                                    dropout=attn_dropout,
                                ),
                            )
                        ),
                        Residual(PreNorm(dim, FeedForward(dim, dropout=ff_dropout))),
                    ]
                )
            )

    def forward(self, x):
        x = self.embeds(x)

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x


# mlp


class MLP(nn.Module):
    def __init__(self, dims, act):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for dim_in, dim_out in dims_pairs:
            # is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)
            layers.append(act)

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


class TabTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int = 16,
        dim_out: int = 1,
        mlp_hidden_mults: Tuple[(int, int)] = (4, 2),
        mlp_act=nn.ReLU(),
        num_special_tokens: int = 2,
        continuous_mean_std=None,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        assert all(
            map(lambda n: n > 0, categories)
        ), "number of each category must be positive"

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position
        #  in the categories embedding table

        categories_offset = F.pad(
            torch.tensor(list(categories)), (1, 0), value=num_special_tokens
        )  # Prepend num_special_tokens.
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer("categories_offset", categories_offset)

        # continuous

        if continuous_mean_std is not None:
            assert continuous_mean_std.shape == (num_continuous, 2,), (
                f"continuous_mean_std must have a shape of ({num_continuous}, 2)"
                f"where the last dimension contains the mean and variance respectively"
            )
        self.register_buffer("continuous_mean_std", continuous_mean_std)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous

        # transformer

        self.transformer = Transformer(
            num_tokens=total_tokens,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

        # mlp to logits

        input_size = (dim * self.num_categories) + num_continuous
        l = input_size // 8

        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)

    def forward(self, x_categ: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TabTransformer.

        Args:
            x_categ (torch.Tensor): tensor with categorical data.
            x_cont (torch.Tensor): tensor with continous data.

        Returns:
            torch.Tensor: predictions with shape [batch, 1]
        """
        # Adaptation to work without categorical data
        if x_categ is not None:
            assert x_categ.shape[-1] == self.num_categories, (
                f"you must pass in {self.num_categories} "
                f"values for your categories input"
            )
            x_categ += self.categories_offset
            x = self.transformer(x_categ)
            flat_categ = x.flatten(1)

        assert x_cont.shape[1] == self.num_continuous, (
            f"you must pass in {self.num_continuous} "
            f"values for your continuous input"
        )

        if self.continuous_mean_std is not None:
            mean, std = self.continuous_mean_std.unbind(dim=-1)
            x_cont = (x_cont - mean) / std

        normed_cont = self.norm(x_cont)

        # Adaptation to work without categorical data
        if x_categ is not None:
            x = torch.cat((flat_categ, normed_cont), dim=-1)
        else:
            x = normed_cont

        return self.mlp(x)
