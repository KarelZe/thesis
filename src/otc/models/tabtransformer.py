"""
Implementation of a TabTransformer.

Based on paper:
https://arxiv.org/abs/2012.06678

Implementation adapted from: https://github.com/lucidrains/tab-transformer-pytorch
and https://github.com/kathrinse/TabSurvey/blob/main/models/tabtransformer.py

Also see: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch import einsum, nn, optim

from otc.data.dataloader import TabDataLoader
from otc.data.dataset import TabDataset
from otc.models.activation import GeGLU
from otc.optim.early_stopping import EarlyStopping


class TransformerClassifier(BaseEstimator, ClassifierMixin):

    epochs = 1024

    def __init__(
        self,
        module: nn.Module,
        module_params: Any,
        optim_params: Any,
        dl_params: Any,
        features: list[str],
        callbacks,
    ) -> None:
        self.module = module

        super().__init__()

        self._clf: nn.Module
        self._module_params = module_params
        self._optim_params = optim_params
        self._dl_params = dl_params
        self._features = features
        self._callbacks = callbacks

    def array_to_dataloader(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> TabDataLoader:

        data = TabDataset(
            X,
            y,
            features=self._features,
            cat_features=self._module_params["cat_features"],
            cat_unique_counts=self._module_params["cat_cardinalities"],
        )

        return TabDataLoader(data.x_cat, data.x_cont, data.y, **self._dl_params)

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        eval_set: tuple[np.ndarray, np.ndarray] | tuple[pd.DataFrame, pd.Series] | None,
    ):

        # use insample instead of validation set, if None is passed
        if eval_set:
            X_val, y_val = eval_set
        else:
            X_val, y_val = X, y

        train_loader = self.array_to_dataloader(X, y)
        val_loader = self.array_to_dataloader(X_val, y_val)

        self._clf = self.module(**self._module_params)

        # use multiple gpus, if available
        self._clf = nn.DataParallel(self._clf).to(self._dl_params["device"])

        # half precision, see https://pytorch.org/docs/stable/amp.html
        scaler = torch.cuda.amp.GradScaler()
        # Generate the optimizers
        optimizer = optim.AdamW(
            self._clf.parameters(),
            lr=self._optim_params["lr"],
            weight_decay=self._optim_params["weight_decay"],
        )

        # see https://stackoverflow.com/a/53628783/5755604
        # no sigmoid required; numerically more stable
        criterion = nn.BCEWithLogitsLoss()

        # keep track of val loss and do early stopping
        early_stopping = EarlyStopping(patience=5)

        for epoch in range(self.epochs):

            # perform training
            loss_in_epoch_train = 0

            self._clf.train()

            for x_cat, x_cont, targets in train_loader:

                # reset the gradients back to zero
                optimizer.zero_grad()

                outputs = self._clf(x_cat, x_cont)
                outputs = outputs.flatten()
                with torch.cuda.amp.autocast():
                    train_loss = criterion(outputs, targets)

                # compute accumulated gradients
                scaler.scale(train_loss).backward()

                # perform parameter update based on current gradients
                scaler.step(optimizer)
                scaler.update()

                # add the mini-batch training loss to epoch loss
                loss_in_epoch_train += train_loss.item()

            self._clf.eval()

            loss_in_epoch_val = 0.0
            # correct = 0

            with torch.no_grad():
                for x_cat, x_cont, targets in val_loader:
                    outputs = self._clf(x_cat, x_cont)
                    outputs = outputs.flatten()

                    val_loss = criterion(outputs, targets)
                    loss_in_epoch_val += val_loss.item()

                    # # convert to propability, then round to nearest integer
                    # outputs = torch.sigmoid(outputs).round()
                    # correct += (outputs == targets).sum().item()

            train_loss = loss_in_epoch_train / len(train_loader)
            val_loss = loss_in_epoch_val / len(val_loader)

            self._callbacks.on_epoch_end(epoch, self.epochs, train_loss, val_loss)

            # return early if val loss doesn't decrease for several iterations
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break

        # is fitted flag
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        # convert probs to classes
        return np.where(probs > 0.5, 1, -1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # check if there are attributes with trailing _
        check_is_fitted(self)
        test_loader = self.array_to_dataloader(X, pd.Series(np.zeros(len(X))))

        self._clf.eval()

        probabilites = []
        with torch.no_grad():
            for x_cat, x_cont, _ in test_loader:
                probability = self._clf(x_cat, x_cont)
                probability = probability.flatten()
                probability = torch.sigmoid(probability)
                probabilites.append(probability.detach().cpu().numpy())

        return np.concatenate(probabilites)


class Residual(nn.Module):
    """
    PyTorch implementation of residual connections.

    Args:
        nn (nn.Module): module
    """

    def __init__(self, fn: nn.Module):
        """
        Residual connection.

        Args:
            fn (nn.Module): network.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass of residual connections.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    """
    PyTorch implementation of pre-normalization.

    Args:
        nn (nn.module): module.
    """

    def __init__(self, dim: int, fn: nn.Module):
        """
        Pre-normalization.

        Consists of layer for layer normalization followed by another network.

        Args:
            dim (int): Number of dimensions of normalized shape.
            fn (nn.Module): network.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass of pre-normalization layers.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    PyTorch implementation of feed forward network.

    Args:
        nn (nn.module): module.
    """

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        """
        Feed forward network.

        Network consists of input layer, GEGLU activation, dropout layer,
        and output layer.

        Args:
            dim (int): dimension of input and output layer.
            mult (int, optional): Scaling factor for output dimension of input layer or
            input dimension of output layer. Defaults to 4.
            dropout (float, optional): Degree of dropout. Defaults to 0.0.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GeGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass of feed forward network.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        return self.net(x)


class Attention(nn.Module):
    """
    Pytorch implementation of attention.

    Args:
        nn (nn.Module): module.
    """

    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 16, dropout: float = 0.0
    ):
        """
        Attention.

        Args:
            dim (int): Number of dimensions.
            heads (int, optional): Number of attention heads. Defaults to 8.
            dim_head (int, optional): Dimension of attention heads. Defaults to 16.
            dropout (float, optional): Degree of dropout. Defaults to 0.0.
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of attention module.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        b, n, _ = q.shape
        # reshape and permute: b n (h d) -> b h n d
        q, k, v = map(
            lambda t: t.reshape(b, n, self.heads, -1).permute(0, 2, 1, 3), (q, k, v)
        )
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        # reshape and permute: b h i j, b h j d -> b h i d
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer.

    Based on paper:
    https://arxiv.org/abs/1706.03762

    Args:
        nn (nn.Module): Module with transformer.
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        attn_dropout: float,
        ff_dropout: float,
    ):
        """
        Classical transformer.

        Args:
            num_tokens (int): Number of tokens i. e., unique classes + special tokens.
            dim (int): Number of dimensions.
            depth (int): Depth of encoder / decoder.
            heads (int): Number of attention heads.
            dim_head (int): Dimensions of attention heads.
            attn_dropout (float): Degree of dropout in attention.
            ff_dropout (float): Degree of dropout in feed-forward network.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        x = self.embeds(x)

        for attn, ff in self.layers:  # type: ignore
            x = attn(x)
            x = ff(x)

        return x


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
        dim_head: int = 16,
        dim_out: int = 1,
        mlp_hidden_mults: tuple[(int, int)] = (4, 2),
        mlp_act: Callable[..., nn.Module] = nn.ReLU,
        num_special_tokens: int = 2,
        continuous_mean_std: torch.Tensor | None = None,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        **kwargs: Any,
    ):
        """
        TabTransformer.

        Originally introduced in https://arxiv.org/abs/2012.06678.

        Args:
            cat_cardinalities ([List[int] | Tuple[()]): List with number of categories
            for each categorical feature. If no categorical variables are present,
            use empty tuple. For categorical variables e. g., option type ('C' or 'P'),
            the list would be `[1]`.
            num_continuous (int): Number of continous features.
            dim (int, optional): Dimensionality of transformer. Defaults to 32.
            depth (int, optional): Depth of encoder / decoder of transformer.
            Defaults to 4.
            heads (int, optional): Number of attention heads. Defaults to 8.
            dim_head (int, optional): Dimensionality of attention head. Defaults to 16.
            dim_out (int, optional): Dimension of output layer of MLP. Set to one for
            binary classification. Defaults to 1.
            mlp_hidden_mults (Tuple[(int, int)], optional): multipliers for dimensions
            of hidden layer in MLP. Defaults to (4, 2).
            mlp_act (Callable[..., nn.Module], optional): Activation function used
            in MLP. Defaults to nn.ReLU().
            num_special_tokens (int, optional): Number of special tokens in transformer.
            Defaults to 2.
            continuous_mean_std (torch.Tensor | None): List with mean and
            std deviation of each continous feature. Shape eq. `[num_continous x 2]`.
            Defaults to None.
            attn_dropout (float, optional): Degree of attention dropout used in
            transformer. Defaults to 0.0.
            ff_dropout (float, optional): Dropout in feed forward net. Defaults to 0.0.
        """
        super().__init__()
        assert all(
            map(lambda n: n > 0, cat_cardinalities)
        ), "number of each category must be positive"

        # categories related calculations

        self.num_categories = len(cat_cardinalities)
        self.cardinality_categories = sum(cat_cardinalities)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.cardinality_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position
        #  in the categories embedding table

        categories_offset = F.pad(
            torch.tensor(list(cat_cardinalities)), (1, 0), value=num_special_tokens
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
        j = input_size // 8

        hidden_dimensions = list(map(lambda t: j * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)

    def forward(self, x_cat: torch.Tensor | None, x_cont: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TabTransformer.

        Args:
            x_cat (torch.Tensor | None): tensor with categorical data.
            x_cont (torch.Tensor): tensor with continous data.

        Returns:
            torch.Tensor: probabilitys
        """
        flat_categ: torch.Tensor | None = None

        if x_cat is not None:
            assert x_cat.shape[-1] == self.num_categories, (
                f"you must pass in {self.num_categories} "
                f"values for your categories input"
            )
            x_cat += self.categories_offset
            x = self.transformer(x_cat)
            flat_categ = x.flatten(1)

        assert x_cont.shape[1] == self.num_continuous, (
            f"you must pass in {self.num_continuous} "
            f"values for your continuous input"
        )

        if self.continuous_mean_std is not None:
            mean, std = self.continuous_mean_std.unbind(dim=-1)  # type: ignore
            x_cont = (x_cont - mean) / std

        normed_cont = self.norm(x_cont)

        # Adaptation to work without categorical data
        x = (
            torch.cat((flat_categ, normed_cont), dim=-1)
            if flat_categ is not None
            else normed_cont
        )

        return self.mlp(x)
