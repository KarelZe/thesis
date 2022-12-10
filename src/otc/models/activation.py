"""
Implementation of GeGLU and ReGLU activation functions.

Adapted from:
https://github.com/Yura52/rtdl/blob/main/rtdl/functional.py
"""
import torch
import torch.nn.functional as F
from torch import nn


class GeGLU(nn.Module):
    r"""
    Implementation of the GeGLU activation function.

    Given by:
    $\operatorname{GeGLU}(x, W, V, b, c)=\operatorname{GELU}(x W+b) \otimes(x V+c)$

    Proposed in https://arxiv.org/pdf/2002.05202v1.pdf.

    Args:
        nn (torch.Tensor): module
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GeGlU activation.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        assert x.shape[-1] % 2 == 0
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class ReGLU(nn.Module):
    r"""
    Implementation of the GeGLU activation function.

    Given by:

    Proposed in https://arxiv.org/pdf/2002.05202v1.pdf.

    Args:
        nn (torch.Tensor): module
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GeGlU activation.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        assert x.shape[-1] % 2 == 0
        x, gates = x.chunk(2, dim=-1)
        return x * F.relu(gates)
