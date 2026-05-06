"""
vector_fields.py
================
Learned and analytic vector fields used with the CFG training framework.

Classes
-------
ConditionalVectorField      – Abstract u_t^θ(x | y).
CFGVectorFieldODE           – ODE that implements classifier-free guidance.
MLP                         – Simple feed-forward network.
MLPConditionalVectorField   – MLP-based conditional vector field (for toy data).
"""

from abc import ABC, abstractmethod
from typing import List, Type

import torch
import torch.nn as nn

from models.simulators import ODE


# ---------------------------------------------------------------------------
# Abstract vector field
# ---------------------------------------------------------------------------

class ConditionalVectorField(nn.Module, ABC):
    """Conditional vector field u_t^θ(x | y)."""

    @abstractmethod
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (b, *)
            t: (b,)
            y: (b,)   – integer class labels
        Returns:
            u_t^θ(x|y): (b, *)
        """


# ---------------------------------------------------------------------------
# CFG ODE
# ---------------------------------------------------------------------------

class CFGVectorFieldODE(ODE):
    """
    Implements classifier-free guidance at inference time:

        ũ_t(x|y) = (1 − w) · u_t(x|∅) + w · u_t(x|y)

    Parameters
    ----------
    net : ConditionalVectorField
    null_label : int
        Integer label used as the "null" (unconditional) class.
    guidance_scale : float
        Guidance strength *w*.  Use 1.0 for standard conditional generation.
    """

    def __init__(
        self,
        net: ConditionalVectorField,
        null_label: int,
        guidance_scale: float = 1.0,
    ):
        self.net = net
        self.guidance_scale = guidance_scale
        self.null_label = null_label

    def drift_coefficient(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        guided = self.net(x, t, y)
        null_y = torch.ones_like(y) * self.null_label
        unguided = self.net(x, t, null_y)
        return (1 - self.guidance_scale) * unguided + self.guidance_scale * guided


# ---------------------------------------------------------------------------
# MLP backbone
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Simple multi-layer perceptron.

    Parameters
    ----------
    dims : list[int]
        Layer sizes including input and output dimensions.
    activation : nn.Module subclass
        Activation inserted between every pair of linear layers.
    """

    def __init__(
        self,
        dims: List[int],
        activation: Type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# MLP-based conditional vector field (toy / sanity-check)
# ---------------------------------------------------------------------------

class MLPConditionalVectorField(ConditionalVectorField):
    """
    MLP-based conditional vector field for low-dimensional toy data (e.g. GMM).

    The input is formed by concatenating [x, class_embed(y), t] along the
    feature dimension and passing through an MLP.

    Parameters
    ----------
    dim : int
        Data dimensionality.
    hidden_dim : int
        Hidden dimension of the MLP.
    class_dim : int
        Embedding dimension for class labels.
    num_classes : int
        Number of real classes (null label will be ``num_classes``).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        class_dim: int,
        num_classes: int,
    ):
        super().__init__()
        # num_classes + 1 to accommodate the null label
        self.class_embedding = nn.Embedding(num_classes + 1, class_dim)
        self.mlp = MLP([dim + class_dim + 1, hidden_dim, hidden_dim, dim])

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        embed = self.class_embedding(y)                    # (b, class_dim)
        x_aug = torch.cat([x, embed, t.unsqueeze(-1)], dim=-1)
        return self.mlp(x_aug)

