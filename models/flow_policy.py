"""
flow_policy.py
==================
Neural network components for the toy 2-D unicycle flow policy.

Architecture: DiT (Diffusion Transformer), reusing components from dit.py.

Each action step (v_i, ω_i) is treated as one token of dimension ACT_DIM.
The sequence shape is (b, H, ACT_DIM).  Time conditioning is injected into
every DiT block via adaLN-Zero – it is never concatenated with the token
features.

Exports
-------
ACT_DIM              – action dimension per step (2: v, omega)
ActionFlowNet     – DiT-based flow-matching vector field
ActionFlowODE         – ODE wrapper for EulerSimulator inference
"""

import torch
import torch.nn as nn


from models.simulators import ODE
from models.dit import FourierEncoder, DiffusionTransformer

# ---------------------------------------------------------------------------
# Fixed constants
# ---------------------------------------------------------------------------

ACT_DIM = 2   # per-step action dimension: [v, omega]  (unicycle, always 2)

# H and DATA_DIM depend on the config (H = ceil(D_bev / v_nominal / dt)).
# Use ActionFlowNet.from_config(cfg) to build a correctly-sized model,
# or pass h explicitly to ActionFlowNet(h=...).


# ---------------------------------------------------------------------------
# Time embedding (kept for backward-compatibility; also used internally)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbed(nn.Module):
    """Fixed-frequency sinusoidal time embedding (alias of FourierEncoder)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.register_buffer("freqs", torch.randn(dim // 2))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        f = 2.0 * torch.pi * self.freqs * t.unsqueeze(-1)
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)


# ---------------------------------------------------------------------------
# DiT-based flow-matching vector field
# ---------------------------------------------------------------------------

class ActionFlowNet(nn.Module):
    """
    DiT-based flow-matching vector field for action sequences.

    Build via the class method::

        net = ActionFlowNet.from_config(cfg)

    or directly::

        net = ActionFlowNet(h=167)

    Parameters
    ----------
    h          : int  – action horizon (number of time steps); read from config
    num_layers : int  – number of DiT blocks (default 4)
    dim        : int  – transformer width (default 128)
    heads      : int  – attention heads (default 4, must divide dim)
    """

    def __init__(
        self,
        h          : int = 167,
        num_layers : int = 4,
        dim        : int = 128,
        heads      : int = 4,
    ):
        super().__init__()
        self.H        = h
        self.act_dim  = ACT_DIM
        self.data_dim = h * ACT_DIM   # convenience attribute

        # -- token projection: ACT_DIM → dim  (applied per time-step independently)
        self.in_proj = nn.Linear(ACT_DIM, dim)

        # -- time conditioning: scalar t → (b, dim)
        self.time_embedder = nn.Sequential(
            FourierEncoder(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # -- DiT backbone (includes learnable positional embeddings)
        self.dit = DiffusionTransformer(
            depth=num_layers,
            n_tokens=h,
            dim=dim,
            heads=heads,
        )

        # -- final adaLN: LayerNorm (no affine) + scale/shift from t_emb
        self.final_norm  = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        # -- output projection: dim → ACT_DIM  (per-token, zero-init)
        self.out_proj = nn.Linear(dim, ACT_DIM)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg        : dict,
        num_layers : int = 4,
        dim        : int = 128,
        heads      : int = 4,
    ) -> "ActionFlowNet":
        """
        Build a :class:`ActionFlowNet` with H derived from *cfg*.

        H = ceil(D_bev / v_nominal / dt)  (same formula as FlowDataset).
        """
        from scripts.data.gen_demos import compute_H
        h = compute_H(cfg)
        return cls(h=h, num_layers=num_layers, dim=dim, heads=heads)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (b, H*ACT_DIM)  – flattened noisy action sequence at flow time t
            t : (b,)            – continuous flow time in [0, 1]
        Returns:
            u : (b, H*ACT_DIM)  – predicted vector field (flattened)
        """
        b = x.shape[0]
        if t.dim() > 1:
            t = t.view(-1)

        # (b, H*ACT_DIM) → (b, H, ACT_DIM)
        x_seq  = x.view(b, self.H, self.act_dim)
        tokens = self.in_proj(x_seq)          # (b, H, dim)
        c      = self.time_embedder(t)         # (b, dim)
        tokens = self.dit(tokens, c)           # (b, H, dim)

        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        out = self.out_proj(tokens)            # (b, H, ACT_DIM)
        return out.view(b, -1)                 # (b, H*ACT_DIM)


# ---------------------------------------------------------------------------
# ODE wrapper (for EulerSimulator inference)
# ---------------------------------------------------------------------------

class ActionFlowODE(ODE):
    """Wraps :class:`ActionFlowNet` as an :class:`ODE` for
    :class:`~simulators.EulerSimulator`.
    """

    def __init__(self, net: ActionFlowNet):
        self.net = net

    def drift_coefficient(
        self, xt: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.net(xt, t)
