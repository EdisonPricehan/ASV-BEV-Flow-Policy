"""
cond_flow_policy.py
=======================
Conditional DiT-based flow-matching vector field for the toy unicycle policy.

Architecture per transformer block:
  1. AdaLN(action_tokens; cond_vec)  →  self-attention over action tokens
  2. Cross-attention: Q=action_tokens, K/V=bev_tokens
  3. FFN

cond_vec = global_vec (from goal) + time_emb   (element-wise add)

The module is kept independent of the training loop; CFG is handled by
cfg_wrapper.py at inference time and by passing global_is_null during training.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.dit import FourierEncoder
from models.simulators import ODE
from models.condition_encoder import ToyConditionEncoder


# ---------------------------------------------------------------------------
# Cross-attention module  (Q from action tokens, K/V from BEV tokens)
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Cross-attention: query from *x*, key/value from *context*.

    Parameters
    ----------
    dim   : int – model dimension
    heads : int – number of attention heads (must divide dim)
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads    = heads
        self.head_dim = dim // heads
        self.scale    = self.head_dim ** -0.5

        self.q_proj  = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x       : [B, Nq, d]  – action tokens (query)
            context : [B, Nc, d]  – BEV tokens    (key / value)
        Returns:
            out     : [B, Nq, d]
        """
        B, Nq, d = x.shape
        h = self.heads

        q  = self.q_proj(x)                               # [B, Nq, d]
        kv = self.kv_proj(context)                        # [B, Nc, 2d]
        k, v = kv.chunk(2, dim=-1)                        # [B, Nc, d] each

        q = q.view(B, Nq, h, self.head_dim).transpose(1, 2)  # [B,h,Nq,hd]
        k = k.view(B, -1, h, self.head_dim).transpose(1, 2)  # [B,h,Nc,hd]
        v = v.view(B, -1, h, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,h,Nq,Nc]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)                       # [B,h,Nq,hd]
        out = out.transpose(1, 2).contiguous().view(B, Nq, d)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Single conditional DiT block
# ---------------------------------------------------------------------------

class CondDiTBlock(nn.Module):
    """
    One block of the conditional action backbone.

    Structure:
        adaLN-Zero + self-attn  →  cross-attn with BEV  →  FFN

    Parameters
    ----------
    dim   : int – model dimension
    heads : int – attention heads
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()

        # AdaLN-Zero modulation MLP: cond_vec → 6 * dim params
        self.adaLN_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        nn.init.zeros_(self.adaLN_mlp[-1].weight)
        nn.init.zeros_(self.adaLN_mlp[-1].bias)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # Self-attention over action tokens
        self.self_attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        # Cross-attention: action queries, BEV keys/values
        self.cross_attn = CrossAttention(dim, heads)
        # FFN  (4× expand)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(
        self,
        x          : torch.Tensor,   # [B, H, d]  action tokens
        cond_vec   : torch.Tensor,   # [B, d]     time + global
        bev_tokens : torch.Tensor,   # [B, Nc, d] BEV tokens
    ) -> torch.Tensor:
        # AdaLN params
        mods = self.adaLN_mlp(cond_vec)                           # [B, 6d]
        g1, b1, a1, g2, b2, a2 = mods.chunk(6, dim=-1)
        g1 = g1.unsqueeze(1); b1 = b1.unsqueeze(1); a1 = a1.unsqueeze(1)
        g2 = g2.unsqueeze(1); b2 = b2.unsqueeze(1); a2 = a2.unsqueeze(1)

        # 1. Self-attention branch (adaLN-Zero)
        x_sa = self.norm1(x) * (1 + g1) + b1
        x_sa, _ = self.self_attn(x_sa, x_sa, x_sa, need_weights=False)
        x = x + a1 * x_sa

        # 2. Cross-attention branch (no adaLN – BEV provides its own info)
        x = x + self.cross_attn(self.norm2(x), bev_tokens)

        # 3. FFN branch (adaLN-Zero)
        x_ff = self.norm3(x) * (1 + g2) + b2
        x = x + a2 * self.ff(x_ff)

        return x


# ---------------------------------------------------------------------------
# Full conditional flow network
# ---------------------------------------------------------------------------

class CondActionFlowNet(nn.Module):
    """
    Conditional DiT flow-matching vector field.

    Build with the class-method factory::

        net = CondActionFlowNet.from_config(cfg)

    Parameters
    ----------
    h          : int – action horizon (H time steps)
    bev_h      : int – BEV grid height in cells
    bev_w      : int – BEV grid width  in cells
    d          : int – shared embedding dimension (default 128)
    num_layers : int – number of CondDiTBlocks    (default 4)
    heads      : int – attention heads           (default 4)
    goal_dim   : int – goal_vec dimension        (default 5)
    """

    def __init__(
        self,
        h          : int,
        bev_h      : int,
        bev_w      : int,
        d          : int = 128,
        num_layers : int = 4,
        heads      : int = 4,
        goal_dim   : int = 5,
    ) -> None:
        super().__init__()
        self.H        = h
        self.act_dim  = 2
        self.data_dim = h * 2
        self.d        = d

        # ── Condition encoder ──────────────────────────────────────────
        self.cond_enc = ToyConditionEncoder(
            bev_h=bev_h, bev_w=bev_w, d=d, goal_dim=goal_dim
        )

        # ── Action token projection: 2 → d ────────────────────────────
        self.action_in = nn.Linear(2, d)

        # ── Learnable positional embeddings for action sequence ────────
        self.action_pos_emb = nn.Parameter(torch.randn(1, h, d) * 0.02)

        # ── Time embedding: scalar t → d ──────────────────────────────
        self.time_embedder = nn.Sequential(
            FourierEncoder(d),
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, d),
        )

        # ── Transformer blocks ─────────────────────────────────────────
        self.blocks = nn.ModuleList([
            CondDiTBlock(d, heads) for _ in range(num_layers)
        ])

        # ── Final adaLN + output head ──────────────────────────────────
        self.final_norm   = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_adaLN  = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        self.out_proj = nn.Linear(d, 2)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg        : dict,
        d          : int = 128,
        num_layers : int = 4,
        heads      : int = 4,
    ) -> "CondActionFlowNet":
        """Instantiate with all dimensions derived from *cfg*."""
        from scripts.data.gen_demos import compute_H
        from perception.bev_builder import BEVBuilder
        h     = compute_H(cfg)
        bev   = BEVBuilder(cfg)
        return cls(
            h=h, bev_h=bev.H_cells, bev_w=bev.W_cells,
            d=d, num_layers=num_layers, heads=heads,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x             : torch.Tensor,              # [B, H*2]  noisy action seq
        t             : torch.Tensor,              # [B] or [B,1]
        bev_id        : torch.Tensor,              # Long[B, Hb, Wb]
        goal_vec      : torch.Tensor,              # Float[B, 5]
        global_is_null: Optional[torch.Tensor] = None,  # Bool[B]
    ) -> torch.Tensor:
        """
        Returns:
            v : [B, H*2]  predicted vector field (flattened)
        """
        B = x.shape[0]
        t = t.view(B)

        # ── Condition encoding ─────────────────────────────────────────
        bev_tokens, global_vec = self.cond_enc(bev_id, goal_vec, global_is_null)
        # [B, Nc, d],  [B, d]

        # ── Time embedding ─────────────────────────────────────────────
        t_emb    = self.time_embedder(t)                  # [B, d]
        cond_vec = global_vec + t_emb                     # [B, d]

        # ── Action tokens ──────────────────────────────────────────────
        x_seq  = x.view(B, self.H, 2)                    # [B, H, 2]
        tokens = self.action_in(x_seq) + self.action_pos_emb  # [B, H, d]

        # ── Transformer blocks ─────────────────────────────────────────
        for block in self.blocks:
            tokens = block(tokens, cond_vec, bev_tokens)

        # ── Final head ────────────────────────────────────────────────
        shift, scale = self.final_adaLN(cond_vec).chunk(2, dim=-1)  # [B,d] each
        tokens = self.final_norm(tokens) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        out = self.out_proj(tokens)                       # [B, H, 2]
        return out.view(B, -1)                            # [B, H*2]


# ---------------------------------------------------------------------------
# ODE wrapper for EulerSimulator
# ---------------------------------------------------------------------------

class CondActionFlowODE(ODE):
    """
    Wraps :class:`CondActionFlowNet` as an :class:`ODE`.

    Extra keyword arguments (bev_id, goal_vec, global_is_null) are forwarded
    to the network on each drift call.
    """

    def __init__(self, net: CondActionFlowNet) -> None:
        self.net = net

    def drift_coefficient(
        self,
        xt  : torch.Tensor,
        t   : torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.net(xt, t, **kwargs)

