"""
condition_encoder.py
====================
Encodes BEV semantic grid and goal vector into conditioning signals for
the conditional flow policy.

  bev_id   : Long[B, Hb, Wb]   – semantic labels {0,1,2}
  goal_vec : Float[B, 5]        – [sin_alpha, cos_alpha, sin_beta, cos_beta, d_norm]

Outputs:
  bev_tokens  : Float[B, Nc, d]  – spatial BEV tokens for cross-attention
  global_vec  : Float[B, d]      – fused goal embedding for AdaLN / FiLM

CFG strategy
------------
Only the *global* condition (goal_vec) is dropped.  BEV is never dropped.
When global_is_null=True, global_vec is replaced by a learnable null vector
(nn.Parameter) that is independent of goal_vec.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyConditionEncoder(nn.Module):
    """
    Lightweight condition encoder for the toy unicycle flow policy.

    Parameters
    ----------
    bev_h    : int  – BEV grid height in cells  (H_cells)
    bev_w    : int  – BEV grid width  in cells  (W_cells)
    d        : int  – shared embedding dimension for both BEV tokens and
                      global_vec (must match the action backbone width)
    goal_dim : int  – dimensionality of goal_vec (default 5)
    """

    def __init__(
        self,
        bev_h   : int,
        bev_w   : int,
        d       : int,
        goal_dim: int = 5,
    ) -> None:
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.d     = d

        # ── BEV tokeniser ──────────────────────────────────────────────
        # Input: one-hot [B, 3, Hb, Wb]
        # Output: [B, d, Hb//4, Wb//4]  (two ×2 downsamples)
        self.bev_cnn = nn.Sequential(
            nn.Conv2d(3,  32, kernel_size=3, stride=1, padding=1),  nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  nn.ReLU(),
            nn.Conv2d(64,  d, kernel_size=3, stride=2, padding=1),  nn.ReLU(),
        )

        # Dynamically infer the spatial size after the CNN (stride-2 conv on
        # odd dimensions rounds differently than simple //4).
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, bev_h, bev_w)
            _out   = self.bev_cnn(_dummy)
            self._bev_h_out: int = int(_out.shape[2])
            self._bev_w_out: int = int(_out.shape[3])
            self._Nc        : int = self._bev_h_out * self._bev_w_out

        # Learnable 2-D positional embedding  [1, Nc, d]
        self.bev_pos_emb = nn.Parameter(
            torch.randn(1, self._Nc, d) * 0.02
        )

        # ── Goal encoder (MLP) ─────────────────────────────────────────
        self.goal_mlp = nn.Sequential(
            nn.Linear(goal_dim, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        # Learnable null global vector (used when global_is_null=True)
        self.null_global = nn.Parameter(torch.zeros(d))

    # ------------------------------------------------------------------

    @property
    def Nc(self) -> int:
        """Number of BEV tokens (H'×W' after CNN downsampling)."""
        return self._Nc

    # ------------------------------------------------------------------

    def forward(
        self,
        bev_id        : torch.Tensor,                 # Long[B, Hb, Wb]
        goal_vec      : torch.Tensor,                 # Float[B, 5]
        global_is_null: Optional[torch.Tensor] = None # Bool[B] or None
    ):
        """
        Returns
        -------
        bev_tokens : Float[B, Nc, d]
        global_vec : Float[B, d]
        """
        B = bev_id.shape[0]

        # ── BEV ────────────────────────────────────────────────────────
        # one-hot:  Long[B,Hb,Wb] → Float[B,3,Hb,Wb]
        bev_oh = F.one_hot(bev_id.long(), num_classes=3).permute(0, 3, 1, 2).float()
        feat   = self.bev_cnn(bev_oh)                     # [B, d, H', W']
        bev_tokens = feat.flatten(2).transpose(1, 2)      # [B, Nc, d]
        bev_tokens = bev_tokens + self.bev_pos_emb        # add pos emb

        # ── Goal ───────────────────────────────────────────────────────
        gv = self.goal_mlp(goal_vec)                      # [B, d]

        if global_is_null is not None:
            # Replace rows where global_is_null=True with the null vector
            null = self.null_global.unsqueeze(0).expand(B, -1)   # [B, d]
            mask = global_is_null.view(B, 1).float()             # [B, 1]
            gv   = gv * (1.0 - mask) + null * mask

        return bev_tokens, gv


