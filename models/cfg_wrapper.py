"""
cfg_wrapper.py
==============
Classifier-Free Guidance (CFG) for the conditional flow policy.

Strategy: global-only CFG.
  - BEV is always conditioned (never dropped).
  - goal_vec is dropped by setting global_is_null=True.

Usage at inference::

    from models.cfg_wrapper import drift_cfg, CondFlowODE
    from models.simulators import EulerSimulator

    ode  = CondFlowODE(model, bev_id, goal_vec, cfg_scale=1.5)
    sim  = EulerSimulator(ode)
    x1   = sim.simulate(x0, ts, use_tqdm=False)
"""

from __future__ import annotations

import torch
from models.simulators import ODE


def drift_cfg(
    model     : "CondActionFlowNet",   # noqa: F821
    x         : torch.Tensor,             # [B, D]
    t         : torch.Tensor,             # [B]
    bev_id    : torch.Tensor,             # Long[B, Hb, Wb]
    goal_vec  : torch.Tensor,             # Float[B, 5]
    cfg_scale : float = 1.0,
) -> torch.Tensor:
    """
    Compute the CFG-blended drift vector.

    v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)

    where:
      v_uncond = model(x, t, bev_id, goal_vec, global_is_null=True)
      v_cond   = model(x, t, bev_id, goal_vec, global_is_null=False)

    For cfg_scale=1.0 this reduces to the fully-conditioned drift v_cond.
    For cfg_scale=0.0 this reduces to the BEV-only (no-goal) drift v_uncond.

    Parameters
    ----------
    model     : CondActionFlowNet
    x         : noisy action sequence  [B, D]
    t         : flow time              [B]
    bev_id    : BEV semantic grid      Long[B, Hb, Wb]
    goal_vec  : goal conditioning vec  Float[B, 5]
    cfg_scale : guidance scale (≥ 1.0 increases goal adherence)

    Returns
    -------
    v_cfg : [B, D]
    """
    B = x.shape[0]

    # Unconditional drift (global dropped, BEV kept)
    null_mask = torch.ones(B, dtype=torch.bool, device=x.device)
    v_uncond  = model(x, t, bev_id, goal_vec, global_is_null=null_mask)

    if cfg_scale == 1.0:
        # Skip second forward pass when guidance = 1 (reduces to v_cond)
        # Actually cfg_scale=1 → v_cfg = v_uncond + 1*(v_cond - v_uncond) = v_cond
        # so we still need v_cond.  But if cfg_scale==1 we can fuse in one pass.
        cond_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
        v_cond    = model(x, t, bev_id, goal_vec, global_is_null=cond_mask)
        return v_cond

    # General case: two separate passes
    cond_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
    v_cond    = model(x, t, bev_id, goal_vec, global_is_null=cond_mask)

    return v_uncond + cfg_scale * (v_cond - v_uncond)


class CondFlowODE(ODE):
    """
    ODE wrapper that applies CFG at every drift call.

    Bind the conditioning inputs once at construction so that
    :class:`~simulators.EulerSimulator` can call ``drift_coefficient``
    with only ``(xt, t)``.

    Parameters
    ----------
    model     : CondActionFlowNet
    bev_id    : Long[B, Hb, Wb]
    goal_vec  : Float[B, 5]
    cfg_scale : float
    """

    def __init__(
        self,
        model    : "CondActionFlowNet",  # noqa: F821
        bev_id   : torch.Tensor,
        goal_vec : torch.Tensor,
        cfg_scale: float = 1.0,
    ) -> None:
        self.model     = model
        self.bev_id    = bev_id
        self.goal_vec  = goal_vec
        self.cfg_scale = cfg_scale

    def drift_coefficient(
        self,
        xt: torch.Tensor,
        t : torch.Tensor,
        **_kwargs,
    ) -> torch.Tensor:
        return drift_cfg(
            self.model, xt, t,
            self.bev_id, self.goal_vec,
            self.cfg_scale,
        )

