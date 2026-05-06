"""
utils.py
========
Shared utility functions for the flow policy project.
"""

from __future__ import annotations

import math
import numpy as np


def wrap_to_pi(angle: float) -> float:
    """Wrap *angle* (radians) to (−π, π]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def build_goal_vec(
    pose     : np.ndarray,   # (3,) [x, y, psi]
    goal_pose: np.ndarray,   # (3,) [x_g, y_g, psi_g]
    D_max    : float,
) -> np.ndarray:
    """
    Build the 5-D goal conditioning vector from current pose and goal pose.

    Definition
    ----------
      line_yaw = atan2(y_g - y_t, x_g - x_t)
      alpha    = wrap_to_pi(line_yaw - psi_t)   # heading error toward goal
      beta     = wrap_to_pi(psi_g - line_yaw)   # goal heading vs. line direction
      d_norm   = clip(dist / D_max, 0, 1)

    Returns
    -------
    float32 (5,): [sin(alpha), cos(alpha), sin(beta), cos(beta), d_norm]
    """
    x_t, y_t, psi_t = float(pose[0]),      float(pose[1]),      float(pose[2])
    x_g, y_g, psi_g = float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2])

    dx       = x_g - x_t
    dy       = y_g - y_t
    dist     = math.sqrt(dx * dx + dy * dy)
    line_yaw = math.atan2(dy, dx)

    alpha  = wrap_to_pi(line_yaw - psi_t)
    beta   = wrap_to_pi(psi_g   - line_yaw)
    d_norm = float(np.clip(dist / D_max, 0.0, 1.0))

    return np.array(
        [math.sin(alpha), math.cos(alpha),
         math.sin(beta),  math.cos(beta),
         d_norm],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Action normalisation
# ---------------------------------------------------------------------------

class ActionNormalizer:
    """
    Min-max normalisation for (v, omega) actions → [-1, 1].

    Parameters are read from cfg["controller"]:
        v_min, v_max   : linear velocity bounds  [m/s]
        omega_max      : angular velocity bound  [rad/s]  (symmetric: ±omega_max)

    Normalisation:
        v_scaled     = clip(v,     v_min,     v_max    ) mapped to [-1, 1]
        omega_scaled = clip(omega, -omega_max, omega_max) mapped to [-1, 1]

    Works on numpy arrays and torch tensors of shape (..., 2).
    """

    def __init__(self, cfg: dict) -> None:
        ctrl = cfg["controller"]
        self.v_min     = float(ctrl["v_min"])
        self.v_max     = float(ctrl["v_max"])
        self.omega_max = float(ctrl["omega_max"])

        # [act_lo, act_hi] per channel
        self._lo = np.array([self.v_min,    -self.omega_max], dtype=np.float32)
        self._hi = np.array([self.v_max,     self.omega_max], dtype=np.float32)

    # ---- numpy --------------------------------------------------------

    def normalize_np(self, actions: np.ndarray) -> np.ndarray:
        """(..., 2) float32 → [..., 2] in [-1, 1]."""
        clipped = np.clip(actions, self._lo, self._hi)
        return (clipped - self._lo) / (self._hi - self._lo) * 2.0 - 1.0

    def denormalize_np(self, actions_norm: np.ndarray) -> np.ndarray:
        """(..., 2) in [-1, 1] → [..., 2] raw actions."""
        return (actions_norm + 1.0) / 2.0 * (self._hi - self._lo) + self._lo

    # ---- torch --------------------------------------------------------

    def normalize(self, actions):
        """(..., 2) tensor → [..., 2] in [-1, 1]."""
        import torch
        lo = torch.as_tensor(self._lo, dtype=torch.float32, device=actions.device)
        hi = torch.as_tensor(self._hi, dtype=torch.float32, device=actions.device)
        clipped = torch.clamp(actions, lo, hi)
        return (clipped - lo) / (hi - lo) * 2.0 - 1.0

    def denormalize(self, actions_norm):
        """(..., 2) in [-1, 1] → [..., 2] raw actions (torch)."""
        import torch
        lo = torch.as_tensor(self._lo, dtype=torch.float32, device=actions_norm.device)
        hi = torch.as_tensor(self._hi, dtype=torch.float32, device=actions_norm.device)
        return (actions_norm + 1.0) / 2.0 * (hi - lo) + lo
