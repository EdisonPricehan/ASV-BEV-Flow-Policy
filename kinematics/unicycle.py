"""
kinematics/unicycle.py
======================
Unicycle kinematic model.

State  : (x, y, yaw)  – position [m] and heading [rad]
Action : (v, omega)   – linear velocity [m/s] and angular rate [rad/s]

Forward-Euler integration:
    x'   = x   + v * cos(yaw) * dt
    y'   = y   + v * sin(yaw) * dt
    yaw' = yaw + omega * dt

Public API
----------
UnicycleSimulator   – stateless simulator class with step() and rollout()
wrap_to_pi()        – angle-wrapping utility (defined in utils.py, re-exported here)
"""

from __future__ import annotations

import numpy as np

from common.utils import wrap_to_pi  # re-export for callers that import from here

__all__ = ["UnicycleSimulator", "wrap_to_pi"]


# ---------------------------------------------------------------------------
# UnicycleSimulator
# ---------------------------------------------------------------------------

class UnicycleSimulator:
    """
    Stateless unicycle kinematic simulator.

    All methods are pure functions of their inputs – no internal state is
    mutated – so a single instance can safely be reused across episodes.

    Parameters
    ----------
    dt : float
        Default time step used when *dt* is not passed explicitly to
        :meth:`step` or :meth:`rollout`.  Defaults to 0.05 s.
    """

    def __init__(self, dt: float = 0.05) -> None:
        self.dt = float(dt)

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        dt: float | None = None,
    ) -> np.ndarray:
        """
        Single forward-Euler step of unicycle kinematics.

        Parameters
        ----------
        state  : (3,) float32 – [x, y, yaw]  in metres / radians
        action : (2,) float32 – [v, omega]   in m/s  / rad/s
        dt     : float | None – time step [s]; uses ``self.dt`` if None

        Returns
        -------
        next_state : (3,) float32
        """
        if dt is None:
            dt = self.dt
        x, y, yaw = state
        v, omega   = action
        return np.array(
            [x + v * np.cos(yaw) * dt,
             y + v * np.sin(yaw) * dt,
             yaw + omega * dt],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Full rollout
    # ------------------------------------------------------------------

    def rollout(
        self,
        init_state: np.ndarray,
        actions: np.ndarray,
        dt: float | None = None,
    ) -> np.ndarray:
        """
        Roll out unicycle kinematics from *init_state* for *H* steps.

        Parameters
        ----------
        init_state : (3,)   float32 – starting [x, y, yaw]
        actions    : (H, 2) float32 – action sequence [[v_0, ω_0], …, [v_{H-1}, ω_{H-1}]]
        dt         : float | None   – time step [s]; uses ``self.dt`` if None

        Returns
        -------
        states : (H+1, 3) float32
            Row 0 is *init_state*; row h+1 is the state after applying actions[h].
        """
        if dt is None:
            dt = self.dt
        H = len(actions)
        states = np.zeros((H + 1, 3), dtype=np.float32)
        states[0] = init_state
        for h in range(H):
            states[h + 1] = self.step(states[h], actions[h], dt)
        return states

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"UnicycleSimulator(dt={self.dt})"

