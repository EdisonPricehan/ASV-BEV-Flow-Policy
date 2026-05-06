"""
simulators.py
=============
ODE / SDE classes and Euler / Euler-Maruyama numerical simulators.

Classes
-------
ODE                  – Abstract ordinary differential equation.
SDE                  – Abstract stochastic differential equation.
Simulator            – Abstract simulator (step + simulate helpers).
EulerSimulator       – First-order Euler integrator for ODEs.
EulerMaruyamaSimulator – Euler-Maruyama integrator for SDEs.

Functions
---------
record_every         – Index helper for sub-sampled trajectory recording.
"""

from abc import ABC, abstractmethod

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Abstract differential equations
# ---------------------------------------------------------------------------

class ODE(ABC):
    """Ordinary differential equation: dx/dt = f(x, t)."""

    @abstractmethod
    def drift_coefficient(
        self, xt: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            xt: (b, *)
            t:  (b,)
        Returns:
            drift: (b, *)
        """


class SDE(ABC):
    """Stochastic differential equation: dx = f(x,t) dt + g(x,t) dW."""

    @abstractmethod
    def drift_coefficient(
        self, xt: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            xt: (b, *)
            t:  (b,)
        Returns:
            drift: (b, *)
        """

    @abstractmethod
    def diffusion_coefficient(
        self, xt: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            xt: (b, *)
            t:  (b,)
        Returns:
            diffusion: (b, *)
        """


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------

class Simulator(ABC):
    """Abstract numerical simulator."""

    @abstractmethod
    def step(
        self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Take one integration step from time *t* to *t + dt*."""

    @torch.no_grad()
    def simulate(
        self,
        x: torch.Tensor,
        ts: torch.Tensor,
        use_tqdm: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Simulate from x[0] to x[-1] using the time grid *ts*.

        Args:
            x:  initial condition (b, *)
            ts: time grid (b, T)
        Returns:
            x_final: (b, *)
        """
        nts = ts.shape[1]
        itr = tqdm(range(nts - 1)) if use_tqdm else range(nts - 1)
        for t_idx in itr:
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(
        self,
        x: torch.Tensor,
        ts: torch.Tensor,
        use_tqdm: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Same as ``simulate`` but also returns the full trajectory.

        Returns:
            x_traj: (b, T, *)
        """
        x_traj = [x.clone()]
        nts = ts.shape[1]
        itr = tqdm(range(nts - 1)) if use_tqdm else range(nts - 1)
        for t_idx in itr:
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            x_traj.append(x.clone())
        return torch.stack(x_traj, dim=1)


class EulerSimulator(Simulator):
    """First-order Euler integrator for an ODE."""

    def __init__(self, ode: ODE):
        self.ode = ode

    def step(
        self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        h = h.view([-1] + [1] * (xt.ndim - 1))
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h


class EulerMaruyamaSimulator(Simulator):
    """Euler-Maruyama integrator for an SDE."""

    def __init__(self, sde: SDE):
        self.sde = sde

    def step(
        self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        h = h.view([-1] + [1] * (xt.ndim - 1))
        drift = self.sde.drift_coefficient(xt, t, **kwargs) * h
        diffusion = (
            self.sde.diffusion_coefficient(xt, t, **kwargs)
            * torch.sqrt(h)
            * torch.randn_like(xt)
        )
        return xt + drift + diffusion


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def record_every(num_timesteps: int, every: int) -> torch.Tensor:
    """
    Return indices to record from a trajectory so that every *every*-th step
    is captured (plus always the last step).
    """
    if every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, every),
            torch.tensor([num_timesteps - 1]),
        ]
    )

