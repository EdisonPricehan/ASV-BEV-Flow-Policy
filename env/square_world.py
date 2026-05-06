"""
square_world.py
===============
Square-world map environment for 2-D flow policy.

All coordinates are in **metres**.  The map spans [0, L] × [0, L]
where  L = scale * D_bev.

Public API
----------
SquareWorld   – square map with circular obstacles, start/goal poses,
           collision checking, serialisation, and visualisation.

Dynamics live in kinematics/unicycle.py.
Shared utilities (wrap_to_pi) live in utils.py.
All parameters are managed in configs/common.json.
"""

from __future__ import annotations

import json
import os
from typing import List

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


class SquareWorld:
    """
    Square 2-D map with multiple circular obstacles and randomised
    start / goal poses, parameterised by a real-world BEV scale.

    All coordinates are stored in **metres**.

    Parameters
    ----------
    D_bev        : float – BEV depth reference [m]; map side L = scale * D_bev
    scale        : float – must be in [scale_min, scale_max]
    scale_min    : float
    scale_max    : float
    n_obstacles  : int
    r_min_frac   : float – min obstacle radius as fraction of L
    r_max_frac   : float – max obstacle radius as fraction of L
    margin_frac  : float – collision margin as fraction of L (default 0.01)
    seed         : int | None – RNG seed
    """

    def __init__(
        self,
        D_bev      : float = 10.0,
        scale      : float = 5.0,
        scale_min  : float = 2.0,
        scale_max  : float = 8.0,
        n_obstacles: int   = 1,
        r_min_frac : float = 0.03,
        r_max_frac : float = 0.08,
        margin_frac: float = 0.01,
        seed       : int | None = None,
    ) -> None:
        if not (scale_min <= scale <= scale_max):
            raise ValueError(f"scale={scale} outside [{scale_min}, {scale_max}]")

        self.D_bev       = float(D_bev)
        self.scale       = float(scale)
        self.scale_min   = float(scale_min)
        self.scale_max   = float(scale_max)
        self.n_obstacles = int(n_obstacles)
        self.r_min_frac  = float(r_min_frac)
        self.r_max_frac  = float(r_max_frac)
        self.margin_frac = float(margin_frac)
        self.seed        = seed

        self.L      = self.scale * self.D_bev  # map side length in metres
        self.margin = self.margin_frac * self.L  # collision margin in metres

        # Populated by randomise() or from_dict() / load()
        self.obstacle_centers: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self.obstacle_radii  : np.ndarray = np.zeros((0,),   dtype=np.float32)
        self.start_state     : np.ndarray = np.zeros(3,       dtype=np.float32)
        self.goal_pose       : np.ndarray = np.zeros(3,       dtype=np.float32)

    # ------------------------------------------------------------------
    # Factory: build from configs/common.json "map" section
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "SquareWorld":
        """
        Instantiate from the ``"map"`` section of *configs/common.json*.

        Example
        -------
        >>> import json
        >>> cfg = json.load(open("configs/common.json"))
        >>> m = SquareWorld.from_config(cfg["map"])
        """
        return cls(
            D_bev       = cfg["D_bev"],
            scale       = cfg["scale"],
            scale_min   = cfg["scale_min"],
            scale_max   = cfg["scale_max"],
            n_obstacles = cfg["n_obstacles"],
            r_min_frac  = cfg["r_min_frac"],
            r_max_frac  = cfg["r_max_frac"],
            margin_frac = cfg.get("margin_frac", 0.01),
            seed        = cfg.get("seed"),
        )

    # ------------------------------------------------------------------
    # Randomisation
    # ------------------------------------------------------------------

    def randomise(self, rng: np.random.Generator | None = None) -> "SquareWorld":
        """
        Randomly place obstacles, start pose, and goal pose.

        Guarantees
        ----------
        * Obstacles do not overlap each other.
        * Start and goal are collision-free and at least 0.25 L apart.
        * Both endpoints are at least ``border_pad`` from the boundary.

        Returns self for method chaining.
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        L          = self.L
        border_pad = 0.08 * L
        pose_clear = 0.05 * L   # extra clearance around obstacles for poses

        # ---- obstacles ------------------------------------------------
        centers: List[np.ndarray] = []
        radii:   List[float]      = []

        for i in range(self.n_obstacles):
            r = float(rng.uniform(self.r_min_frac * L, self.r_max_frac * L))
            for _ in range(5000):
                cx = float(rng.uniform(border_pad + r, L - border_pad - r))
                cy = float(rng.uniform(border_pad + r, L - border_pad - r))
                c  = np.array([cx, cy], dtype=np.float32)
                if all(np.linalg.norm(c - oc) > r + or_ + self.margin
                       for oc, or_ in zip(centers, radii)):
                    centers.append(c)
                    radii.append(r)
                    break
            else:
                raise RuntimeError(
                    f"Could not place obstacle {i+1}/{self.n_obstacles} "
                    "after 5000 attempts. Try fewer obstacles or a larger map."
                )

        self.obstacle_centers = np.array(centers, dtype=np.float32)
        self.obstacle_radii   = np.array(radii,   dtype=np.float32)

        # ---- start pose -----------------------------------------------
        self.start_state = self._sample_free_pose(rng, border_pad, pose_clear)

        # ---- goal pose ------------------------------------------------
        for _ in range(10000):
            goal = self._sample_free_pose(rng, border_pad, pose_clear)
            if np.linalg.norm(goal[:2] - self.start_state[:2]) > 0.25 * L:
                self.goal_pose = goal
                break
        else:
            raise RuntimeError(
                "Could not place goal far enough from start after 10 000 attempts."
            )

        return self

    def _sample_free_pose(
        self,
        rng        : np.random.Generator,
        border_pad : float,
        extra_clear: float,
    ) -> np.ndarray:
        """Sample [x, y, yaw] free of obstacles and boundary."""
        L = self.L
        for _ in range(50000):
            x   = float(rng.uniform(border_pad, L - border_pad))
            y   = float(rng.uniform(border_pad, L - border_pad))
            yaw = float(rng.uniform(-np.pi, np.pi))
            if self._point_free(np.array([x, y], dtype=np.float32), extra=extra_clear):
                return np.array([x, y, yaw], dtype=np.float32)
        raise RuntimeError("Could not sample a collision-free pose after 50 000 attempts.")

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------

    def _point_free(self, xy: np.ndarray, extra: float = 0.0) -> bool:
        """True if *xy* is inside the map and outside every inflated obstacle."""
        if np.any(xy < 0.0) or np.any(xy > self.L):
            return False
        for c, r in zip(self.obstacle_centers, self.obstacle_radii):
            if np.linalg.norm(xy - c) < r + self.margin + extra:
                return False
        return True

    def collision_check(
        self,
        traj_xy: np.ndarray,
        margin : float | None = None,
    ) -> bool:
        """
        Return True if *any* point of *traj_xy* (T, 2) collides with an
        obstacle (including margin) or exits the map boundary.
        """
        m = self.margin if margin is None else float(margin)
        if np.any(traj_xy < 0.0) or np.any(traj_xy > self.L):
            return True
        for c, r in zip(self.obstacle_centers, self.obstacle_radii):
            if np.any(np.linalg.norm(traj_xy - c, axis=-1) < r + m):
                return True
        return False

    # ------------------------------------------------------------------
    # Convenience property
    # ------------------------------------------------------------------

    @property
    def goal_xy(self) -> np.ndarray:
        """Goal (x, y) in metres, shape (2,)."""
        return self.goal_pose[:2].copy()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "D_bev"          : self.D_bev,
            "scale"          : self.scale,
            "scale_min"      : self.scale_min,
            "scale_max"      : self.scale_max,
            "n_obstacles"    : self.n_obstacles,
            "r_min_frac"     : self.r_min_frac,
            "r_max_frac"     : self.r_max_frac,
            "margin_frac"    : self.margin_frac,
            "seed"           : self.seed,
            "obstacle_centers": self.obstacle_centers.tolist(),
            "obstacle_radii"  : self.obstacle_radii.tolist(),
            "start_state"    : self.start_state.tolist(),
            "goal_pose"      : self.goal_pose.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SquareWorld":
        m = cls(
            D_bev       = d["D_bev"],
            scale       = d["scale"],
            scale_min   = d["scale_min"],
            scale_max   = d["scale_max"],
            n_obstacles = d["n_obstacles"],
            r_min_frac  = d["r_min_frac"],
            r_max_frac  = d["r_max_frac"],
            margin_frac = d.get("margin_frac", 0.01),
            seed        = d.get("seed"),
        )
        m.obstacle_centers = np.array(d["obstacle_centers"], dtype=np.float32)
        m.obstacle_radii   = np.array(d["obstacle_radii"],   dtype=np.float32)
        m.start_state      = np.array(d["start_state"],      dtype=np.float32)
        m.goal_pose        = np.array(d["goal_pose"],        dtype=np.float32)
        return m

    def save(self, path: str) -> None:
        """Save to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SquareWorld":
        """Load from JSON previously saved with :meth:`save`."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualise(
        self,
        ax               : plt.Axes | None    = None,
        trajectories     : List[np.ndarray] | None = None,
        traj_colors      : List       | None  = None,
        traj_alpha       : float               = 0.5,
        traj_lw          : float               = 1.0,
        show_goal_heading: bool                = True,
        title            : str | None          = None,
        save_path        : str | None          = None,
    ) -> plt.Axes:
        """
        Draw the map with optional trajectory overlays.

        Parameters
        ----------
        ax                : reuse an existing Axes; new figure if None
        trajectories      : list of (T, 2) or (T, 3) arrays – drawn without
                            individual legend entries to keep the plot clean
        traj_colors       : colour for each trajectory; cycles through a
                            colormap if None
        traj_alpha        : line transparency
        traj_lw           : line width
        show_goal_heading : draw arrow for goal yaw
        title             : override default axes title
        save_path         : save figure here and close; show interactively if None
        """
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(7, 7))

        L = self.L
        arrow_len = 0.05 * L   # length of heading arrows

        # ── boundary ──────────────────────────────────────────────────
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, 0), L, L,
            boxstyle="square,pad=0",
            linewidth=2, edgecolor="#2c3e50",
            facecolor="#f8f9fa", zorder=0,
        ))

        # ── obstacles + margin rings ───────────────────────────────────
        for c, r in zip(self.obstacle_centers, self.obstacle_radii):
            ax.add_patch(plt.Circle(
                c, r, color="#e74c3c", alpha=0.80, zorder=3,
            ))
            ax.add_patch(plt.Circle(
                c, r + self.margin,
                color="#e74c3c", alpha=0.18, zorder=2,
                fill=True, linewidth=0,
            ))

        # ── trajectories (no legend) ───────────────────────────────────
        if trajectories:
            cmap = matplotlib.colormaps["viridis"]
            n    = len(trajectories)
            for i, traj in enumerate(trajectories):
                xy    = np.asarray(traj)[:, :2]
                color = (traj_colors[i] if traj_colors and i < len(traj_colors)
                         else cmap(i / max(n - 1, 1)))
                ax.plot(xy[:, 0], xy[:, 1],
                        color=color, alpha=traj_alpha, linewidth=traj_lw,
                        zorder=4)

        # ── start: green circle + heading arrow ───────────────────────
        sx, sy, syaw = self.start_state
        ax.scatter(sx, sy, s=100, color="#27ae60", zorder=7,
                   label="Start", edgecolors="white", linewidths=0.8)
        ax.annotate(
            "", xy=(sx + arrow_len * np.cos(syaw),
                    sy + arrow_len * np.sin(syaw)),
            xytext=(sx, sy),
            arrowprops=dict(arrowstyle="-|>", color="#27ae60",
                            lw=2.0, mutation_scale=12),
            zorder=8,
        )

        # ── goal: orange circle + heading arrow ────────────────────────
        gx, gy, gyaw = self.goal_pose
        ax.scatter(gx, gy, s=100, color="#e67e22", zorder=7,
                   label="Goal", edgecolors="white", linewidths=0.8)
        if show_goal_heading:
            ax.annotate(
                "", xy=(gx + arrow_len * np.cos(gyaw),
                        gy + arrow_len * np.sin(gyaw)),
                xytext=(gx, gy),
                arrowprops=dict(arrowstyle="-|>", color="#e67e22",
                                lw=2.0, mutation_scale=12),
                zorder=8,
            )

        # ── axes decoration ───────────────────────────────────────────
        ax.set_xlim(-0.05 * L, 1.05 * L)
        ax.set_ylim(-0.05 * L, 1.05 * L)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.7)
        ax.set_title(title or (
            f"SquareWorld  L={L:.1f} m  (D_bev={self.D_bev} m × {self.scale}×)"
            f"  |  {self.n_obstacles} obstacle(s)"
        ), fontsize=10)

        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
        elif standalone:
            plt.tight_layout()
            plt.show()

        return ax

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SquareWorld(L={self.L:.2f} m, D_bev={self.D_bev} m, "
            f"scale={self.scale}, n_obstacles={self.n_obstacles}, "
            f"obstacles placed={len(self.obstacle_radii)})"
        )

