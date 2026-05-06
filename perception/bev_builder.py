"""
bev_builder.py
==============
BEV (Bird's-Eye View) semantic map builder for the toy unicycle policy.

Semantic labels
---------------
  0 – UNKNOWN   (outside the visible triangle, but inside the BEV rectangle)
  1 – FREE
  2 – OBSTACLE

Visibility model
----------------
Only a single forward-facing camera is assumed.  The visible region is the
triangle formed by:
  • the ship's current centre  (apex)
  • the two far corners of the BEV rectangle

Everything inside the BEV rectangle but outside that triangle is labelled
UNKNOWN.

BEV rectangle (in the ship's body frame)
-----------------------------------------
  • forward extent  : [0, D_bev]  (along the ship's heading)
  • lateral extent  : [-W_bev/2, +W_bev/2]  (perpendicular to heading)
  • resolution      : res  metres per cell
  • grid size       : (H_cells, W_cells) = (D_bev/res, W_bev/res)

All BEV parameters are read from the ``"bev"`` section of *configs/common.json*.

Public API
----------
BEVBuilder          – builds a BEV grid for a given (SquareWorld, pose)
visualise_bev_gif   – renders a GIF of BEV evolution along a trajectory
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from env.square_world import SquareWorld

# Semantic label constants
UNKNOWN  = 0
FREE     = 1
OBSTACLE = 2

# Colours for visualisation: UNKNOWN=grey, FREE=light-green, OBSTACLE=red
_LABEL_COLOURS = np.array([
    [0.55, 0.55, 0.55, 1.0],   # 0  UNKNOWN  – grey
    [0.72, 0.96, 0.72, 1.0],   # 1  FREE     – light green
    [0.95, 0.25, 0.25, 1.0],   # 2  OBSTACLE – red
], dtype=np.float32)


# ---------------------------------------------------------------------------
# BEVBuilder
# ---------------------------------------------------------------------------

class BEVBuilder:
    """
    Builds a 2-D semantic BEV grid given a SquareWorld and a ship pose.

    Parameters (all read from cfg["bev"])
    --------------------------------------
    D_bev : float  – forward depth of BEV rectangle [m]   (default 10.0)
    W_bev : float  – lateral width of BEV rectangle [m]   (default  6.0)
    res   : float  – grid resolution [m/cell]              (default  0.2)

    Grid layout
    -----------
    Row 0  ↔  farthest row from the ship  (top of the image = far end)
    Row -1 ↔  closest row                 (bottom of image = near end)
    Col 0  ↔  left edge   (-W_bev/2)
    Col -1 ↔  right edge  (+W_bev/2)

    This follows the natural "forward = up" convention for a top-down BEV.
    """

    def __init__(self, cfg: dict) -> None:
        bev_cfg    = cfg.get("bev", {})
        self.D_bev = float(bev_cfg.get("D_bev", 10.0))
        self.W_bev = float(bev_cfg.get("W_bev",  6.0))
        self.res   = float(bev_cfg.get("res",     0.2))

        # Grid dimensions
        self.H_cells = int(round(self.D_bev / self.res))   # depth  (rows)
        self.W_cells = int(round(self.W_bev / self.res))   # width  (cols)

        # Pre-compute cell centres in the body frame (forward=+x, left=+y)
        # Row i: forward distance (i + 0.5) * res  — row 0 will be flipped to far end
        # Col j: lateral offset  (j + 0.5) * res - W_bev/2   (negative = right side)
        #        After col-flip in build(): col 0 → left (+W_bev/2), col -1 → right (-W_bev/2)
        rows = (np.arange(self.H_cells) + 0.5) * self.res          # (H,)
        cols = (np.arange(self.W_cells) + 0.5) * self.res \
               - self.W_bev / 2.0                                    # (W,)  left<0, right>0
        fwd, lat = np.meshgrid(rows, cols, indexing="ij")            # (H, W)
        self._cell_fwd = fwd.astype(np.float32)   # (H_cells, W_cells)
        self._cell_lat = lat.astype(np.float32)   # (H_cells, W_cells)

        # Visibility triangle in body frame
        # Apex: (0, 0)
        # Far-left corner:  (D_bev, -W_bev/2)
        # Far-right corner: (D_bev, +W_bev/2)
        # A cell at (fwd, lat) is visible iff:
        #   |lat| <= fwd * (W_bev/2) / D_bev
        self._vis_slope = (self.W_bev / 2.0) / self.D_bev
        self._vis_mask  = (
            np.abs(self._cell_lat) <= self._cell_fwd * self._vis_slope
        )                                              # (H_cells, W_cells)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def build(self, world: SquareWorld, pose: np.ndarray) -> np.ndarray:
        """
        Build a BEV semantic grid for *pose* on *world*.

        Parameters
        ----------
        world : SquareWorld
        pose    : (3,) array  [x, y, yaw]  in world frame [m, m, rad]

        Returns
        -------
        grid : (H_cells, W_cells) uint8 array
               Row 0 = farthest from ship, Row -1 = closest to ship.
               Semantic labels: 0=UNKNOWN, 1=FREE, 2=OBSTACLE.
        """
        px, py, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        # Rotate body-frame cell centres to world frame
        # world_x = px + fwd * cos_yaw - lat * sin_yaw
        # world_y = py + fwd * sin_yaw + lat * cos_yaw
        wx = px + self._cell_fwd * cos_y - self._cell_lat * sin_y   # (H, W)
        wy = py + self._cell_fwd * sin_y + self._cell_lat * cos_y   # (H, W)

        # Start with all cells FREE
        grid = np.full((self.H_cells, self.W_cells), FREE, dtype=np.uint8)

        # Mark cells outside visibility triangle as UNKNOWN
        grid[~self._vis_mask] = UNKNOWN

        # Mark cells outside map boundary as UNKNOWN
        in_map = (
            (wx >= 0.0) & (wx <= world.L) &
            (wy >= 0.0) & (wy <= world.L)
        )
        grid[~in_map & (grid == FREE)] = UNKNOWN

        # Mark obstacle cells (only within visible, in-map cells)
        visible_free = (grid == FREE)
        for c, r in zip(world.obstacle_centers, world.obstacle_radii):
            dist = np.sqrt((wx - c[0]) ** 2 + (wy - c[1]) ** 2)
            grid[(dist < r) & visible_free] = OBSTACLE

        # Flip rows so that row 0 = far end   (top    of image = farthest from ship)
        # Flip cols so that col 0 = left side  (left   of image = port / left of ship)
        # Without col-flip: col=0 → lat=-W_bev/2 (right), which mirrors the image.
        return grid[::-1, ::-1].copy()

    # ------------------------------------------------------------------
    # Convenience: build + colour image
    # ------------------------------------------------------------------

    def to_rgb(self, grid: np.ndarray) -> np.ndarray:
        """
        Convert a label grid to an RGBA uint8 image.

        Returns
        -------
        rgba : (H_cells, W_cells, 4) uint8 array
        """
        return (_LABEL_COLOURS[grid] * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# GIF / video visualisation
# ---------------------------------------------------------------------------

def visualise_bev_gif(
    world    : SquareWorld,
    states     : np.ndarray,
    bev_builder: BEVBuilder,
    out_path   : str   = "outputs/bev_rollout.gif",
    fps        : int   = 10,
    stride     : int   = 1,
    map_ax_size: float = 4.0,
    bev_ax_size: float = 3.0,
) -> None:
    """
    Render a GIF showing the BEV semantic grid evolving as the ship follows
    a trajectory.

    Left panel  – global map view with ship pose overlaid on the trajectory.
    Right panel – current BEV grid (far end at top, near end at bottom).

    Parameters
    ----------
    world     : SquareWorld
    states      : (T+1, 3) array of [x, y, yaw] poses
    bev_builder : BEVBuilder
    out_path    : output file path (.gif)
    fps         : frames per second
    stride      : render every *stride*-th state (to reduce GIF size)
    map_ax_size : width of the map panel in inches
    bev_ax_size : width of the BEV panel in inches
    """
    import matplotlib.animation as animation

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    L         = world.L
    D_bev     = bev_builder.D_bev
    W_bev     = bev_builder.W_bev
    arrow_len = 0.04 * L

    frames_idx = list(range(0, len(states), stride))

    # ── figure layout ──────────────────────────────────────────────────
    fig_w = map_ax_size + bev_ax_size + 1.0
    fig_h = max(map_ax_size, bev_ax_size * (D_bev / W_bev)) + 0.8
    fig, (ax_map, ax_bev) = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [map_ax_size, bev_ax_size]},
    )
    fig.tight_layout(pad=1.4)

    # ── static map elements ────────────────────────────────────────────
    _draw_static_map(ax_map, world)
    traj_xy = states[:, :2]
    ax_map.plot(traj_xy[:, 0], traj_xy[:, 1],
                color="#aaaaaa", lw=0.8, zorder=3, alpha=0.6)

    # ── BEV axes decoration ────────────────────────────────────────────
    ax_bev.set_title("BEV semantic grid", fontsize=9)
    ax_bev.set_xlabel("lateral [m]", fontsize=8)
    ax_bev.set_ylabel("forward [m]  (↑ = far)", fontsize=8)
    ax_bev.set_xlim(-W_bev / 2, W_bev / 2)
    ax_bev.set_ylim(0, D_bev)
    ax_bev.set_aspect("equal")
    ax_bev.set_xticks(np.arange(-W_bev / 2, W_bev / 2 + 0.01, 1.0))
    ax_bev.set_yticks(np.arange(0, D_bev + 0.01, 1.0))
    ax_bev.tick_params(labelsize=6)

    # BEV image extent: x in [-W/2, W/2], y in [0, D_bev] (far=top)
    extent = [-W_bev / 2, W_bev / 2, 0, D_bev]

    # Legend patches for BEV
    legend_patches = [
        mpatches.Patch(color=_LABEL_COLOURS[UNKNOWN, :3], label="Unknown"),
        mpatches.Patch(color=_LABEL_COLOURS[FREE,    :3], label="Free"),
        mpatches.Patch(color=_LABEL_COLOURS[OBSTACLE, :3], label="Obstacle"),
    ]
    ax_bev.legend(handles=legend_patches, loc="upper right",
                  fontsize=7, framealpha=0.8)

    # BEV FOV boundary lines (in BEV axes coords – x=lateral, y=forward)
    slope = W_bev / 2.0 / D_bev
    ax_bev.plot([ slope * D_bev, 0], [D_bev, 0], color="black", lw=0.8, ls="--", alpha=0.5)
    ax_bev.plot([-slope * D_bev, 0], [D_bev, 0], color="black", lw=0.8, ls="--", alpha=0.5)

    # ── mutable artists ────────────────────────────────────────────────
    ship_dot, = ax_map.plot([], [], "o", color="#2980b9", ms=6, zorder=7)
    ship_arr  = ax_map.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="#2980b9", lw=1.8, mutation_scale=10),
        zorder=8,
    )
    bev_poly = plt.Polygon(
        np.zeros((4, 2)), closed=True,
        fill=False, edgecolor="#2980b9", lw=1.0, ls=":", zorder=6,
    )
    ax_map.add_patch(bev_poly)

    dummy_grid = np.zeros((bev_builder.H_cells, bev_builder.W_cells), dtype=np.uint8)
    bev_img = ax_bev.imshow(
        bev_builder.to_rgb(dummy_grid),
        origin="upper",
        extent=extent,
        interpolation="nearest",
        aspect="auto",
        zorder=2,
    )
    step_txt = ax_bev.text(
        0.02, 0.02, "", transform=ax_bev.transAxes,
        fontsize=7, color="black", va="bottom",
    )

    def _update(frame_i: int):
        pose = states[frames_idx[frame_i]]
        px, py, yaw = float(pose[0]), float(pose[1]), float(pose[2])

        # ship dot + arrow on map
        ship_dot.set_data([px], [py])
        ship_arr.xy     = (px + arrow_len * np.cos(yaw),
                           py + arrow_len * np.sin(yaw))
        ship_arr.xytext = (px, py)

        # BEV rectangle footprint on map
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        hw = W_bev / 2.0
        corners_body = np.array([
            [0,     -hw],
            [0,      hw],
            [D_bev,  hw],
            [D_bev, -hw],
        ], dtype=np.float32)
        corners_world = np.column_stack([
            px + corners_body[:, 0] * cos_y - corners_body[:, 1] * sin_y,
            py + corners_body[:, 0] * sin_y + corners_body[:, 1] * cos_y,
        ])
        bev_poly.set_xy(corners_world)

        # BEV grid
        grid = bev_builder.build(world, pose)
        bev_img.set_data(bev_builder.to_rgb(grid))
        step_txt.set_text(f"step {frames_idx[frame_i]}")

        return ship_dot, bev_img, step_txt, bev_poly

    ani = animation.FuncAnimation(
        fig, _update,
        frames=len(frames_idx),
        interval=int(1000 / fps),
        blit=False,
    )

    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"BEV animation saved → {out_path}  ({len(frames_idx)} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_static_map(ax: plt.Axes, world: SquareWorld) -> None:
    """Draw map boundary, obstacles, start and goal onto *ax*."""
    L = world.L
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), L, L, boxstyle="square,pad=0",
        linewidth=2, edgecolor="#2c3e50", facecolor="#f8f9fa", zorder=0,
    ))
    for c, r in zip(world.obstacle_centers, world.obstacle_radii):
        ax.add_patch(plt.Circle(c, r, color="#e74c3c", alpha=0.80, zorder=3))
        ax.add_patch(plt.Circle(c, r + world.margin,
                                color="#e74c3c", alpha=0.15, zorder=2,
                                fill=True, linewidth=0))
    arrow_len = 0.04 * L
    sx, sy, syaw = world.start_state
    ax.scatter(sx, sy, s=80, color="#27ae60", zorder=7, edgecolors="white", lw=0.8)
    ax.annotate("",
                xy=(sx + arrow_len * np.cos(syaw), sy + arrow_len * np.sin(syaw)),
                xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=1.8, mutation_scale=10),
                zorder=8)
    gx, gy, gyaw = world.goal_pose
    ax.scatter(gx, gy, s=80, color="#e67e22", zorder=7, edgecolors="white", lw=0.8)
    ax.annotate("",
                xy=(gx + arrow_len * np.cos(gyaw), gy + arrow_len * np.sin(gyaw)),
                xytext=(gx, gy),
                arrowprops=dict(arrowstyle="-|>", color="#e67e22", lw=1.8, mutation_scale=10),
                zorder=8)
    ax.set_xlim(-0.05 * L, 1.05 * L)
    ax.set_ylim(-0.05 * L, 1.05 * L)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.set_title(f"Global map  L={L:.0f} m", fontsize=9)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from project_config import POLICY_DIR as _POLICY_DIR, OUTPUTS_DIR as _OUTPUTS_DIR
    parser = argparse.ArgumentParser(
        description="Visualise BEV evolution along a demo trajectory."
    )
    parser.add_argument("--map",    required=True,  help="Path to map.json")
    parser.add_argument("--demos",  required=True,  help="Path to demos.npz")
    parser.add_argument("--idx",    type=int, default=0,
                        help="Demo index to visualise (default: 0)")
    parser.add_argument("--cfg",    default=str(_POLICY_DIR / "configs" / "common.json"),
                        help="Path to configs/common.json")
    parser.add_argument("--out",    default=str(_OUTPUTS_DIR / "bev_rollout.gif"),
                        help="Output GIF path")
    parser.add_argument("--fps",    type=int, default=10)
    parser.add_argument("--stride", type=int, default=3,
                        help="Render every N-th step to reduce GIF size (default: 3)")
    args = parser.parse_args()

    from project_config import load_config
    cfg = load_config("common")
    world  = SquareWorld.load(args.map)
    data     = np.load(args.demos)
    states   = data["demos_states"]
    lengths  = data["demo_lengths"]

    idx = args.idx
    T   = int(lengths[idx])
    traj_states = states[idx, :T + 1]   # (T+1, 3) – strip padding

    bev = BEVBuilder(cfg)
    print(f"BEV grid : {bev.H_cells} × {bev.W_cells} cells  "
          f"(D={bev.D_bev} m, W={bev.W_bev} m, res={bev.res} m/cell)")
    print(f"Demo #{idx}: {T} steps  ({T * cfg['simulator']['dt']:.1f} s)")

    visualise_bev_gif(
        world     = world,
        states      = traj_states,
        bev_builder = bev,
        out_path    = args.out,
        fps         = args.fps,
        stride      = args.stride,
    )

