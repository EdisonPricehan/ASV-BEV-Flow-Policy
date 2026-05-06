"""
trajectory_metrics.py
=====================
Scalar scoring metrics for a single candidate trajectory in the toy
unicycle environment.

Metric groups
-------------
safety  - collision avoidance and clearance from obstacles
goal    - positional and heading error at the final state
smooth  - angular-rate jerkiness
total   - weighted combination of the three

All costs are clipped to [0, 1] so the individual score ranges are
[-1, 0]  (except total which is the sum, so ≥ -4).

Public API
----------
default_metrics_cfg()               -> dict
compute_trajectory_metrics(...)     -> dict
visualize_trajectory_metrics(...)   -> None
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


from common.utils import wrap_to_pi
from env.square_world import SquareWorld
from kinematics.unicycle import UnicycleSimulator
from perception.bev_builder import BEVBuilder, _LABEL_COLOURS, UNKNOWN, FREE, OBSTACLE


# ---------------------------------------------------------------------------
# Default metrics config
# ---------------------------------------------------------------------------

def default_metrics_cfg() -> dict:
    """
    Return a dict with sensible defaults for all metric hyperparameters.

    Override individual keys before passing to compute_trajectory_metrics().
    """
    return {
        # Safety
        "clearance_safe"  : 0.5,    # [m] desired minimum clearance to obstacle surface
        "clearance_weight": 1.0,    # weight of clearance score in secondary ranking
        # Goal
        "pos_ref"         : 2.0,    # [m] distance reference for normalising pos error
        # Smooth
        "delta_omega_ref" : 0.3,    # [rad/s] reference delta-omega for normalisation
        # Total
        "lambda_smooth"   : 0.5,    # weight on smooth_score in total
    }


# ---------------------------------------------------------------------------
# Clearance helpers
# ---------------------------------------------------------------------------

def _min_clearance_to_obstacles(
    traj_xy        : np.ndarray,   # (T, 2)
    obstacle_centers: np.ndarray,  # (N, 2)
    obstacle_radii  : np.ndarray,  # (N,)
) -> tuple[float, int]:
    """
    Compute the minimum clearance (signed distance from obstacle surface)
    across all trajectory points and all obstacles.

    Returns
    -------
    min_clearance : float  - positive = outside obstacle, negative = inside
    argmin_idx    : int    - index in traj_xy with minimum clearance
    """
    if len(obstacle_centers) == 0:
        return np.inf, 0

    # dist_to_center[i, j] = distance from traj point i to obstacle j centre
    diff = traj_xy[:, np.newaxis, :] - obstacle_centers[np.newaxis, :, :]   # (T, N, 2)
    dist_to_center = np.linalg.norm(diff, axis=-1)                           # (T, N)

    # signed clearance: positive outside, negative inside
    clearance = dist_to_center - obstacle_radii[np.newaxis, :]               # (T, N)

    min_all  = clearance.min()
    flat_idx = clearance.argmin()
    row_idx  = flat_idx // len(obstacle_radii)
    return float(min_all), int(row_idx)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_trajectory_metrics(
    map_obj    : SquareWorld,
    bev_builder: BEVBuilder,
    init_pose  : np.ndarray,    # (3,) [x, y, yaw]
    goal_pose  : np.ndarray,    # (3,) [xg, yg, psig]
    action_seq : np.ndarray,    # (H, 2) [v, omega]  raw (not normalised)
    dt         : float,
    config     : Optional[dict] = None,
) -> dict:
    """
    Roll out the unicycle from *init_pose* with *action_seq* and compute
    scoring metrics.

    Parameters
    ----------
    map_obj     : SquareWorld with obstacles already placed
    bev_builder : BEVBuilder used for Fig 3 (BEV projection)
    init_pose   : (3,) starting [x, y, yaw]
    goal_pose   : (3,) target   [xg, yg, psig]
    action_seq  : (H, 2) raw action sequence [v, omega]
    dt          : time step [s]
    config      : metric hyperparameters (uses default_metrics_cfg() if None)

    Returns
    -------
    metrics : dict with keys
        safety, goal, smooth, total, rollout
    """
    cfg = default_metrics_cfg()
    if config is not None:
        cfg.update(config)

    # ------------------------------------------------------------------ #
    # 1. Rollout
    # ------------------------------------------------------------------ #
    sim        = UnicycleSimulator(dt=dt)
    state_seq  = sim.rollout(init_pose, action_seq, dt=dt)   # (H+1, 3)
    traj_xy    = state_seq[:, :2]                            # (H+1, 2)

    # ------------------------------------------------------------------ #
    # 2. Safety
    # ------------------------------------------------------------------ #
    collision_any = bool(map_obj.collision_check(traj_xy))

    min_clr, argmin_idx = _min_clearance_to_obstacles(
        traj_xy,
        map_obj.obstacle_centers,
        map_obj.obstacle_radii,
    )

    clearance_safe = cfg["clearance_safe"]
    collision_cost = 1.0 if collision_any else 0.0
    clearance_cost = float(np.clip(
        (clearance_safe - min_clr) / clearance_safe, 0.0, 1.0
    ))
    safety_score   = -(collision_cost + clearance_cost)

    # ------------------------------------------------------------------ #
    # 3. Goal
    # ------------------------------------------------------------------ #
    final_state    = state_seq[-1]
    final_pos_err  = float(np.linalg.norm(final_state[:2] - goal_pose[:2]))
    final_yaw_err  = float(abs(wrap_to_pi(final_state[2] - goal_pose[2])))

    pos_ref        = cfg["pos_ref"]
    pos_cost       = float(np.clip(final_pos_err / pos_ref,   0.0, 1.0))
    yaw_cost       = float(np.clip(final_yaw_err / np.pi,      0.0, 1.0))
    goal_score     = -(pos_cost + yaw_cost)

    # ------------------------------------------------------------------ #
    # 4. Smooth
    # ------------------------------------------------------------------ #
    omegas            = action_seq[:, 1]                        # (H,)
    delta_omega       = np.abs(np.diff(omegas))                 # (H-1,)
    mean_abs_d_omega  = float(delta_omega.mean()) if len(delta_omega) else 0.0

    d_omega_ref  = cfg["delta_omega_ref"]
    smooth_cost  = float(np.clip(mean_abs_d_omega / d_omega_ref, 0.0, 1.0))
    smooth_score = -smooth_cost

    # ------------------------------------------------------------------ #
    # 5. Total
    # ------------------------------------------------------------------ #
    lam         = cfg["lambda_smooth"]
    total_score = safety_score + goal_score + lam * smooth_score

    # ------------------------------------------------------------------ #
    # 6. BEV at init_pose (stored for visualisation)
    # ------------------------------------------------------------------ #
    bev_grid = bev_builder.build(map_obj, init_pose)   # (Hb, Wb) uint8

    return {
        "safety": {
            "collision_any" : collision_any,
            "min_clearance" : min_clr,
            "safety_score"  : safety_score,
            # internals
            "_collision_cost" : collision_cost,
            "_clearance_cost" : clearance_cost,
            "_argmin_idx"     : argmin_idx,
        },
        "goal": {
            "final_pos_error" : final_pos_err,
            "final_yaw_error" : final_yaw_err,
            "goal_score"      : goal_score,
        },
        "smooth": {
            "mean_abs_delta_omega" : mean_abs_d_omega,
            "smooth_score"         : smooth_score,
        },
        "total": {
            "total_score" : total_score,
        },
        "rollout": {
            "state_seq" : state_seq,    # (H+1, 3)
            "traj_xy"   : traj_xy,      # (H+1, 2)
            "bev_grid"  : bev_grid,     # (Hb, Wb) uint8
        },
    }


# ---------------------------------------------------------------------------
# Best-action selection
# ---------------------------------------------------------------------------

def select_best_action_seq(
    map_obj      : SquareWorld,
    bev_builder  : BEVBuilder,
    init_pose    : np.ndarray,          # (3,) [x, y, yaw]
    goal_pose    : np.ndarray,          # (3,) [xg, yg, psig]
    action_seqs  : np.ndarray,          # (M, H, 2) raw (not normalised)
    dt           : float,
    config       : Optional[dict] = None,
    exec_horizon : Optional[int]  = None,
) -> tuple[int, np.ndarray, list[dict]]:
    """
    Select the best action sequence from *M* candidates.

    Selection rule
    --------------
    1. Compute full metrics for every candidate.
    2. **Safety gate (exec_horizon)**:  a candidate is inadmissible if
       the first ``exec_horizon`` steps collide with any obstacle.
       Using exec_horizon < H means a trajectory that is safe for the
       *immediately executed* portion is still admissible even if it
       would later enter an obstacle — the policy will re-plan before
       those later steps are reached.
       If exec_horizon is None the full horizon H is checked (old behaviour).
    3. Among admissible candidates rank by:
           goal_score + lambda_smooth * smooth_score
                      + clearance_weight * clearance_score
       where clearance_score = clip(min_clearance / clearance_safe, 0, 1) - 1
       (0 when clearance ≥ clearance_safe, -1 when touching the obstacle).
    4. **Fallback**: if *all* candidates fail the exec-horizon safety gate,
       select the one with the highest safety_score (least bad).

    Parameters
    ----------
    exec_horizon : int or None
        Number of leading steps to check for collision.  Recommended value:
        the exec_steps used in the MPC loop.  None = check full H.
    """
    cfg = default_metrics_cfg()
    if config is not None:
        cfg.update(config)

    lam              = cfg["lambda_smooth"]
    clearance_weight = cfg.get("clearance_weight", 1.0)
    clearance_safe   = cfg["clearance_safe"]
    M                = len(action_seqs)

    all_metrics: list[dict] = []
    for seq in action_seqs:
        m = compute_trajectory_metrics(
            map_obj     = map_obj,
            bev_builder = bev_builder,
            init_pose   = init_pose,
            goal_pose   = goal_pose,
            action_seq  = seq,
            dt          = dt,
            config      = cfg,
        )
        all_metrics.append(m)

    # ── safety gate on the exec window only ───────────────────────────
    if exec_horizon is not None and exec_horizon < action_seqs.shape[1]:
        # re-check collision on the truncated trajectory
        sim = UnicycleSimulator(dt=dt)
        exec_safe = []
        for seq in action_seqs:
            trunc_states = sim.rollout(init_pose, seq[:exec_horizon], dt=dt)
            exec_safe.append(
                not map_obj.collision_check(trunc_states[:, :2])
            )
        safe_mask = np.array(exec_safe)
    else:
        # full-horizon collision flag (original behaviour)
        safe_mask = np.array(
            [not m["safety"]["collision_any"] for m in all_metrics]
        )

    def _secondary_score(m: dict) -> float:
        """goal + smooth + clearance composite (higher = better)."""
        clr      = m["safety"]["min_clearance"]
        clr_score = float(np.clip(clr / clearance_safe, 0.0, 1.0)) - 1.0  # ∈ [-1, 0]
        return (m["goal"]["goal_score"]
                + lam             * m["smooth"]["smooth_score"]
                + clearance_weight * clr_score)

    if safe_mask.any():
        scores = np.array([
            _secondary_score(m) if safe_mask[i] else -np.inf
            for i, m in enumerate(all_metrics)
        ])
        best_idx = int(np.argmax(scores))
    else:
        # all fail exec-horizon gate → least bad full-horizon safety
        safety_scores = np.array(
            [m["safety"]["safety_score"] for m in all_metrics]
        )
        best_idx = int(np.argmax(safety_scores))

    return best_idx, action_seqs[best_idx], all_metrics


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_trajectory_metrics(
    metrics    : dict,
    map_obj    : SquareWorld,
    bev_builder: BEVBuilder,
    init_pose  : np.ndarray,   # (3,)
    goal_pose  : np.ndarray,   # (3,)
    action_seq : np.ndarray,   # (H, 2) raw
    dt         : float,
    save_path  : Optional[str] = None,
    label      : str           = "",
) -> None:
    """
    Produce three figures:

    Fig 1 - Global map + rollout trajectory
    Fig 2 - Control sequence (v and omega)
    Fig 3 - Egocentric BEV at the start pose with trajectory projected onto it

    Parameters
    ----------
    metrics    : output of compute_trajectory_metrics()
    map_obj    : SquareWorld
    bev_builder: BEVBuilder
    init_pose  : (3,) starting [x, y, yaw]
    goal_pose  : (3,) target   [xg, yg, psig]
    action_seq : (H, 2) raw
    dt         : time step
    save_path  : if given, save to this path (PNG) and close; else plt.show()
    label      : short string added to every title for identification
    """
    sf   = metrics["safety"]
    gf   = metrics["goal"]
    sm   = metrics["smooth"]
    tot  = metrics["total"]
    ro   = metrics["rollout"]

    traj_xy   = ro["traj_xy"]    # (H+1, 2)
    state_seq = ro["state_seq"]  # (H+1, 3)
    bev_grid  = ro["bev_grid"]   # (Hb, Wb) uint8

    L         = map_obj.L
    arrow_len = 0.04 * L

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        f"Trajectory Metrics  {('— ' + label) if label else ''}   "
        f"total={tot['total_score']:.3f}",
        fontsize=12, fontweight="bold",
    )

    # ================================================================== #
    # Fig 1 - Global map + trajectory
    # ================================================================== #
    ax1 = fig.add_subplot(1, 3, 1)

    # Map boundary
    ax1.add_patch(mpatches.FancyBboxPatch(
        (0, 0), L, L, boxstyle="square,pad=0",
        linewidth=2, edgecolor="#2c3e50", facecolor="#f8f9fa", zorder=0,
    ))

    # Obstacles
    for c, r in zip(map_obj.obstacle_centers, map_obj.obstacle_radii):
        ax1.add_patch(plt.Circle(c, r, color="#e74c3c", alpha=0.80, zorder=3))
        ax1.add_patch(plt.Circle(
            c, r + map_obj.margin,
            color="#e74c3c", alpha=0.15, zorder=2, fill=True, linewidth=0,
        ))

    # Trajectory
    color_traj = "#e74c3c" if sf["collision_any"] else "#2980b9"
    ax1.plot(traj_xy[:, 0], traj_xy[:, 1],
             color=color_traj, lw=1.8, zorder=4, alpha=0.85, label="Trajectory")

    # Collision points (red dots)
    if sf["collision_any"]:
        for pt in traj_xy:
            in_obs = any(
                np.linalg.norm(pt - c) < r + map_obj.margin
                for c, r in zip(map_obj.obstacle_centers, map_obj.obstacle_radii)
            )
            if in_obs:
                ax1.scatter(pt[0], pt[1], s=20, color="red", zorder=6, alpha=0.7)

    # Min-clearance point (orange star)
    idx_mc = sf["_argmin_idx"]
    mc_pt  = traj_xy[idx_mc]
    ax1.scatter(mc_pt[0], mc_pt[1], s=90, marker="*",
                color="#f39c12", zorder=7, label=f"min-clr={sf['min_clearance']:.2f} m")

    # Start
    sx, sy, syaw = init_pose
    ax1.scatter(sx, sy, s=100, color="#27ae60", zorder=8,
                edgecolors="white", lw=0.8, label="Start")
    ax1.annotate("",
                 xy=(sx + arrow_len * np.cos(syaw), sy + arrow_len * np.sin(syaw)),
                 xytext=(sx, sy),
                 arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=2.0, mutation_scale=12),
                 zorder=9)

    # Goal
    gx, gy, gyaw = goal_pose
    ax1.scatter(gx, gy, s=100, color="#e67e22", zorder=8,
                edgecolors="white", lw=0.8, label="Goal")
    ax1.annotate("",
                 xy=(gx + arrow_len * np.cos(gyaw), gy + arrow_len * np.sin(gyaw)),
                 xytext=(gx, gy),
                 arrowprops=dict(arrowstyle="-|>", color="#e67e22", lw=2.0, mutation_scale=12),
                 zorder=9)

    ax1.set_xlim(-0.05 * L, 1.05 * L)
    ax1.set_ylim(-0.05 * L, 1.05 * L)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.legend(fontsize=7, loc="upper left", framealpha=0.7)
    ax1.set_title(
        f"collision={sf['collision_any']}  min_clr={sf['min_clearance']:.2f} m\n"
        f"pos_err={gf['final_pos_error']:.2f} m  yaw_err={np.degrees(gf['final_yaw_error']):.1f}°  "
        f"Δω={sm['mean_abs_delta_omega']:.3f}\n"
        f"safety={sf['safety_score']:.2f}  goal={gf['goal_score']:.2f}  "
        f"smooth={sm['smooth_score']:.2f}  total={tot['total_score']:.2f}",
        fontsize=8,
    )

    # ================================================================== #
    # Fig 2 - Control sequence
    # ================================================================== #
    H      = len(action_seq)
    t_axis = np.arange(H) * dt

    ax2_top = fig.add_subplot(2, 3, 2)
    ax2_bot = fig.add_subplot(2, 3, 5)

    ax2_top.plot(t_axis, action_seq[:, 0], color="#27ae60", lw=1.5)
    ax2_top.set_ylabel("v [m/s]")
    ax2_top.set_title(
        f"Control sequence\n"
        f"Δω_mean={sm['mean_abs_delta_omega']:.3f}  smooth={sm['smooth_score']:.2f}",
        fontsize=8,
    )
    ax2_top.grid(True, alpha=0.3)
    ax2_top.tick_params(labelbottom=False)

    ax2_bot.plot(t_axis, action_seq[:, 1], color="#8e44ad", lw=1.5)
    ax2_bot.set_xlabel("time [s]")
    ax2_bot.set_ylabel("ω [rad/s]")
    ax2_bot.grid(True, alpha=0.3)

    # Shade delta-omega magnitude
    if H > 1:
        dom = np.abs(np.diff(action_seq[:, 1]))
        t_mid = (t_axis[:-1] + t_axis[1:]) / 2.0
        ax2_bot.bar(t_mid, dom, width=dt * 0.9,
                    color="#c0392b", alpha=0.25, label="|Δω|")
        ax2_bot.legend(fontsize=7)

    # ================================================================== #
    # Fig 3 - BEV at start + projected trajectory
    # ================================================================== #
    ax3 = fig.add_subplot(1, 3, 3)

    D_bev = bev_builder.D_bev
    W_bev = bev_builder.W_bev

    # Draw BEV image (row-0 = far end = top of image)
    extent = (-W_bev / 2, W_bev / 2, 0, D_bev)
    ax3.imshow(
        bev_builder.to_rgb(bev_grid),
        origin="upper",
        extent=extent,
        interpolation="nearest",
        aspect="auto",
        zorder=1,
    )

    # FOV boundary lines
    slope = (W_bev / 2.0) / D_bev
    ax3.plot([ slope * D_bev, 0], [D_bev, 0], color="black", lw=0.8, ls="--", alpha=0.5)
    ax3.plot([-slope * D_bev, 0], [D_bev, 0], color="black", lw=0.8, ls="--", alpha=0.5)

    # Project future trajectory into body frame
    sx_w, sy_w, syaw_w = float(init_pose[0]), float(init_pose[1]), float(init_pose[2])

    world_pts = traj_xy - np.array([sx_w, sy_w])      # (H+1, 2) shifted
    # body-frame: fwd (forward along heading), lat (+left / port, -right / starboard)
    body_fwd =  world_pts[:, 0] * np.cos(syaw_w) + world_pts[:, 1] * np.sin(syaw_w)
    body_lat = -world_pts[:, 0] * np.sin(syaw_w) + world_pts[:, 1] * np.cos(syaw_w)

    # BEV image x-axis convention (after bev_builder's col-flip):
    #   col 0  (image left,  x = -W/2) = lat = +W/2  (port / left)
    #   col -1 (image right, x = +W/2) = lat = -W/2  (starboard / right)
    # So image-x = -body_lat  (negate to match the col-flip in build())
    bev_x = -body_lat

    # Only plot points within the BEV rectangle
    in_bev = (
        (body_fwd >= 0) & (body_fwd <= D_bev) &
        (np.abs(body_lat) <= W_bev / 2)
    )
    ax3.plot(bev_x[in_bev], body_fwd[in_bev],
             color="#2980b9", lw=1.5, zorder=5, alpha=0.85)
    if in_bev.any():
        first = np.argmax(in_bev)
        ax3.scatter(bev_x[first], body_fwd[first],
                    s=60, color="#2980b9", zorder=6)

    # ── Obstacle visibility diagnostics ──────────────────────────────
    # Check each obstacle: classify as visible / out-of-FOV-triangle / behind / out-of-range
    vis_slope = W_bev / 2.0 / D_bev
    n_obs_total   = len(map_obj.obstacle_centers)
    n_obs_visible = int(np.sum(bev_grid == OBSTACLE) > 0)  # any cells visible

    obs_status_lines = []
    for oi, (oc, or_) in enumerate(zip(map_obj.obstacle_centers, map_obj.obstacle_radii)):
        dx = oc[0] - sx_w;  dy = oc[1] - sy_w
        ofwd =  dx * np.cos(syaw_w) + dy * np.sin(syaw_w)
        olat = -dx * np.sin(syaw_w) + dy * np.cos(syaw_w)
        if ofwd <= 0:
            reason = "behind"
        elif ofwd > D_bev:
            reason = f"fwd={ofwd:.1f}>{D_bev:.0f}m"
        elif abs(olat) > W_bev / 2:
            reason = f"|lat|={abs(olat):.1f}>{W_bev/2:.0f}m"
        else:
            max_lat_at_fwd = ofwd * vis_slope
            margin = max_lat_at_fwd - abs(olat)
            if margin < 0:
                reason = f"FOV tri (Δ={margin:.1f}m)"
            else:
                reason = "VISIBLE"
        obs_status_lines.append(f"obs{oi}: {reason}")

    # Draw a text box with obstacle status in bottom-left of BEV axes
    status_text = "\n".join(obs_status_lines)
    ax3.text(
        0.02, 0.02, status_text,
        transform=ax3.transAxes,
        fontsize=6, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, ec="gray"),
        zorder=10,
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color=_LABEL_COLOURS[UNKNOWN, :3], label="Unknown"),
        mpatches.Patch(color=_LABEL_COLOURS[FREE,    :3], label="Free"),
        mpatches.Patch(color=_LABEL_COLOURS[OBSTACLE, :3], label="Obstacle"),
    ]
    ax3.legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.8)

    ax3.set_xlim(-W_bev / 2, W_bev / 2)
    ax3.set_ylim(0, D_bev)
    ax3.set_xlabel("lateral [m]  (← port/left · starboard/right →)")
    ax3.set_ylabel("forward [m]  (↑ = far)")
    ax3.set_title(
        f"BEV at init_pose  (FOV half-angle={np.degrees(np.arctan(vis_slope)):.1f}°)\n"
        f"blue = future traj  |  {n_obs_total} obstacles total",
        fontsize=8,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.94))

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → saved: {save_path}")
    else:
        plt.show()



