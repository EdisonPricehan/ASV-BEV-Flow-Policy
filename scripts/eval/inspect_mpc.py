"""
inspect_mpc.py
==============
Single-agent MPC inspection tool.  At every MPC query cycle the script
records all M candidate trajectories sampled from the policy, labels each
one as safe / unsafe / best-selected, and saves everything as an annotated
GIF so you can visually inspect:

  * What the policy "thinks" at each location along the executed path
  * Whether best-selection (use_best_selection) changes the chosen trajectory
  * How many of the M candidates are collision-free at each step

Layout of each GIF frame
------------------------
  Left  : global map
           - grey  : executed path so far
           - thin  : all M candidate rollouts from the current pose
                     (green = safe, red = collision, orange thick = best)
           - dots  : animated agent position
  Right : BEV at the current pose

Usage::

    cd /home/naslab-5090/asv-bev-flow-policy

    python -m scripts.eval.inspect_mpc \\
        --map   outputs/maps/map_s5_n3_r1 \\
        --ckpt  outputs/runs/cond_flow_jitter_fix/best_model.pt \\
        --M 8 --cfg_scale 2.0 --use_best_selection \\
        --out   outputs/eval/inspect_map_s5_n3_r1.gif

    # Disable best-selection to compare
    python -m scripts.eval.inspect_mpc \\
        --map   outputs/maps/map_s5_n3_r1 \\
        --no-use_best_selection \\
        --out   outputs/eval/inspect_map_s5_n3_r1_nosel.gif
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from project_config import OUTPUTS_DIR as _OUTPUTS_DIR, POLICY_DIR as _POLICY_DIR, load_config
from env.square_world import SquareWorld
from kinematics.unicycle import UnicycleSimulator
from perception.bev_builder import BEVBuilder, UNKNOWN, FREE, OBSTACLE
from models.cond_flow_policy import CondActionFlowNet
from common.utils import ActionNormalizer
from common.mpc import mpc_rollout as _mpc_rollout
from scripts.eval.eval_cond_policy import query_policy_batch  # noqa: F401


# ---------------------------------------------------------------------------
# Data collection: run MPC and record every query cycle
# ---------------------------------------------------------------------------

def run_and_record(
    model      : CondActionFlowNet,
    norm       : ActionNormalizer,
    bev_b      : BEVBuilder,
    sim_uni    : UnicycleSimulator,
    world      : SquareWorld,
    start_pose : np.ndarray,
    goal_pose  : np.ndarray,
    D_max      : float,
    cfg        : dict,
    args,
    device     : torch.device,
) -> dict:
    """
    Run a single-agent MPC episode and record every query cycle for
    frame-by-frame visualisation.  Delegates to ``common.mpc.mpc_rollout``
    with ``record=True``.

    Returns
    -------
    dict with keys:
        executed_states : np.ndarray [T+1, 3]
        query_records   : list of dicts, one per MPC cycle
        success         : bool
        collided        : bool
        goal_dist       : float
    """
    return _mpc_rollout(
        model      = model,
        norm       = norm,
        bev_b      = bev_b,
        sim_uni    = sim_uni,
        world      = world,
        start_pose = start_pose,
        goal_pose  = goal_pose,
        D_max      = D_max,
        cfg        = cfg,
        M                  = args.M,
        cfg_scale          = args.cfg_scale,
        ode_steps          = args.ode_steps,
        exec_steps         = args.exec_steps,
        max_steps          = args.max_steps,
        use_best_selection = args.use_best_selection,
        device     = device,
        seed       = args.seed,    # fix seed so sel vs nosel are comparable
        record     = True,
        pbar_desc  = "recording",
    )


# ── colours ──────────────────────────────────────────────────────────────
_BEV_COLOURS = np.array([
    [0.55, 0.55, 0.55],   # UNKNOWN
    [0.72, 0.96, 0.72],   # FREE
    [0.95, 0.25, 0.25],   # OBSTACLE
], dtype=np.float32)

_COL_SAFE    = "#2ecc71"   # safe (but not selected)
_COL_COLL    = "#e74c3c"   # collision candidate
_COL_BEST    = "#ff9900"   # selected best
_COL_EXEC    = "#2c3e50"   # executed path so far
_COL_START   = "#27ae60"
_COL_GOAL    = "#e67e22"


def render_gif(
    world        : SquareWorld,
    bev_b        : BEVBuilder,
    start_pose   : np.ndarray,
    goal_pose    : np.ndarray,
    result       : dict,
    out_path     : str | Path,
    fps          : int = 4,
    use_selection: bool = True,
) -> None:
    """
    Render one frame per MPC query cycle.  Each frame shows:
      - Left:  global map with all M candidate trajectories + executed path
      - Right: BEV at the current query pose
    """
    L         = world.L
    D_bev     = bev_b.D_bev
    W_bev     = bev_b.W_bev
    arrow_len = 0.04 * L

    executed  = result["executed_states"]    # [T+1, 3]
    records   = result["query_records"]
    success   = result["success"]
    collided  = result["collided"]
    goal_dist = result["goal_dist"]
    n_cycles  = len(records)

    if n_cycles == 0:
        print("No query cycles recorded — nothing to render.")
        return

    fig, (ax_map, ax_bev) = plt.subplots(
        1, 2, figsize=(13, 6),
        gridspec_kw={"width_ratios": [1.15, 0.85]},
    )
    fig.tight_layout(pad=1.5)

    # ── static map background ─────────────────────────────────────────
    ax_map.add_patch(mpatches.FancyBboxPatch(
        (0, 0), L, L, boxstyle="square,pad=0",
        linewidth=2, edgecolor="#2c3e50", facecolor="#f8f9fa", zorder=0))
    for c, r in zip(world.obstacle_centers, world.obstacle_radii):
        ax_map.add_patch(plt.Circle(c, r, color="#e74c3c", alpha=0.75, zorder=3))
        ax_map.add_patch(plt.Circle(c, r + world.margin,
                                    color="#e74c3c", alpha=0.10, zorder=2))

    # start / goal markers (static)
    sx, sy, syaw = start_pose
    gx, gy, gyaw = goal_pose
    ax_map.scatter(sx, sy, s=100, color=_COL_START, zorder=11, edgecolors="white", lw=1)
    ax_map.annotate("", xy=(sx + arrow_len*np.cos(syaw), sy + arrow_len*np.sin(syaw)),
                    xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color=_COL_START, lw=2, mutation_scale=12),
                    zorder=12)
    ax_map.scatter(gx, gy, s=100, color=_COL_GOAL, zorder=11, edgecolors="white", lw=1)
    ax_map.annotate("", xy=(gx + arrow_len*np.cos(gyaw), gy + arrow_len*np.sin(gyaw)),
                    xytext=(gx, gy),
                    arrowprops=dict(arrowstyle="-|>", color=_COL_GOAL, lw=2, mutation_scale=12),
                    zorder=12)

    # full executed path (faint static reference)
    ax_map.plot(executed[:, 0], executed[:, 1],
                color=_COL_EXEC, lw=0.6, alpha=0.20, zorder=4, ls="--")

    ax_map.set_xlim(-0.05*L, 1.05*L); ax_map.set_ylim(-0.05*L, 1.05*L)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("x [m]", fontsize=8); ax_map.set_ylabel("y [m]", fontsize=8)

    # legend handles
    _leg = [
        mpatches.Patch(color=_COL_SAFE, alpha=0.7,  label="Candidate (safe)"),
        mpatches.Patch(color=_COL_COLL, alpha=0.5,  label="Candidate (collision)"),
        mpatches.Patch(color=_COL_EXEC,              label="Executed path"),
    ]
    if use_selection:
        _leg.insert(2, mpatches.Patch(color=_COL_BEST, label="Selected best"))
    ax_map.legend(handles=_leg, loc="lower right", fontsize=6, framealpha=0.85)

    # ── animated objects (map) ────────────────────────────────────────
    # candidate trajectory lines — re-used each frame
    M = len(records[0]["cand_xys"])
    cand_lines = [ax_map.plot([], [], lw=0.9, alpha=0.0, zorder=5)[0] for _ in range(M)]
    exec_line, = ax_map.plot([], [], color=_COL_EXEC, lw=2.0, zorder=6)
    agent_dot, = ax_map.plot([], [], "o", color=_COL_EXEC, ms=7, zorder=10)
    title_txt  = ax_map.set_title("", fontsize=8)

    # ── BEV panel ─────────────────────────────────────────────────────
    extent = [-W_bev/2, W_bev/2, 0, D_bev]
    slope  = W_bev / 2.0 / D_bev
    ax_bev.plot([ slope*D_bev, 0], [D_bev, 0], "k--", lw=0.8, alpha=0.5, zorder=5)
    ax_bev.plot([-slope*D_bev, 0], [D_bev, 0], "k--", lw=0.8, alpha=0.5, zorder=5)
    ax_bev.set_xlim(-W_bev/2, W_bev/2); ax_bev.set_ylim(0, D_bev)
    ax_bev.set_aspect("equal")
    ax_bev.set_xlabel("lateral [m]", fontsize=8)
    ax_bev.set_ylabel("forward [m]  (↑ = far)", fontsize=8)
    bev_leg = [
        mpatches.Patch(color=_BEV_COLOURS[UNKNOWN], label="Unknown"),
        mpatches.Patch(color=_BEV_COLOURS[FREE],    label="Free"),
        mpatches.Patch(color=_BEV_COLOURS[OBSTACLE],label="Obstacle"),
    ]
    ax_bev.legend(handles=bev_leg, loc="upper right", fontsize=6, framealpha=0.8)
    dummy   = np.zeros((bev_b.H_cells, bev_b.W_cells), dtype=np.uint8)
    bev_img = ax_bev.imshow(
        (_BEV_COLOURS[dummy]*255).astype(np.uint8),
        origin="upper", extent=extent, interpolation="nearest",
        aspect="auto", zorder=2,
    )
    bev_title = ax_bev.set_title("BEV at query pose", fontsize=8)

    # ── update function ───────────────────────────────────────────────
    def _update(frame_i: int):
        rec         = records[frame_i]
        pose        = rec["pose"]
        bev_grid    = rec["bev_grid"]
        cand_xys    = rec["cand_xys"]
        coll_flags  = rec["coll_flags"]
        best_idx    = rec["best_idx"]
        exec_start  = rec["exec_start"]

        # executed path up to this cycle
        exec_xy = executed[:exec_start + 1, :2]
        exec_line.set_data(exec_xy[:, 0], exec_xy[:, 1])
        agent_dot.set_data([pose[0]], [pose[1]])

        # candidate trajectories
        for m, line in enumerate(cand_lines):
            xy = cand_xys[m]
            is_best = (m == best_idx) and use_selection
            is_coll = coll_flags[m]

            if is_best:
                col, lw, alpha, zo = _COL_BEST, 2.5, 1.0, 8
            elif is_coll:
                col, lw, alpha, zo = _COL_COLL, 0.8, 0.45, 5
            else:
                col, lw, alpha, zo = _COL_SAFE, 0.9, 0.60, 5

            line.set_data(xy[:, 0], xy[:, 1])
            line.set_color(col); line.set_linewidth(lw)
            line.set_alpha(alpha); line.set_zorder(zo)

        # BEV
        bev_img.set_data((_BEV_COLOURS[bev_grid]*255).astype(np.uint8))

        n_safe = sum(not c for c in coll_flags)
        sel_tag = f"  best=#{best_idx}" if use_selection else "  (no selection)"
        status  = "COLLIDED" if (frame_i == n_cycles-1 and result["collided"]) \
                  else ("SUCCESS" if result["success"] else "running")
        title_txt.set_text(
            f"Cycle {frame_i+1}/{n_cycles}  |  "
            f"safe {n_safe}/{M}{sel_tag}  |  {status}"
        )
        bev_title.set_text(
            f"BEV  pose=({pose[0]:.1f},{pose[1]:.1f})  "
            f"dist={np.linalg.norm(pose[:2]-goal_pose[:2]):.1f}m"
        )
        return [exec_line, agent_dot, bev_img, title_txt, bev_title] + cand_lines

    ani = animation.FuncAnimation(
        fig, _update, frames=n_cycles,
        interval=int(1000 / fps), blit=False,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(out_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"GIF saved → {out_path}  ({n_cycles} frames)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    _e = load_config("common", "eval").get("eval", {})
    p  = argparse.ArgumentParser(
        description="Inspect single-agent MPC: visualise all M candidate "
                    "trajectories at every query cycle."
    )
    p.add_argument("--map",     type=str,
                   default=str(_OUTPUTS_DIR / "maps" / "map_s8_n1_r1"),
                   help="Path to a map subdirectory (contains map.json + demos.npz)")
    p.add_argument("--ckpt",    type=str,
                   default=_e.get("model_path", None),
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--traj_idx",type=int, default=0,
                   help="Index of the demo trajectory to use as start pose")
    p.add_argument("--M",       type=int,   default=_e.get("M",          8))
    p.add_argument("--cfg_scale",type=float,default=_e.get("cfg_scale",  2.0))
    p.add_argument("--ode_steps",type=int,  default=_e.get("ode_steps",  20))
    p.add_argument("--exec_steps",type=int, default=_e.get("exec_steps", 20))
    p.add_argument("--max_steps", type=int, default=_e.get("max_steps",  2000))
    p.add_argument("--use_best_selection", action=argparse.BooleanOptionalAction,
                   default=bool(_e.get("use_best_selection", True)),
                   help="Enable best-candidate selection (--no-use_best_selection to disable)")
    p.add_argument("--fps",     type=int,   default=3,
                   help="GIF frames per second (lower = easier to inspect)")
    p.add_argument("--seed",    type=int,   default=0)
    p.add_argument("--out",     type=str,   default=None,
                   help="Output GIF path (auto-named if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = load_config("common", "eval")

    # ── resolve output path ────────────────────────────────────────────
    if args.out is None:
        map_name = Path(args.map).name
        sel_tag  = "sel" if args.use_best_selection else "nosel"
        args.out = str(_OUTPUTS_DIR / "eval" / f"inspect_{map_name}_{sel_tag}.gif")

    # ── load map & demos ───────────────────────────────────────────────
    map_json  = os.path.join(args.map, "map.json")
    demos_npz = os.path.join(args.map, "demos.npz")
    world     = SquareWorld.load(map_json)
    data      = np.load(demos_npz)
    states    = data["demos_states"]
    n_trajs   = states.shape[0]
    traj_idx  = args.traj_idx % n_trajs
    start_pose = states[traj_idx, 0].astype(np.float32)
    goal_pose  = np.array(world.goal_pose, dtype=np.float32)

    print(f"Map        : {map_json}")
    print(f"Start pose : {start_pose}")
    print(f"Goal pose  : {goal_pose}")
    print(f"M={args.M}  cfg_scale={args.cfg_scale}  "
          f"exec_steps={args.exec_steps}  "
          f"use_best_selection={args.use_best_selection}")
    print(f"Output     : {args.out}\n")

    # ── ckpt ──────────────────────────────────────────────────────────
    if args.ckpt is None:
        raise ValueError(
            "No checkpoint specified. "
            "Set eval.model_path in configs/eval.json or pass --ckpt."
        )

    # ── model ─────────────────────────────────────────────────────────
    model = CondActionFlowNet.from_config(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"Model loaded  params={sum(p.numel() for p in model.parameters()):,}  "
          f"H={model.H}\n")

    norm    = ActionNormalizer(cfg)
    bev_b   = BEVBuilder(cfg)
    sim_uni = UnicycleSimulator(dt=float(cfg["simulator"]["dt"]))
    D_max   = float(cfg.get("dataset", {}).get("D_max", 120.0))

    # ── run & record ──────────────────────────────────────────────────
    result = run_and_record(
        model, norm, bev_b, sim_uni, world,
        start_pose, goal_pose, D_max, cfg, args, device,
    )
    status = "SUCCESS" if result["success"] else \
             ("COLLIDED" if result["collided"] else "TIMEOUT")
    print(f"\nEpisode: {status}  "
          f"goal_dist={result['goal_dist']:.2f}m  "
          f"steps={result['executed_states'].shape[0]-1}  "
          f"cycles={len(result['query_records'])}")

    # ── render ────────────────────────────────────────────────────────
    render_gif(
        world, bev_b, start_pose, goal_pose,
        result, args.out,
        fps=args.fps,
        use_selection=args.use_best_selection,
    )
