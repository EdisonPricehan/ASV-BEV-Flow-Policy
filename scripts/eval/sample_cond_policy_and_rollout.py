"""
sample_cond_policy_and_rollout.py
==================================
Sample M action sequences from the trained conditional flow policy, rollout
the unicycle dynamics, and visualise trajectories + BEV.

Usage::

    cd /home/naslab-5090/asv-bev-flow-policy
    python -m scripts.eval.sample_cond_policy_and_rollout \
        --ckpt    outputs/runs/cond_flow/best_model.pt \
        --map     outputs/maps/map_s5_n3_r1/map.json \
        --demos   outputs/maps/map_s5_n3_r1/demos.npz \
        --cfg     configs/common.json \
        --M       50 \\
        --cfg_scale 1.5 \\
        --out     outputs/cond_rollout.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch


from project_config import OUTPUTS_DIR as _OUTPUTS_DIR
from env.square_world import SquareWorld
from kinematics.unicycle import UnicycleSimulator
from perception.bev_builder    import BEVBuilder
from datasets.flow_dataset import FlowDataset
from models.cond_flow_policy import CondActionFlowNet
from models.cfg_wrapper           import CondFlowODE
from models.simulators            import EulerSimulator
from common.utils                 import ActionNormalizer
from metrics.trajectory_metrics   import select_best_action_seq, default_metrics_cfg


# ---------------------------------------------------------------------------
# Sample helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_action_sequences(
    model    : CondActionFlowNet,
    bev_id   : torch.Tensor,     # Long[1, Hb, Wb]  or  [M, Hb, Wb]
    goal_vec : torch.Tensor,     # Float[1, 5]       or  [M, 5]
    M        : int,
    cfg_scale: float,
    ode_steps: int,
    device   : torch.device,
) -> torch.Tensor:
    """
    Sample M action sequences and return them as Float[M, H*2].

    BEV and goal_vec are tiled from shape [1,...] to [M,...] if needed.
    """
    D   = model.data_dim

    # tile conditioning to [M, ...]
    bev_id_m  = bev_id.expand(M, -1, -1).to(device)
    goal_vec_m= goal_vec.expand(M, -1).to(device)

    # initial noise
    x0 = torch.randn(M, D, device=device)

    # time grid
    ts = torch.linspace(0, 1, ode_steps + 1, device=device).unsqueeze(0).expand(M, -1)

    ode = CondFlowODE(model, bev_id_m, goal_vec_m, cfg_scale=cfg_scale)
    sim = EulerSimulator(ode)
    x1  = sim.simulate(x0, ts, use_tqdm=False)   # [M, D]
    return x1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    from project_config import load_config
    cfg = load_config("common", "eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load map & demo ────────────────────────────────────────────────
    world   = SquareWorld.load(args.map)
    data      = np.load(args.demos)
    lengths   = data["demo_lengths"]
    states    = data["demos_states"]
    actions   = data["demos_actions"]
    dt        = float(cfg["simulator"]["dt"])

    # ── Dataset (for goal_vec + BEV) ───────────────────────────────────
    map_id  = os.path.basename(os.path.dirname(args.map))
    ds = FlowDataset(
        maps_dir  = os.path.dirname(os.path.dirname(args.map)),
        cfg       = cfg,
        map_ids   = [map_id],
        stride    = 1,
        K_windows = 9999,
        seed      = 0,
    )

    # Pick the sample closest to the start of a random trajectory
    rng = np.random.default_rng(args.seed)
    traj_idx = int(rng.integers(0, len(lengths)))

    # Find the dataset index corresponding to (map_id, traj_idx, t=0)
    sample = None
    for idx in range(len(ds)):
        entry = ds._index[idx]
        if entry[1] == traj_idx and entry[2] == 0:
            sample = ds[idx]
            break
    if sample is None:
        # fallback: first sample of this map
        sample = ds[0]
    assert sample is not None

    bev_id   = sample["bev_id"].unsqueeze(0).to(device)    # [1, Hb, Wb]
    goal_vec = sample["goal_vec"].unsqueeze(0).to(device)  # [1, 5]
    pose_t0  = sample["meta"]["pose_t"]                    # (3,)

    # ── Load model ─────────────────────────────────────────────────────
    model = CondActionFlowNet.from_config(cfg).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Model loaded from {args.ckpt}  (H={model.H}, D={model.data_dim})")

    # ── Sample ─────────────────────────────────────────────────────────
    norm = ActionNormalizer(cfg)
    x1   = sample_action_sequences(
        model, bev_id, goal_vec,
        M=args.M, cfg_scale=args.cfg_scale,
        ode_steps=args.ode_steps, device=device,
    )  # [M, H*2]

    x1_np      = x1.cpu().numpy()                           # [M, H*2]
    act_norm   = x1_np.reshape(args.M, model.H, 2)          # [M, H, 2]
    act_raw    = norm.denormalize_np(act_norm.reshape(-1, 2)).reshape(args.M, model.H, 2)

    # ── Rollout ────────────────────────────────────────────────────────
    sim  = UnicycleSimulator(dt=dt)
    traj_xys   = []
    goal_dists = []

    for m in range(args.M):
        pose = pose_t0.copy()
        xy   = [pose[:2].copy()]
        for step in range(model.H):
            v, omega = float(act_raw[m, step, 0]), float(act_raw[m, step, 1])
            pose     = sim.step(pose, np.array([v, omega]))
            xy.append(pose[:2].copy())
        xy = np.array(xy)                                   # [H+1, 2]
        traj_xys.append(xy)
        goal_dists.append(float(np.linalg.norm(xy[-1] - world.goal_pose[:2])))

    # ── Best-sequence selection (optional) ─────────────────────────────
    best_idx = None
    if args.use_best_selection:
        best_idx, _, all_m = select_best_action_seq(
            map_obj     = world,
            bev_builder = BEVBuilder(cfg),
            init_pose   = pose_t0,
            goal_pose   = np.array(world.goal_pose, dtype=np.float32),
            action_seqs = act_raw,
            dt          = dt,
            config      = default_metrics_cfg(),
        )
        n_safe = int(sum(not m["safety"]["collision_any"] for m in all_m))
        print(f"  Best sequence: idx={best_idx}  safe candidates: {n_safe}/{args.M}")

    # ── Statistics ────────────────────────────────────────────────────
    success_eps = float(cfg["demo"]["goal_eps_m"])
    n_success   = sum(d < success_eps for d in goal_dists)
    n_coll      = sum(world.collision_check(xy) for xy in traj_xys)

    print(f"\n[cfg_scale={args.cfg_scale}]  M={args.M}")
    print(f"  Success  (dist < {success_eps:.1f} m) : {n_success}/{args.M}  "
          f"({100*n_success/args.M:.0f}%)")
    print(f"  Collision-free                       : {args.M - n_coll}/{args.M}  "
          f"({100*(args.M-n_coll)/args.M:.0f}%)")
    print(f"  Mean goal dist: {np.mean(goal_dists):.2f} m")

    # ── Visualise ─────────────────────────────────────────────────────
    L         = world.L
    arrow_len = 0.03 * L

    fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                             gridspec_kw={"width_ratios": [1, 1]})
    ax_map, ax_bev = axes

    # --- map panel ---
    ax_map.add_patch(mpatches.FancyBboxPatch(
        (0, 0), L, L, boxstyle="square,pad=0",
        linewidth=2, edgecolor="#2c3e50", facecolor="#f8f9fa", zorder=0))
    for c, r in zip(world.obstacle_centers, world.obstacle_radii):
        ax_map.add_patch(plt.Circle(c, r, color="#e74c3c", alpha=0.80, zorder=3))
        ax_map.add_patch(plt.Circle(c, r + world.margin,
                                    color="#e74c3c", alpha=0.12, zorder=2))

    # demo trajectories (up to 5, grey)
    for i in range(min(5, len(lengths))):
        T   = int(lengths[i])
        xy  = states[i, :T + 1, :2]
        ax_map.plot(xy[:, 0], xy[:, 1], color="#cccccc", lw=0.8, zorder=4, alpha=0.7)

    # sampled trajectories (colour-coded; best highlighted if selected)
    cmap = plt.get_cmap("tab20")
    for m, xy in enumerate(traj_xys):
        is_best = (best_idx is not None and m == best_idx)
        col  = "#ff9900" if is_best else cmap(m % 20)
        lw   = 2.5       if is_best else 0.9
        zo   = 8         if is_best else 5
        alpha= 1.0       if is_best else 0.5
        ax_map.plot(xy[:, 0], xy[:, 1], color=col, lw=lw, alpha=alpha, zorder=zo)
    if best_idx is not None:
        ax_map.plot([], [], color="#ff9900", lw=2.5, label="Best")

    # start / goal
    sx, sy, syaw = pose_t0
    ax_map.scatter(sx, sy, s=100, color="#27ae60", zorder=9, edgecolors="white", lw=1)
    ax_map.annotate("", xy=(sx + arrow_len * np.cos(syaw), sy + arrow_len * np.sin(syaw)),
                    xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=2, mutation_scale=12),
                    zorder=10)
    gx, gy, gyaw = world.goal_pose
    ax_map.scatter(gx, gy, s=100, color="#e67e22", zorder=9, edgecolors="white", lw=1)
    ax_map.annotate("", xy=(gx + arrow_len * np.cos(gyaw), gy + arrow_len * np.sin(gyaw)),
                    xytext=(gx, gy),
                    arrowprops=dict(arrowstyle="-|>", color="#e67e22", lw=2, mutation_scale=12),
                    zorder=10)

    ax_map.set_xlim(-0.05 * L, 1.05 * L); ax_map.set_ylim(-0.05 * L, 1.05 * L)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("x [m]"); ax_map.set_ylabel("y [m]")
    ax_map.set_title(
        f"Sampled trajectories  (M={args.M}, cfg={args.cfg_scale})\n"
        f"success={n_success}/{args.M}  coll-free={args.M-n_coll}/{args.M}"
    )

    # ── BEV panel ─────────────────────────────────────────────────────
    bev_builder = BEVBuilder(cfg)
    bev_grid    = bev_builder.build(world, pose_t0)
    _COLOURS    = np.array([[0.55, 0.55, 0.55], [0.72, 0.96, 0.72], [0.95, 0.25, 0.25]])
    D_bev = bev_builder.D_bev; W_bev = bev_builder.W_bev
    ax_bev.imshow(
        (_COLOURS[bev_grid] * 255).astype(np.uint8),
        origin="upper", extent=[-W_bev/2, W_bev/2, 0, D_bev],
        interpolation="nearest", aspect="auto",
    )
    slope = W_bev / 2.0 / D_bev
    ax_bev.plot([ slope * D_bev, 0], [D_bev, 0], "k--", lw=0.8, alpha=0.5)
    ax_bev.plot([-slope * D_bev, 0], [D_bev, 0], "k--", lw=0.8, alpha=0.5)
    ax_bev.set_xlabel("lateral [m]"); ax_bev.set_ylabel("forward [m]")
    ax_bev.set_title("BEV at start pose")
    legend_patches = [
        mpatches.Patch(color=_COLOURS[0], label="Unknown"),
        mpatches.Patch(color=_COLOURS[1], label="Free"),
        mpatches.Patch(color=_COLOURS[2], label="Obstacle"),
    ]
    ax_bev.legend(handles=legend_patches, loc="upper right", fontsize=7)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved → {args.out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    from project_config import load_config
    _e = load_config("common", "eval").get("eval", {})
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      default=_e.get("model_path", None),
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--map",       required=True, help="Path to map.json")
    p.add_argument("--demos",     required=True, help="Path to demos.npz")
    p.add_argument("--M",         type=int,   default=_e.get("M",          50))
    p.add_argument("--cfg_scale", type=float, default=_e.get("cfg_scale",  1.5))
    p.add_argument("--ode_steps", type=int,   default=_e.get("ode_steps",  20))
    p.add_argument("--use_best_selection", action=argparse.BooleanOptionalAction,
                   default=bool(_e.get("use_best_selection", True)),
                   help="Pick the best of M samples via trajectory_metrics "
                        "(--no-use_best_selection to disable).")
    p.add_argument("--out",       default=str(_OUTPUTS_DIR / "cond_rollout.png"))
    p.add_argument("--seed",      type=int,   default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())

