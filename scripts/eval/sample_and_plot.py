"""
sample_and_plot.py
==================
Sample action sequences from the trained **unconditional** flow model
(ActionFlowNet), rollout unicycle trajectories from a fixed start pose,
and visualise multimodal avoidance.

Usage
-----
    python -m scripts.eval.sample_and_plot \\
        --map  outputs/maps/map_s5_n3_r1 \\
        --ckpt outputs/runs/<run>/final_model.pt

    # auto-detect latest checkpoint:
    python -m scripts.eval.sample_and_plot \\
        --map outputs/maps/map_s5_n3_r1

Output
------
    outputs/flow_policy_samples.png   (default)
"""

import argparse
import glob
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
from models.flow_policy import ActionFlowNet, ActionFlowODE, ACT_DIM
from models.simulators import EulerSimulator
from common.utils import ActionNormalizer


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_actions(
    model    : ActionFlowNet,
    norm     : ActionNormalizer,
    n_samples: int = 50,
    ode_steps: int = 100,
    device   : str = "cpu",
) -> np.ndarray:
    """
    Sample *n_samples* action sequences via ODE integration x0→x1.

    Returns
    -------
    actions : float32 (n_samples, H, 2)  in raw (denormalised) units
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    D  = model.H * ACT_DIM                               # flattened dim
    x0 = torch.randn(n_samples, D, device=dev)
    ts = (torch.linspace(0.0, 1.0, ode_steps + 1)
          .unsqueeze(0).expand(n_samples, -1).to(dev))

    x1_norm = EulerSimulator(ActionFlowODE(model)).simulate(
        x0, ts, use_tqdm=True)                           # (M, D) in [-1,1]

    act_norm = x1_norm.cpu().numpy().reshape(n_samples, model.H, ACT_DIM)
    act_raw  = norm.denormalize_np(act_norm.reshape(-1, 2)).reshape(
        n_samples, model.H, ACT_DIM)
    return act_raw.astype(np.float32)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(
    world          : SquareWorld,
    sim            : UnicycleSimulator,
    start_pose     : np.ndarray,          # (3,)
    sampled_actions: np.ndarray,          # (M, H, 2)
    demo_states    : np.ndarray,          # (N_show, T+1, 3)
    save_path      : str,
    goal_eps       : float = 0.5,
) -> None:
    """Rollout sampled actions and produce the comparison figure."""
    M  = sampled_actions.shape[0]
    L  = world.L
    gx, gy, _ = world.goal_pose
    goal_xy    = np.array([gx, gy])
    arrow_len  = 0.03 * L

    # rollout all sampled trajectories
    traj_xys   = []
    collisions = []
    goal_hits  = []
    for i in range(M):
        states = sim.rollout(start_pose, sampled_actions[i])   # (H+1, 3)
        xy     = states[:, :2]
        traj_xys.append(xy)
        collisions.append(world.collision_check(xy))
        goal_hits.append(float(np.linalg.norm(xy[-1] - goal_xy)) < goal_eps)

    n_coll = sum(collisions)
    n_ok   = sum(goal_hits)

    fig, ax = plt.subplots(figsize=(8, 8))

    # map boundary
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), L, L, boxstyle="square,pad=0",
        linewidth=2, edgecolor="#2c3e50", facecolor="#f8f9fa", zorder=0))

    # obstacles
    for c, r in zip(world.obstacle_centers, world.obstacle_radii):
        ax.add_patch(plt.Circle(c, r,              color="#e74c3c", alpha=0.80, zorder=3))
        ax.add_patch(plt.Circle(c, r + world.margin,
                                color="#e74c3c", alpha=0.12, zorder=2))

    # demo trajectories (grey background)
    for s in demo_states:
        ax.plot(s[:, 0], s[:, 1], color="gray", alpha=0.30, lw=0.8, zorder=1)

    # sampled trajectories
    cmap = matplotlib.colormaps["viridis"]
    for i, xy in enumerate(traj_xys):
        hit   = collisions[i]
        color = cmap(i / max(M - 1, 1))
        ax.plot(xy[:, 0], xy[:, 1],
                color=color,
                alpha=0.45 if not hit else 0.20,
                lw=1.4 if not hit else 0.7,
                ls="-"  if not hit else "--",
                zorder=2)

    # start
    sx, sy, syaw = start_pose
    ax.scatter(sx, sy, s=120, color="#27ae60", zorder=9, edgecolors="white", lw=1)
    ax.annotate("", xy=(sx + arrow_len*np.cos(syaw), sy + arrow_len*np.sin(syaw)),
                xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=2, mutation_scale=12),
                zorder=10)

    # goal
    ax.scatter(gx, gy, s=120, color="#e67e22", zorder=9, edgecolors="white", lw=1)
    ax.annotate("",
                xy=(gx + arrow_len*np.cos(world.goal_pose[2]),
                    gy + arrow_len*np.sin(world.goal_pose[2])),
                xytext=(gx, gy),
                arrowprops=dict(arrowstyle="-|>", color="#e67e22", lw=2, mutation_scale=12),
                zorder=10)

    ax.set_xlim(-0.05*L, 1.05*L)
    ax.set_ylim(-0.05*L, 1.05*L)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(
        f"Unconditional flow policy – {M} samples\n"
        f"collision-free: {M-n_coll}/{M}  |  "
        f"goal reached (<{goal_eps:.1f}m): {n_ok}/{M}",
        fontsize=12,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, M-1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="Sample index")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")
    print(f"  collision-free: {M-n_coll}/{M}  |  goal reached: {n_ok}/{M}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_model(runs_dir: str) -> str:
    candidates = sorted(
        glob.glob(os.path.join(runs_dir, "*", "final_model.pt")),
        key=os.path.getmtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No final_model.pt found under {runs_dir}. "
            "Run train_flow_policy.py first."
        )
    return candidates[-1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Sample from the unconditional ActionFlowNet and visualise rollouts."
    )
    p.add_argument("--map",        type=str,
                   default=str(_OUTPUTS_DIR / "maps" / "map_s5_n3_r1"),
                   help="Path to a map subdirectory containing map.json and demos.npz")
    p.add_argument("--ckpt",       type=str, default=None,
                   help="Path to model checkpoint (.pt); auto-detected if omitted")
    p.add_argument("--n_samples",  type=int,   default=50)
    p.add_argument("--ode_steps",  type=int,   default=100)
    p.add_argument("--n_demo_show",type=int,   default=20,
                   help="Number of expert demo trajectories to show in background")
    p.add_argument("--device",     type=str,   default=None)
    p.add_argument("--out",        type=str,
                   default=str(_OUTPUTS_DIR / "flow_policy_samples.png"))
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from project_config import load_config
    cfg    = load_config("common")

    # ── load map & demos ──────────────────────────────────────────────
    map_json  = os.path.join(args.map, "map.json")
    demos_npz = os.path.join(args.map, "demos.npz")
    world     = SquareWorld.load(map_json)
    data      = np.load(demos_npz)
    lengths   = data["demo_lengths"]
    states    = data["demos_states"]

    # background demo trajectories (up to n_demo_show)
    n_show = min(args.n_demo_show, len(lengths))
    idx    = np.linspace(0, len(lengths)-1, n_show, dtype=int)
    demo_states = [states[i, :lengths[i]+1] for i in idx]   # list of (T+1,3)

    # start pose: first pose of the first demo
    start_pose = states[0, 0].copy()

    # ── model ─────────────────────────────────────────────────────────
    if args.ckpt is None:
        runs_dir   = str(_OUTPUTS_DIR / "runs")
        args.ckpt  = _latest_model(runs_dir)
    print(f"Loading model from {args.ckpt} …")

    model = ActionFlowNet.from_config(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    print(f"  H={model.H}  params={sum(p.numel() for p in model.parameters()):,}")

    # ── sample ────────────────────────────────────────────────────────
    norm    = ActionNormalizer(cfg)
    sim     = UnicycleSimulator(dt=float(cfg["simulator"]["dt"]))
    print(f"Sampling {args.n_samples} action sequences (ode_steps={args.ode_steps}) …")
    sampled = sample_actions(model, norm,
                             n_samples=args.n_samples,
                             ode_steps=args.ode_steps,
                             device=device)

    # ── plot ──────────────────────────────────────────────────────────
    goal_eps = float(cfg["demo"]["goal_eps_m"])
    plot_results(
        world           = world,
        sim             = sim,
        start_pose      = start_pose,
        sampled_actions = sampled,
        demo_states     = demo_states,
        save_path       = args.out,
        goal_eps        = goal_eps,
    )
