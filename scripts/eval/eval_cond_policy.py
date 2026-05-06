"""
eval_cond_policy.py
===================
Evaluate the trained conditional flow policy across multiple maps using a
**rolling MPC loop**: at each control step the policy is queried for H future
actions, only the first `exec_steps` are executed, then BEV and goal_vec are
recomputed and the policy is queried again — until goal is reached or a
maximum number of steps is exceeded.

Outputs per map  →  outputs/eval/<map_id>_rollout.gif
Global summary   →  outputs/eval/summary.png  +  metrics.json

Usage::

    python -m scripts.eval.eval_cond_policy \\
        --ckpt     outputs/runs/cond_flow_full/best_model.pt \\
        --maps_dir outputs/maps \\
        --cfg      configs/common.json \\
        --out_dir  outputs/eval \\
        --M 8 --cfg_scale 1.5 --ode_steps 20 --exec_steps 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


from env.square_world import SquareWorld
from kinematics.unicycle import UnicycleSimulator
from perception.bev_builder import BEVBuilder, UNKNOWN, FREE, OBSTACLE
from datasets.flow_dataset import FlowDataset
from models.cond_flow_policy import CondActionFlowNet
from models.cfg_wrapper import CondFlowODE
from models.simulators import EulerSimulator
from common.utils import ActionNormalizer
from common.mpc import mpc_rollout as _mpc_rollout

# ── Colour maps ────────────────────────────────────────────────────────────
_TRAJ_CMAP   = plt.get_cmap("tab20")
_BEV_COLOURS = np.array([
    [0.55, 0.55, 0.55],   # 0 UNKNOWN
    [0.72, 0.96, 0.72],   # 1 FREE
    [0.95, 0.25, 0.25],   # 2 OBSTACLE
], dtype=np.float32)



# ---------------------------------------------------------------------------
# Single policy query
# ---------------------------------------------------------------------------


@torch.no_grad()
def query_policy_batch(
    model    : CondActionFlowNet,
    norm     : ActionNormalizer,
    bev_grid : np.ndarray,          # uint8 [Hb, Wb]  (same BEV for all M)
    goal_vec : np.ndarray,          # float32 [5]
    M        : int,
    cfg_scale: float,
    ode_steps: int,
    device   : torch.device,
) -> np.ndarray:
    """
    Sample M action sequences from the policy given a single BEV + goal_vec.
    Returns float32 [M, H, 2] *raw* (denormalised) actions.
    """
    D = model.data_dim

    bev_t    = torch.from_numpy(bev_grid).long().unsqueeze(0).expand(M, -1, -1).to(device)
    goal_t   = torch.from_numpy(goal_vec).float().unsqueeze(0).expand(M, -1).to(device)

    x0 = torch.randn(M, D, device=device)
    ts = torch.linspace(0, 1, ode_steps + 1, device=device).unsqueeze(0).expand(M, -1)

    ode = CondFlowODE(model, bev_t, goal_t, cfg_scale=cfg_scale)
    sim = EulerSimulator(ode)
    x1  = sim.simulate(x0, ts, use_tqdm=False)           # [M, D]

    act_norm = x1.cpu().numpy().reshape(M, model.H, 2)
    act_raw  = norm.denormalize_np(act_norm.reshape(-1, 2)).reshape(M, model.H, 2)
    return act_raw                                        # [M, H, 2]


# ---------------------------------------------------------------------------
# Rolling MPC rollout for one agent  (thin wrapper around common.mpc)
# ---------------------------------------------------------------------------

def mpc_rollout(
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
    rng        : np.random.Generator,
) -> dict:
    """Thin wrapper: unpacks ``args`` and delegates to ``common.mpc.mpc_rollout``."""
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
        seed       = None,   # eval runs many agents; don't fix seed per-agent
        record     = False,
        pbar_desc  = "MPC",
    )


# ---------------------------------------------------------------------------
# Per-map evaluation
# ---------------------------------------------------------------------------

def eval_one_map(
    map_id  : str,
    world : SquareWorld,
    ds      : FlowDataset,
    model   : CondActionFlowNet,
    norm    : ActionNormalizer,
    bev_b   : BEVBuilder,
    sim_uni : UnicycleSimulator,
    args,
    device  : torch.device,
    out_dir : Path,
    rng     : np.random.Generator,
    D_max   : float,
    cfg     : dict,
) -> dict:
    """Evaluate M independent MPC rollouts on one map, save GIF."""

    map_demos = ds._demos.get(map_id)
    if map_demos is None:
        return {}

    # pick start pose from a random demo trajectory
    lengths = map_demos["lengths"]
    states  = map_demos["states"]
    traj_idx   = int(rng.integers(0, len(lengths)))
    start_pose = states[traj_idx, 0].copy()          # pose at step 0
    goal_pose  = world.goal_pose                   # (3,)

    M = args.M
    results = []
    for m in tqdm(range(M), desc=f"  {map_id}", leave=False,
                  unit="agent", dynamic_ncols=True):
        res = mpc_rollout(
            model, norm, bev_b, sim_uni, world,
            start_pose, goal_pose, D_max, cfg, args, device, rng,
        )
        results.append(res)

    n_success  = sum(r["success"]  for r in results)
    n_coll     = sum(r["collided"] for r in results)
    mean_dist  = float(np.mean([r["goal_dist"]    for r in results]))
    mean_steps = float(np.mean([r["steps"]        for r in results]))
    mean_safe  = float(np.mean([r["n_safe_mean"]  for r in results]))

    metrics = {
        "map_id"               : map_id,
        "n_samples"            : M,
        "success_rate"         : n_success / M,
        "collision_free_rate"  : (M - n_coll) / M,
        "mean_goal_dist_m"     : mean_dist,
        "mean_steps"           : mean_steps,
        "mean_safe_per_query"  : mean_safe,
        "cfg_scale"            : args.cfg_scale,
    }

    # build GIF
    gif_path = out_dir / f"{map_id}_rollout.gif"
    _make_rollout_gif(
        world, results, bev_b, start_pose, goal_pose, metrics, gif_path,
        fps=args.fps, stride=args.stride,
    )

    tqdm.write(f"  {map_id:22s}  success={n_success}/{M}  "
               f"coll-free={M-n_coll}/{M}  "
               f"mean_dist={mean_dist:.1f}m  "
               f"safe/query={mean_safe:.1f}/{M}")
    return metrics


# ---------------------------------------------------------------------------
# GIF renderer
# ---------------------------------------------------------------------------

def _make_rollout_gif(
    world  : SquareWorld,
    results  : list,
    bev_b    : BEVBuilder,
    start_pose: np.ndarray,
    goal_pose : np.ndarray,
    metrics  : dict,
    gif_path : Path,
    fps      : int = 8,
    stride   : int = 8,
) -> None:
    L         = world.L
    D_bev     = bev_b.D_bev
    W_bev     = bev_b.W_bev
    arrow_len = 0.03 * L
    M         = len(results)

    # find longest trajectory for frame count
    max_T = max(r["states"].shape[0] for r in results)
    frames_idx = list(range(0, max_T, stride))
    if frames_idx[-1] != max_T - 1:
        frames_idx.append(max_T - 1)

    fig, (ax_map, ax_bev) = plt.subplots(
        1, 2, figsize=(11, 5.5),
        gridspec_kw={"width_ratios": [1.1, 0.9]},
    )
    fig.tight_layout(pad=1.3)

    # ── static map elements ───────────────────────────────────────────
    ax_map.add_patch(mpatches.FancyBboxPatch(
        (0, 0), L, L, boxstyle="square,pad=0",
        linewidth=2, edgecolor="#2c3e50", facecolor="#f8f9fa", zorder=0))
    for c, r in zip(world.obstacle_centers, world.obstacle_radii):
        ax_map.add_patch(plt.Circle(c, r,              color="#e74c3c", alpha=0.75, zorder=3))
        ax_map.add_patch(plt.Circle(c, r + world.margin,
                                    color="#e74c3c", alpha=0.10, zorder=2))
    # full trajectories (faint)
    for m, res in enumerate(results):
        s = res["states"]
        ax_map.plot(s[:, 0], s[:, 1],
                    color=_TRAJ_CMAP(m % 20), lw=0.7, alpha=0.30, zorder=4)

    sx, sy, syaw = start_pose
    gx, gy, gyaw = goal_pose
    ax_map.scatter(sx, sy, s=90, color="#27ae60", zorder=9, edgecolors="white", lw=1)
    ax_map.annotate("", xy=(sx + arrow_len*np.cos(syaw), sy + arrow_len*np.sin(syaw)),
                    xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=2, mutation_scale=12),
                    zorder=10)
    ax_map.scatter(gx, gy, s=90, color="#e67e22", zorder=9, edgecolors="white", lw=1)
    ax_map.annotate("", xy=(gx + arrow_len*np.cos(gyaw), gy + arrow_len*np.sin(gyaw)),
                    xytext=(gx, gy),
                    arrowprops=dict(arrowstyle="-|>", color="#e67e22", lw=2, mutation_scale=12),
                    zorder=10)

    ax_map.set_xlim(-0.05*L, 1.05*L); ax_map.set_ylim(-0.05*L, 1.05*L)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("x [m]", fontsize=8); ax_map.set_ylabel("y [m]", fontsize=8)
    suc_pct = metrics["success_rate"] * 100
    cf_pct  = metrics["collision_free_rate"] * 100
    ax_map.set_title(
        f"{metrics['map_id']}  cfg={metrics['cfg_scale']}\n"
        f"success={suc_pct:.0f}%  coll-free={cf_pct:.0f}%  "
        f"mean_dist={metrics['mean_goal_dist_m']:.1f}m",
        fontsize=8,
    )

    # animated ship dots
    ship_dots = []
    for m in range(M):
        col = "#27ae60" if results[m]["success"] else (
              "#e74c3c" if results[m]["collided"] else _TRAJ_CMAP(m % 20))
        dot, = ax_map.plot([], [], "o", color=col, ms=5, zorder=7)
        ship_dots.append(dot)

    # ── BEV panel ─────────────────────────────────────────────────────
    ax_bev.set_xlim(-W_bev/2, W_bev/2); ax_bev.set_ylim(0, D_bev)
    ax_bev.set_aspect("equal")
    ax_bev.set_xlabel("lateral [m]", fontsize=8)
    ax_bev.set_ylabel("forward [m]  (↑ = far)", fontsize=8)
    ax_bev.set_title("BEV (agent #0)", fontsize=8)
    slope = W_bev / 2.0 / D_bev
    ax_bev.plot([ slope*D_bev, 0], [D_bev, 0], "k--", lw=0.8, alpha=0.5)
    ax_bev.plot([-slope*D_bev, 0], [D_bev, 0], "k--", lw=0.8, alpha=0.5)
    extent = [-W_bev/2, W_bev/2, 0, D_bev]
    legend_patches = [
        mpatches.Patch(color=_BEV_COLOURS[UNKNOWN], label="Unknown"),
        mpatches.Patch(color=_BEV_COLOURS[FREE],    label="Free"),
        mpatches.Patch(color=_BEV_COLOURS[OBSTACLE],label="Obstacle"),
    ]
    ax_bev.legend(handles=legend_patches, loc="upper right", fontsize=6, framealpha=0.8)
    dummy   = np.zeros((bev_b.H_cells, bev_b.W_cells), dtype=np.uint8)
    bev_img = ax_bev.imshow(
        (_BEV_COLOURS[dummy]*255).astype(np.uint8),
        origin="upper", extent=extent, interpolation="nearest", aspect="auto", zorder=2,
    )
    step_txt = ax_bev.text(0.02, 0.02, "", transform=ax_bev.transAxes,
                           fontsize=7, color="black", va="bottom")
    time_txt = ax_map.text(0.02, 0.97, "", transform=ax_map.transAxes,
                           fontsize=8, va="top", color="#2c3e50")
    dt_cfg   = 0.1  # will be overwritten below if available

    def _update(frame_i: int):
        fi = frames_idx[frame_i]
        for m, dot in enumerate(ship_dots):
            s  = results[m]["states"]
            fi_m = min(fi, s.shape[0] - 1)
            dot.set_data([s[fi_m, 0]], [s[fi_m, 1]])
        # BEV from agent #0
        fi_0 = min(fi, results[0]["states"].shape[0] - 1)
        pose_fi = results[0]["states"][fi_0]
        grid    = bev_b.build(world, pose_fi)
        bev_img.set_data((_BEV_COLOURS[grid]*255).astype(np.uint8))
        step_txt.set_text(f"step {fi}")
        time_txt.set_text(f"t = {fi*dt_cfg:.1f} s")
        return [*ship_dots, bev_img, step_txt, time_txt]

    ani = animation.FuncAnimation(
        fig, _update, frames=len(frames_idx),
        interval=int(1000/fps), blit=False,
    )
    ani.save(str(gif_path), writer="pillow", fps=fps)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

def make_summary_plot(all_metrics: list, out_path: Path) -> None:
    if not all_metrics:
        return
    map_ids   = [m["map_id"]                    for m in all_metrics]
    successes = [m["success_rate"]  * 100        for m in all_metrics]
    coll_free = [m["collision_free_rate"] * 100  for m in all_metrics]
    dists     = [m["mean_goal_dist_m"]           for m in all_metrics]

    x   = np.arange(len(map_ids))
    fig, axes = plt.subplots(3, 1,
                             figsize=(max(10, int(len(map_ids)*0.55)), 9),
                             sharex=True)
    axes[0].bar(x, successes,  color="#3498db", alpha=0.8)
    axes[0].axhline(50, color="red",    lw=0.8, ls="--")
    axes[0].set_ylabel("Success rate [%]"); axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, coll_free, color="#2ecc71", alpha=0.8)
    axes[1].axhline(80, color="orange", lw=0.8, ls="--")
    axes[1].set_ylabel("Collision-free [%]"); axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, dists, color="#e67e22", alpha=0.8)
    axes[2].set_ylabel("Mean goal dist [m]")
    axes[2].grid(axis="y", alpha=0.3)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(map_ids, rotation=45, ha="right", fontsize=7)
    cfg_s = all_metrics[0]["cfg_scale"] if all_metrics else "?"
    fig.suptitle(f"Conditional flow policy evaluation  (MPC, cfg_scale={cfg_s})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Summary plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    from project_config import load_config
    cfg    = load_config("common", "eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Resolve checkpoint path ────────────────────────────────────────
    if args.ckpt is None:
        raise ValueError(
            "No model checkpoint specified. "
            "Set eval.model_path in configs/eval.json or pass --ckpt <path>."
        )

    # ── Model ──────────────────────────────────────────────────────────
    model = CondActionFlowNet.from_config(cfg).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params  "
          f"H={model.H}  D={model.data_dim}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"exec_steps={args.exec_steps}  max_steps={args.max_steps}  "
          f"cfg_scale={args.cfg_scale}\n")

    # ── Helpers ────────────────────────────────────────────────────────
    norm    = ActionNormalizer(cfg)
    bev_b   = BEVBuilder(cfg)
    sim_uni = UnicycleSimulator(dt=float(cfg["simulator"]["dt"]))
    D_max   = float(cfg.get("dataset", {}).get("D_max", 120.0))

    # ── Dataset (for map/demo loading only) ────────────────────────────
    ds = FlowDataset(
        maps_dir  = args.maps_dir,
        cfg       = cfg,
        stride    = 1,
        K_windows = 1,          # minimal; only need demo meta
        seed      = args.seed,
    )

    map_ids = sorted(ds.map_ids)
    print(f"Evaluating {len(map_ids)} maps  (M={args.M} agents each) …\n")

    all_metrics = []
    for map_id in tqdm(map_ids, desc="Maps", unit="map", dynamic_ncols=True):
        world = ds._worlds.get(map_id)
        if world is None:
            continue
        metrics = eval_one_map(
            map_id, world, ds, model, norm, bev_b, sim_uni,
            args, device, out_dir, rng, D_max, cfg,
        )
        if metrics:
            all_metrics.append(metrics)

    # ── Metrics JSON ───────────────────────────────────────────────────
    metrics_path = out_dir / "metrics.json"
    json.dump(all_metrics, open(metrics_path, "w"), indent=2)
    tqdm.write(f"\nMetrics → {metrics_path}")

    if all_metrics:
        mean_suc  = np.mean([m["success_rate"]         for m in all_metrics])
        mean_cf   = np.mean([m["collision_free_rate"]  for m in all_metrics])
        mean_dist = np.mean([m["mean_goal_dist_m"]     for m in all_metrics])
        tqdm.write(f"\n{'='*55}")
        tqdm.write(f"Overall  ({len(all_metrics)} maps, MPC, cfg={args.cfg_scale})")
        tqdm.write(f"  Mean success rate    : {mean_suc*100:.1f}%")
        tqdm.write(f"  Mean coll-free rate  : {mean_cf*100:.1f}%")
        tqdm.write(f"  Mean goal dist       : {mean_dist:.2f} m")
        tqdm.write(f"{'='*55}")

    make_summary_plot(all_metrics, out_dir / "summary.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    # Pre-load config to use as defaults so the user rarely needs CLI flags.
    from project_config import OUTPUTS_DIR as _OUTPUTS_DIR, load_config
    try:
        _cfg  = load_config("common", "eval")
        _e    = _cfg.get("eval",  {})
        _tcfg = _cfg.get("train", {})
    except Exception:
        _e, _tcfg = {}, {}

    p = argparse.ArgumentParser(
        description="Evaluate conditional flow policy (rolling MPC). "
                    "Defaults come from configs/common.json + configs/eval.json."
    )
    p.add_argument("--ckpt",       default=_e.get("model_path", None),
                   help="Path to model checkpoint (.pt). "
                        "Defaults to eval.model_path in configs/eval.json.")
    p.add_argument("--maps_dir",   default=_e.get("maps_dir",  _tcfg.get("maps_dir", str(_OUTPUTS_DIR / "maps"))))
    p.add_argument("--out_dir",    default=_e.get("out_dir",   str(_OUTPUTS_DIR / "eval")))
    p.add_argument("--M",          type=int,   default=_e.get("M",          8))
    p.add_argument("--use_best_selection", action=argparse.BooleanOptionalAction,
                   default=bool(_e.get("use_best_selection", True)),
                   help="Filter M candidates via trajectory_metrics and execute "
                        "the best one (--no-use_best_selection to disable).")
    p.add_argument("--cfg_scale",  type=float, default=_e.get("cfg_scale",  1.5))
    p.add_argument("--ode_steps",  type=int,   default=_e.get("ode_steps",  20))
    p.add_argument("--exec_steps", type=int,   default=_e.get("exec_steps", 20))
    p.add_argument("--max_steps",  type=int,   default=_e.get("max_steps",  2000))
    p.add_argument("--fps",        type=int,   default=_e.get("fps",        8))
    p.add_argument("--stride",     type=int,   default=_e.get("stride",     15))
    p.add_argument("--seed",       type=int,   default=_e.get("seed",       0))
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())

