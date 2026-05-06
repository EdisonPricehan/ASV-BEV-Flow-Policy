"""
scripts/test_trajectory_metrics.py
===================================
Quick smoke-test for trajectory_metrics.py.

Loads one map and one expert demo trajectory from the maps/ folder,
constructs three candidate trajectories, scores each one, and saves
visualisation figures under outputs/metrics_test/.

Usage (from repo root):
    python -m scripts.test.test_trajectory_metrics
    python -m scripts.test.test_trajectory_metrics --map map_s5_n3_r1

Trajectories tested
-------------------
1. Expert  : first H steps of a real demo trajectory
2. Noisy   : expert + small Gaussian noise on omega
3. Bad     : expert with exaggerated omega oscillations (likely to collide)
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from project_config import POLICY_DIR as _POLICY_DIR, OUTPUTS_DIR as _OUTPUTS_DIR
from env.square_world import SquareWorld
from perception.bev_builder import BEVBuilder
from metrics.trajectory_metrics import (
    compute_trajectory_metrics,
    visualize_trajectory_metrics,
    default_metrics_cfg,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test trajectory metrics on toy maps.")
    p.add_argument(
        "--map", default="map_s5_n3_r1",
        help="Map sub-folder name under outputs/maps/ (default: map_s5_n3_r1)",
    )
    p.add_argument(
        "--demo-idx", type=int, default=0,
        help="Which demo trajectory to use as 'expert' (default: 0)",
    )
    p.add_argument(
        "--H", type=int, default=60,
        help="Action-sequence horizon to evaluate (default: 60)",
    )
    p.add_argument(
        "--cfg", default=str(_POLICY_DIR / "configs" / "common.json"),
        help="Path to configs/common.json",
    )
    p.add_argument(
        "--out", default=str(_OUTPUTS_DIR / "metrics_test"),
        help="Output directory for visualisation figures",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Build test trajectories
# ---------------------------------------------------------------------------

def make_trajectories(
    expert_actions: np.ndarray,   # (H, 2) raw
    rng           : np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Return a dict of named (H, 2) raw action sequences.

    expert : verbatim expert slice
    noisy  : expert + small Gaussian noise on omega
    bad    : expert omega replaced by high-amplitude sine (zigzag)
    """
    H = len(expert_actions)

    # -- Noisy: small perturbation on omega only
    noise        = rng.normal(0.0, 0.08, size=(H,)).astype(np.float32)
    noisy        = expert_actions.copy()
    noisy[:, 1] += noise

    # -- Bad: replace omega with a high-frequency oscillation that
    #    ignores the correct heading → likely to veer into obstacles
    t             = np.linspace(0, 2 * np.pi * 4, H)
    bad           = expert_actions.copy()
    bad[:, 1]     = (np.sin(t) * 1.2).astype(np.float32)   # ±1.2 rad/s zigzag

    return {
        "expert": expert_actions.copy(),
        "noisy" : noisy,
        "bad"   : bad,
    }


# ---------------------------------------------------------------------------
# Print a compact score table
# ---------------------------------------------------------------------------

def print_scores(label: str, metrics: dict) -> None:
    sf  = metrics["safety"]
    gf  = metrics["goal"]
    sm  = metrics["smooth"]
    tot = metrics["total"]
    print(
        f"  [{label:6s}]  "
        f"safety={sf['safety_score']:+.3f}  "
        f"goal={gf['goal_score']:+.3f}  "
        f"smooth={sm['smooth_score']:+.3f}  "
        f"total={tot['total_score']:+.3f}  "
        f"| collision={sf['collision_any']}  "
        f"min_clr={sf['min_clearance']:.2f} m  "
        f"pos_err={gf['final_pos_error']:.2f} m  "
        f"yaw_err={np.degrees(gf['final_yaw_error']):.1f}°"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── load config ──────────────────────────────────────────────────────
    from project_config import load_config
    env_cfg = load_config("common")

    dt   = float(env_cfg["simulator"]["dt"])
    H    = args.H

    # ── load map ─────────────────────────────────────────────────────────
    maps_dir = str(_OUTPUTS_DIR / "maps")
    map_dir  = os.path.join(maps_dir, args.map)
    if not os.path.isdir(map_dir):
        raise FileNotFoundError(f"Map directory not found: {map_dir}")

    map_json = os.path.join(map_dir, "map.json")
    demos_npz = os.path.join(map_dir, "demos.npz")

    print(f"\nLoading map  : {map_json}")
    world = SquareWorld.load(map_json)
    print(f"  {world}")

    print(f"Loading demos: {demos_npz}")
    data    = np.load(demos_npz)
    all_actions = data["demos_actions"]   # (N, T_max, 2)
    all_states  = data["demos_states"]    # (N, T_max+1, 3)
    lengths     = data["demo_lengths"]    # (N,)

    demo_idx = args.demo_idx
    T_demo   = int(lengths[demo_idx])
    if T_demo < H:
        raise ValueError(
            f"Demo #{demo_idx} has only {T_demo} steps, but H={H} requested."
        )

    expert_actions = all_actions[demo_idx, :H].astype(np.float32)   # (H, 2)
    init_pose      = all_states[demo_idx, 0].astype(np.float32)      # (3,)
    goal_pose      = world.goal_pose.astype(np.float32)             # (3,)

    print(f"\nDemo #{demo_idx}: T={T_demo} steps  init={init_pose}  goal={goal_pose}")
    print(f"Evaluating H={H} steps  dt={dt} s")

    # ── build BEV builder ─────────────────────────────────────────────────
    bev_builder = BEVBuilder(env_cfg)

    # ── metrics config ────────────────────────────────────────────────────
    metrics_cfg = default_metrics_cfg()
    # Adjust pos_ref to map scale (2 m is fine for smaller maps, scale up for large)
    metrics_cfg["pos_ref"] = max(2.0, world.L * 0.05)

    # ── construct three trajectories ──────────────────────────────────────
    rng   = np.random.default_rng(42)
    trajs = make_trajectories(expert_actions, rng)

    # ── compute metrics ───────────────────────────────────────────────────
    print("\n── Scores ─────────────────────────────────────────────────────────")
    all_metrics: dict[str, dict] = {}
    for name, act_seq in trajs.items():
        m = compute_trajectory_metrics(
            map_obj     = world,
            bev_builder = bev_builder,
            init_pose   = init_pose,
            goal_pose   = goal_pose,
            action_seq  = act_seq,
            dt          = dt,
            config      = metrics_cfg,
        )
        all_metrics[name] = m
        print_scores(name, m)

    # ── visualise ─────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    print(f"\nSaving figures to: {args.out}/")

    for name, act_seq in trajs.items():
        save_path = os.path.join(args.out, f"{args.map}_{name}.png")
        visualize_trajectory_metrics(
            metrics     = all_metrics[name],
            map_obj     = world,
            bev_builder = bev_builder,
            init_pose   = init_pose,
            goal_pose   = goal_pose,
            action_seq  = act_seq,
            dt          = dt,
            save_path   = save_path,
            label       = name,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

