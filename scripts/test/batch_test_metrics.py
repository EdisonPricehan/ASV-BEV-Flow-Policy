"""
scripts/batch_test_metrics.py
==============================
Run trajectory_metrics on ALL maps under outputs/maps/ and save:

  outputs/metrics_batch/
    <map_id>/
      expert.png
      noisy.png
      bad.png
    summary_table.png     – heatmap of all scores across all maps
    summary_scores.csv    – raw numbers

Usage (from repo root):
    python -m scripts.test.batch_test_metrics
    python -m scripts.test.batch_test_metrics --H 80 --demo-idx 2
    python -m scripts.test.batch_test_metrics --maps map_s2_n5_r0 map_s8_n3_r1
"""

from __future__ import annotations

import argparse
import csv
import os
import traceback
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_config import POLICY_DIR as _POLICY_DIR, OUTPUTS_DIR as _OUTPUTS_DIR
from env.square_world import SquareWorld
from perception.bev_builder import BEVBuilder, OBSTACLE as _BEV_OBSTACLE
from metrics.trajectory_metrics import (
    compute_trajectory_metrics,
    visualize_trajectory_metrics,
    default_metrics_cfg,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch trajectory metrics across all maps.")
    p.add_argument("--maps", nargs="*", default=None,
                   help="Subset of map names to test (default: all)")
    p.add_argument("--H", type=int, default=60,
                   help="Action-sequence horizon (default: 60)")
    p.add_argument("--demo-idx", type=int, default=0,
                   help="Demo index to use as 'expert' base (default: 0)")
    p.add_argument("--best-bev", action="store_true",
                   help="Auto-select the timestep with most obstacle cells in BEV "
                        "as the eval start (useful when t=0 has no visible obstacles)")
    p.add_argument("--cfg", default=str(_POLICY_DIR / "configs" / "common.json"))
    p.add_argument("--out", default=str(_OUTPUTS_DIR / "metrics_batch"))
    p.add_argument("--no-per-map-figs", action="store_true",
                   help="Skip per-map 3-panel figures (only produce summary)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Trajectory constructors
# ---------------------------------------------------------------------------

def make_trajectories(
    expert_actions: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    expert : verbatim first-H slice of demo
    noisy  : expert + σ=0.08 Gaussian noise on ω
    bad    : high-amplitude sine on ω (zigzag, ignores heading)
    """
    H = len(expert_actions)
    noise        = rng.normal(0.0, 0.08, size=(H,)).astype(np.float32)
    noisy        = expert_actions.copy()
    noisy[:, 1] += noise

    t         = np.linspace(0, 2 * np.pi * 4, H)
    bad       = expert_actions.copy()
    bad[:, 1] = (np.sin(t) * 1.2).astype(np.float32)

    return {"expert": expert_actions.copy(), "noisy": noisy, "bad": bad}


# ---------------------------------------------------------------------------
# Find the timestep where the most obstacle cells are visible in BEV
# ---------------------------------------------------------------------------

def find_best_bev_pose(
    world    : SquareWorld,
    all_states : np.ndarray,   # (T+1, 3)
    T          : int,
    bev_builder: BEVBuilder,
    H          : int,
) -> int:
    """
    Scan timesteps 0..T-H and return the t that maximises the number of
    OBSTACLE-labelled cells in the BEV grid.
    Falls back to t=0 if no obstacle is ever visible.
    """
    best_t     = 0
    best_count = 0
    for t in range(0, T - H + 1):
        pose  = all_states[t]
        grid  = bev_builder.build(world, pose)
        count = int(np.sum(grid == _BEV_OBSTACLE))
        if count > best_count:
            best_count = count
            best_t     = t
    return best_t


# ---------------------------------------------------------------------------
# Per-map runner
# ---------------------------------------------------------------------------

def run_one_map(
    map_id     : str,
    maps_dir   : str,
    env_cfg    : dict,
    H          : int,
    demo_idx   : int,
    out_dir    : str,
    bev_builder: BEVBuilder,
    save_figs  : bool,
    best_bev   : bool = False,
) -> Optional[dict]:
    """
    Score 3 trajectories on one map.
    Returns a row dict or None if the map cannot be loaded / demos too short.
    """
    map_dir   = os.path.join(maps_dir, map_id)
    map_json  = os.path.join(map_dir, "map.json")
    demos_npz = os.path.join(map_dir, "demos.npz")

    if not os.path.exists(map_json) or not os.path.exists(demos_npz):
        print(f"  [{map_id}] SKIP – missing map.json or demos.npz")
        return None

    world = SquareWorld.load(map_json)
    data    = np.load(demos_npz)
    lengths = data["demo_lengths"]

    # pick a demo that is long enough; fall back to longest if needed
    idx = demo_idx if int(lengths[demo_idx]) >= H else int(np.argmax(lengths))
    T_demo = int(lengths[idx])
    if T_demo < H:
        print(f"  [{map_id}] SKIP – longest demo has only {T_demo} < H={H} steps")
        return None

    all_states = data["demos_states"][idx]   # (T_max+1, 3)
    dt         = float(env_cfg["simulator"]["dt"])

    # Choose starting timestep
    if best_bev:
        t_start = find_best_bev_pose(world, all_states, T_demo, bev_builder, H)
    else:
        t_start = 0

    expert_actions = data["demos_actions"][idx, t_start : t_start + H].astype(np.float32)
    init_pose      = all_states[t_start].astype(np.float32)
    goal_pose      = world.goal_pose.astype(np.float32)

    metrics_cfg = default_metrics_cfg()
    metrics_cfg["pos_ref"] = max(2.0, world.L * 0.05)

    rng   = np.random.default_rng(42)
    trajs = make_trajectories(expert_actions, rng)

    row = {"map_id": map_id, "t_start": t_start}
    map_out = os.path.join(out_dir, map_id)
    if save_figs:
        os.makedirs(map_out, exist_ok=True)

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
        sf  = m["safety"]
        gf  = m["goal"]
        sm  = m["smooth"]
        tot = m["total"]

        row[f"{name}_safety"]  = round(sf["safety_score"],  4)
        row[f"{name}_goal"]    = round(gf["goal_score"],    4)
        row[f"{name}_smooth"]  = round(sm["smooth_score"],  4)
        row[f"{name}_total"]   = round(tot["total_score"],  4)
        row[f"{name}_collide"] = int(sf["collision_any"])
        row[f"{name}_min_clr"] = round(sf["min_clearance"], 3)
        row[f"{name}_pos_err"] = round(gf["final_pos_error"], 3)

        if save_figs:
            fig_path = os.path.join(map_out, f"{name}.png")
            visualize_trajectory_metrics(
                metrics     = m,
                map_obj     = world,
                bev_builder = bev_builder,
                init_pose   = init_pose,
                goal_pose   = goal_pose,
                action_seq  = act_seq,
                dt          = dt,
                save_path   = fig_path,
                label       = f"{map_id} · {name}",
            )

    return row


# ---------------------------------------------------------------------------
# Summary heatmap
# ---------------------------------------------------------------------------

def _parse_map_id(map_id: str) -> tuple[int, int, int]:
    """Extract (scale, n_obs, replicate) from 'map_sX_nY_rZ'."""
    parts = map_id.split("_")
    s = int(parts[1][1:]) if len(parts) > 1 else 0
    n = int(parts[2][1:]) if len(parts) > 2 else 0
    r = int(parts[3][1:]) if len(parts) > 3 else 0
    return s, n, r


def save_summary_table(rows: list[dict], out_dir: str) -> None:
    """
    Save:
      summary_scores.csv   – all raw numbers
      summary_table.png    – compact heatmap of total_score per map × traj_type
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── CSV ────────────────────────────────────────────────────────────
    if not rows:
        return
    csv_path = os.path.join(out_dir, "summary_scores.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV saved → {csv_path}")

    # ── Heatmap (total_score) ───────────────────────────────────────────
    map_ids   = [r["map_id"] for r in rows]
    traj_cols = ["expert", "noisy", "bad"]

    data_total   = np.array([[r[f"{t}_total"]   for t in traj_cols] for r in rows])
    data_collide = np.array([[r[f"{t}_collide"] for t in traj_cols] for r in rows])

    n_maps = len(map_ids)
    fig_h  = max(6, n_maps * 0.38)
    fig, axes = plt.subplots(1, 2, figsize=(13, fig_h))
    fig.suptitle("Batch Trajectory Metrics – All Maps", fontsize=13, fontweight="bold")

    # -- total_score heatmap
    ax = axes[0]
    vmin, vmax = data_total.min(), 0.0
    im = ax.imshow(data_total, aspect="auto", cmap="RdYlGn",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(traj_cols)))
    ax.set_xticklabels(traj_cols, fontsize=9)
    ax.set_yticks(range(n_maps))
    ax.set_yticklabels(map_ids, fontsize=7)
    ax.set_title("total_score  (green = better)", fontsize=9)
    for i in range(n_maps):
        for j in range(len(traj_cols)):
            ax.text(j, i, f"{data_total[i,j]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="black" if data_total[i, j] > (vmin + vmax) / 2 else "white")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # -- collision heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(data_collide, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax2.set_xticks(range(len(traj_cols)))
    ax2.set_xticklabels(traj_cols, fontsize=9)
    ax2.set_yticks(range(n_maps))
    ax2.set_yticklabels(map_ids, fontsize=7)
    ax2.set_title("collision_any  (red = collided)", fontsize=9)
    for i in range(n_maps):
        for j in range(len(traj_cols)):
            ax2.text(j, i, str(data_collide[i, j]),
                     ha="center", va="center", fontsize=8,
                     color="black" if data_collide[i, j] == 0 else "white")
    plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02)

    plt.tight_layout()
    png_path = os.path.join(out_dir, "summary_table.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary heatmap saved → {png_path}")

    # ── per-scale bar chart ─────────────────────────────────────────────
    # Group by scale (s2 / s5 / s8) and show average total_score
    scales = sorted(set(_parse_map_id(m)[0] for m in map_ids))
    traj_types = traj_cols

    fig2, ax3 = plt.subplots(figsize=(9, 4))
    bar_w = 0.22
    x     = np.arange(len(scales))
    colors = {"expert": "#2980b9", "noisy": "#27ae60", "bad": "#e74c3c"}

    for ji, ttype in enumerate(traj_types):
        avgs = []
        for sc in scales:
            idxs  = [i for i, m in enumerate(map_ids) if _parse_map_id(m)[0] == sc]
            avgs.append(np.mean([data_total[i, ji] for i in idxs]))
        ax3.bar(x + ji * bar_w, avgs, bar_w, label=ttype, color=colors[ttype], alpha=0.85)

    ax3.set_xticks(x + bar_w)
    ax3.set_xticklabels([f"scale={s}" for s in scales], fontsize=10)
    ax3.set_ylabel("avg total_score")
    ax3.set_title("Average total_score by map scale and trajectory type", fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.3)
    ax3.axhline(0, color="black", lw=0.8, ls="--")

    plt.tight_layout()
    bar_path = os.path.join(out_dir, "summary_by_scale.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Scale bar chart saved → {bar_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print(f"{'MAP':20s}  {'expert_tot':>10}  {'noisy_tot':>9}  {'bad_tot':>8}  "
          f"{'exp_col':>7}  {'noisy_col':>9}  {'bad_col':>7}  "
          f"{'exp_pos_err':>11}  {'bad_pos_err':>11}")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['map_id']:20s}  "
            f"{r['expert_total']:+10.3f}  "
            f"{r['noisy_total']:+9.3f}  "
            f"{r['bad_total']:+8.3f}  "
            f"{r['expert_collide']:7d}  "
            f"{r['noisy_collide']:9d}  "
            f"{r['bad_collide']:7d}  "
            f"{r['expert_pos_err']:11.2f}  "
            f"{r['bad_pos_err']:11.2f}"
        )
    print("=" * 100)

    # aggregate stats
    n = len(rows)
    if n == 0:
        return
    for ttype in ("expert", "noisy", "bad"):
        col_rate = sum(r[f"{ttype}_collide"] for r in rows) / n * 100
        avg_tot  = np.mean([r[f"{ttype}_total"] for r in rows])
        print(f"  [{ttype:6s}]  avg_total={avg_tot:+.3f}   collision_rate={col_rate:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    from project_config import load_config
    env_cfg = load_config("common")

    bev_builder = BEVBuilder(env_cfg)
    maps_dir    = str(_OUTPUTS_DIR / "maps")
    os.makedirs(args.out, exist_ok=True)

    # discover maps
    if args.maps:
        all_map_ids = sorted(args.maps)
    else:
        all_map_ids = sorted(
            d for d in os.listdir(maps_dir)
            if os.path.isdir(os.path.join(maps_dir, d)) and d.startswith("map_")
        )

    print(f"Maps to test : {len(all_map_ids)}")
    print(f"Horizon H    : {args.H}")
    print(f"Demo index   : {args.demo_idx}")
    print(f"Output dir   : {args.out}")
    print(f"Per-map figs : {not args.no_per_map_figs}\n")

    rows: list[dict] = []
    for i, map_id in enumerate(all_map_ids, 1):
        print(f"[{i:2d}/{len(all_map_ids)}] {map_id} ...", end=" ", flush=True)
        try:
            row = run_one_map(
                map_id      = map_id,
                maps_dir    = maps_dir,
                env_cfg     = env_cfg,
                H           = args.H,
                demo_idx    = args.demo_idx,
                out_dir     = args.out,
                bev_builder = bev_builder,
                save_figs   = not args.no_per_map_figs,
                best_bev    = args.best_bev,
            )
            if row is not None:
                rows.append(row)
                print(
                    f"t={row['t_start']:4d}  "
                    f"expert={row['expert_total']:+.2f}  "
                    f"noisy={row['noisy_total']:+.2f}  "
                    f"bad={row['bad_total']:+.2f}  "
                    f"col=[{row['expert_collide']},{row['noisy_collide']},{row['bad_collide']}]"
                )
        except Exception:
            print("ERROR")
            traceback.print_exc()

    print_summary(rows)
    save_summary_table(rows, args.out)
    print("\nAll done.")


if __name__ == "__main__":
    main()

