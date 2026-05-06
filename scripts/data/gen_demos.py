"""
gen_demos.py
============
Generate multimodal unicycle demonstration dataset for the toy 2-D flow policy.

All environment, simulator, and controller parameters are read from
configs/common.json.  The map is randomised once and saved alongside the demos
so that training and evaluation use the same geometry.

Horizon design
--------------
Two separate horizon concepts are used:

  H      – policy prediction horizon: the fixed sliding-window size that the
            flow policy will be trained on.
            H = ceil(D_bev / v_nominal / dt) * h_scale
            Only covers ~one BEV-depth of forward travel.

  T_max  – maximum demo trajectory length (steps).
            T_max = ceil(L * path_scale / v_nominal / dt)
            Each demo runs until the goal is reached OR T_max steps elapse.
            Actual trajectory length T varies per demo (T >= H).
            Longer and more varied T values → richer sliding-window dataset.

Variable-length storage
-----------------------
Demos are padded to T_max and stored with a ``demo_lengths`` array so that
training can recover each demo's true length and apply sliding windows only
over valid steps.

Multimodal strategy
-------------------
Side (left / right) is sampled 50/50.  The bypass waypoint is placed
perpendicular to the start→goal line, offset by ~0.25 L, with Gaussian noise
for diversity.  After the waypoint is reached the controller heads for goal.

Feasibility criteria
--------------------
A demo is accepted only if:
  1. No collision with any obstacle or boundary.
  2. Final position error  < goal_eps_m  (absolute metres, not scaled by L)
  3. Final heading error   < goal_yaw_eps_deg

Usage
-----
    python -m scripts.data.gen_demos
    python -m scripts.data.gen_demos --config configs/common.json --n_demos 500
    python -m scripts.data.gen_demos --map_path outputs/data/my_map.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

from project_config import POLICY_DIR as _POLICY_DIR, OUTPUTS_DIR as _OUTPUTS_DIR
from env.square_world import SquareWorld
from kinematics.unicycle import UnicycleSimulator
from common.utils import wrap_to_pi


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

from project_config import load_config as _pkg_load_config


def load_config(path: str = None) -> dict:
    """Load common config via the package helper (path argument kept for API compat)."""
    return _pkg_load_config("common")


def compute_H(cfg: dict) -> int:
    """
    Policy prediction horizon.

    H = ceil(D_bev / v_nominal / dt)

    D_bev     comes from cfg["map"]["D_bev"]       (map geometry).
    v_nominal comes from cfg["controller"]["v_nominal"] (single source of truth).
    dt        comes from cfg["simulator"]["dt"].
    """
    D_bev     = cfg["map"]["D_bev"]
    v_nominal = cfg["controller"]["v_nominal"]
    dt        = cfg["simulator"]["dt"]
    return math.ceil(D_bev / v_nominal / dt)


def compute_T_max(cfg: dict, world: "SquareWorld | None" = None) -> int:
    """
    Maximum demo trajectory length (steps).

    T_max = ceil(L * path_scale / v_nominal / dt)

    L is taken from *world* if provided (actual map), otherwise from cfg.
    v_nominal comes from cfg["controller"]["v_nominal"] (single source of truth).
    """
    L          = world.L if world is not None else cfg["map"]["D_bev"] * cfg["map"]["scale"]
    v_nominal  = cfg["controller"]["v_nominal"]
    dt         = cfg["simulator"]["dt"]
    path_scale = cfg["horizon"]["path_scale"]
    return math.ceil(L * path_scale / v_nominal / dt)


# ---------------------------------------------------------------------------
# Waypoint / path planning helpers
# ---------------------------------------------------------------------------

def _line_blocked(
    start   : np.ndarray,
    goal    : np.ndarray,
    world : SquareWorld,
    n_check : int = 40,
) -> bool:
    """Return True if the straight line start→goal passes through any obstacle."""
    for t in np.linspace(0.0, 1.0, n_check):
        pt = (start + t * (goal - start)).astype(np.float32)
        if not world._point_free(pt, extra=0.0):
            return True
    return False


def _sample_waypoint(
    side     : str,
    world  : SquareWorld,
    rng      : np.random.Generator,
    noise_std: float,
) -> np.ndarray | None:
    """
    Sample a collision-free bypass waypoint perpendicular to start→goal.

    Tries candidate anchor positions spread along the start→goal segment
    (not only the midpoint) so that dense obstacle maps are handled correctly.

    Returns None if no free waypoint can be found after all attempts.
    """
    L     = world.L
    start = world.start_state[:2]
    goal  = world.goal_xy
    sg    = goal - start
    sg_len = np.linalg.norm(sg)
    sg_unit = sg / sg_len if sg_len > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
    perp    = np.array([-sg_unit[1], sg_unit[0]], dtype=np.float32)
    sign    = 1.0 if side == "left" else -1.0

    # Candidate anchor points: sample positions along the start→goal line,
    # biased toward the middle third.  Try multiple perpendicular offsets.
    for _ in range(2000):
        t      = rng.uniform(0.25, 0.75)
        anchor = start + t * sg
        frac   = rng.uniform(0.15, 0.35)
        noise  = rng.normal(0.0, noise_std, size=2).astype(np.float32)
        wp     = np.clip(
            anchor + sign * frac * L * perp + noise,
            0.08 * L, 0.92 * L,
        ).astype(np.float32)
        if (world._point_free(wp, extra=0.0)
                and not _line_blocked(start, wp, world)
                and not _line_blocked(wp, goal, world)):
            return wp

    return None


# ---------------------------------------------------------------------------
# Single demo generator  (variable-length, early-stop at goal)
# ---------------------------------------------------------------------------

def _generate_one_demo(
    world : SquareWorld,
    sim     : UnicycleSimulator,
    H       : int,
    T_max   : int,
    cfg_ctrl: dict,
    cfg_demo: dict,
    rng     : np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray | None, int | None]:
    """
    Simulate one demo trajectory.

    Strategy
    --------
    1. Check whether the direct start→goal line is blocked by an obstacle.
    2. If NOT blocked: drive straight to goal (small lateral noise for diversity).
    3. If blocked: sample a bypass waypoint on a random left/right side,
       then drive start → waypoint → goal.

    Near the goal, yaw is smoothly blended toward goal_pose[2] to avoid
    sharp in-place turns at the endpoint.

    Feasibility checks (all must pass):
      - No collision with any obstacle or boundary.
      - Final position error < goal_eps_m  (absolute metres).
      - Final heading error  < goal_yaw_eps_deg.
      - Trajectory length T >= H  (at least one sliding window).
    """
    L = world.L

    v_nominal      = cfg_ctrl["v_nominal"]
    v_noise_std    = cfg_ctrl["v_noise_std"]
    v_max          = cfg_ctrl["v_max"]
    v_min          = cfg_ctrl["v_min"]
    k_yaw          = cfg_ctrl["k_yaw"]
    omega_max      = cfg_ctrl["omega_max"]
    wp_switch_dist = cfg_ctrl["wp_switch_dist_frac"] * L
    noise_std      = cfg_ctrl["wp_noise_std_frac"] * L

    goal_eps     = cfg_demo["goal_eps_m"]                     # absolute [m], not scaled by L
    goal_yaw_eps = math.radians(cfg_demo["goal_yaw_eps_deg"])

    start = world.start_state[:2]
    goal  = world.goal_xy

    # ── decide whether a bypass waypoint is needed ────────────────────
    blocked = _line_blocked(start, goal, world)

    # Even when the straight line is clear, a large initial yaw error causes
    # the controller to arc widely before aligning – that arc may hit an
    # obstacle.  Detect this and add a near-start steering waypoint.
    ideal_yaw  = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    init_yaw   = float(world.start_state[2])
    yaw_error  = abs(wrap_to_pi(ideal_yaw - init_yaw))
    large_yaw_error = yaw_error > math.pi / 2   # > 90 deg

    if blocked or large_yaw_error:
        side     = "left" if rng.random() < 0.5 else "right"
        waypoint = _sample_waypoint(side, world, rng, noise_std)
        if waypoint is None:
            # try the other side before giving up
            other = "right" if side == "left" else "left"
            waypoint = _sample_waypoint(other, world, rng, noise_std)
        if waypoint is None:
            return None, None, None
    else:
        # Direct path clear and heading well-aligned: go straight to goal.
        waypoint = goal.copy()

    v_bias = float(rng.uniform(-0.05 * v_nominal, 0.05 * v_nominal))

    state_list   = [world.start_state.copy()]
    action_list  = []
    reached_wp   = False
    reached_pos  = False   # reached goal xy within goal_eps
    reached_goal = False

    for _ in range(T_max):
        state = state_list[-1]
        pos   = state[:2]
        yaw   = state[2]

        # ── phase 1: drive toward waypoint ────────────────────────────
        if not reached_wp:
            if np.linalg.norm(pos - waypoint) < wp_switch_dist:
                reached_wp = True

        # ── phase 2: drive toward goal position ───────────────────────
        if reached_wp and not reached_pos:
            dist_to_goal_check = float(np.linalg.norm(pos - goal))
            if dist_to_goal_check < goal_eps * 1.05:   # slight tolerance for fp
                reached_pos = True

        # ── compute desired yaw ───────────────────────────────────────
        if not reached_wp:
            diff        = waypoint - pos
            desired_yaw = float(np.arctan2(diff[1], diff[0]))
        else:
            diff        = goal - pos
            dist_to_g   = float(np.linalg.norm(diff))
            desired_yaw = (float(np.arctan2(diff[1], diff[0]))
                           if dist_to_g > 1e-3 else yaw)

        yaw_err = wrap_to_pi(desired_yaw - yaw)
        omega   = float(np.clip(k_yaw * yaw_err, -omega_max, omega_max))

        # Linear speed taper: smoothly reduce v as we approach goal.
        # v_taper = clip(dist / taper_start, 0, 1)
        # → at dist >= taper_start: full speed
        # → at dist = 0:            v = 0   (only AT the goal centre)
        # → at dist = goal_eps:     v ≈ goal_eps/taper_start  (still positive)
        # This ensures the ship can cross the goal_eps boundary.
        dist_to_goal = float(np.linalg.norm(pos - goal))
        taper_start  = 3.0 * goal_eps
        v_taper      = float(np.clip(dist_to_goal / taper_start, 0.0, 1.0))

        v_raw = float(np.clip(
            v_nominal + v_bias + rng.normal(0.0, v_noise_std),
            0.0, v_max,           # v_min = 0 to allow full taper
        ))
        v = v_raw * v_taper

        action = np.array([v, omega], dtype=np.float32)
        action_list.append(action)
        state_list.append(sim.step(state, action))

        # ── early-stop: position reached (no in-place rotation) ───────
        # We do NOT do in-place heading alignment here; the dataset will
        # exclude windows that start at or after T_active (inside goal_eps).
        if reached_pos:
            reached_goal = True
            break

    T       = len(action_list)
    actions = np.stack(action_list)   # (T, 2)
    states  = np.stack(state_list)    # (T+1, 3)
    traj_xy = states[:, :2]

    if not reached_goal:
        return None, None, None
    if world.collision_check(traj_xy):
        return None, None, None

    return actions, states, T


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_demos(
    cfg     : dict,
    world : SquareWorld,
    n_demos : int | None = None,
    seed    : int | None = None,
) -> dict:
    """
    Generate *n_demos* variable-length feasible demonstrations on *world*.

    Trajectories are padded to T_max and stored together with a
    ``demo_lengths`` array that records each demo's true step count T,
    so that training can reconstruct sliding windows over valid steps only.

    Returns
    -------
    dict with keys:
        demos_actions  : float32 [N, T_max, 2]   – zero-padded
        demos_states   : float32 [N, T_max+1, 3] – zero-padded
        demo_lengths   : int32   [N]              – actual T per demo
        H              : int32                    – policy horizon
        T_max          : int32                    – max trajectory length
        dt             : float32
    """
    cfg_demo = cfg["demo"]
    cfg_ctrl = cfg["controller"]

    n_demos  = n_demos if n_demos is not None else cfg_demo["n_demos"]
    seed     = seed    if seed    is not None else cfg_demo["seed"]
    max_att  = n_demos * cfg_demo["max_attempts_mul"]

    H     = compute_H(cfg)
    T_max = compute_T_max(cfg, world)
    dt    = cfg["simulator"]["dt"]
    sim   = UnicycleSimulator(dt=dt)
    rng   = np.random.default_rng(seed)

    print(f"Map   : {world}")
    print(f"H     = {H}   (policy horizon, dt={dt} s → {H*dt:.1f} s)")
    print(f"T_max = {T_max}  (max demo steps → {T_max*dt:.1f} s)")
    print(f"Generating {n_demos} demos …")

    all_actions : list[np.ndarray] = []
    all_states  : list[np.ndarray] = []
    all_lengths : list[int]        = []
    attempts = 0

    while len(all_actions) < n_demos and attempts < max_att:
        acts, sts, T = _generate_one_demo(
            world, sim, H, T_max, cfg_ctrl, cfg_demo, rng
        )
        attempts += 1
        if acts is not None:
            all_actions.append(acts)
            all_states.append(sts)
            all_lengths.append(T)
            n = len(all_actions)
            if n % 50 == 0:
                mean_T = sum(all_lengths) / len(all_lengths)
                print(f"  Collected {n}/{n_demos}  "
                      f"(attempt {attempts}, "
                      f"success {100*n/attempts:.1f} %, "
                      f"mean T={mean_T:.0f})")

    collected = len(all_actions)
    if collected == 0:
        raise RuntimeError(
            "No feasible demos generated. "
            "Check controller / map / feasibility parameters in configs/common.json."
        )
    if collected < n_demos:
        print(f"WARNING: only {collected}/{n_demos} demos in {attempts} attempts.")
    else:
        mean_T = sum(all_lengths) / collected
        print(f"Done: {collected} demos in {attempts} attempts "
              f"(success {100*collected/attempts:.1f} %, mean T={mean_T:.0f} steps)")

    # Pad to T_max
    actions_pad = np.zeros((collected, T_max,     2), dtype=np.float32)
    states_pad  = np.zeros((collected, T_max + 1, 3), dtype=np.float32)
    for i, (a, s, T) in enumerate(zip(all_actions, all_states, all_lengths)):
        actions_pad[i, :T]     = a
        states_pad [i, :T + 1] = s

    return {
        "demos_actions": actions_pad,
        "demos_states" : states_pad,
        "demo_lengths" : np.array(all_lengths, dtype=np.int32),
        "H"            : np.int32(H),
        "T_max"        : np.int32(T_max),
        "dt"           : np.float32(dt),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate toy unicycle demonstration dataset.\n\n"
                    "Two modes:\n"
                    "  --maps_dir  (recommended) batch-generate demos for every\n"
                    "              map_* subdirectory under the given folder.\n"
                    "  --map_path  single-map mode (legacy).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--config",   type=str, default=_DEFAULT_CONFIG)
    p.add_argument("--maps_dir", type=str, default=None,
                   help="Root dir containing map_* subdirs (batch mode).\n"
                        "Defaults to cfg[train][maps_dir] if not given.")
    p.add_argument("--n_demos",  type=int, default=None,
                   help="Override cfg[demo][n_demos] (per map)")
    p.add_argument("--seed",     type=int, default=None,
                   help="Override cfg[demo][seed] (base seed; each map gets seed+i)")
    p.add_argument("--map_path", type=str, default=None,
                   help="Single-map mode: path to an existing map JSON.")
    p.add_argument("--out",      type=str, default=None,
                   help="Single-map mode: override output .npz path.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-generate demos even if demos.npz already exists.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-map save helper (shared by both modes)
# ---------------------------------------------------------------------------

def _save_and_report(
    data    : dict,
    world : SquareWorld,
    out_npz : str,
) -> None:
    """Save demos.npz and demos_preview.png next to it."""
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    np.savez(out_npz, **data)

    lengths   = data["demo_lengths"]
    H         = int(data["H"])
    T_max     = int(data["T_max"])
    N         = len(lengths)
    n_windows = int(sum(max(0, T - H + 1) for T in lengths))

    print(f"  Saved  → {out_npz}")
    print(f"    demos_actions : {data['demos_actions'].shape}  (padded to T_max={T_max})")
    print(f"    demo_lengths  : min={lengths.min()}  max={lengths.max()}  "
          f"mean={lengths.mean():.0f}  (H={H})")
    print(f"    Sliding-window samples: {n_windows}")

    # mode split
    start  = world.start_state[:2]
    goal   = world.goal_xy
    sg     = goal - start
    sg_len = np.linalg.norm(sg)
    perp   = (np.array([-sg[1], sg[0]]) / sg_len
               if sg_len > 1e-6 else np.array([0.0, 1.0]))
    states  = data["demos_states"]
    mid_idx = np.minimum(lengths // 2, T_max)
    mid_pos = np.array([states[i, mid_idx[i], :2] for i in range(N)])
    proj    = (mid_pos - start) @ perp
    n_left  = int((proj > 0).sum())
    n_right = int((proj <= 0).sum())
    print(f"    Mode split: {n_left} left / {n_right} right")

    # visualisation
    import matplotlib; matplotlib.use("Agg")
    vis_path = os.path.join(os.path.dirname(os.path.abspath(out_npz)),
                            "demos_preview.png")
    vis_idx  = np.linspace(0, N - 1, min(40, N), dtype=int)
    world.visualise(
        trajectories=[states[i, :lengths[i]+1, :2] for i in vis_idx],
        save_path=vis_path,
        title=f"{N} demos  |  H={H}  T∈[{lengths.min()},{lengths.max()}]  "
              f"{n_windows} windows",
    )
    print(f"    Preview → {vis_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    cfg  = load_config(args.config)

    base_seed = args.seed if args.seed is not None else cfg["demo"]["seed"]

    # ── Batch mode: --maps_dir ─────────────────────────────────────────
    maps_dir = args.maps_dir or cfg.get("train", {}).get("maps_dir")
    if maps_dir and os.path.isdir(maps_dir) and args.map_path is None:
        map_subdirs = sorted(
            d for d in os.listdir(maps_dir)
            if os.path.isdir(os.path.join(maps_dir, d)) and d.startswith("map_")
        )
        if not map_subdirs:
            print(f"No map_* subdirectories found in {maps_dir}. Exiting.")
            sys.exit(1)

        print(f"Batch mode: {len(map_subdirs)} maps in {maps_dir}")
        print(f"n_demos per map: {args.n_demos or cfg['demo']['n_demos']}\n")

        total_windows = 0
        for i, subdir_name in enumerate(map_subdirs):
            subdir   = os.path.join(maps_dir, subdir_name)
            map_json = os.path.join(subdir, "map.json")
            out_npz  = os.path.join(subdir, "demos.npz")

            if not os.path.isfile(map_json):
                print(f"[{i+1}/{len(map_subdirs)}] {subdir_name}: no map.json, skipping.")
                continue

            if os.path.isfile(out_npz) and not args.overwrite:
                print(f"[{i+1}/{len(map_subdirs)}] {subdir_name}: demos.npz exists, skipping "
                      f"(use --overwrite to regenerate).")
                continue

            print(f"[{i+1}/{len(map_subdirs)}] {subdir_name}")
            world = SquareWorld.load(map_json)
            # give each map a reproducible but distinct seed
            map_seed = base_seed + i
            try:
                data = generate_demos(cfg, world,
                                      n_demos=args.n_demos,
                                      seed=map_seed)
                _save_and_report(data, world, out_npz)
                total_windows += int(sum(
                    max(0, T - int(data["H"]) + 1)
                    for T in data["demo_lengths"]
                ))
            except RuntimeError as e:
                print(f"  WARNING: {e}")
            print()

        print(f"Batch complete. Total sliding-window samples across all maps: {total_windows}")

    # ── Single-map mode (legacy) ───────────────────────────────────────
    else:
        if args.map_path and os.path.exists(args.map_path):
            print(f"Loading existing map from {args.map_path}")
            world = SquareWorld.load(args.map_path)
        else:
            world = SquareWorld.from_config(cfg["map"])
            world.randomise()
            map_save = args.map_path or cfg["demo"]["map_path"]
            world.save(map_save)
            print(f"New map saved to {map_save}")

        data    = generate_demos(cfg, world, n_demos=args.n_demos, seed=base_seed)
        out_npz = args.out or cfg["demo"]["output_path"]
        _save_and_report(data, world, out_npz)

