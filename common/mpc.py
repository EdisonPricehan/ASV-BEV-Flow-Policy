"""
mpc.py
======
Shared rolling-MPC loop used by both eval_cond_policy.py and inspect_mpc.py.

The single entry point is ``mpc_rollout()``.  Pass ``record=True`` to also
collect per-cycle candidate trajectories for visualisation (inspect_mpc);
leave it ``False`` for fast bulk evaluation (eval_cond_policy).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from env.square_world            import SquareWorld
from kinematics.unicycle         import UnicycleSimulator
from perception.bev_builder      import BEVBuilder
from models.cond_flow_policy     import CondActionFlowNet
from common.utils                import ActionNormalizer, build_goal_vec
from metrics.trajectory_metrics  import select_best_action_seq, default_metrics_cfg


# ---------------------------------------------------------------------------
# Shared adaptive exec-step helper
# ---------------------------------------------------------------------------

def _adaptive_exec(dist: float, goal_eps: float, exec_steps: int) -> int:
    """
    Return the number of steps to execute in this MPC cycle.
    Fewer steps are executed when the agent is close to the goal so that
    the policy can re-plan more frequently near the target.
    """
    far_thresh   = 4.0 * goal_eps
    close_thresh = 2.0 * goal_eps
    if dist >= far_thresh:
        return exec_steps
    elif dist <= close_thresh:
        return 1
    alpha = (dist - close_thresh) / (far_thresh - close_thresh)
    return max(1, int(alpha * exec_steps))


# ---------------------------------------------------------------------------
# Core MPC rollout
# ---------------------------------------------------------------------------

def mpc_rollout(
    model      : CondActionFlowNet,
    norm       : ActionNormalizer,
    bev_b      : BEVBuilder,
    sim_uni    : UnicycleSimulator,
    world      : SquareWorld,
    start_pose : np.ndarray,          # (3,) [x, y, yaw]
    goal_pose  : np.ndarray,          # (3,) [xg, yg, psig]
    D_max      : float,
    cfg        : dict,
    # MPC hyper-parameters
    M              : int   = 8,
    cfg_scale      : float = 2.0,
    ode_steps      : int   = 20,
    exec_steps     : int   = 20,
    max_steps      : int   = 2000,
    use_best_selection: bool = True,
    # misc
    device     : torch.device = None,
    seed       : Optional[int] = None,
    record     : bool = False,        # True  → also return query_records for viz
    pbar_desc  : str  = "MPC",
) -> dict:
    """
    Run one agent with a rolling MPC loop until goal is reached or
    ``max_steps`` is exceeded.

    Parameters
    ----------
    record : bool
        When True, every MPC cycle additionally records:
          - all M candidate rollout xy-traces
          - per-candidate collision flags
          - index of the selected best candidate
        This data is returned under the ``query_records`` key and is used
        by inspect_mpc.py for frame-by-frame visualisation.
        Has no effect on the executed trajectory.

    Returns
    -------
    dict with keys:
        states       : np.ndarray [T+1, 3]  full executed trajectory
        success      : bool
        collided     : bool
        steps        : int
        goal_dist    : float
        n_safe_mean  : float   mean number of safe candidates per cycle
        query_records: list[dict]  (only present when record=True)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optionally fix random seed for reproducibility across sel/nosel runs
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    goal_eps   = float(cfg["demo"]["goal_eps_m"])
    dt         = float(cfg["simulator"]["dt"])
    metric_cfg = default_metrics_cfg()

    pose        = start_pose.copy()
    traj        = [pose.copy()]
    collided    = False
    n_safe_log  : list[int]  = []
    query_records: list[dict] = []
    total_steps = 0
    n_queries   = max_steps // max(exec_steps, 1)

    pbar = tqdm(range(n_queries), leave=False, unit="query",
                desc=f"    {pbar_desc}", dynamic_ncols=True)

    for _ in pbar:
        if total_steps >= max_steps:
            break

        dist = float(np.linalg.norm(pose[:2] - goal_pose[:2]))
        if dist < goal_eps:
            break

        # ── BEV + goal conditioning ────────────────────────────────────
        bev_grid = bev_b.build(world, pose)
        gv       = build_goal_vec(pose, goal_pose, D_max)

        # ── sample M candidate action sequences ────────────────────────
        from scripts.eval.eval_cond_policy import query_policy_batch
        act_raw_M = query_policy_batch(
            model, norm, bev_grid, gv,
            M=M, cfg_scale=cfg_scale, ode_steps=ode_steps, device=device,
        )                                                         # [M, H, 2]

        # ── compute how many steps we will execute this cycle ──────────
        n_exec = min(_adaptive_exec(dist, goal_eps, exec_steps),
                     model.H,
                     max_steps - total_steps)

        # ── optional: rollout all candidates for recording ─────────────
        cand_xys: list[np.ndarray] = []
        if record:
            for m in range(M):
                p  = pose.copy()
                xy = [p[:2].copy()]
                for h in range(model.H):
                    p = sim_uni.step(p, act_raw_M[m, h])
                    xy.append(p[:2].copy())
                cand_xys.append(np.array(xy))                   # [H+1, 2]

        # ── select best (or take first) ────────────────────────────────
        if use_best_selection:
            best_idx, best_seq, all_m = select_best_action_seq(
                map_obj      = world,
                bev_builder  = bev_b,
                init_pose    = pose,
                goal_pose    = goal_pose,
                action_seqs  = act_raw_M,
                dt           = dt,
                config       = metric_cfg,
                exec_horizon = n_exec,
            )
            coll_flags = [m["safety"]["collision_any"] for m in all_m]
            n_safe     = int(sum(not c for c in coll_flags))
        else:
            best_idx   = 0
            best_seq   = act_raw_M[0]
            coll_flags = (
                [world.collision_check(xy) for xy in cand_xys]
                if record
                else [False] * M          # unknown when not recording
            )
            n_safe = M

        n_safe_log.append(n_safe)

        # ── record this cycle (inspect_mpc only) ───────────────────────
        if record:
            query_records.append({
                "pose"      : pose.copy(),
                "bev_grid"  : bev_grid.copy(),
                "cand_xys"  : cand_xys,
                "coll_flags": coll_flags,
                "best_idx"  : best_idx,
                "exec_start": len(traj) - 1,
            })

        # ── execute ────────────────────────────────────────────────────
        for s in range(n_exec):
            v, w  = float(best_seq[s, 0]), float(best_seq[s, 1])
            pose  = sim_uni.step(pose, np.array([v, w]))
            traj.append(pose.copy())
            total_steps += 1
            if world.collision_check(np.array(traj)[:, :2]):
                collided = True
                break

        dist = float(np.linalg.norm(pose[:2] - goal_pose[:2]))
        pbar.set_postfix(dist=f"{dist:.1f}m",
                         safe=f"{n_safe}/{M}",
                         exec=n_exec,
                         steps=total_steps,
                         status="COLL" if collided else "ok")
        if collided:
            break

    states  = np.array(traj)
    dist_f  = float(np.linalg.norm(states[-1, :2] - goal_pose[:2]))
    success = (not collided) and (dist_f < goal_eps)

    result = {
        "states"     : states,
        "success"    : success,
        "collided"   : collided,
        "steps"      : len(traj) - 1,
        "goal_dist"  : dist_f,
        "n_safe_mean": float(np.mean(n_safe_log)) if n_safe_log else 0.0,
    }
    if record:
        result["query_records"]    = query_records
        result["executed_states"]  = states   # alias expected by render_gif
    return result

