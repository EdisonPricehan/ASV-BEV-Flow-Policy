"""
flow_dataset.py
===================
PyTorch Dataset for training a conditional flow policy on the unicycle
demonstration data.

Each sample (__getitem__) returns a dict:
    bev_id      : torch.uint8   [H_cells, W_cells]  – 0=unknown,1=free,2=obstacle
    goal_vec    : torch.float32 [5]                  – [a_sin,a_cos,b_sin,b_cos,d_norm]
    action_seq  : torch.float32 [H, 2]               – normalised (v,omega) ∈ [-1,1]
    meta        : dict  {map_id, traj_id, t_start, pose_t, goal_pose}

goal_vec definition
-------------------
  line_yaw = atan2(y_g - y_t, x_g - x_t)
  alpha    = wrap_to_pi(line_yaw - psi_t)      # heading error toward goal
  beta     = wrap_to_pi(psi_g - line_yaw)      # goal heading vs. line direction
  d_norm   = clip(dist / D_max, 0, 1)
  goal_vec = [sin(alpha), cos(alpha), sin(beta), cos(beta), d_norm]

Action normalisation
--------------------
  v_scaled     = clip(v     / v_max,     -1, 1)
  omega_scaled = clip(omega / omega_max, -1, 1)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from perception.bev_builder import BEVBuilder
from env.square_world import SquareWorld
from common.utils import ActionNormalizer, build_goal_vec


# ---------------------------------------------------------------------------
# FlowDataset
# ---------------------------------------------------------------------------

class FlowDataset(Dataset):
    """
    Sliding-window dataset over unicycle demonstration trajectories.

    Parameters
    ----------
    maps_dir  : root directory that contains map_*/  subdirectories
    cfg       : loaded config dict (from load_config("common"))
    map_ids   : list of map subdirectory names to include (None = all)
    stride    : sliding-window step size (default 3)
    K_windows : max windows to sample per trajectory (default 60)
    seed      : RNG seed for window subsampling
    """

    def __init__(
        self,
        maps_dir : str,
        cfg      : dict,
        map_ids  : Optional[List[str]] = None,
        stride   : int = 3,
        K_windows: int = 60,
        seed     : int = 0,
    ) -> None:
        super().__init__()

        self.maps_dir  = maps_dir
        self.cfg       = cfg
        self.stride    = stride
        self.K_windows = K_windows
        self.rng       = np.random.default_rng(seed)

        self.action_normalizer = ActionNormalizer(cfg)
        self.v_max     = self.action_normalizer.v_max
        self.omega_max = self.action_normalizer.omega_max

        ds_cfg     = cfg.get("dataset", {})
        self.D_max = float(ds_cfg.get("D_max", 120.0))

        from scripts.data.gen_demos import compute_H
        self.H = compute_H(cfg)

        self.bev_builder = BEVBuilder(cfg)

        if map_ids is None:
            map_ids = sorted(
                d for d in os.listdir(maps_dir)
                if os.path.isdir(os.path.join(maps_dir, d))
                and d.startswith("map_")
            )
        self.map_ids = map_ids

        self._worlds: Dict[str, SquareWorld] = {}
        self._demos : Dict[str, dict]        = {}
        self._index : List[Tuple[str, int, int]] = []
        self._build_index()

    # ------------------------------------------------------------------
    # index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        rng = self.rng
        for map_id in self.map_ids:
            subdir    = os.path.join(self.maps_dir, map_id)
            map_json  = os.path.join(subdir, "map.json")
            demos_npz = os.path.join(subdir, "demos.npz")
            if not (os.path.isfile(map_json) and os.path.isfile(demos_npz)):
                continue

            world = SquareWorld.load(map_json)
            data  = np.load(demos_npz)

            self._worlds[map_id] = world
            self._demos[map_id]  = {
                "actions": data["demos_actions"],   # [N, T_max, 2]
                "states" : data["demos_states"],    # [N, T_max+1, 3]
                "lengths": data["demo_lengths"],    # [N]
            }

            lengths  = data["demo_lengths"]
            H        = self.H
            goal_xy  = world.goal_xy
            goal_eps = float(self.cfg.get("demo", {}).get("goal_eps_m", 0.5))

            for traj_id in range(len(lengths)):
                T      = int(lengths[traj_id])
                states = data["demos_states"][traj_id]

                # Exclude windows that start at or after the agent first
                # enters the goal region (those steps contain in-place
                # heading-alignment actions not seen during policy execution).
                dists     = np.linalg.norm(states[:T, :2] - goal_xy, axis=1)
                goal_hits = np.where(dists < goal_eps)[0]
                T_active  = int(goal_hits[0]) if len(goal_hits) > 0 else T

                all_starts = list(range(0, T_active - H + 1, self.stride))
                if not all_starts:
                    continue
                if len(all_starts) > self.K_windows:
                    chosen = rng.choice(
                        all_starts, size=self.K_windows, replace=False
                    ).tolist()
                else:
                    chosen = all_starts
                for t in chosen:
                    self._index.append((map_id, traj_id, t))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        map_id, traj_id, t_start = self._index[idx]

        world = self._worlds[map_id]
        demos = self._demos[map_id]
        H     = self.H

        pose_t = demos["states"][traj_id, t_start].astype(np.float32)  # (3,)
        goal_p = world.goal_pose.astype(np.float32)                     # (3,)

        bev_grid   = self.bev_builder.build(world, pose_t)              # uint8 [Hb, Wb]
        bev_tensor = torch.from_numpy(bev_grid)

        goal_vec = build_goal_vec(pose_t, goal_p, self.D_max)           # float32 (5,)

        raw_actions = demos["actions"][traj_id, t_start : t_start + H]  # (H, 2)
        action_seq  = self.action_normalizer.normalize_np(raw_actions)   # (H, 2)

        return {
            "bev_id"    : bev_tensor,
            "goal_vec"  : torch.from_numpy(goal_vec),
            "action_seq": torch.from_numpy(action_seq),
            "meta"      : {
                "map_id"   : map_id,
                "traj_id"  : traj_id,
                "t_start"  : t_start,
                "pose_t"   : pose_t,
                "goal_pose": goal_p,
            },
        }

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def bev_shape(self) -> Tuple[int, int]:
        """(H_cells, W_cells) of the BEV grid."""
        return self.bev_builder.H_cells, self.bev_builder.W_cells

    @property
    def action_scale(self) -> dict:
        """Scale factors used for normalisation."""
        n = self.action_normalizer
        return {"v_min": n.v_min, "v_max": n.v_max, "omega_max": n.omega_max}


# ---------------------------------------------------------------------------
# Self-test / sanity check
# ---------------------------------------------------------------------------

def _selftest(maps_dir: str, cfg: dict, n_samples: int = 5) -> None:
    import random

    print("=" * 60)
    print("FlowDataset self-test")
    print("=" * 60)

    ds = FlowDataset(maps_dir, cfg, stride=3, K_windows=60, seed=42)
    print(f"Total samples : {len(ds)}")
    print(f"H             : {ds.H}")
    print(f"BEV shape     : {ds.bev_shape}  (H_cells × W_cells)")
    print(f"Action scale  : v_max={ds.v_max}  omega_max={ds.omega_max}")
    print(f"D_max         : {ds.D_max}  (from cfg[dataset][D_max])")
    print()

    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
    all_pass = True

    for rank, idx in enumerate(indices):
        sample = ds[idx]
        gv     = sample["goal_vec"]
        act    = sample["action_seq"]
        bev    = sample["bev_id"]
        meta   = sample["meta"]

        a_sin, a_cos = gv[0].item(), gv[1].item()
        b_sin, b_cos = gv[2].item(), gv[3].item()
        d_norm       = gv[4].item()

        norm_a = a_sin**2 + a_cos**2
        norm_b = b_sin**2 + b_cos**2
        act_ok = bool((act.abs() <= 1.0 + 1e-5).all())
        d_ok   = 0.0 <= d_norm <= 1.0
        shape_ok = (act.shape == (ds.H, 2)) and (bev.shape == ds.bev_shape)

        ok = (abs(norm_a - 1.0) < 1e-4
              and abs(norm_b - 1.0) < 1e-4
              and act_ok and d_ok and shape_ok)
        all_pass = all_pass and ok

        print(f"  sample [{rank}]  map={meta['map_id']}  "
              f"traj={meta['traj_id']}  t={meta['t_start']}")
        print(f"    goal_vec  = [{a_sin:.3f}, {a_cos:.3f}, "
              f"{b_sin:.3f}, {b_cos:.3f}, {d_norm:.3f}]")
        print(f"    |alpha|²  = {norm_a:.6f}  (want 1.0)")
        print(f"    |beta|²   = {norm_b:.6f}  (want 1.0)")
        print(f"    d_norm    = {d_norm:.4f}  ∈ [0,1]: {d_ok}")
        print(f"    act shape = {tuple(act.shape)}  ∈ [-1,1]: {act_ok}")
        print(f"    bev shape = {tuple(bev.shape)}  "
              f"labels: {bev.unique().tolist()}")
        print(f"    {'PASS' if ok else 'FAIL <<<'}")
        print()

    print("=" * 60)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from project_config import POLICY_DIR as _POLICY_DIR, OUTPUTS_DIR as _OUTPUTS_DIR

    parser = argparse.ArgumentParser(
        description="Self-test for FlowDataset."
    )
    parser.add_argument("--maps_dir", default=str(_OUTPUTS_DIR / "maps"))
    parser.add_argument("--cfg",      default=str(_POLICY_DIR  / "configs/common.json"))
    parser.add_argument("--n",        type=int, default=5,
                        help="Number of random samples to check")
    args = parser.parse_args()

    from project_config import load_config
    cfg = load_config("common")
    _selftest(args.maps_dir, cfg, n_samples=args.n)

