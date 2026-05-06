"""
gen_maps.py
===========
Generate a collection of SquareWorld instances covering the parameter space and
optionally produce demonstration trajectories for each map.

Map naming convention
---------------------
Each map lives in its own subdirectory under ``maps/``:

    maps/
      map_s{scale}_n{n_obs}_r{seed}/
        map.json          ← SquareWorld geometry
        map.png           ← visualisation (map only)
        demos.npz         ← demo trajectories (if --gen_demos)
        demos_preview.png ← demo visualisation (if --gen_demos)

The directory name encodes the key parameters so the map's characteristics
are visible from the file path alone.

Parameter grid
--------------
The grid is defined by MAP_SCALES × N_OBSTACLES × SEEDS.  Every combination
is generated, giving full coverage of the parameter space.

Usage
-----
    # Generate maps only
    python -m scripts.data.gen_maps

    # Generate maps AND demos for each map
    python -m scripts.data.gen_maps --gen_demos

    # Custom grid or output directory
    python -m scripts.data.gen_maps --scales 2 5 8 --n_obs 1 2 4 --seeds 0 1 2 --gen_demos
    python -m scripts.data.gen_maps --out_dir outputs/maps --gen_demos --n_demos 200
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import numpy as np

from project_config import POLICY_DIR as _POLICY_DIR, OUTPUTS_DIR as _OUTPUTS_DIR
from env.square_world import SquareWorld
from scripts.data.gen_demos import load_config, generate_demos, compute_H, compute_T_max

# ---------------------------------------------------------------------------
# Default parameter grid
# ---------------------------------------------------------------------------

# scale values:  small (2×), medium (5×), large (8×)  → L = 20 / 50 / 80 m
DEFAULT_SCALES     = [2, 5, 8]

# obstacle counts: sparse, moderate, dense
DEFAULT_N_OBS      = [1, 3, 5]

# random seeds: three independent draws per (scale, n_obs) combination
DEFAULT_SEEDS      = [0, 1, 2]

_DEFAULT_CONFIG    = str(_POLICY_DIR / "configs" / "common.json")
_DEFAULT_OUT_DIR   = str(_OUTPUTS_DIR / "maps")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_dir(out_dir: str, scale: int, n_obs: int, seed: int) -> str:
    """Return the subdirectory path for one map."""
    return os.path.join(out_dir, f"map_s{scale}_n{n_obs}_r{seed}")


def _build_cfg(base_cfg: dict, scale: int, n_obs: int, seed: int) -> dict:
    """Deep-copy base_cfg and override the varying map parameters."""
    cfg = copy.deepcopy(base_cfg)
    cfg["map"]["scale"] = float(scale)
    cfg["map"]["n_obstacles"] = int(n_obs)
    cfg["map"]["seed"] = int(seed)
    # Keep scale within declared bounds
    cfg["map"]["scale_min"] = min(cfg["map"]["scale_min"], float(scale))
    cfg["map"]["scale_max"] = max(cfg["map"]["scale_max"], float(scale))
    return cfg


def generate_one_map(
    cfg    : dict,
    out_dir: str,
    scale  : int,
    n_obs  : int,
    seed   : int,
) -> tuple[SquareWorld, str]:
    """
    Randomise and save one SquareWorld.

    Returns
    -------
    (world, map_subdir)
    """
    subdir = _map_dir(out_dir, scale, n_obs, seed)
    os.makedirs(subdir, exist_ok=True)

    world = SquareWorld.from_config(cfg["map"])
    world.randomise()

    # Save JSON
    json_path = os.path.join(subdir, "map.json")
    world.save(json_path)

    # Save PNG
    png_path = os.path.join(subdir, "map.png")
    world.visualise(
        save_path=png_path,
        title=(
            f"scale={scale}×  L={world.L:.0f} m  "
            f"n_obs={n_obs}  seed={seed}"
        ),
    )

    return world, subdir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    base_cfg  : dict,
    scales    : List[int]  | None = None,
    n_obs_list: List[int]  | None = None,
    seeds     : List[int]  | None = None,
    out_dir   : str               = _DEFAULT_OUT_DIR,
    gen_demos : bool              = False,
    n_demos   : int | None        = None,
) -> None:
    scales     = scales     if scales     is not None else DEFAULT_SCALES
    n_obs_list = n_obs_list if n_obs_list is not None else DEFAULT_N_OBS
    seeds      = seeds      if seeds      is not None else DEFAULT_SEEDS
    combos = list(itertools.product(scales, n_obs_list, seeds))
    print(f"Generating {len(combos)} maps  "
          f"(scales={scales}, n_obs={n_obs_list}, seeds={seeds})")
    print(f"Output directory: {out_dir}")
    if gen_demos:
        print(f"Demo generation: ENABLED  (n_demos={n_demos or base_cfg['demo']['n_demos']})")
    print()

    results = []   # (scale, n_obs, seed, success, n_demos_collected)

    for i, (scale, n_obs, seed) in enumerate(combos):
        tag = f"map_s{scale}_n{n_obs}_r{seed}"
        print(f"[{i+1:2d}/{len(combos)}]  {tag}")

        cfg = _build_cfg(base_cfg, scale, n_obs, seed)

        # --- generate map ---
        try:
            world, subdir = generate_one_map(cfg, out_dir, scale, n_obs, seed)
            H     = compute_H(cfg)
            T_max = compute_T_max(cfg, world)
            print(f"         L={world.L:.0f} m  "
                  f"H={H} ({H*cfg['simulator']['dt']:.1f} s)  "
                  f"T_max={T_max} ({T_max*cfg['simulator']['dt']:.1f} s)  "
                  f"→ {subdir}")
        except RuntimeError as e:
            print(f"         FAILED to generate map: {e}")
            results.append((scale, n_obs, seed, False, 0))
            continue

        # --- generate demos (optional) ---
        if gen_demos:
            try:
                data = generate_demos(
                    cfg, world,
                    n_demos=n_demos,
                    seed=seed,
                )
                lengths = data["demo_lengths"]
                n_windows = int(sum(max(0, T - H + 1) for T in lengths))

                # Save npz
                npz_path = os.path.join(subdir, "demos.npz")
                np.savez(npz_path, **data)

                # Save demo preview
                states   = data["demos_states"]
                vis_idx  = np.linspace(0, len(lengths)-1, min(40, len(lengths)), dtype=int)
                png_path = os.path.join(subdir, "demos_preview.png")
                world.visualise(
                    trajectories=[states[j, :lengths[j]+1, :2] for j in vis_idx],
                    save_path=png_path,
                    title=(
                        f"{len(lengths)} demos  scale={scale}×  n_obs={n_obs}  seed={seed}"
                        f"  |  H={H}  T∈[{lengths.min()},{lengths.max()}]"
                        f"  |  {n_windows} windows"
                    ),
                )
                print(f"         demos: N={len(lengths)}  "
                      f"T∈[{lengths.min()},{lengths.max()}]  "
                      f"windows={n_windows}  → {npz_path}")
                results.append((scale, n_obs, seed, True, len(lengths)))

            except RuntimeError as e:
                print(f"         FAILED to generate demos: {e}")
                results.append((scale, n_obs, seed, True, 0))
        else:
            results.append((scale, n_obs, seed, True, 0))

        print()

    # --- summary ---
    print("=" * 60)
    print(f"Summary: {len(combos)} maps")
    map_ok    = sum(1 for r in results if r[3])
    map_fail  = len(combos) - map_ok
    print(f"  Maps generated:  {map_ok}/{len(combos)}")
    if map_fail:
        print(f"  Maps failed:     {map_fail}")
    if gen_demos:
        demo_ok   = sum(1 for r in results if r[4] > 0)
        demo_fail = sum(1 for r in results if r[3] and r[4] == 0)
        print(f"  Demos generated: {demo_ok}/{map_ok}")
        if demo_fail:
            print(f"  Demos failed:    {demo_fail}")
    print("=" * 60)

    # Save summary JSON
    summary = {
        "maps": [
            {
                "scale"    : r[0],
                "n_obs"    : r[1],
                "seed"     : r[2],
                "map_ok"   : r[3],
                "n_demos"  : r[4],
                "subdir"   : _map_dir(out_dir, r[0], r[1], r[2]),
            }
            for r in results
        ]
    }
    summary_path = os.path.join(out_dir, "summary.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate a grid of SquareWorld instances with optional demos."
    )
    p.add_argument("--out_dir",   type=str,   default=_DEFAULT_OUT_DIR,
                   help="Root output directory (default: outputs/maps)")
    p.add_argument("--scales",    type=int,   nargs="+", default=DEFAULT_SCALES,
                   help="Map scale multipliers to include")
    p.add_argument("--n_obs",     type=int,   nargs="+", default=DEFAULT_N_OBS,
                   help="Obstacle counts to include")
    p.add_argument("--seeds",     type=int,   nargs="+", default=DEFAULT_SEEDS,
                   help="Random seeds to include")
    p.add_argument("--gen_demos", action="store_true",
                   help="Also generate demo trajectories for each map")
    p.add_argument("--n_demos",   type=int,   default=None,
                   help="Override n_demos per map (default: from config)")
    return p.parse_args()


if __name__ == "__main__":
    args     = _parse_args()
    base_cfg = load_config()
    main(
        base_cfg   = base_cfg,
        scales     = args.scales,
        n_obs_list = args.n_obs,
        seeds      = args.seeds,
        out_dir    = args.out_dir,
        gen_demos  = args.gen_demos,
        n_demos    = args.n_demos,
    )





