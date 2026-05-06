"""
gen_bev_gifs.py
===============
For every map under outputs/maps/, pick one random demo trajectory and
render a BEV evolution GIF saved into the same map subfolder.

Usage:
    python -m scripts.data.gen_bev_gifs [--maps_dir outputs/maps] [--cfg configs/common.json]
                                   [--fps 10] [--stride 5] [--seed 0]
"""
import argparse
import os
import numpy as np

from project_config import POLICY_DIR as _POLICY_DIR, OUTPUTS_DIR as _OUTPUTS_DIR
from env.square_world import SquareWorld
from perception.bev_builder import BEVBuilder, visualise_bev_gif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps_dir", default=str(_OUTPUTS_DIR / "maps"))
    parser.add_argument("--cfg",      default=str(_POLICY_DIR  / "configs/common.json"))
    parser.add_argument("--fps",      type=int, default=10)
    parser.add_argument("--stride",   type=int, default=5)
    parser.add_argument("--seed",     type=int, default=0)
    args = parser.parse_args()

    from project_config import load_config
    cfg = load_config("common")
    bev = BEVBuilder(cfg)
    rng = np.random.default_rng(args.seed)

    map_dirs = sorted(
        d for d in os.listdir(args.maps_dir)
        if os.path.isdir(os.path.join(args.maps_dir, d)) and d.startswith("map_")
    )

    print(f"Found {len(map_dirs)} map directories.")
    print(f"BEV grid: {bev.H_cells}×{bev.W_cells} cells  "
          f"(D={bev.D_bev}m, W={bev.W_bev}m, res={bev.res}m/cell)\n")

    ok = 0
    for name in map_dirs:
        subdir    = os.path.join(args.maps_dir, name)
        map_json  = os.path.join(subdir, "map.json")
        demos_npz = os.path.join(subdir, "demos.npz")
        out_gif   = os.path.join(subdir, "bev_rollout.gif")

        if not (os.path.isfile(map_json) and os.path.isfile(demos_npz)):
            print(f"  [{name}]  SKIP – missing map.json or demos.npz")
            continue

        world = SquareWorld.load(map_json)
        data    = np.load(demos_npz)
        lengths = data["demo_lengths"]
        states  = data["demos_states"]

        # pick a random demo index
        idx = int(rng.integers(0, len(lengths)))
        T   = int(lengths[idx])
        traj_states = states[idx, :T + 1]

        print(f"  [{name}]  demo #{idx}  T={T} steps → {out_gif}")
        visualise_bev_gif(
            world     = world,
            states      = traj_states,
            bev_builder = bev,
            out_path    = out_gif,
            fps         = args.fps,
            stride      = args.stride,
        )
        ok += 1

    print(f"\nDone: {ok}/{len(map_dirs)} GIFs generated.")


if __name__ == "__main__":
    main()

