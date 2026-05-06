"""
train_flow_policy.py
====================
Train a flow-matching model on the unicycle action-sequence dataset.

The model (ActionFlowNet) and its constants live in flow_policy.py.
This script only contains the dataset helper, the training loop, and the CLI.

Usage
-----
    python -m scripts.train.train_flow_policy
    python -m scripts.train.train_flow_policy --num_steps 10000 --batch_size 64

Output
------
    outputs/runs/<run_name>/
        final_model.pt
        loss_curve.png
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow imports from parent and current directories

# Model, ODE wrapper, and shared constants come from flow_policy
from models.flow_policy import ActionFlowNet, ActionFlowODE, ACT_DIM
from common.utils import ActionNormalizer
from project_config import POLICY_DIR as _POLICY_DIR, OUTPUTS_DIR as _OUTPUTS_DIR


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ActionSeqDataset(torch.utils.data.Dataset):
    """
    Wraps a numpy array of action sequences as a torch Dataset.

    Actions are normalised to [-1, 1] per dimension before being stored,
    so the flow model always operates in a standardised space.
    """

    def __init__(self, actions: np.ndarray, cfg: dict):
        # actions: [N, H, 2]  raw values
        norm = ActionNormalizer(cfg)
        data = torch.from_numpy(actions.astype(np.float32))   # (N, H, 2)
        data = norm.normalize(data)                            # → [-1, 1]
        N, _H, A = data.shape
        self.data = data.reshape(N, _H * A)                   # (N, H*2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    num_steps: int = 8000,
    batch_size: int = 128,
    lr: float = 3e-4,
    warmup_steps: int = 500,
    num_layers: int = 4,
    dim: int = 128,
    heads: int = 4,
    grad_clip: float = 1.0,
    ckpt_every: int = 2000,
    run_name: str = None,
    eps: float = 1e-3,
    device: str = None,
    n_train: int = None,
) -> str:
    """Train the flow model and return the path to the saved checkpoint."""
    from project_config import load_config, POLICY_DIR as _POLICY_DIR
    cfg = load_config("common", "train")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Using device: {dev}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"Loading demos from {data_path} …")
    npz = np.load(data_path)
    actions = npz["demos_actions"]   # [N, H, 2]

    # Optionally subsample a fixed number of trajectories
    if n_train is not None:
        if n_train > len(actions):
            raise ValueError(f"n_train={n_train} exceeds dataset size {len(actions)}")
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(actions), size=n_train, replace=False)
        actions = actions[idx]
        print(f"  Subsampled {n_train}/{npz['demos_actions'].shape[0]} trajectories")
    print(f"  Dataset: {actions.shape}  (N={actions.shape[0]}, H={actions.shape[1]})")

    dataset = ActionSeqDataset(actions, cfg)
    eff_batch = min(batch_size, len(dataset))
    drop_last = eff_batch == batch_size
    if eff_batch < batch_size:
        print(f"  Note: batch_size shrunk to {eff_batch} (dataset size={len(dataset)})")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=eff_batch, shuffle=True,
        drop_last=drop_last, pin_memory=True
    )
    loader_iter = iter(loader)

    def next_batch():
        nonlocal loader_iter
        try:
            return next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            return next(loader_iter)

    # ------------------------------------------------------------------
    # Model  (h is derived from cfg via from_config)
    # ------------------------------------------------------------------
    model = ActionFlowNet.from_config(
        cfg, num_layers=num_layers, dim=dim, heads=heads
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: H={model.H}  dim={dim}  layers={num_layers}  params={n_params:,}")

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ------------------------------------------------------------------
    # Run directory
    # ------------------------------------------------------------------
    import uuid
    if run_name is None:
        run_name = f"toy-policy-{str(uuid.uuid4())[:8]}"
    run_dir = str(_OUTPUTS_DIR / "runs" / run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # ------------------------------------------------------------------
    # Training loop  (linear flow matching: x_t = t*x1 + (1-t)*x0)
    # ------------------------------------------------------------------
    from tqdm import tqdm

    model.train()
    losses = []

    # Warmup: start lr at 0
    for pg in opt.param_groups:
        pg["lr"] = 0.0

    pbar = tqdm(range(num_steps))
    for step in pbar:
        # LR schedule: linear warm-up then constant
        if warmup_steps > 0 and step < warmup_steps:
            cur_lr = lr * (step + 1) / warmup_steps
        else:
            cur_lr = lr
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        # Sample data (x1) and noise (x0)
        x1 = next_batch().to(dev)                     # (b, H*2)
        x0 = torch.randn_like(x1)                     # N(0,I)

        # Sample time
        t  = eps + (1.0 - eps) * torch.rand(x1.shape[0], device=dev)  # (b,)

        # Interpolate: x_t = t*x1 + (1-t)*x0  (linear path)
        t_bc = t.unsqueeze(-1)                         # (b, 1)
        xt   = t_bc * x1 + (1.0 - t_bc) * x0         # (b, H*2)

        # Target vector field for linear path: u_t = x1 - x0
        u_ref  = x1 - x0                              # (b, H*2)
        u_pred = model(xt, t)                         # (b, H*2)

        loss = torch.mean((u_pred - u_ref) ** 2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        losses.append(float(loss.detach()))
        pbar.set_description(
            f"step={step} lr={cur_lr:.1e} loss={loss.item():.4f}"
        )

        if ckpt_every > 0 and step > 0 and step % ckpt_every == 0:
            ckpt_path = os.path.join(run_dir, f"step_{step:06d}_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"\n  Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(run_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved: {final_path}")

    # Loss curve
    plt.figure(figsize=(8, 4))
    window   = max(1, len(losses) // 100)
    smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
    plt.plot(smoothed, linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("FM Loss (MSE)")
    plt.title("Flow Matching Training Loss – Toy Action Policy")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=120)
    plt.close()

    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Train flow model for toy unicycle action sequences."
    )
    p.add_argument("--data",         type=str,
                   default=str(_OUTPUTS_DIR / "data" / "demos.npz"))
    p.add_argument("--num_steps",    type=int,   default=8000)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int,   default=500)
    p.add_argument("--num_layers",   type=int,   default=4)
    p.add_argument("--dim",          type=int,   default=128)
    p.add_argument("--heads",        type=int,   default=4)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--ckpt_every",   type=int,   default=2000)
    p.add_argument("--run_name",     type=str,   default=None)
    p.add_argument("--device",       type=str,   default=None)
    p.add_argument("--n_train",      type=int,   default=None,
                   help="Randomly subsample this many trajectories for training.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        data_path    = args.data,
        num_steps    = args.num_steps,
        batch_size   = args.batch_size,
        lr           = args.lr,
        warmup_steps = args.warmup_steps,
        num_layers   = args.num_layers,
        dim          = args.dim,
        heads        = args.heads,
        grad_clip    = args.grad_clip,
        ckpt_every   = args.ckpt_every,
        run_name     = args.run_name,
        device       = args.device,
        n_train      = args.n_train,
    )
