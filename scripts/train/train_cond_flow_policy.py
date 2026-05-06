"""
train_cond_flow_policy.py
====================
Training script for the conditional flow policy with CFG dropout.

All hyperparameters default to cfg["train"] / cfg["model"] in configs/common.json.
CLI flags only *override* the config values – you rarely need them.

Rectified flow objective:
    x0 ~ N(0,I),   x1 = normalised expert action sequence
    x_t = (1-t)*x0 + t*x1
    target = x1 - x0
    loss   = ||model(x_t, t, bev_id, goal_vec, global_is_null) - target||^2

Usage::

    # use everything from config
    python -m scripts.train.train_cond_flow_policy

    # override a few values at the command line
    python -m scripts.train.train_cond_flow_policy --epochs 200 --batch 128 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


from datasets.flow_dataset import FlowDataset
from models.cond_flow_policy import CondActionFlowNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    bev_id     = torch.stack([b["bev_id"].long() for b in batch])
    goal_vec   = torch.stack([b["goal_vec"]       for b in batch])
    action_seq = torch.stack([b["action_seq"]     for b in batch])
    return bev_id, goal_vec, action_seq


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    from project_config import load_config
    cfg = load_config("common", "train")

    # ── Merge config defaults with CLI overrides ────────────────────────
    # CLI wins when the user explicitly provides a flag; otherwise the
    # value already in `args` came from the config default set in _parse().
    tcfg = cfg["train"]
    mcfg = cfg["model"]

    epochs     = args.epochs
    batch      = args.batch
    lr         = args.lr
    weight_decay = args.weight_decay
    grad_clip  = args.grad_clip
    p_drop     = args.p_drop
    seed       = args.seed
    log_every  = args.log_every
    save_every = args.save_every
    maps_dir   = args.maps_dir
    out_dir    = Path(args.out_dir)

    dim        = args.dim
    num_layers = args.layers
    heads      = args.heads

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ── Dataset ────────────────────────────────────────────────────────
    ds = FlowDataset(
        maps_dir  = maps_dir,
        cfg       = cfg,
        stride    = cfg["dataset"]["stride"],
        K_windows = cfg["dataset"]["K_windows"],
        seed      = seed,
    )
    val_frac = tcfg.get("val_frac", 0.05)
    val_n  = max(1, int(val_frac * len(ds)))
    trn_n  = len(ds) - val_n
    trn_ds, val_ds = random_split(
        ds, [trn_n, val_n],
        generator=torch.Generator().manual_seed(seed),
    )
    trn_dl = DataLoader(trn_ds, batch_size=batch, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False,
                        collate_fn=collate_fn, num_workers=0, pin_memory=False)
    print(f"Dataset : {len(trn_ds)} train  |  {len(val_ds)} val"
          f"  |  H={ds.H}  data_dim={ds.H*2}")

    # ── Model ──────────────────────────────────────────────────────────
    model = CondActionFlowNet.from_config(
        cfg, d=dim, num_layers=num_layers, heads=heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model   : {n_params:,} params  "
          f"(dim={dim}, layers={num_layers}, heads={heads})")
    print(f"Device  : {device}")
    print(f"Training: epochs={epochs}  batch={batch}  lr={lr:.1e}"
          f"  p_drop={p_drop}  grad_clip={grad_clip}")

    # ── Optimiser ──────────────────────────────────────────────────────
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_eta_min = lr * tcfg.get("lr_eta_min_frac", 0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * len(trn_dl), eta_min=lr_eta_min,
    )

    # ── Output dir ─────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    # save effective config (with CLI overrides) for reproducibility
    effective = {
        "cfg_file"  : args.cfg,
        "maps_dir"  : maps_dir,
        "out_dir"   : str(out_dir),
        "model"     : {"dim": dim, "num_layers": num_layers, "heads": heads},
        "train"     : {
            "epochs": epochs, "batch": batch, "lr": lr,
            "weight_decay": weight_decay, "grad_clip": grad_clip,
            "p_drop": p_drop, "seed": seed,
            "log_every": log_every, "save_every": save_every,
            "val_frac": val_frac, "lr_eta_min_frac": tcfg.get("lr_eta_min_frac", 0.01),
        },
    }
    json.dump(effective, open(out_dir / "effective_config.json", "w"), indent=2)

    # ── Training ───────────────────────────────────────────────────────
    best_val = float("inf")
    train_losses, val_losses = [], []
    t0 = time.time()

    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs", unit="ep",
                     dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        ep_loss = 0.0

        batch_bar = tqdm(trn_dl, desc=f"  Train {epoch:4d}/{epochs}",
                         leave=False, unit="batch", dynamic_ncols=True)
        for bev_id, goal_vec, action_seq in batch_bar:
            bev_id     = bev_id.to(device)
            goal_vec   = goal_vec.to(device)
            action_seq = action_seq.to(device)
            B = bev_id.shape[0]

            x1     = action_seq.view(B, -1)
            x0     = torch.randn_like(x1)
            t      = torch.rand(B, device=device)
            t_bc   = t.view(B, 1)
            x_t    = (1.0 - t_bc) * x0 + t_bc * x1
            target = x1 - x0

            global_is_null = torch.rand(B, device=device) < p_drop
            v_pred = model(x_t, t, bev_id, goal_vec, global_is_null)

            loss = nn.functional.mse_loss(v_pred, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()

            ep_loss += loss.item() * B
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        ep_loss /= len(trn_ds)
        train_losses.append(ep_loss)

        # ── Validation ─────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for bev_id, goal_vec, action_seq in tqdm(
                    val_dl, desc=f"  Val   {epoch:4d}/{epochs}",
                    leave=False, unit="batch", dynamic_ncols=True):
                bev_id     = bev_id.to(device)
                goal_vec   = goal_vec.to(device)
                action_seq = action_seq.to(device)
                B = bev_id.shape[0]
                x1     = action_seq.view(B, -1)
                x0     = torch.randn_like(x1)
                t      = torch.rand(B, device=device)
                t_bc   = t.view(B, 1)
                x_t    = (1.0 - t_bc) * x0 + t_bc * x1
                target = x1 - x0
                v_pred = model(x_t, t, bev_id, goal_vec,
                               global_is_null=torch.zeros(B, dtype=torch.bool, device=device))
                v_loss += nn.functional.mse_loss(v_pred, target).item() * B
        v_loss /= len(val_ds)
        val_losses.append(v_loss)

        # update epoch-level progress bar suffix
        elapsed = time.time() - t0
        epoch_bar.set_postfix(
            train=f"{ep_loss:.4f}",
            val=f"{v_loss:.4f}",
            lr=f"{sched.get_last_lr()[0]:.1e}",
            elapsed=f"{elapsed/60:.1f}min",
        )

        if epoch % log_every == 0 or epoch == epochs:
            tqdm.write(
                f"Epoch {epoch:4d}/{epochs}  "
                f"train={ep_loss:.5f}  val={v_loss:.5f}  "
                f"lr={sched.get_last_lr()[0]:.2e}  "
                f"elapsed={elapsed/60:.1f}min"
            )

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        if epoch % save_every == 0 or epoch == epochs:
            torch.save(model.state_dict(), out_dir / f"epoch_{epoch:04d}_model.pt")

    # ── Final stats ────────────────────────────────────────────────────
    total_time = time.time() - t0
    peak_mem_gb = (torch.cuda.max_memory_allocated() / 1024**3
                   if torch.cuda.is_available() else 0.0)

    np.save(out_dir / "train_losses.npy", np.array(train_losses))
    np.save(out_dir / "val_losses.npy",   np.array(val_losses))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses,   label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss")
    ax.set_title("Conditional flow policy – loss curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    summary = {
        "best_val_loss"  : best_val,
        "total_time_min" : round(total_time / 60, 2),
        "peak_gpu_mem_gb": round(peak_mem_gb, 3),
        "n_params"       : n_params,
    }
    json.dump(summary, open(out_dir / "train_summary.json", "w"), indent=2)

    print(f"\nBest val loss    : {best_val:.6f}")
    print(f"Total train time : {total_time/60:.1f} min  ({total_time:.0f} s)")
    print(f"Peak GPU memory  : {peak_mem_gb:.3f} GB")
    print(f"Outputs saved to : {out_dir}")


# ---------------------------------------------------------------------------
# CLI  – all defaults come from config; flags only override
# ---------------------------------------------------------------------------

def _parse():
    # Load config first to use as defaults
    import sys
    from project_config import OUTPUTS_DIR as _OUTPUTS_DIR, load_config
    try:
        _cfg = load_config("common", "train")
        _t   = _cfg.get("train", {})
        _m   = _cfg.get("model", {})
    except Exception:
        _t, _m = {}, {}

    p = argparse.ArgumentParser(
        description="Train conditional flow policy. "
                    "Defaults come from configs/common.json + configs/train.json."
    )
    p.add_argument("--maps_dir",   default=_t.get("maps_dir",   str(_OUTPUTS_DIR / "maps")))
    p.add_argument("--out_dir",    default=_t.get("out_dir",    str(_OUTPUTS_DIR / "runs" / "cond_flow")))
    # train hyperparams
    p.add_argument("--epochs",     type=int,   default=_t.get("epochs",     500))
    p.add_argument("--batch",      type=int,   default=_t.get("batch",      256))
    p.add_argument("--lr",         type=float, default=_t.get("lr",         3e-4))
    p.add_argument("--weight_decay",type=float,default=_t.get("weight_decay",1e-4))
    p.add_argument("--grad_clip",  type=float, default=_t.get("grad_clip",   1.0))
    p.add_argument("--p_drop",     type=float, default=_t.get("p_drop",      0.2))
    p.add_argument("--seed",       type=int,   default=_t.get("seed",         42))
    p.add_argument("--log_every",  type=int,   default=_t.get("log_every",    20))
    p.add_argument("--save_every", type=int,   default=_t.get("save_every",  100))
    # model hyperparams
    p.add_argument("--dim",        type=int,   default=_m.get("dim",        128))
    p.add_argument("--layers",     type=int,   default=_m.get("num_layers",   4))
    p.add_argument("--heads",      type=int,   default=_m.get("heads",        4))
    return p.parse_args()


if __name__ == "__main__":
    train(_parse())

