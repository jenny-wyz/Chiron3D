#!/usr/bin/env python3
"""Print per-epoch train loss, val loss and val insulation r for several runs side by side,
from Lightning's CSV logs. Shows where each run peaked and how fast it overfits.

Usage (from the repo root):
    python3 scripts/curves.py checkpoints_dcnn_v4c_s80 checkpoints_dcnn_v6a_max checkpoints_dcnn_v6a_meanmax
"""
import glob, os, sys
import pandas as pd

for run in sys.argv[1:]:
    files = sorted(glob.glob(os.path.join(run, "csv", "**", "metrics.csv"), recursive=True))
    if not files:
        print(f"{run}: no metrics.csv found"); continue
    df = pd.concat([pd.read_csv(f) for f in files])
    lr_cols = [c for c in df.columns if c.startswith("lr-")][:1]
    cols = [c for c in ("train_loss", "val_loss", "pearson_corr_val") if c in df.columns] + lr_cols
    last_valid = lambda s: s.dropna().iloc[-1] if s.notna().any() else float("nan")
    g = df.groupby("epoch")[cols].agg(last_valid)
    best = g["val_loss"].idxmin() if "val_loss" in g else None
    tail = g["val_loss"].tail(20)
    print(f"\n=== {run}  ({len(g)} epochs, best val_loss {g['val_loss'].min():.4f} at epoch {best}; "
          f"last-20-epoch val_loss median {tail.median():.4f}, min {tail.min():.4f}, max {tail.max():.4f}) ===")
    print(f"{'epoch':>5} " + " ".join(f"{c:>16}" for c in cols) + "   train-val gap")
    for ep, row in g.iterrows():
        gap = row["train_loss"] - row["val_loss"] if {"train_loss", "val_loss"} <= set(cols) else float("nan")
        flag = "  <- best" if ep == best else ""
        print(f"{int(ep):>5} " + " ".join(f"{row[c]:16.4f}" for c in cols) + f" {gap:+14.4f}{flag}")