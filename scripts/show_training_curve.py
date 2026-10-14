"""Usage: python3 scripts/show_training_curve.py <checkpoint_dir> [<checkpoint_dir> ...]
Prints one row per epoch from Lightning's CSV log and saves <dir>/training_curve.png"""
import sys, glob, os
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
for run in sys.argv[1:]:
    f = sorted(glob.glob(os.path.join(run, "csv", "lightning_logs", "version_*", "metrics.csv")))[-1]
    df = pd.read_csv(f)
    lr_col = [c for c in df.columns if c.startswith("lr")]
    cols = [c for c in ["train_loss", "val_loss", "pearson_corr_val"] + lr_col if c in df.columns]
    g = df.groupby("epoch")[cols].first()        # train and val metrics live on separate rows; merge per epoch
    best = g["val_loss"].idxmin()
    print(f"\n=== {run}  ({f})\nbest val_loss at epoch {best}")
    print(g.to_string(float_format=lambda x: f"{x:.4g}"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    ax[0].plot(g.index, g["train_loss"], label="train"); ax[0].plot(g.index, g["val_loss"], label="val (chr2L)")
    ax[0].axvline(best, c="grey", ls="--"); ax[0].set_title("loss (as trained, i.e. weighted for v4c)"); ax[0].legend()
    ax[1].plot(g.index, g["pearson_corr_val"]); ax[1].axvline(best, c="grey", ls="--"); ax[1].set_title("val insulation Pearson")
    for a in ax: a.set_xlabel("epoch")
    fig.savefig(os.path.join(run, "training_curve.png"), dpi=120)
