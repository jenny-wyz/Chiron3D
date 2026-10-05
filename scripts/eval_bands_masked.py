"""Like eval_bands.py, but ignores bad-bin pixels when the dump carries a 'mask' key (new balanced dumps).
Falls back to all pixels for old dumps.   Usage: python3 scripts/eval_bands_masked.py <dump_dir>"""
import numpy as np, glob, sys, os
dump = sys.argv[1]
files = sorted(glob.glob(os.path.join(dump, "matrices_*.npz")))
P = np.concatenate([np.load(f)["pred"].astype(np.float32) for f in files])
O = np.concatenate([np.load(f)["obs"].astype(np.float32) for f in files])
K = np.concatenate([np.load(f)["mask"] if "mask" in np.load(f).files else np.ones(np.load(f)["obs"].shape, bool) for f in files])
N = O.shape[-1]
d = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
prof = np.array([O[:, d == k][K[:, d == k]].mean() for k in range(N)])
print(f"{dump}: {len(P)} windows, {N} bins, {100*(~K).mean():.1f}% pixels masked")
print(f"{'band':>14} {'MSE':>8} {'decay':>8} {'skill':>8}")
for lo, hi in [(1, 10), (10, 50), (50, 150), (150, N)]:
    m = (d >= lo) & (d < hi)
    sel = K[:, m]
    mse = ((P[:, m] - O[:, m]) ** 2)[sel].mean()
    base = ((np.broadcast_to(prof[d][m], O[:, m].shape) - O[:, m]) ** 2)[sel].mean()
    print(f"{lo*800//1000:>5}-{hi*800//1000:<5}kb {mse:8.4f} {base:8.4f} {1-mse/base:+8.3f}")
