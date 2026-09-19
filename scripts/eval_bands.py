import numpy as np, glob, sys, os
dump = sys.argv[1]
files = sorted(glob.glob(os.path.join(dump, "matrices_*.npz")))
P = np.concatenate([np.load(f)["pred"].astype(np.float32) for f in files])
O = np.concatenate([np.load(f)["obs"].astype(np.float32)  for f in files])
N = O.shape[-1]
d = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
prof = np.array([O[:, d == k].mean() for k in range(N)])
print(f"{dump}: {len(P)} windows, {N} bins")
print(f"{'band':>14} {'MSE':>8} {'decay':>8} {'skill':>8}")
for lo, hi in [(1, 10), (10, 50), (50, 150), (150, N)]:
    m = (d >= lo) & (d < hi)
    mse  = ((P[:, m] - O[:, m]) ** 2).mean()
    base = ((prof[d][m] - O[:, m]) ** 2).mean()
    print(f"{lo*800//1000:>5}-{hi*800//1000:<5}kb {mse:8.4f} {base:8.4f} {1-mse/base:+8.3f}")


# python3 scripts/eval_bands.py dump_dcnn_v2
# python3 scripts/eval_bands.py dump_borzoi