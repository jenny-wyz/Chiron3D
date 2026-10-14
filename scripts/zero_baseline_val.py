"""Val loss a model would get by predicting 0 everywhere (= distance decay only) on chr2L, in the v4b/v4c target.
Usage: python3 scripts/zero_baseline_val.py <mcool::resolutions/800> <matrix_scale> [weights.npy ...]
Without weight files: plain masked MSE (comparable to v4b's val_loss). With: the v4c weighted loss for each file."""
import sys, cooler, numpy as np, pandas as pd
mcool, scale, wfiles = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
RES = 800
c = cooler.Cooler(mcool)
prof = np.load("data/expected_log_800_bal.npy")
z = np.load("data/bad_bins_800_bal.npz", allow_pickle=True)
bad = {(ch, s): b for ch, s, b in zip(z["chrom"].astype(str), z["start"], z["bad"])}
bed = pd.read_csv("data/windows_dm6_C523200_f1024.bed", sep="\t", names=["chr", "start", "end"])
bed = bed[bed.chr == "chr2L"]
N = (bed.end.iloc[0] - bed.start.iloc[0]) // RES
d = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
W = {"plain (v4b)": np.ones((N, N))}
for f in wfiles:
    w = np.load(f); W[f] = w[d]
num = {k: 0.0 for k in W}; den = {k: 0.0 for k in W}
for r in bed.itertuples():
    t = np.log(np.nan_to_num(c.matrix(balance=True).fetch(f"{r.chr}:{r.start}-{r.end}")) * scale + 1.0) - prof[d]
    good = ~np.array([bad[(r.chr, s)] for s in range(r.start, r.end, RES)])
    m = good[:, None] & good[None, :]
    for k, w in W.items():
        num[k] += np.sum(w * m * t ** 2); den[k] += np.sum(w * m)
print(f"{len(bed)} chr2L windows")
for k in W:
    print(f"predict-zero val loss, {k}: {num[k] / den[k]:.4f}")
