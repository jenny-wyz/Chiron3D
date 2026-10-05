"""Balanced-data versions of expected_log_800.npy and bad_bins_800.npz.
Usage: python3 scripts/make_expected_balanced.py <cool_path> <matrix_scale> [bed]
  cool_path     e.g. /path/larval_brain_merge.mcool::resolutions/800   (must have a 'weight' column)
  matrix_scale  the same number you pass to --matrix-scale (1/psi, e.g. 2621.2)
Writes data/expected_log_800_bal.npy (mean of log(balanced*scale+1) per diagonal, training chroms only)
and    data/bad_bins_800_bal.npz   (bad = bins ICE could not balance, i.e. weight is NaN)."""
import sys, cooler, numpy as np, pandas as pd
cool_path, scale = sys.argv[1], float(sys.argv[2])
bed_path = sys.argv[3] if len(sys.argv) > 3 else "data/windows_dm6_C523200_f1024.bed"
c = cooler.Cooler(cool_path)
assert "weight" in c.bins().columns, "no 'weight' column: run `cooler balance` first"
bed = pd.read_csv(bed_path, sep="\t", names=["chr", "start", "end"])
bed = bed[~bed.chr.isin(["chr2L", "chrX"])]                     # training chromosomes only
M = np.stack([np.log(np.nan_to_num(c.matrix(balance=True).fetch(f"{r.chr}:{r.start}-{r.end}"), nan=0.0) * scale + 1.0)
              for r in bed.itertuples()])
N = M.shape[-1]
d = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
w = c.bins()[:]
bad = ~np.isfinite(w["weight"].values)
# exclude bad bins from the expected profile so masked zeros do not drag the mean down
good_rows = []
for r in bed.itertuples():
    sel = (w.chrom == r.chr) & (w.start >= r.start) & (w.start < r.end)
    good_rows.append(~bad[sel.values])
G = np.stack(good_rows)                                         # (windows, N)
G2 = G[:, :, None] & G[:, None, :]
prof = np.array([M[:, d == k][G2[:, d == k]].mean() for k in range(N)])
np.save("data/expected_log_800_bal.npy", prof)
np.savez("data/bad_bins_800_bal.npz", chrom=w.chrom.astype(str).values, start=w.start.values, bad=bad)
print("expected profile:", prof[:5], "...", prof[-3:])
print(f"bad bins (weight NaN): {bad.sum()} / {len(bad)} = {bad.mean()*100:.1f}%")
