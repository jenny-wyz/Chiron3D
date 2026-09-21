import cooler, numpy as np, pandas as pd
c = cooler.Cooler("data/lbm.800.cool")
bed = pd.read_csv("data/windows_dm6_C523200_f1024.bed", sep="\t", names=["chr","start","end"])
bed = bed[~bed.chr.isin(["chr2L","chrX"])]          # training chroms only
M = np.stack([np.log(c.matrix(balance=False).fetch(f"{r.chr}:{r.start}-{r.end}") + 1.0)
              for r in bed.itertuples()])
N = M.shape[-1]
d = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
prof = np.array([M[:, d == k].mean() for k in range(N)])
np.save("data/expected_log_800.npy", prof)
print("expected profile:", prof[:5], "...", prof[-3:])
# bad bins: zero coverage genome-wide
bins = c.bins()[:]
cov = np.asarray(c.info.get("sum", 0))
marg = np.zeros(len(bins))
for ch in bins.chrom.unique():
    m = c.matrix(balance=False, sparse=True).fetch(str(ch)).sum(axis=1).A1
    marg[(bins.chrom == ch).values] = m
bad = (marg == 0)
np.savez("data/bad_bins_800.npz",
         chrom=bins.chrom.astype(str).values, start=bins.start.values, bad=bad)
print("bad bins:", bad.sum(), "/", len(bad))


# python3 scripts/make_expected.py