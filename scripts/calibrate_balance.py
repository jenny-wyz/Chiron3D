'''
python3 scripts/calibrate_balance.py --cool data/lbm.800.cool --chrom chr2L
'''

import argparse, cooler, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--cool", required=True)      # e.g. data/dm6.mcool::/resolutions/800
p.add_argument("--chrom", default="chr2L")
a = p.parse_args()

clr = cooler.Cooler(a.cool)
assert "weight" in clr.bins().columns, "No 'weight' column — run `cooler balance` at this resolution."

w = clr.bins().fetch(a.chrom)["weight"].values
print(f"[{a.chrom}] bins={len(w)}  NaN weights={np.isnan(w).mean():.3%}")

raw = clr.matrix(balance=False).fetch(a.chrom)
bal = clr.matrix(balance=True).fetch(a.chrom)
ok = np.isfinite(bal) & (bal > 0) & (raw > 0)
scale = float(np.median(raw[ok] / bal[ok]))
print(f"MATRIX_SCALE = {scale:.6g}")

print(f"log1p(raw)        mean={np.log1p(raw).mean():.4f}  std={np.log1p(raw).std():.4f}")
lb = np.log1p(np.nan_to_num(bal, nan=0.0) * scale)
print(f"log1p(bal*scale)  mean={lb.mean():.4f}  std={lb.std():.4f}")