#!/usr/bin/env python3
"""
Visualise the O/E target used in v3a, and check how the model handles zero-count pixels.

Works on any dump written by evaluation.py --dump-matrices (v2, v3a, v3b alike), because
every dump is in log(count+1) space (v3a has the expected profile added back).

usage:
  python3 scripts/show_oe.py dump_dcnn_v3a
  python3 scripts/show_oe.py dump_dcnn_v3a --chrom chrX --window 40 --out oe_v3a_w40.png

Outputs:
  a PNG with observed / predicted / expected in log space (top row) and
  observed-E / predicted-E (the O/E target) plus the expected profile (bottom row),
  and a per-distance-band table: var(O/E), MSE, skill, and zero-count statistics.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("dump", help="dump directory, e.g. dump_dcnn_v3a")
p.add_argument("--expected", default="data/expected_log_800.npy")
p.add_argument("--chrom", default="chrX")
p.add_argument("--window", type=int, default=None,
               help="window index; default = the window with the most contacts")
p.add_argument("--out", default=None)
a = p.parse_args()

# ---------------------------------------------------------------- load
z = np.load(os.path.join(a.dump, f"matrices_{a.chrom}.npz"), allow_pickle=True)
P = z["pred"].astype(np.float32)          # (W, n, n)  log(count+1), clamped >= 0
O = z["obs"].astype(np.float32)           # (W, n, n)  log(count+1)
res, n = int(z["resolution"]), int(z["n_bins"])
starts = z["region_start"]

E1 = np.load(a.expected)                  # (n,)  mean log(count+1) per diagonal, training chroms
assert len(E1) >= n, f"expected profile has {len(E1)} entries, need {n}"
d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])   # distance in bins, (n, n)
E = E1[d]                                                    # (n, n)

w = a.window if a.window is not None else int(O.sum(axis=(1, 2)).argmax())
o, pr = O[w], P[w]
oe_obs, oe_pred = o - E, pr - E

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 3, figsize=(16, 9.5), layout="constrained")
kw_log = dict(cmap="Reds", vmin=0, vmax=float(np.percentile(o, 99.5)))
lim = float(np.percentile(np.abs(oe_obs), 99))
kw_oe = dict(cmap="RdBu_r", vmin=-lim, vmax=lim)

im0 = ax[0, 0].imshow(o, **kw_log)
ax[0, 0].set_title(f"observed  log(count+1)\n{a.chrom}:{int(starts[w]):,}-{int(starts[w]) + n * res:,}")
ax[0, 1].imshow(pr, **kw_log)
ax[0, 1].set_title(f"predicted  log(count+1)\n{os.path.basename(a.dump)}")
ax[0, 2].imshow(E, **kw_log)
ax[0, 2].set_title("expected E(d)\n(what v3a subtracts from every pixel)")
fig.colorbar(im0, ax=ax[0, :].tolist(), location="right", fraction=0.03, pad=0.02, shrink=0.9)

im1 = ax[1, 0].imshow(oe_obs, **kw_oe)
ax[1, 0].set_title("observed − E   (the v3a target)\nred = more contact than typical, blue = less")
ax[1, 1].imshow(oe_pred, **kw_oe)
ax[1, 1].set_title("predicted − E")
fig.colorbar(im1, ax=ax[1, :].tolist(), location="right", fraction=0.03, pad=0.02, shrink=0.9)

ax[1, 2].plot(np.arange(n) * res / 1000, E1[:n])
ax[1, 2].set_xlabel("genomic distance (kb)")
ax[1, 2].set_ylabel("mean log(count+1)")
ax[1, 2].set_title("expected profile E(d)")
ax[1, 2].grid(alpha=0.3)
ax[1, 2].set_box_aspect(1)          # square, like the map panels
ax[1, 2].tick_params(labelsize=8)

for x in ax.ravel()[:5]:
    x.set_xticks([])
    x.set_yticks([])

out = a.out or f"oe_{os.path.basename(a.dump)}_{a.chrom}_w{w}.png"
plt.savefig(out, dpi=130)
print("wrote", out)

# ---------------------------------------------------------------- numbers
bands = [(1, 10, "0-8 kb"), (10, 50, "8-40 kb"), (50, 150, "40-120 kb"), (150, n, "120-523 kb")]
OE_obs, OE_pred = O - E, P - E
ZERO_LOG = 0.35        # log(1.5) ~ 0.405; below this a prediction rounds to zero contacts

print(f"\n{a.chrom}: {len(O)} windows, {n} bins at {res} bp. All numbers pooled over windows.\n")
hdr = f"{'band':>12} | {'var(O/E)':>8} {'MSE':>7} {'skill':>6} | {'obs==0':>7} {'pred==0':>8} {'pred<0.35':>9} {'pred|obs==0':>11}"
print(hdr)
print("-" * len(hdr))
for lo, hi, name in bands:
    m = (d >= lo) & (d < hi)
    t, q = OE_obs[:, m], OE_pred[:, m]
    var = float(t.var())
    mse = float(((q - t) ** 2).mean())
    ob, pb = O[:, m], P[:, m]
    zo = float((ob == 0).mean())
    zp = float((pb == 0).mean())
    zl = float((pb < ZERO_LOG).mean())
    pz = float(pb[ob == 0].mean()) if zo > 0 else float("nan")
    print(f"{name:>12} | {var:8.4f} {mse:7.4f} {1 - mse / var:6.3f} | {zo:7.3f} {zp:8.4f} {zl:9.4f} {pz:11.3f}")

print("""
var(O/E)     how much variation is left after distance decay is removed; this is what there is to explain
             (it equals the decay-MSE floor for that band)
MSE          model error on the O/E target (identical to error on log counts)
skill        1 - MSE / var(O/E): the fraction of the remaining variation the model explains (same as eval_bands)
obs==0       fraction of pixels with zero observed contacts
pred==0      fraction of pixels the model puts at exactly 0 (i.e. predicted O/E <= -E(d), then clamped)
pred<0.35    fraction of pixels the model puts below half a contact (rounds to 0 counts)
pred|obs==0  mean prediction (log units) at pixels where the truth is 0; a zero-predicting model gives ~0,
             an MSE model gives a smooth positive haze
""")


# python3 scripts/show_oe.py dump_dcnn_v3a
# python3 scripts/show_oe.py dump_dcnn_v2