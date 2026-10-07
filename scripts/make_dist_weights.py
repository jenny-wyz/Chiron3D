"""Per-distance loss weights for v4c (Chimaera-style focus on the near-diagonal band).
Usage: python3 scripts/make_dist_weights.py <n_bins> <resolution> <band_kb> <band_share>
  e.g.  python3 scripts/make_dist_weights.py 654 800 54 0.5
Writes data/dist_weights_<res>_band<band_kb>_s<share>.npy : one weight per diagonal d = 0..n_bins-1.
  - diagonals 0 and 1 (0 - 1.6 kb): weight 0 (ligation artefacts; cooler balance and Chimaera both skip them)
  - 2 <= d <= band_kb: constant weight f  (chosen so this band carries band_share of the total loss)
  - beyond:            weight 1
  - finally rescaled so the mean weight over all pixels is 1 (loss magnitude comparable to v4b)
With plain MSE the <= 54 kb band carries ~20% of the loss on a 654-bin square; Chimaera gives it 100%."""
import sys, numpy as np
N, RES, band_kb, share = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
K = int(band_kb * 1000 // RES)
npix = N - np.arange(N)                          # pixels per diagonal (upper triangle)
w = np.ones(N); w[:2] = 0.0
inb = (np.arange(N) >= 2) & (np.arange(N) <= K)
S_in, S_out = npix[inb].sum(), npix[~inb & (np.arange(N) >= 2)].sum()
w[inb] = (share / (1 - share)) * S_out / S_in
w = w / (np.sum(w * npix) / npix.sum())
out = f"data/dist_weights_{RES}_band{int(band_kb)}_s{int(round(share*100))}.npy"
np.save(out, w)
print(f"wrote {out}")
print(f"weight for 1.6 kb <= d <= {band_kb:g} kb : {w[2]:.3f}   weight beyond : {w[-1]:.3f}   (d=0,1: 0)")
print(f"share of total loss weight at <= {band_kb:g} kb : {np.sum(w[inb]*npix[inb])/np.sum(w*npix):.2f}  (plain MSE: {npix[inb].sum()/npix.sum():.2f})")
