#!/usr/bin/env python3
"""Step 0a: test-time averaging of overlapping windows, scored against the single-shot predictions.

Reads the per-chromosome dump written by evaluation.py --dump-matrices (matrices_<chrom>.npz with
pred, obs, mask, region_start, resolution, n_bins). Windows overlap 4x (stride 130,400 bp), so every
pixel of the chromosome band is predicted up to 4 times. This script averages those predictions on
the chromosome, cuts the averaged band back into the same windows, and scores both versions with the
same metrics (same bands as eval_bands_masked.py, plus pooled Pearson per band and interior
insulation Pearson at 25 kb radius).

Usage (from the repo root, inside the chiron conda env):
    python3 scripts/step0a_tta.py dump_dcnn_v4c_s80_chrX
    python3 scripts/step0a_tta.py dump_dcnn_v4c_s80_chrX --out results/step0/tta_v4c_s80.json
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.getcwd())
from src.models.evaluation.metrics import chr_score   # noqa: E402

BANDS = [(1, 10), (10, 50), (50, 150), (150, None)]   # in bins; (150, None) = 150 .. N-1


def stitch_average(P, starts, res):
    """Average overlapping window predictions on the chromosome, return per-window averaged maps."""
    W, N, _ = P.shape
    s0 = (starts // res).astype(np.int64)
    L = int(s0.max()) + N + 1
    acc = [np.zeros(L, np.float64) for _ in range(N)]
    cnt = [np.zeros(L, np.float64) for _ in range(N)]
    for w in range(W):
        for d in range(N):
            x = np.diagonal(P[w], offset=d)
            acc[d][s0[w]:s0[w] + N - d] += x
            cnt[d][s0[w]:s0[w] + N - d] += 1.0
    A = np.zeros_like(P)
    for w in range(W):
        for d in range(N):
            v = acc[d][s0[w]:s0[w] + N - d] / np.maximum(cnt[d][s0[w]:s0[w] + N - d], 1.0)
            idx = np.arange(N - d)
            A[w, idx, idx + d] = v
            A[w, idx + d, idx] = v
    return A


def score(P, O, K, res, radius=25000):
    N = O.shape[-1]
    d = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
    prof = np.array([O[:, d == k][K[:, d == k]].mean() for k in range(N)])
    out = {}
    for lo, hi in BANDS:
        hi_ = N if hi is None else hi
        m = (d >= lo) & (d < hi_)
        sel = K[:, m]
        if sel.sum() < 2:
            continue
        p, o = P[:, m][sel], O[:, m][sel]
        base = np.broadcast_to(prof[d][m], O[:, m].shape)[sel]
        mse = float(((p - o) ** 2).mean())
        bmse = float(((base - o) ** 2).mean())
        name = f"{lo*res//1000}-{hi_*res//1000}kb"
        out[name] = {"mse": mse, "decay_mse": bmse, "skill": 1 - mse / bmse,
                     "pearson": float(pearsonr(p, o)[0])}
    pix = int(radius / res)
    rs = []
    for w in range(len(P)):
        a = np.asarray(chr_score(P[w], res=res, radius=radius))[pix:-pix]
        b = np.asarray(chr_score(O[w], res=res, radius=radius))[pix:-pix]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 2:
            rs.append(pearsonr(a[ok], b[ok])[0])
    out["insulation_interior_r"] = float(np.mean(rs)) if rs else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    results = {}
    for f in sorted(glob.glob(os.path.join(a.dump, "matrices_*.npz"))):
        z = np.load(f)
        P = z["pred"].astype(np.float32); O = z["obs"].astype(np.float32)
        K = z["mask"] if "mask" in z.files else np.ones(O.shape, bool)
        res = int(z["resolution"]); starts = z["region_start"]
        chrom = os.path.basename(f)[9:-4]
        print(f"[{chrom}] {len(P)} windows, N={O.shape[-1]}, res={res}")
        A = stitch_average(P, starts, res)
        single, avg = score(P, O, K, res), score(A, O, K, res)
        results[chrom] = {"single": single, "averaged": avg}
        print(f"{'band':>12} {'skill 1x':>9} {'skill avg':>10} {'r 1x':>7} {'r avg':>7}")
        for band in single:
            if band == "insulation_interior_r":
                continue
            s, v = single[band], avg[band]
            print(f"{band:>12} {s['skill']:+9.3f} {v['skill']:+10.3f} {s['pearson']:7.3f} {v['pearson']:7.3f}")
        print(f"insulation interior r (25 kb): single {single['insulation_interior_r']:.3f}  "
              f"averaged {avg['insulation_interior_r']:.3f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()