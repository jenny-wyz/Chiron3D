#!/usr/bin/env python3
"""Rebuild per-window obs/pred matrices from an existing metrics_<chrom>.npz.

distance_stratified_correlation stores, for every diagonal offset d, the pred
and obs values of that diagonal concatenated across windows in dataloader order
(shuffle=False, batch_size=1). That is enough to reconstruct the upper triangle
of every window. The lower triangle is filled by mirroring.

Produces a file with the same keys as the --dump-matrices patch, so the plotting
scripts consume it unchanged.

Usage:
-------
python3 reconstruct_matrices.py \
    --metrics metrics_chrX_5k.npz \
    --regions data/windows_dm6.bed \
    --chrom chrX \
    --out dumps/matrices_chrX_5k.npz
"""

import argparse
import os

import numpy as np
import pandas as pd


def load_diags(path):
    z = np.load(path, allow_pickle=True)
    keys = set(z.files)

    if "diag_x" in keys:                       # main branch: dicts pickled whole
        xs = z["diag_x"].item()
        ys = z["diag_y"].item()
        xs = {int(d): np.asarray(v, dtype=np.float64) for d, v in xs.items()}
        ys = {int(d): np.asarray(v, dtype=np.float64) for d, v in ys.items()}
    else:                                      # dm6 branch: diag_x_<d> arrays
        xs, ys = {}, {}
        for k in z.files:
            if k.startswith("diag_x_"):
                xs[int(k[len("diag_x_"):])] = np.asarray(z[k], dtype=np.float64)
            elif k.startswith("diag_y_"):
                ys[int(k[len("diag_y_"):])] = np.asarray(z[k], dtype=np.float64)
        if not xs:
            raise SystemExit(
                f"{path} has no diag_x/diag_y entries (keys: {sorted(keys)}). "
                "Nothing to reconstruct -- re-run evaluation with --dump-matrices.")
    return xs, ys, z


def infer_geometry(xs):
    """len(xs[0]) = W*N and len(xs[1]) = W*(N-1), so W = len0 - len1."""
    if 0 not in xs or 1 not in xs:
        raise SystemExit("Need diagonals 0 and 1 to infer the geometry.")
    len0, len1 = len(xs[0]), len(xs[1])
    W = len0 - len1
    if W <= 0 or len0 % W != 0:
        raise SystemExit(f"Inconsistent lengths: |d=0|={len0}, |d=1|={len1}")
    N = len0 // W
    dmax = max(xs)
    if dmax not in (N - 2, N - 1):
        print(f"[warn] highest stored diagonal is {dmax}, expected {N-2} for N={N}")
    return W, N


def rebuild(diag, W, N):
    """diag[d] -> (W, N, N) with the upper triangle filled and mirrored."""
    M = np.zeros((W, N, N), dtype=np.float32)
    for d, vals in sorted(diag.items()):
        L = N - d
        if L < 2:
            continue
        v = np.asarray(vals, dtype=np.float32)
        if v.size != W * L:
            raise SystemExit(
                f"diagonal {d}: got {v.size} values, expected {W}*{L}={W*L}. "
                "Window count or matrix size is not what was inferred.")
        v = v.reshape(W, L)
        rows = np.arange(L)
        M[:, rows, rows + d] = v
        if d > 0:
            M[:, rows + d, rows] = v          # mirror
    return M


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True, help="metrics_<chrom>.npz you already have")
    p.add_argument("--regions", required=True, help="the BED passed to --regions-file")
    p.add_argument("--chrom", required=True, help="chromosome that was evaluated, e.g. chrX")
    p.add_argument("--out", required=True, help="output matrices_<chrom>.npz")
    p.add_argument("--resolution", type=int, default=None,
                   help="bp per bin; inferred from the BED width and N if omitted")
    p.add_argument("--float32", action="store_true", help="store float32 instead of float16")
    a = p.parse_args()

    xs, ys, z = load_diags(a.metrics)
    W, N = infer_geometry(xs)
    print(f"[infer] {W} windows, {N}x{N} bins, diagonals 0..{max(xs)}")

    if "mse" in z.files and len(z["mse"]) != W:
        print(f"[warn] metrics file lists {len(z['mse'])} windows but the diagonals imply {W}")

    bed = pd.read_csv(a.regions, sep="\t", header=None,
                      names=["chr", "start", "end"], usecols=[0, 1, 2])
    bed = bed[bed["chr"] == a.chrom].reset_index(drop=True)
    if len(bed) != W:
        raise SystemExit(
            f"{a.chrom} has {len(bed)} regions in {a.regions} but the metrics file "
            f"implies {W} windows. Wrong BED, or the eval filtered differently.")

    width = int((bed["end"] - bed["start"]).iloc[0])
    res = a.resolution or (width // N)
    if a.resolution is None:
        print(f"[infer] resolution = {width} bp / {N} bins = {res} bp")
    if width % N:
        print(f"[warn] window width {width} is not divisible by N={N}; "
              f"pass --resolution explicitly if {res} is wrong")
    off = bed["start"] % res
    if (off != 0).any():
        print(f"[warn] {int((off != 0).sum())} window starts are not multiples of {res}; "
              "cooler snapped the fetch outward, so bin 0 may be up to one bin left "
              "of region_start")

    pred = rebuild(xs, W, N)
    obs = rebuild(ys, W, N)
    dt = np.float32 if a.float32 else np.float16

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez_compressed(
        a.out,
        pred=pred.astype(dt), obs=obs.astype(dt),
        chrom=np.array([a.chrom] * W),
        region_start=bed["start"].to_numpy(np.int64),
        region_end=bed["end"].to_numpy(np.int64),
        resolution=np.int64(res),
        n_bins=np.int64(N),
    )
    print(f"[write] {a.out}  pred={pred.shape}  "
          f"obs_max={obs.max():.3f}  pred_max={pred.max():.3f}")
    print("[note] lower triangle is mirrored from the upper; pixel (0, N-1) is 0 "
          "because that diagonal has size 1 and was skipped")


if __name__ == "__main__":
    main()