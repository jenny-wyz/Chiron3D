#!/usr/bin/env python
"""
Merge per-chromosome evaluation outputs from Chiron3D
(`src/models/evaluation/evaluation.py`) into a single aggregated file.

Input files are the `metrics_<chrom>.npz` archives written by evaluation.py:

    insu_pearson         (W,)     per-window insulation-profile Pearson r
    insu_spearman        (W,)     per-window insulation-profile Spearman rho
    mse                  (W,)     per-window MSE
    dist_strat_pearson   (W, D)   per-window, per-diagonal Pearson r
    dist_strat_spearman  (W, D)   per-window, per-diagonal Spearman rho
    diag_x               dict     {offset -> list of predicted values},  pooled over windows
    diag_y               dict     {offset -> list of observed  values},  pooled over windows

W = number of test windows on that chromosome, D = matrix_size - 1
(104 for the clipped C.Origami comparison, 208 for the Borzoi head).
`diag_x` / `diag_y` are Python dicts, so numpy stores them as 0-d object
arrays and they can only be read back with allow_pickle=True.

Usage
-----
python merge_metrics.py \
    '../metrics_chr2.npz' \
    '../metrics_chr6.npz' \
    '../metrics_chr19.npz' \
    -o '../metrics_hg19_agg.npz'

    # or let it glob
python merge_metrics.py --glob 'metrics_chr*.npz' -o metrics_hg19_avg.npz

    # drop the raw pooled diagonals to keep the output small
python merge_metrics.py --glob 'metrics_chr*.npz' --diag-mode drop
"""

from __future__ import annotations

import argparse
import glob as globmod
import os
import re
import sys
import warnings

import numpy as np

try:
    from scipy.stats import pearsonr, spearmanr
    _HAVE_SCIPY = True
except ImportError:  # pooled recomputation is then skipped
    _HAVE_SCIPY = False

# Per-window scalar metrics: concatenated across chromosomes.
SCALAR_KEYS = ("mse", "insu_pearson", "insu_spearman")
# Per-window x per-diagonal metrics: row-stacked (NaN-padded) across chromosomes.
MATRIX_KEYS = ("dist_strat_pearson", "dist_strat_spearman")
# Pooled raw values, stored as dicts.
DIAG_KEYS = ("diag_x", "diag_y")


# --------------------------------------------------------------------------- #
# loading helpers
# --------------------------------------------------------------------------- #
def parse_chrom(path: str) -> str:
    """Recover the chromosome label from a `metrics_<chrom>.npz` filename."""
    base = os.path.basename(path)
    m = re.match(r"^metrics[_\-](.+?)\.npz$", base)
    if m:
        return m.group(1)
    m = re.search(r"(chr[0-9]+|chr[XYM]|chr[0-9]+[LR])", base)
    return m.group(1) if m else os.path.splitext(base)[0]


def as_dict(obj) -> dict:
    """Unwrap a dict that numpy stored as a 0-d object array."""
    if isinstance(obj, np.ndarray):
        if obj.dtype == object and obj.ndim == 0:
            obj = obj.item()
        elif obj.dtype == object and obj.size == 1:
            obj = obj.reshape(-1)[0]
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise TypeError(f"expected a dict of diagonals, got {type(obj)}")
    # keys may come back as np.int64; values as Python lists
    return {int(k): np.asarray(v, dtype=np.float64).ravel() for k, v in obj.items()}


def as_matrix(arr) -> np.ndarray:
    """
    Coerce a per-window x per-diagonal metric to a 2-D float array.

    evaluation.py builds these with np.asarray(list_of_lists, float), which
    yields a clean (W, D) array when every window has the same matrix size.
    A ragged run (mixed matrix sizes) lands here as a 1-D object array, so
    handle that too by NaN-padding to the longest row.
    """
    arr = np.asarray(arr)
    if arr.dtype != object and arr.ndim == 2:
        return arr.astype(np.float64, copy=False)
    if arr.dtype != object and arr.ndim == 1:
        # a single window, or an empty run
        return arr.astype(np.float64, copy=False).reshape(1, -1) if arr.size else np.empty((0, 0))
    rows = [np.asarray(r, dtype=np.float64).ravel() for r in arr]
    if not rows:
        return np.empty((0, 0))
    width = max(r.size for r in rows)
    out = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        out[i, : r.size] = r
    return out


def pad_to(mat: np.ndarray, width: int) -> np.ndarray:
    """NaN-pad a (W, D) block on the right so it can be stacked with others."""
    if mat.shape[1] == width:
        return mat
    out = np.full((mat.shape[0], width), np.nan)
    out[:, : mat.shape[1]] = mat
    return out


# --------------------------------------------------------------------------- #
# stats helpers
# --------------------------------------------------------------------------- #
def finite_stats(a: np.ndarray) -> dict:
    """mean / std / sem / n over the finite entries only, NaN-safe when empty."""
    a = np.asarray(a, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "sem": np.nan, "median": np.nan, "n": 0}
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    return {
        "mean": float(a.mean()),
        "std": std,
        "sem": std / np.sqrt(n) if n > 1 else 0.0,
        "median": float(np.median(a)),
        "n": n,
    }


def column_stats(mat: np.ndarray):
    """Per-diagonal mean / std / count over finite entries of a (W, D) block."""
    if mat.size == 0:
        empty = np.empty(0)
        return empty, empty, empty.astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros(mat.shape[1])
    n = np.isfinite(mat).sum(axis=0).astype(np.int64)
    mean = np.where(n > 0, mean, np.nan)
    std = np.where(n > 1, std, np.nan)
    return mean, std, n


def pooled_corr(x: np.ndarray, y: np.ndarray, max_n: int, rng: np.random.Generator,
                do_spearman: bool = True):
    """
    Correlate the pooled raw values for one diagonal.

    This is the reason diag_x / diag_y exist: averaging per-window
    correlations is not the same as correlating the pooled values, and the
    pooled version is the one that is comparable across chromosomes with
    different window counts.
    """
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = int(x.size)
    if n < 2:
        return np.nan, np.nan, n
    if max_n and n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
        x, y = x[idx], y[idx]
    # a constant vector makes r undefined
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, np.nan, n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rp = float(pearsonr(x, y)[0])
        rs = float(spearmanr(x, y)[0]) if do_spearman else np.nan
    return rp, rs, n


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(
        description="Merge Chiron3D per-chromosome metrics_*.npz into one aggregate file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("files", nargs="*", help="metrics_<chrom>.npz files to merge")
    p.add_argument("--glob", default=None,
                   help="glob pattern used instead of / in addition to positional files")
    p.add_argument("-o", "--output", default="metrics_hg19_avg.npz",
                   help="output archive")
    p.add_argument("--diag-mode", choices=["full", "subsample", "drop"], default="full",
                   help="what to do with the pooled diag_x/diag_y values in the output")
    p.add_argument("--diag-subsample", type=int, default=200_000,
                   help="values kept per diagonal when --diag-mode subsample")
    p.add_argument("--pool-max-n", type=int, default=2_000_000,
                   help="cap on values used per diagonal when recomputing pooled "
                        "correlations (0 = no cap); guards Spearman runtime")
    p.add_argument("--no-pooled", action="store_true",
                   help="skip recomputing pooled per-diagonal correlations")
    p.add_argument("--no-pooled-spearman", action="store_true",
                   help="recompute pooled Pearson only (Spearman is the slow one)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling")
    p.add_argument("--strict", action="store_true",
                   help="fail instead of warn on missing keys or shape mismatches")
    p.add_argument("--allow-overwrite", action="store_true",
                   help="overwrite the output file if it already exists")
    args = p.parse_args()

    # ---------------- collect input files ----------------
    paths = list(args.files)
    if args.glob:
        paths.extend(globmod.glob(args.glob))
    # de-duplicate by realpath, keep order
    seen, files = set(), []
    for f in paths:
        rp = os.path.realpath(f)
        if rp not in seen:
            seen.add(rp)
            files.append(f)
    if not files:
        p.error("no input files (pass paths positionally or use --glob)")
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        p.error("input file(s) not found: " + ", ".join(missing))
    if os.path.exists(args.output) and not args.allow_overwrite:
        p.error(f"{args.output} already exists (use --allow-overwrite)")

    def problem(msg: str):
        if args.strict:
            raise ValueError(msg)
        print(f"[warn] {msg}", file=sys.stderr)

    rng = np.random.default_rng(args.seed)

    # ---------------- accumulators ----------------
    chroms, n_windows = [], []
    scalars = {k: [] for k in SCALAR_KEYS}
    matrices = {k: [] for k in MATRIX_KEYS}
    per_chrom = {k: {"mean": [], "std": [], "n": []} for k in SCALAR_KEYS}
    diag_acc = {k: {} for k in DIAG_KEYS}          # offset -> list of arrays
    window_chrom = []

    print(f"Merging {len(files)} file(s)")
    for path in files:
        chrom = parse_chrom(path)
        with np.load(path, allow_pickle=True) as z:
            keys = set(z.files)

            # per-window scalars decide the window count for this file
            w = None
            local_scalars = {}
            for k in SCALAR_KEYS:
                if k not in keys:
                    problem(f"{path}: missing '{k}'")
                    local_scalars[k] = np.empty(0)
                    continue
                v = np.asarray(z[k], dtype=np.float64).ravel()
                local_scalars[k] = v
                if w is None:
                    w = v.size
                elif v.size != w:
                    problem(f"{path}: '{k}' has {v.size} entries but expected {w}")
            w = w or 0

            for k in SCALAR_KEYS:
                v = local_scalars[k]
                if v.size != w:                      # keep columns alignable
                    padded = np.full(w, np.nan)
                    padded[: min(w, v.size)] = v[: min(w, v.size)]
                    v = padded
                scalars[k].append(v)
                st = finite_stats(v)
                per_chrom[k]["mean"].append(st["mean"])
                per_chrom[k]["std"].append(st["std"])
                per_chrom[k]["n"].append(st["n"])

            for k in MATRIX_KEYS:
                if k not in keys:
                    problem(f"{path}: missing '{k}'")
                    matrices[k].append(np.full((w, 0), np.nan))
                    continue
                mat = as_matrix(z[k])
                if mat.shape[0] != w:
                    problem(f"{path}: '{k}' has {mat.shape[0]} rows but {w} windows")
                matrices[k].append(mat)

            for k in DIAG_KEYS:
                if k not in keys:
                    problem(f"{path}: missing '{k}'")
                    continue
                try:
                    d = as_dict(z[k])
                except TypeError as e:
                    problem(f"{path}: could not read '{k}' ({e})")
                    continue
                for off, vals in d.items():
                    diag_acc[k].setdefault(off, []).append(vals)

        chroms.append(chrom)
        n_windows.append(w)
        window_chrom.append(np.full(w, chrom, dtype=object))
        print(f"  {os.path.basename(path)}: chrom={chrom}  windows={w}  "
              f"MSE={finite_stats(local_scalars['mse'])['mean']:.4f}  "
              f"insu_r={finite_stats(local_scalars['insu_pearson'])['mean']:.4f}")

    # ---------------- concatenate ----------------
    out = {}
    for k in SCALAR_KEYS:
        out[k] = np.concatenate(scalars[k]) if scalars[k] else np.empty(0)

    widths = [m.shape[1] for k in MATRIX_KEYS for m in matrices[k] if m.size]
    width = max(widths) if widths else 0
    if len(set(widths)) > 1:
        problem(f"distance-stratified blocks have differing widths {sorted(set(widths))}; "
                f"NaN-padding to {width}")
    for k in MATRIX_KEYS:
        blocks = [pad_to(m, width) if m.shape[1] != width else m for m in matrices[k]]
        blocks = [b for b in blocks if b.shape[1] == width]
        out[k] = np.vstack(blocks) if blocks else np.empty((0, width))

    out["files"] = np.array([os.path.abspath(f) for f in files], dtype=object)
    out["chroms"] = np.array(chroms, dtype=object)
    out["n_windows_per_chrom"] = np.asarray(n_windows, dtype=np.int64)
    out["window_chrom"] = (np.concatenate(window_chrom) if window_chrom
                           else np.empty(0, dtype=object))

    # ---------------- scalar summaries ----------------
    # window-weighted: every test window counts once, regardless of chromosome
    # chrom-weighted:  unweighted mean of the three per-chromosome means
    for k in SCALAR_KEYS:
        st = finite_stats(out[k])
        out[f"{k}_mean"] = st["mean"]
        out[f"{k}_std"] = st["std"]
        out[f"{k}_sem"] = st["sem"]
        out[f"{k}_median"] = st["median"]
        out[f"{k}_n"] = st["n"]

        pm = np.asarray(per_chrom[k]["mean"], dtype=np.float64)
        out[f"{k}_per_chrom_mean"] = pm
        out[f"{k}_per_chrom_std"] = np.asarray(per_chrom[k]["std"], dtype=np.float64)
        out[f"{k}_per_chrom_n"] = np.asarray(per_chrom[k]["n"], dtype=np.int64)
        finite_pm = pm[np.isfinite(pm)]
        out[f"{k}_mean_chrom_weighted"] = float(finite_pm.mean()) if finite_pm.size else np.nan
        out[f"{k}_std_chrom_weighted"] = (float(finite_pm.std(ddof=1))
                                          if finite_pm.size > 1 else np.nan)

    # ---------------- distance-stratified summaries ----------------
    out["dist_offsets"] = np.arange(width, dtype=np.int64)
    for k in MATRIX_KEYS:
        mean, std, n = column_stats(out[k])
        out[f"{k}_mean"] = mean
        out[f"{k}_std"] = std
        out[f"{k}_n"] = n

    # ---------------- pooled diagonals ----------------
    merged_diag = {}
    for k in DIAG_KEYS:
        merged_diag[k] = {off: np.concatenate(v) for off, v in sorted(diag_acc[k].items())}

    xs, ys = merged_diag["diag_x"], merged_diag["diag_y"]
    common = sorted(set(xs) & set(ys))
    if set(xs) != set(ys):
        problem(f"diag_x and diag_y cover different offsets "
                f"({len(xs)} vs {len(ys)}); using the {len(common)} shared ones")

    if common and not args.no_pooled:
        if not _HAVE_SCIPY:
            problem("scipy not available; skipping pooled correlation recomputation")
        else:
            pooled_p = np.full(width if width else len(common), np.nan)
            pooled_s = np.full_like(pooled_p, np.nan)
            pooled_n = np.zeros(pooled_p.size, dtype=np.int64)
            for off in common:
                if off >= pooled_p.size:
                    continue
                x, y = xs[off], ys[off]
                if x.size != y.size:
                    problem(f"offset {off}: diag_x has {x.size} values, diag_y has {y.size}; "
                            f"truncating to the shorter")
                    m = min(x.size, y.size)
                    x, y = x[:m], y[:m]
                rp, rs, n = pooled_corr(x, y, args.pool_max_n, rng,
                                        do_spearman=not args.no_pooled_spearman)
                pooled_p[off], pooled_s[off], pooled_n[off] = rp, rs, n
            out["pooled_dist_pearson"] = pooled_p
            out["pooled_dist_spearman"] = pooled_s
            out["pooled_dist_n"] = pooled_n

    # ---------------- raw pooled values ----------------
    if args.diag_mode == "drop":
        print("  (diag_x / diag_y dropped from the output)")
    else:
        for k in DIAG_KEYS:
            d = merged_diag[k]
            if args.diag_mode == "subsample":
                cap = args.diag_subsample
                sub = {}
                for off, v in d.items():
                    if v.size > cap:
                        # same seed per offset for both x and y so pairs stay aligned
                        idx = np.random.default_rng(args.seed + off).choice(
                            v.size, size=cap, replace=False)
                        idx.sort()
                        v = v[idx]
                    sub[off] = v.astype(np.float32)
                d = sub
            else:
                d = {off: v.astype(np.float32) for off, v in d.items()}
            out[k] = d

    # ---------------- write ----------------
    np.savez_compressed(args.output, **out)

    # ---------------- report ----------------
    print()
    print(f"Chromosomes : {', '.join(chroms)}")
    print(f"Windows     : {int(np.sum(n_windows))} "
          f"({', '.join(f'{c}={n}' for c, n in zip(chroms, n_windows))})")
    print(f"Diagonals   : {width}")
    print()
    label = {"mse": "MSE", "insu_pearson": "Insulation Pearson",
             "insu_spearman": "Insulation Spearman"}
    print(f"{'metric':<22}{'mean':>10}{'std':>10}{'sem':>10}{'n':>8}   chrom-weighted")
    for k in SCALAR_KEYS:
        print(f"{label[k]:<22}{out[f'{k}_mean']:>10.4f}{out[f'{k}_std']:>10.4f}"
              f"{out[f'{k}_sem']:>10.4f}{out[f'{k}_n']:>8d}   "
              f"{out[f'{k}_mean_chrom_weighted']:.4f}")
    for k in MATRIX_KEYS:
        m = out[f"{k}_mean"]
        m = m[np.isfinite(m)]
        if m.size:
            name = "Dist-strat " + k.rsplit("_", 1)[1].capitalize()
            print(f"{name:<22}{m.mean():>10.4f}{'':>10}{'':>10}{m.size:>8d}"
                  "   (mean over diagonals)")
    if "pooled_dist_pearson" in out:
        v = out["pooled_dist_pearson"]
        v = v[np.isfinite(v)]
        if v.size:
            print(f"{'pooled Pearson':<22}{v.mean():>10.4f}{'':>10}{'':>10}{v.size:>8d}"
                  "   (correlated on pooled values, not averaged)")
    print()
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Wrote {args.output}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())