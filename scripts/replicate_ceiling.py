"""Measure the replicate ceiling for a Micro-C cooler.

Three subcommands:

  depth   -- how many reads per pixel do we have, as a function of distance?

  split   -- binomially split every pixel count into two half-depth coolers.
             Equivalent to randomly halving the read list.

  ceiling -- distance-stratified Pearson between the two halves, on the same
             windows the model is evaluated on, with the same log1p transform.
             Applies the Spearman-Brown correction and the sqrt attenuation
             correction to give the ceiling for a full-depth prediction.

Example:
    python3 scripts/replicate_ceiling.py depth   --cool data/lbm.800.cool \
        --regions data/windows_dm6_C523200.bed --chrom chrX --resolution 800 --n-bins 654
    python3 scripts/replicate_ceiling.py split   --cool data/lbm.800.cool \
        --out1 data/lbm.800.h1.cool --out2 data/lbm.800.h2.cool
    python3 scripts/replicate_ceiling.py ceiling --cool1 data/lbm.800.h1.cool \
        --cool2 data/lbm.800.h2.cool --regions data/windows_dm6_C523200.bed \
        --chrom chrX --resolution 800 --n-bins 654 --out ceiling_chrX_800.npz
"""

import argparse
import numpy as np
import pandas as pd
import cooler
from scipy.stats import pearsonr

BORZOI_INPUT = 524288


def resolve_chrom(c, chrom):
    if chrom in c.chromnames:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else "chr" + chrom
    if alt in c.chromnames:
        return alt
    raise SystemExit(f"{chrom!r} not in cooler (have: {c.chromnames[:5]} ...)")

# ─────────────────────────────────────────────────────────────────────────────


def load_windows(regions, chrom, n_expect):
    df = pd.read_csv(regions, sep="\t", names=["chr", "start", "end"])
    df = df[df["chr"] == chrom].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"no windows for {chrom} in {regions}")
    print(f"{len(df)} windows on {chrom}, width {df.end.iloc[0] - df.start.iloc[0]}")
    return df


def fetch(c, chrom, s, e, n_bins):
    m = c.matrix(balance=False).fetch(f"{chrom}:{s}-{e}")
    assert m.shape == (n_bins, n_bins), f"got {m.shape}, expected {n_bins}"
    return m.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
def cmd_depth(a):
    c = cooler.Cooler(a.cool)
    assert c.binsize == a.resolution, f"cooler binsize {c.binsize} != {a.resolution}"
    cchrom = resolve_chrom(c, a.chrom)
    W = load_windows(a.regions, a.chrom, a.n_bins)
    W = W.iloc[:: max(1, len(W) // a.n_sample)]

    acc = np.zeros(a.n_bins)
    cnt = np.zeros(a.n_bins)
    for _, r in W.iterrows():
        m = fetch(c, cchrom, r.start, r.end, a.n_bins)
        for d in range(a.n_bins):
            v = np.diagonal(m, offset=d)
            acc[d] += v.sum()
            cnt[d] += v.size
    mean_counts = acc / np.maximum(cnt, 1)

    print(f"\nmean RAW counts per pixel, {len(W)} windows on {a.chrom}, r={a.resolution}")
    print(f"{'distance':>12} {'mean count':>12} {'expected r_repl':>16}")
    for d_kb in [0, 5, 10, 25, 50, 100, 200, 300, 400, 500]:
        d = int(d_kb * 1000 / a.resolution)
        if d >= a.n_bins:
            continue
        lam = mean_counts[d]
        # crude Poisson expectation assuming Var(lambda) ~ lambda (overdispersion 1)
        approx = lam / (lam + 1.0)
        print(f"{d_kb:>10} kb {lam:>12.2f} {approx:>16.2f}")
    print("\nRule of thumb: mean count below ~1 per pixel means the ceiling is")
    print("driving your correlation, not the model. Run `split` + `ceiling` to measure it.")
    np.savez_compressed(a.out, mean_counts=mean_counts, resolution=a.resolution)
    print(f"wrote {a.out}")


# ─────────────────────────────────────────────────────────────────────────────
def cmd_split(a):
    c = cooler.Cooler(a.cool)
    bins = c.bins()[:][["chrom", "start", "end"]]
    nnz = c.info["nnz"]
    print(f"{a.cool}: binsize={c.binsize} nnz={nnz:,} total_reads={c.info.get('sum', 'n/a')}")

    def stream(take_first_half):
        # SAME seed both passes -> identical binomial draws -> h1 + h2 == count
        rng = np.random.default_rng(a.seed)
        for lo in range(0, nnz, a.chunksize):
            p = c.pixels()[lo: lo + a.chunksize]
            k = rng.binomial(p["count"].values.astype(np.int64), 0.5)
            out = p[["bin1_id", "bin2_id"]].copy()
            out["count"] = k if take_first_half else (p["count"].values - k)
            out = out[out["count"] > 0]
            yield out

    for path, first in [(a.out1, True), (a.out2, False)]:
        cooler.create_cooler(path, bins, stream(first), ordered=True,
                             dtypes={"count": np.int32})
        cc = cooler.Cooler(path)
        print(f"wrote {path}: nnz={cc.info['nnz']:,}")


# ─────────────────────────────────────────────────────────────────────────────
def cmd_ceiling(a):
    c1, c2 = cooler.Cooler(a.cool1), cooler.Cooler(a.cool2)
    assert c1.binsize == c2.binsize == a.resolution
    cchrom = resolve_chrom(c1, a.chrom)
    assert resolve_chrom(c2, a.chrom) == cchrom
    W = load_windows(a.regions, a.chrom, a.n_bins)

    per_window = []
    for i, r in W.iterrows():
        m1 = np.log1p(fetch(c1, cchrom, r.start, r.end, a.n_bins))
        m2 = np.log1p(fetch(c2, cchrom, r.start, r.end, a.n_bins))
        row = np.full(a.n_bins, np.nan)
        for d in range(a.n_bins):
            x, y = np.diagonal(m1, offset=d), np.diagonal(m2, offset=d)
            if x.size < 2 or x.std() == 0 or y.std() == 0:
                continue
            row[d] = pearsonr(x, y)[0]
        per_window.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(W)}")

    M = np.asarray(per_window)
    r_half = np.nanmean(M, axis=0)
    n_ok = np.isfinite(M).sum(axis=0)
    r_half[n_ok < 0.5 * len(M)] = np.nan

    # half-depth -> full-depth reliability
    r_full = 2 * r_half / (1 + r_half)
    # a perfect predictor of the truth correlates sqrt(reliability) with a noisy measurement
    ceiling = np.sqrt(np.clip(r_full, 0, 1))

    np.savez_compressed(a.out, r_half=r_half, r_full=r_full, ceiling=ceiling,
                        n_ok=n_ok, resolution=a.resolution, n_bins=a.n_bins)
    print(f"\nwrote {a.out}")
    print(f"{'distance':>12} {'r_half':>9} {'r_full':>9} {'ceiling':>9}")
    for d_kb in [0, 10, 25, 50, 100, 200, 300, 400, 500]:
        d = int(d_kb * 1000 / a.resolution)
        if d >= a.n_bins or not np.isfinite(ceiling[d]):
            continue
        print(f"{d_kb:>10} kb {r_half[d]:>9.3f} {r_full[d]:>9.3f} {ceiling[d]:>9.3f}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("depth")
    d.add_argument("--cool", required=True)
    d.add_argument("--regions", required=True)
    d.add_argument("--chrom", default="chrX")
    d.add_argument("--resolution", type=int, required=True)
    d.add_argument("--n-bins", dest="n_bins", type=int, required=True)
    d.add_argument("--n-sample", dest="n_sample", type=int, default=20)
    d.add_argument("--out", default="depth_profile.npz")
    d.set_defaults(func=cmd_depth)

    s = sub.add_parser("split")
    s.add_argument("--cool", required=True)
    s.add_argument("--out1", required=True)
    s.add_argument("--out2", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--chunksize", type=int, default=10_000_000)
    s.set_defaults(func=cmd_split)

    c = sub.add_parser("ceiling")
    c.add_argument("--cool1", required=True)
    c.add_argument("--cool2", required=True)
    c.add_argument("--regions", required=True)
    c.add_argument("--chrom", default="chrX")
    c.add_argument("--resolution", type=int, required=True)
    c.add_argument("--n-bins", dest="n_bins", type=int, required=True)
    c.add_argument("--out", default="ceiling.npz")
    c.set_defaults(func=cmd_ceiling)

    a = p.parse_args()
    a.func(a)