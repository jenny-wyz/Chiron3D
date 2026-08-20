"""Generate dm6 training windows for Chiron3D, optionally filtered against a blacklist.

Each BED row is the TARGET span (n_bins * resolution). GenomicDataset expands it to the
524,288 bp Borzoi input by adding a flank of (524288 - C) // 2 on each side, so the windows
here are already positioned to keep that full input inside the chromosome.

  --max-blacklist-frac   bounds the TOTAL masked share of the target span
  --max-blacklist-run    bounds the LONGEST UNBROKEN masked stretch, in bp

Without --blacklist the output is identical to the unfiltered generator.
"""

import argparse
import numpy as np
import pandas as pd

BORZOI_INPUT = 524288

DM6 = {"chr2L": 23513712, "chr2R": 25286936, "chr3L": 28110227,
       "chr3R": 32079331, "chr4": 1348131, "chrX": 23542271}


def load_blacklist(paths, resolution, chromsizes):
    """Rasterise one or more blacklist BEDs onto the resolution grid, one mask per chromosome."""
    masks = {chrom: np.zeros(-(-L // resolution), dtype=bool) for chrom, L in chromsizes.items()}
    for path in paths:
        bl = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2],
                         names=["chr", "start", "end"])
        for chrom, start, end in bl.itertuples(index=False):
            if chrom not in masks:
                continue
            masks[chrom][start // resolution: -(-end // resolution)] = True
    return masks


def max_run(mask):
    """Length of the longest consecutive True run, in bins."""
    m = np.asarray(mask, dtype=np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(d == 1)[0]
    if starts.size == 0:
        return 0
    return int((np.where(d == -1)[0] - starts).max())


def make_windows(out_bed, resolution, n_bins, stride=None, chromsizes=DM6,
                 blacklist=None, max_bl_frac=0.20, max_bl_run=10000, shift_bins=9):
    C = n_bins * resolution
    F = (BORZOI_INPUT - C) // 2
    stride = stride or (C // 4)

    assert C % 64 == 0, f"C={C} must be a multiple of 64"
    assert (F - 512) % 32 == 0, f"flank {F} does not land on an embedding-bin edge"
    assert stride % resolution == 0, f"stride {stride} must be a multiple of {resolution}"

    masks = load_blacklist(blacklist, resolution, chromsizes) if blacklist else None
    # shift_aug moves the target by up to +-shift_bins during training, so a window is only safe
    # if every position it can reach is clean, not just the one written to the bed
    pad = shift_bins * resolution
    max_run_bins = max_bl_run // resolution

    rows = []
    stats = []
    for chrom, L in chromsizes.items():
        lo = -(-F // resolution) * resolution      # ceil to resolution grid
        hi = L - C - F                             # keeps the full 524288 bp input in-bounds
        t = lo
        n_total = n_drop_frac = n_drop_run = 0
        while t <= hi:
            n_total += 1
            if masks is not None:
                sub = masks[chrom][max(t - pad, 0) // resolution: (t + C + pad) // resolution]
                if sub.mean() > max_bl_frac:
                    n_drop_frac += 1
                    t += stride
                    continue
                if max_run(sub) > max_run_bins:
                    n_drop_run += 1
                    t += stride
                    continue
            rows.append((chrom, t, t + C))
            t += stride
        stats.append((chrom, n_total, n_total - n_drop_frac - n_drop_run, n_drop_frac, n_drop_run))

    df = pd.DataFrame(rows, columns=["chr", "start", "end"])
    df.to_csv(out_bed, sep="\t", header=False, index=False)
    print(f"{out_bed}: {len(df)} windows  r={resolution} N={n_bins} C={C} F={F} S={stride}")

    if masks is None:
        print(df.groupby("chr").size().to_string())
        return

    print(f"blacklist: {blacklist}  max_frac={max_bl_frac} "
          f"max_run={max_bl_run}bp ({max_run_bins} bins)  shift_pad=+-{pad}bp")
    sd = pd.DataFrame(stats, columns=["chr", "candidates", "kept", "drop_frac", "drop_run"])
    sd["kept_%"] = (100 * sd["kept"] / sd["candidates"]).round(1)
    print(sd.to_string(index=False))
    total = sd[["candidates", "kept"]].sum()
    print(f"TOTAL kept {total.kept}/{total.candidates} "
          f"({100 * total.kept / total.candidates:.1f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", type=int, default=800)
    p.add_argument("--n-bins", type=int, default=512)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--blacklist", action="append", default=None,
                   help="blacklist BED; repeat the flag to union several")
    p.add_argument("--max-blacklist-frac", type=float, default=0.20,
                   help="drop a window if more than this fraction of its target span is masked")
    p.add_argument("--max-blacklist-run", type=int, default=10000,
                   help="drop a window whose longest unbroken masked run exceeds this, in bp")
    p.add_argument("--shift-bins", type=int, default=9,
                   help="pad the target each side by this many bins before testing, matching shift_aug")
    a = p.parse_args()
    make_windows(a.out, a.resolution, a.n_bins, a.stride,
                 blacklist=a.blacklist, max_bl_frac=a.max_blacklist_frac,
                 max_bl_run=a.max_blacklist_run, shift_bins=a.shift_bins)
