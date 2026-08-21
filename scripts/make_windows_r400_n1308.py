"""Generate TARGET-interval BED files for Chiron3D-dm6.

The BED holds intervals of width C = n_bins * resolution.  The Borzoi input
window is derived by the dataset as [t - F, t - F + 524288) where
F = (524288 - C) // 2, so every window keeps full 524,288 bp of context in bounds.

Usage:
    python3 scripts/make_windows_dm6.py --out data/windows_dm6_C523200.bed \
        --resolution 400 --n-bins 1308 --stride 130400
"""

import argparse
import pandas as pd

BORZOI_INPUT = 524288
EMB_BINS = 16352
EMB_BIN = 32
EMB_HEAD_OFF = 512
MAX_SPAN = EMB_BINS * EMB_BIN          # 523,264 bp -- hard ceiling on C

DM6 = {"chr2L": 23513712, "chr2R": 25286936, "chr3L": 28110227,
       "chr3R": 32079331, "chr4": 1348131, "chrX": 23542271}


def make_windows(out_bed, resolution, n_bins, stride=None, chromsizes=DM6):
    C = n_bins * resolution
    F = (BORZOI_INPUT - C) // 2
    stride = stride or (C // 4)

    # --- the same three asserts Chiron3D.__init__ will run ---
    assert C <= MAX_SPAN, (
        f"C = {n_bins} x {resolution} = {C} exceeds the Borzoi embedding span "
        f"{MAX_SPAN}. Largest legal n_bins at r={resolution} is "
        f"{max(n for n in range(1, MAX_SPAN // resolution + 1) if (n * resolution) % 64 == 0)}")
    assert C % 64 == 0, f"C = {C} must be a multiple of 64"
    assert (F - EMB_HEAD_OFF) % EMB_BIN == 0, f"flank {F} is not on an embedding-bin edge"
    assert stride % resolution == 0, f"stride {stride} must be a multiple of {resolution}"
 
    emb_lo = (F - EMB_HEAD_OFF) // EMB_BIN
    emb_hi = emb_lo + C // EMB_BIN
    pool = (C // EMB_BIN) / n_bins

    rows = []
    for chrom, L in sorted(chromsizes.items()):
        lo = -(-F // resolution) * resolution     # ceil onto the resolution grid
        hi = L - C - F                            # full input window stays in bounds
        t = lo
        while t <= hi:
            rows.append((chrom, t, t + C))
            t += stride

    df = pd.DataFrame(rows, columns=["chr", "start", "end"])
    df.to_csv(out_bed, sep="\t", header=False, index=False)

    print(f"{out_bed}")
    print(f"  r={resolution}  N={n_bins}  C={C}  F={F}  S={stride}  first_start={lo}")
    print(f"  emb[{emb_lo}:{emb_hi}]  pool_factor={pool}  exact_pool={resolution % EMB_BIN == 0}")
    print(f"  {len(df)} windows total")
    print(df.groupby("chr").size().to_string())
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", type=int, required=True)
    p.add_argument("--n-bins", dest="n_bins", type=int, required=True)
    p.add_argument("--stride", type=int, default=None)
    a = p.parse_args()
    make_windows(a.out, a.resolution, a.n_bins, a.stride)