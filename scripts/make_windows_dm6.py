import argparse
import pandas as pd

BORZOI_INPUT = 524288

DM6 = {"chr2L": 23513712, "chr2R": 25286936, "chr3L": 28110227,
       "chr3R": 32079331, "chr4": 1348131, "chrX": 23542271}


def make_windows(out_bed, resolution, n_bins, stride=None, chromsizes=DM6):
    C = n_bins * resolution
    F = (BORZOI_INPUT - C) // 2
    stride = stride or (C // 4)

    assert C % 64 == 0, f"C={C} must be a multiple of 64"
    assert (F - 512) % 32 == 0, f"flank {F} does not land on an embedding-bin edge"
    assert stride % resolution == 0, f"stride {stride} must be a multiple of {resolution}"

    rows = []
    for chrom, L in chromsizes.items():
        lo = -(-F // resolution) * resolution      # ceil to resolution grid
        hi = L - C - F                             # keeps the full 524288 bp input in-bounds
        t = lo
        while t <= hi:
            rows.append((chrom, t, t + C))
            t += stride
    df = pd.DataFrame(rows, columns=["chr", "start", "end"])
    df.to_csv(out_bed, sep="\t", header=False, index=False)
    print(f"{out_bed}: {len(df)} windows  r={resolution} N={n_bins} C={C} F={F} S={stride}")
    print(df.groupby("chr").size().to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", type=int, default=800)
    p.add_argument("--n-bins", type=int, default=256)
    p.add_argument("--stride", type=int, default=None)
    a = p.parse_args()
    make_windows(a.out, a.resolution, a.n_bins, a.stride)