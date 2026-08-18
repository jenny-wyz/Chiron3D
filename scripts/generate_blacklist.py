#!/usr/bin/env python3
"""
Resolution-independent blacklist: contiguous stretches of genome with no Micro-C signal.
The ENCODE genome blacklist is deliberately NOT included by default. Pass --encode-blacklist to union it in anyway.

To be included in the blacklist, a region must have no contacts across more than --min-gap-bp, 
the stretch must not be interrupted by more that --bridge-bp.

Usage
-----
    python generate_blacklist_gaps.py \
        --cool /path/lbm.mcool::resolutions/100 \
        --min-gap-bp 5000 --bridge-bp 500 \
        --fasta-dir /path/dmel_chromosomes \
        --out dm6_lbm_gaps_5kb.bed --report reports/gaps_5kb.txt
"""

import argparse
import os
import sys
import time

import cooler
import h5py
import numpy as np
import pandas as pd

CHUNK = 20_000_000  # pixel rows per streaming step

DM6_MAIN = ["chr2L", "chr2R", "chr3L", "chr3R", "chr4", "chrX"]


def split_uri(uri):
    """Split 'file.mcool::/resolutions/800' into (path, group). Group is '' for a flat .cool."""
    if "::" in uri:
        path, grp = uri.split("::", 1)
        return path, grp.strip("/")
    return uri, ""


def compute_marginals(uri, cache_path=None):
    """Per-bin sum of all contacts with one endpoint in that bin (diagonal counted once).

    Streams the pixel table so memory stays flat regardless of library size.  Cached to
    `cache_path`, because this is the only expensive step and every threshold sweep reuses it.
    """
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path)
        print(f"[marginals] loaded cache {cache_path}")
        return d["marg"], d["diag"]

    clr = cooler.Cooler(uri)
    n = clr.info["nbins"]
    marg = np.zeros(n, dtype=np.float64)
    diag = np.zeros(n, dtype=np.float64)

    path, grp = split_uri(uri)
    t0 = time.time()
    with h5py.File(path, "r") as f:
        px = f[f"{grp}/pixels"] if grp else f["pixels"]
        total = px["bin1_id"].shape[0]
        for start in range(0, total, CHUNK):
            end = min(start + CHUNK, total)
            b1 = px["bin1_id"][start:end]
            b2 = px["bin2_id"][start:end]
            cnt = px["count"][start:end].astype(np.float64)

            marg += np.bincount(b1, weights=cnt, minlength=n)
            marg += np.bincount(b2, weights=cnt, minlength=n)

            # coolers store symmetric-upper, so a diagonal pixel was just added twice
            on_diag = b1 == b2
            if on_diag.any():
                d = np.bincount(b1[on_diag], weights=cnt[on_diag], minlength=n)
                diag += d
                marg -= d
            print(f"[marginals] {end:,}/{total:,}", flush=True)

    print(f"[marginals] done in {time.time() - t0:.0f}s")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(cache_path, marg=marg, diag=diag)
        print(f"[marginals] cached -> {cache_path}")
    return marg, diag


def chrom_offsets(clr):
    """Map chromosome -> (first bin index, n bins) in the global bin table."""
    off, out = 0, {}
    for chrom, length in clr.chromsizes.items():
        nb = -(-int(length) // clr.binsize)
        out[chrom] = (off, nb)
        off += nb
    return out


def chrom_slice(offsets, chrom):
    lo, nb = offsets[chrom]
    return slice(lo, lo + nb)


def find_runs(mask):
    """Start indices and lengths of consecutive True runs."""
    m = np.asarray(mask, dtype=np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return starts, ends - starts


def mask_ngap(fasta_dir, clr, offsets, chroms, ngap_frac):
    """Bins whose reference sequence is more than `ngap_frac` ambiguous bases."""
    import pyfaidx
    out = np.zeros(clr.info["nbins"], dtype=bool)
    res = clr.binsize
    total_n = 0
    for chrom in chroms:
        fa_path = os.path.join(fasta_dir, f"{chrom}.fa")
        if not os.path.exists(fa_path):
            print(f"[ngap] WARNING no FASTA for {chrom}, skipping")
            continue
        fa = pyfaidx.Fasta(fa_path)
        seq = np.frombuffer(str(fa[chrom][:]).upper().encode(), dtype="S1")
        is_n = seq == b"N"
        total_n += int(is_n.sum())
        nb_seq = -(-len(seq) // res)
        padded = np.zeros(nb_seq * res, dtype=bool)
        padded[:len(seq)] = is_n
        frac = padded.reshape(nb_seq, res).mean(axis=1)
        lo, nb = offsets[chrom]
        out[lo:lo + min(nb, nb_seq)] = frac[:min(nb, nb_seq)] > ngap_frac
    return out, total_n


def gap_intervals(empty, res, min_gap_bp, bridge_bp):
    """Empty runs, merged across short interruptions, filtered to a minimum genomic length.

    `empty` is a boolean array for ONE chromosome.  Returns (start_bp, end_bp) pairs.
    """
    starts, lengths = find_runs(empty)
    if starts.size == 0:
        return []

    merged = []
    cur_s, cur_e = starts[0], starts[0] + lengths[0]
    for s, ln in zip(starts[1:], lengths[1:]):
        if (s - cur_e) * res < bridge_bp:
            cur_e = s + ln
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, s + ln
    merged.append((cur_s, cur_e))

    return [(s * res, e * res) for s, e in merged if (e - s) * res >= min_gap_bp]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cool", required=True, help="finest-resolution cooler URI")
    p.add_argument("--min-gap-bp", type=int, default=5000,
                   help="only stretches at least this long are blacklisted")
    p.add_argument("--bridge-bp", type=int, default=500,
                   help="merge empty runs separated by less than this much signal")
    p.add_argument("--empty-frac", type=float, default=0.0,
                   help="a bin counts as empty at or below this fraction of the per-chromosome "
                        "median nonzero marginal (0 = require an exact zero)")
    p.add_argument("--fasta-dir", default=None, help="union in assembly N-gaps")
    p.add_argument("--encode-blacklist", default=None,
                   help="optional; NOT recommended for Micro-C, see module docstring")
    p.add_argument("--chroms", nargs="+", default=DM6_MAIN)
    p.add_argument("--out", required=True)
    p.add_argument("--report", default=None)
    p.add_argument("--marginals-cache", default=None)
    p.add_argument("--sweep", action="store_true",
                   help="also report mask size across a range of --min-gap-bp")
    args = p.parse_args()

    clr = cooler.Cooler(args.cool)
    res = clr.binsize
    offsets = chrom_offsets(clr)
    chroms = [c for c in args.chroms if c in offsets]
    print(f"[init] {args.cool}  base resolution {res} bp  nbins {clr.info['nbins']:,}")

    cache = args.marginals_cache or os.path.join(
        os.path.dirname(args.out) or ".", "cache", f"marginals_{res}.npz")
    marg, _ = compute_marginals(args.cool, cache)

    ngap = None
    if args.fasta_dir:
        ngap, _ = mask_ngap(args.fasta_dir, clr, offsets, chroms, 0.5)

    enc_iv = {}
    if args.encode_blacklist:
        bl = pd.read_csv(args.encode_blacklist, sep="\t", header=None, usecols=[0, 1, 2],
                         names=["chrom", "start", "end"])
        for chrom, sub in bl.groupby("chrom"):
            enc_iv[chrom] = list(zip(sub.start, sub.end))

    rows, per_chrom = [], []
    for chrom in chroms:
        sl = chrom_slice(offsets, chrom)
        m = marg[sl]
        nz = m[m > 0]
        thr = args.empty_frac * (np.median(nz) if nz.size else 0)
        empty = m <= thr
        if ngap is not None:
            empty |= ngap[sl]

        ivs = gap_intervals(empty, res, args.min_gap_bp, args.bridge_bp)
        ivs += enc_iv.get(chrom, [])
        ivs.sort()

        merged = []
        for s, e in ivs:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        L = int(clr.chromsizes[chrom])
        covered = 0
        for s, e in merged:
            e = min(e, L)
            rows.append((chrom, s, e))
            covered += e - s
        per_chrom.append((chrom, L, len(merged), covered, 100 * covered / L,
                          max((e - s for s, e in merged), default=0)))

    bed = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    bed.to_csv(args.out, sep="\t", header=False, index=False)

    out = []
    out.append(f"Gap blacklist -- {args.cool}")
    out.append("=" * 78)
    out.append(f"detection resolution {res} bp   min gap {args.min_gap_bp:,} bp   "
               f"bridge {args.bridge_bp} bp   empty-frac {args.empty_frac}")
    out.append(f"assembly N-gaps: {'included' if args.fasta_dir else 'no'}   "
               f"ENCODE: {'INCLUDED' if args.encode_blacklist else 'excluded (recommended)'}")
    out.append("")
    out.append(f"{'chrom':8} {'length':>12} {'gaps':>7} {'masked bp':>12} {'%':>7} {'longest':>11}")
    tot_bp = tot_len = 0
    for chrom, L, n, cov, pct, mx in per_chrom:
        out.append(f"{chrom:8} {L:12,} {n:7,} {cov:12,} {pct:6.2f}% {mx / 1000:9.1f}kb")
        tot_bp += cov
        tot_len += L
    out.append(f"{'TOTAL':8} {tot_len:12,} {len(bed):7,} {tot_bp:12,} "
               f"{100 * tot_bp / tot_len:6.2f}%")
    out.append("")

    if args.sweep:
        out.append("-- min-gap sweep " + "-" * 60)
        out.append(f"{'min gap':>10} {'intervals':>11} {'masked bp':>13} {'% genome':>10}")
        for mg in [1000, 2000, 5000, 10000, 20000, 50000]:
            n = bp = 0
            for chrom in chroms:
                sl = chrom_slice(offsets, chrom)
                m = marg[sl]
                nz = m[m > 0]
                empty = m <= args.empty_frac * (np.median(nz) if nz.size else 0)
                if ngap is not None:
                    empty |= ngap[sl]
                iv = gap_intervals(empty, res, mg, args.bridge_bp)
                n += len(iv)
                bp += sum(e - s for s, e in iv)
            mark = "  <- selected" if mg == args.min_gap_bp else ""
            out.append(f"{mg:10,} {n:11,} {bp:13,} {100 * bp / tot_len:9.2f}%{mark}")
        out.append("")

    report = "\n".join(out)
    print(report)
    print(f"\n[out] {args.out}: {len(bed):,} intervals")
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as fh:
            fh.write(report + "\n")
        print(f"[out] {args.report}")


if __name__ == "__main__":
    sys.exit(main())
