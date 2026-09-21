import numpy as np, pandas as pd, glob, os, sys
DUMP  = sys.argv[1]
LOOPS = sys.argv[2] if len(sys.argv) > 2 else "data/all_loops.tsv"
PAD, SHIFT = 10, 40                      # SHIFT bins along the diagonal = distance-matched null

lp = pd.read_csv(LOOPS, sep="\t")
g = (lp.groupby("loop.ID")
       .agg(chr=("anchor.chr","first"), nchr=("anchor.chr","nunique"),
            s1=("anchor.summit","min"), s2=("anchor.summit","max"),
            typ=("loop.type","first")).reset_index())
g = g[(g.nchr == 1) & (g.typ == "intra_TAD")]

ids, real, null = [], [], []
for f in sorted(glob.glob(os.path.join(DUMP, "matrices_*.npz"))):
    z = np.load(f, allow_pickle=True)
    P = z["pred"].astype(np.float32); res = int(z["resolution"]); n = int(z["n_bins"])
    chroms, starts = z["chrom"], z["region_start"]
    d = np.abs(np.arange(n)[:,None] - np.arange(n)[None,:])
    Pe = P - np.array([P[:, d == k].mean() for k in range(n)])[d]
    for w in range(len(P)):
        sub = g[g.chr == str(chroms[w])]
        i = (sub.s1.values - starts[w]) // res
        j = (sub.s2.values - starts[w]) // res
        ok = (i >= PAD) & (j >= PAD) & (i < n-PAD) & (j < n-PAD) & (j-i >= 5)
        for lid, a, b in zip(sub["loop.ID"].values[ok], i[ok], j[ok]):
            ids.append(lid); real.append(Pe[w, a-PAD:a+PAD+1, b-PAD:b+PAD+1])
            for s in (SHIFT, -SHIFT):
                if PAD <= a+s < n-PAD and PAD <= b+s < n-PAD:
                    null.append(Pe[w, a+s-PAD:a+s+PAD+1, b+s-PAD:b+s+PAD+1])

def enrich(A):
    c = A[PAD-1:PAD+2, PAD-1:PAD+2].mean()
    k = np.concatenate([A[:5,:5].ravel(), A[:5,-5:].ravel(), A[-5:,:5].ravel(), A[-5:,-5:].ravel()])
    return c - k.mean()

real = np.array(real); ids = np.array(ids); uniq = np.unique(ids)
rng = np.random.default_rng(0)
boot = [enrich(real[np.isin(ids, rng.choice(uniq, len(uniq), replace=True))].mean(0))
        for _ in range(1000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"{DUMP}: {len(real)} instances from {len(uniq)} loops")
print(f"  predicted APA : {enrich(real.mean(0)):+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"  shifted null  : {enrich(np.array(null).mean(0)):+.4f}  ({len(null)} patches)")


# python3 scripts/apa_stats.py dump_dcnn_v2 /cluster/work/boeva/Gambetta_collaboration/Loops/all_loops.tsv
# python3 scripts/apa_stats.py dump_borzoi_f1024 /cluster/work/boeva/Gambetta_collaboration/Loops/all_loops.tsv