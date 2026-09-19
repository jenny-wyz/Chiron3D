import numpy as np, pandas as pd, glob, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DUMP  = sys.argv[1]                       # e.g. dump_dcnn_v2
LOOPS = sys.argv[2] if len(sys.argv) > 2 else "data/all_loops.tsv"
PAD   = 10                                # +/- 10 bins = +/- 8 kb

lp = pd.read_csv(LOOPS, sep="\t")
g = (lp.groupby("loop.ID")
       .agg(chr=("anchor.chr", "first"), nchr=("anchor.chr", "nunique"),
            s1=("anchor.summit", "min"), s2=("anchor.summit", "max"),
            typ=("loop.type", "first"))
       .reset_index())
g = g[(g.nchr == 1) & (g.typ == "intra_TAD")]

obs_patches, pred_patches = [], []
for f in sorted(glob.glob(os.path.join(DUMP, "matrices_*.npz"))):
    z = np.load(f, allow_pickle=True)
    P, O = z["pred"].astype(np.float32), z["obs"].astype(np.float32)
    res, n = int(z["resolution"]), int(z["n_bins"])
    chroms, starts = z["chrom"], z["region_start"]
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    # remove distance decay so the patch isn't just a gradient
    Oe = O - np.array([O[:, d == k].mean() for k in range(n)])[d]
    Pe = P - np.array([P[:, d == k].mean() for k in range(n)])[d]
    for w in range(len(P)):
        sub = g[g.chr == str(chroms[w])]
        i = (sub.s1.values - starts[w]) // res
        j = (sub.s2.values - starts[w]) // res
        ok = (i >= PAD) & (j >= PAD) & (i < n - PAD) & (j < n - PAD) & (j - i >= 5)
        for a, b in zip(i[ok], j[ok]):
            obs_patches.append (Oe[w, a-PAD:a+PAD+1, b-PAD:b+PAD+1])
            pred_patches.append(Pe[w, a-PAD:a+PAD+1, b-PAD:b+PAD+1])

A_o, A_p = np.mean(obs_patches, 0), np.mean(pred_patches, 0)

def enrich(A):
    centre  = A[PAD-1:PAD+2, PAD-1:PAD+2].mean()
    corners = np.concatenate([A[:5,:5].ravel(), A[:5,-5:].ravel(),
                              A[-5:,:5].ravel(), A[-5:,-5:].ravel()])
    return centre - corners.mean()

print(f"{len(obs_patches)} loop instances (loops counted once per overlapping window)")
print(f"observed  APA enrichment: {enrich(A_o):+.4f}")
print(f"predicted APA enrichment: {enrich(A_p):+.4f}")

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for a, A, t in zip(ax, [A_o, A_p], ["observed", "predicted"]):
    im = a.imshow(A, cmap="Reds"); a.set_title(f"{t}  ({enrich(A):+.3f})")
    plt.colorbar(im, ax=a, fraction=0.046)
plt.tight_layout(); plt.savefig(f"apa_{os.path.basename(DUMP)}.png", dpi=150)
print(f"wrote apa_{os.path.basename(DUMP)}.png")


# python3 scripts/apa.py dump_dcnn_v2 /cluster/work/boeva/Gambetta_collaboration/Loops/all_loops.tsv
# python3 scripts/apa.py dump_borzoi /cluster/work/boeva/Gambetta_collaboration/Loops/all_loops.tsv