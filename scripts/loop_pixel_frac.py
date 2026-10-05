import pandas as pd, numpy as np
lp = pd.read_csv("/cluster/work/boeva/Gambetta_collaboration/Loops/all_loops.tsv", sep="\t")
g = (lp.groupby("loop.ID").agg(chr=("anchor.chr","first"), nchr=("anchor.chr","nunique"),
                               s1=("anchor.summit","min"), s2=("anchor.summit","max"),
                               typ=("loop.type","first")).reset_index())
g = g[(g.nchr == 1) & (g.typ == "intra_TAD") & (~g.chr.isin(["chrX"]))]
bed = pd.read_csv("data/windows_dm6_C523200_f1024.bed", sep="\t", names=["chr","start","end"])
bed = bed[~bed.chr.isin(["chr2L","chrX"])]
n = 0
for r in bed.itertuples():
    sub = g[(g.chr == r.chr) & (g.s1 >= r.start) & (g.s2 < r.end)]
    n += len(sub) * 18          # 3x3 blob, both triangles
print("positive pixels per window:", n / len(bed), " fraction:", n / len(bed) / 654**2)