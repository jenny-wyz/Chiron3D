#!/usr/bin/env python3
"""Step 0b: dump the 800 bp bin embeddings of a trained Chiron3D_DCNN checkpoint, plus per-bin k-mer
counts and the observed insulation score, for one or more chromosomes.

For every window it saves
    emb_mean  (N, 128)  the tensor the transformer actually receives (whatever pooling the checkpoint uses)
    emb_max   (N, 128)  v4 checkpoints only: preview of GELU + max pool through the same trained projector
    kmer      (N, 80)   2-mer (16) + 3-mer (64) frequencies of the 800 bp bin (A,C,G,T order)
    ins_obs   (N,)      observed insulation score (chr_score, radius 25 kb) of the window's target map
    interior  (N,) bool bins whose insulation diamond is not clipped (use these for probes)

Needs a GPU (sbatch it, see step0.sbatch). Usage from the repo root:
    python3 scripts/step0b_dump_embeddings.py \
        --ckpt checkpoints_dcnn_v4c_s80/models/epoch=XX-step=YYYY.ckpt \
        --regions-file data/windows_dm6_C523200_f1024.bed \
        --fasta-dir data/dmel_chromosomes \
        --cool-file /cluster/work/boeva/Gambetta_collaboration/Micro-C/larval_brain_merge.mcool::resolutions/800 \
        --chroms chr3R chrX --out results/step0/emb_v4c_s80
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
torch.serialization.add_safe_globals([argparse.Namespace])
from src.models.training.module import TrainModule                 # noqa: E402
from src.models.dataset.genomic_dataset import GenomicDataset       # noqa: E402
from src.models.evaluation.metrics import chr_score                 # noqa: E402

RADIUS = 25000


def kmer_features(onehot, n_bins, pool_bp):
    """onehot: (4, span) numpy, A,C,G,T. Returns (n_bins, 80) frequencies of 2-mers and 3-mers."""
    present = onehot.sum(0) > 0
    code = onehot.argmax(0).astype(np.int64)
    code[~present] = -1
    feats = []
    for k, nk in ((2, 16), (3, 64)):
        idx = np.zeros(len(code) - k + 1, np.int64); ok = np.ones(len(code) - k + 1, bool)
        for j in range(k):
            c = code[j:len(code) - k + 1 + j]
            idx = idx * 4 + np.maximum(c, 0); ok &= c >= 0
        idx[~ok] = -1
        out = np.zeros((n_bins, nk), np.float32)
        for b in range(n_bins):
            seg = idx[b * pool_bp:(b + 1) * pool_bp]
            seg = seg[seg >= 0]
            if len(seg):
                out[b] = np.bincount(seg, minlength=nk) / len(seg)
        feats.append(out)
    return np.concatenate(feats, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--regions-file", required=True)
    ap.add_argument("--fasta-dir", required=True)
    ap.add_argument("--cool-file", required=True)
    ap.add_argument("--chroms", nargs="+", default=["chr3R", "chrX"])
    ap.add_argument("--out", required=True, help="output prefix; writes <out>_<chrom>.npz")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = TrainModule.load_from_checkpoint(a.ckpt, map_location=dev).to(dev).eval()
    ck = module.hparams["args"]
    m = module.model
    print(f"[step0b] trunk={ck.trunk} res={ck.resolution} N={ck.n_bins} flank={ck.flank} "
          f"balance={getattr(ck,'balance',False)} scale={getattr(ck,'matrix_scale',1.0)} "
          f"oe={getattr(ck,'oe_target',False)}")
    assert ck.trunk == "dcnn"
    res, N, pool = int(ck.resolution), int(ck.n_bins), int(m.pool_factor)
    pix = int(RADIUS / res)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    for chrom in a.chroms:
        ds = GenomicDataset(regions_file_path=a.regions_file, cool_file_path=a.cool_file,
                            fasta_dir=a.fasta_dir, mode="test", test_chroms=[chrom],
                            use_pretrained_backbone=True, resolution=res, n_bins=N, flank=int(ck.flank),
                            balance=getattr(ck, "balance", False),
                            matrix_scale=getattr(ck, "matrix_scale", 1.0))   # NOTE: no oe_target: raw log map
        E_mean, E_max, KM, INS, ST = [], [], [], [], []
        for i in range(len(ds)):
            item = ds[i]
            x = item["sequence"].unsqueeze(0).to(dev)
            with torch.no_grad():
                if x.shape[-1] != m.input_width:
                    off = (x.shape[-1] - m.input_width) // 2
                    x = x[..., off:off + m.input_width]
                h = m.trunk(x)
                h = h[..., m.trim:h.shape[-1] - m.trim]
                b, c, l = h.shape
                mode = getattr(m, "pool_mode", "mean")
                if mode == "mean":                               # v4: project, then average
                    hp = m.projector(h)
                    e_mean = hp.view(b, 128, N, pool).mean(-1)[0].T          # what the model uses (N, 128)
                    e_max = F.gelu(hp).view(b, 128, N, pool).amax(-1)[0].T   # preview only
                else:                                            # v6a: GELU, pool, project
                    g = F.gelu(h).view(b, c, N, pool)
                    pooled = g.amax(-1) if mode == "max" else torch.cat([g.mean(-1), g.amax(-1)], 1)
                    e_mean = m.projector(pooled)[0].T            # what the model uses (N, 128)
                    e_max = e_mean                               # no separate preview for v6a models
            E_mean.append(e_mean.float().cpu().numpy().astype(np.float16))
            E_max.append(e_max.float().cpu().numpy().astype(np.float16))
            seq = item["sequence"].numpy()[:, ds.flank:ds.flank + N * res]
            KM.append(kmer_features(seq, N, res))
            ins = np.asarray(chr_score(item["matrix"].numpy(), res=res, radius=RADIUS), np.float32)
            INS.append(ins)
            ST.append(int(item["region_start"]))
            if i % 20 == 0:
                print(f"  {chrom} window {i}/{len(ds)}")
        interior = np.zeros(N, bool); interior[pix:N - pix] = True
        path = f"{a.out}_{chrom}.npz"
        np.savez_compressed(path, emb_mean=np.stack(E_mean), emb_max=np.stack(E_max), kmer=np.stack(KM),
                            ins_obs=np.stack(INS), interior=interior, region_start=np.array(ST),
                            resolution=res, n_bins=N)
        print(f"[step0b] wrote {path}: {len(ST)} windows")


if __name__ == "__main__":
    main()