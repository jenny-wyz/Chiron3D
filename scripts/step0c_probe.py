#!/usr/bin/env python3
"""Step 0c: linear probes on the embeddings dumped by step0b.

Test 1 (fingerprint): ridge regression  k-mer frequencies (80) -> embedding (128).  Fit on the training
    chromosome, R2 on the test chromosome.  High R2 (> ~0.8) means the bin embedding is mostly a composition
    fingerprint of the bin.
Test 2 (whose fault): ridge regression  embedding -> observed insulation score, fit on the training
    chromosome, Pearson r on the test chromosome (interior bins only).  Compared with
      (a) the same probe from k-mer frequencies only, and
      (b) the model's own predicted insulation on the test chromosome (if --dump is given).

Usage (CPU is fine, from the repo root):
    python3 scripts/step0c_probe.py --train results/step0/emb_v4c_s80_chr3R.npz \
        --test results/step0/emb_v4c_s80_chrX.npz --dump dump_dcnn_v4c_s80_chrX
"""
import argparse, glob, os, sys
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.getcwd())

LAMBDAS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]


def ridge_fit(X, Y, lam):
    XtX = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ Y)


def standardize(train, *others):
    mu, sd = train.mean(0), train.std(0) + 1e-8
    return [(t - mu) / sd for t in (train,) + others]


def r2(Y, Yhat):
    ss_res = ((Y - Yhat) ** 2).sum(0); ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    per_dim = 1 - ss_res / np.maximum(ss_tot, 1e-12)
    return float(per_dim.mean()), float(1 - ss_res.sum() / ss_tot.sum())


def fit_select(Xtr, Ytr, Xte, Yte, n_val_windows, per_window):
    """Choose lambda on the last 20% of training windows, refit on all, evaluate on test."""
    cut = len(Xtr) - n_val_windows * per_window
    best, best_lam = -np.inf, None
    for lam in LAMBDAS:
        W = ridge_fit(Xtr[:cut], Ytr[:cut], lam)
        s = r2(Ytr[cut:], Xtr[cut:] @ W)[1]
        if s > best:
            best, best_lam = s, lam
    W = ridge_fit(Xtr, Ytr, best_lam)
    return W, best_lam, Xte @ W


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def flat(a):
    return a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--dump", default=None, help="evaluation dump dir of the TEST chromosome (optional)")
    a = ap.parse_args()
    tr, te = load(a.train), load(a.test)
    N = int(tr["n_bins"]); res = int(tr["resolution"])
    Wtr, Wte = tr["emb_mean"].shape[0], te["emb_mean"].shape[0]
    n_val = max(1, Wtr // 5)
    print(f"train windows {Wtr}, test windows {Wte}, N={N}")

    # ---------- Test 1: k-mer -> embedding ----------
    Ktr, Kte = flat(tr["kmer"]).astype(np.float64), flat(te["kmer"]).astype(np.float64)
    Ktr_s, Kte_s = standardize(Ktr, Kte)
    print("\nTest 1  fingerprint: k-mer frequencies -> embedding (ridge, fit train chrom, R2 on test chrom)")
    for key in ("emb_mean", "emb_max"):
        Etr, Ete = flat(tr[key]).astype(np.float64), flat(te[key]).astype(np.float64)
        Etr_s, Ete_s = standardize(Etr, Ete)
        W, lam, pred = fit_select(Ktr_s, Etr_s, Kte_s, Ete_s, n_val, N)
        m, tot = r2(Ete_s, pred)
        tag = "(what the model uses)" if key == "emb_mean" else "(preview, GELU+max through same projector)"
        print(f"  {key:9s} {tag:45s} R2 mean over dims = {m:.3f}   variance-weighted R2 = {tot:.3f}   lambda={lam:g}")

    # ---------- Test 2: -> observed insulation ----------
    print("\nTest 2  whose fault: X -> observed insulation (interior bins), fit train chrom, Pearson r on test chrom")
    itr = np.tile(tr["interior"], Wtr); ite = np.tile(te["interior"], Wte)
    ytr, yte = flat(tr["ins_obs"]).astype(np.float64), flat(te["ins_obs"]).astype(np.float64)
    ok_tr = itr & np.isfinite(ytr); ok_te = ite & np.isfinite(yte)
    ytr_l, yte_l = np.log(ytr[ok_tr]), np.log(yte[ok_te])        # insulation ratio: log makes it symmetric
    feats = {"kmer only": (Ktr_s, Kte_s), "emb_mean": None, "emb_max": None}
    for key in ("emb_mean", "emb_max"):
        Etr, Ete = flat(tr[key]).astype(np.float64), flat(te[key]).astype(np.float64)
        feats[key] = standardize(Etr, Ete)
    feats["kmer + emb_mean"] = (np.concatenate([Ktr_s, feats["emb_mean"][0]], 1),
                                np.concatenate([Kte_s, feats["emb_mean"][1]], 1))
    rows = []
    for name, (Xtr, Xte) in feats.items():
        Xtr_, Xte_ = Xtr[ok_tr], Xte[ok_te]
        n_val_rows = int(0.2 * len(Xtr_))
        best, best_lam = -np.inf, None
        for lam in LAMBDAS:
            W = ridge_fit(Xtr_[:-n_val_rows], ytr_l[:-n_val_rows], lam)
            s = pearsonr(Xtr_[-n_val_rows:] @ W, ytr_l[-n_val_rows:])[0]
            if s > best:
                best, best_lam = s, lam
        W = ridge_fit(Xtr_, ytr_l, best_lam)
        r_tr = pearsonr(Xtr_ @ W, ytr_l)[0]; r_te = pearsonr(Xte_ @ W, yte_l)[0]
        rows.append((name, r_tr, r_te, best_lam))
    print(f"  {'features':18s} {'r train':>8} {'r test':>8}  lambda")
    for name, r_tr, r_te, lam in rows:
        print(f"  {name:18s} {r_tr:8.3f} {r_te:8.3f}  {lam:g}")

    if a.dump:
        from src.models.evaluation.metrics import chr_score
        f = sorted(glob.glob(os.path.join(a.dump, "matrices_*.npz")))
        z = np.load(f[0]); P = z["pred"].astype(np.float32)
        pix = int(25000 / res)
        rs = []
        for w in range(len(P)):
            p = np.asarray(chr_score(P[w], res=res, radius=25000))[pix:N - pix]
            o = te["ins_obs"][w][pix:N - pix]
            ok = np.isfinite(p) & np.isfinite(o) & (p > 0) & (o > 0)
            rs.append(pearsonr(np.log(p[ok]), np.log(o[ok]))[0])
        p_all = np.concatenate([np.asarray(chr_score(P[w], res=res, radius=25000))[pix:N - pix] for w in range(len(P))])
        o_all = np.concatenate([te["ins_obs"][w][pix:N - pix] for w in range(len(P))])
        ok = np.isfinite(p_all) & np.isfinite(o_all) & (p_all > 0) & (o_all > 0)
        print(f"\n  model's own predicted insulation vs observed on test chrom: "
              f"per-window mean r = {np.mean(rs):.3f}, pooled r = {pearsonr(np.log(p_all[ok]), np.log(o_all[ok]))[0]:.3f}")

    print("\nHow to read it:")
    print("  Test 1 R2 > ~0.8      -> embeddings are composition fingerprints (pooling is the problem; steps 1-2).")
    print("  Test 2 emb r_test >= model's own r -> head (transformer+decoder) overfits; regularization (step 3) matters most.")
    print("  Test 2 emb r_test ~ kmer-only r     -> trunk+pool carry nothing beyond composition; tower (step 2) essential.")
    print("  Test 2 r_train >> r_test            -> even the linear probe memorizes: features are window-specific.")


if __name__ == "__main__":
    main()