import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch

def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def mse(pred, target):
    pred = _to_np(pred)
    target = _to_np(target)
    return float(np.mean((pred - target) ** 2))

def point_score(locus, radius, matrix, pseudocount):
    n = matrix.shape[0]
    l_edge = max(locus - radius, 0)
    r_edge = min(locus + radius, n)

    L = matrix[l_edge:locus, l_edge:locus]
    R = matrix[locus:r_edge, locus:r_edge]
    C = matrix[l_edge:locus, locus:r_edge]

    if C.size == 0:
        return np.nan

    Lm = np.mean(L) if L.size else np.nan
    Rm = np.mean(R) if R.size else np.nan
    Cm = np.mean(C)  

    num = np.nanmax([Lm, Rm]) + pseudocount
    den = Cm + pseudocount

    if not np.isfinite(den) or den == 0.0:
        return np.nan
    if not np.isfinite(num):
        return np.nan

    return num / den

def chr_score(matrix, res=5000, radius=125000, pseudocount_coeff=30):
    mat = _to_np(matrix).astype(np.float64, copy=False)
    pseudocount = float(mat.mean()) * pseudocount_coeff
    pix = int(radius / res)
    return [point_score(loc, pix, mat, pseudocount) for loc in range(mat.shape[0])]

def insulation_corr(pred, target, res=5000, radius=125000):
    pred_insu  = np.asarray(chr_score(pred, res=res, radius=radius), dtype=np.float64)
    targ_insu  = np.asarray(chr_score(target, res=res, radius=radius), dtype=np.float64)
    m = np.isfinite(pred_insu) & np.isfinite(targ_insu)

    if m.sum() < 2:   
        return np.nan, np.nan

    rp, _ = pearsonr(pred_insu[m], targ_insu[m])
    rs, _ = spearmanr(pred_insu[m], targ_insu[m])
    return float(rp), float(rs)

def distance_stratified_correlation(pred, target, xs, ys, max_offset=None, store_diag=True):
    pears, spears = [], []
    n = len(pred)
    dmax = n if max_offset is None else min(n, max_offset)
    for d in range(dmax):
        x = np.diagonal(pred, offset=d)
        y = np.diagonal(target, offset=d)
        if x.size < 2:
            continue
        if store_diag:
            xs.setdefault(d, []).append(np.asarray(x, dtype=np.float32))
            ys.setdefault(d, []).append(np.asarray(y, dtype=np.float32))
        rp, _ = pearsonr(x, y)
        rs, _ = spearmanr(x, y)
        pears.append(float(rp))
        spears.append(float(rs))
    return pears, spears, xs, ys