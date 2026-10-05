import numpy as np
z = np.load("dump_dcnn_v3a/matrices_chrX.npz")
P, O = z["pred"].astype(float), z["obs"].astype(float)
i = np.arange(654)
print("main diagonal   pred %.3f   obs %.3f" % (P[:, i, i].mean(), O[:, i, i].mean()))
print("whole map       pred %.3f   obs %.3f" % (P.mean(), O.mean()))

n = int(z["n_bins"]); d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
pp = np.array([P[:, d == k].mean() for k in range(n)])
po = np.array([O[:, d == k].mean() for k in range(n)])
print("\nbias by distance:", np.round((pp - po)[[0, 1, 5, 10, 25, 50, 100, 200, 400, 600]], 3))
print("MSE cost of the offset alone:", float(((pp - po)[d] ** 2).mean()))

A = np.load("apa_patches_dump_dcnn_v3a.npz")["pred"]
k = np.arange(-10, 11)
print("\nalong-diagonal (distance-matched):", np.round(A[10 + k, 10 + k], 4))
print("row profile:", np.round(A.mean(1), 4))
print("col profile:", np.round(A.mean(0), 4))