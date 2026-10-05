import numpy as np
z = np.load("dump_dcnn_v3a/matrices_chrX.npz", allow_pickle=True)
print("keys:", list(z.keys()))
P, O = z["pred"].astype(np.float32), z["obs"].astype(np.float32)
print("shapes:", P.shape, O.shape, " res:", int(z["resolution"]), " n_bins:", int(z["n_bins"]))

for name, A in (("obs", O), ("pred", P)):
    print(f"\n{name}:  min {A.min():.6f}  max {A.max():.3f}  mean {A.mean():.4f}")
    print(f"  exactly 0      : {(A == 0).mean()*100:7.3f} %")
    print(f"  < 0.01         : {(A < 0.01).mean()*100:7.3f} %")
    print(f"  < 0.35 (~0 cts): {(A < 0.35).mean()*100:7.3f} %")
    print("  10 smallest values:", np.sort(A.ravel())[:10])

# a far-off-diagonal strip, where the truth is almost all zeros
w, i, j = 0, 100, 500
print(f"\nwindow {w}, row {i}, cols {j}..{j+12}  (distance ~{(j-i)*int(z['resolution'])/1000:.0f} kb)")
print("  obs :", np.round(O[w, i, j:j+12], 3))
print("  pred:", np.round(P[w, i, j:j+12], 3))