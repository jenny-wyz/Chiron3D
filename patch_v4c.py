import sys
def edit(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        if new.strip() in s: print(f"already patched: {path}"); return
        if old not in s: sys.exit(f"ABORT: expected text not found in {path}:\n{old}")
        s = s.replace(old, new, 1)
    open(path, "w").write(s); print("patched", path)

edit("src/models/training/module.py", [
('''        self.args = args
        self.save_hyperparameters()
''',
 '''        self.args = args
        self.save_hyperparameters()
        # v4c: optional per-distance loss weights (one value per diagonal), broadcast to an (N, N) buffer
        self.dist_w = None
        dw = getattr(args, "dist_weight_file", None)
        if dw:
            if os.path.exists(dw):
                w = np.load(dw).astype(np.float32)
                idx = np.abs(np.arange(len(w))[:, None] - np.arange(len(w))[None, :])
                self.register_buffer("dist_w", torch.tensor(w[idx]), persistent=False)
                print(f"[TrainModule] distance loss weights from {dw}: "
                      f"w(d=2)={w[2]:.3f} w(d=last)={w[-1]:.3f} ({len(w)} diagonals)")
            else:
                print(f"[TrainModule] WARNING: {dw} not found; training WITHOUT distance weights")
'''),
('''        mask = batch["mask"].to(outputs.device)
        if getattr(self.args, "loop_file", None):
''',
 '''        mask = batch["mask"].to(outputs.device)
        if self.dist_w is not None:
            mask = mask * self.dist_w                # weighted, masked MSE: sum(w*err^2) / sum(w)
        if getattr(self.args, "loop_file", None):
''')])

edit("src/models/training/train.py", [
('''    parser.add_argument('--bad-bins-file', dest='bad_bins_file', default=None,
''',
 '''    parser.add_argument('--dist-weight-file', dest='dist_weight_file', default=None,
                        help='v4c: .npy with one loss weight per diagonal (see scripts/make_dist_weights.py)')
    parser.add_argument('--bad-bins-file', dest='bad_bins_file', default=None,
''')])
print("ALL PATCHES APPLIED")
