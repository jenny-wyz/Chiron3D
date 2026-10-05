import sys
def edit(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        if new.strip() in s:
            print(f"already patched: {path}"); return
        if old not in s:
            sys.exit(f"ABORT: expected text not found in {path}:\n{old}")
        s = s.replace(old, new, 1)
    open(path, "w").write(s); print("patched", path)

edit("src/models/dataset/utils.py", [(
'''def get_matrix(cool, chrom, start, end):
    matrix = cool.matrix(balance=False).fetch(f"{chrom}:{start}-{end}")
    matrix = torch.tensor(matrix, dtype=torch.float32)
''',
'''def get_matrix(cool, chrom, start, end, balance=False, matrix_scale=1.0):
    """log(count + 1) of the raw map, or, with balance=True, log(balanced * matrix_scale + 1).
    matrix_scale = 1/psi puts ICE-balanced values back on a 'counts' scale so the same
    pseudocount logic applies (identical to Chimaera's log(balanced + psi) up to a constant).
    Balanced NaN pixels (bad bins) become 0 here and must be masked out by the caller."""
    matrix = cool.matrix(balance=balance).fetch(f"{chrom}:{start}-{end}")
    matrix = np.asarray(matrix, dtype=np.float32)
    if balance:
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0) * float(matrix_scale)
    matrix = torch.tensor(matrix, dtype=torch.float32)
''')])

edit("src/models/dataset/genomic_dataset.py", [
('''                 use_aug=False, resolution=400, n_bins=256, flank=None, oe_target=False,
                 loop_file=None):''',
 '''                 use_aug=False, resolution=400, n_bins=256, flank=None, oe_target=False,
                 loop_file=None, balance=False, matrix_scale=1.0, expected_file=None, bad_bins_file=None):'''),
('''        self.oe_target = oe_target
        if oe_target:
            prof = np.load("data/expected_log_800.npy")
            d = np.abs(np.arange(self.n_bins)[:, None] - np.arange(self.n_bins)[None, :])
            self.expected_2d = torch.tensor(prof[d], dtype=torch.float32)
            z = np.load("data/bad_bins_800.npz", allow_pickle=True)
            self.bad_bins = {"chrom": z["chrom"], "start": z["start"], "bad": z["bad"]}
        else:
            self.expected_2d = None
''',
 '''        self.oe_target = oe_target
        self.balance = bool(balance)
        self.matrix_scale = float(matrix_scale)
        self.bad_bins = None
        self.expected_2d = None
        if bad_bins_file is None and (oe_target or self.balance):
            bad_bins_file = "data/bad_bins_800_bal.npz" if self.balance else "data/bad_bins_800.npz"
        if expected_file is None and oe_target:
            expected_file = "data/expected_log_800_bal.npy" if self.balance else "data/expected_log_800.npy"
        if oe_target:
            prof = np.load(expected_file)
            assert len(prof) >= self.n_bins, f"{expected_file} has {len(prof)} diagonals, need {self.n_bins}"
            d = np.abs(np.arange(self.n_bins)[:, None] - np.arange(self.n_bins)[None, :])
            self.expected_2d = torch.tensor(prof[d], dtype=torch.float32)
        if oe_target or self.balance:
            z = np.load(bad_bins_file, allow_pickle=True)
            self.bad_bins = {"chrom": z["chrom"], "start": z["start"], "bad": z["bad"]}
            print(f"[GenomicDataset] mode={mode} balance={self.balance} scale={self.matrix_scale:g} "
                  f"oe_target={oe_target} expected={expected_file} bad_bins={bad_bins_file}")
'''),
('''        matrix = get_matrix(self.cool, chrom, target_start, target_end)
''',
 '''        matrix = get_matrix(self.cool, chrom, target_start, target_end,
                            balance=self.balance, matrix_scale=self.matrix_scale)
'''),
('''        if self.oe_target:
            matrix = matrix - self.expected_2d
            bb = self.bad_bins
''',
 '''        if self.oe_target:
            matrix = matrix - self.expected_2d
        if self.bad_bins is not None:
            bb = self.bad_bins
''')])

edit("src/models/training/train.py", [(
'''    parser.add_argument('--oe-target', dest='oe_target', action='store_true',
                        help='predict log observed/expected with unmappable bins masked')
''',
'''    parser.add_argument('--oe-target', dest='oe_target', action='store_true',
                        help='predict log observed/expected with unmappable bins masked')
    # v4: ICE-balanced target
    parser.add_argument('--balance', dest='balance', action='store_true',
                        help='use ICE-balanced contacts (cool must have a weight column); bad bins masked')
    parser.add_argument('--matrix-scale', dest='matrix_scale', type=float, default=1.0,
                        help='with --balance: target = log(balanced * scale + 1); use 1/psi (e.g. 2621.2)')
    parser.add_argument('--expected-file', dest='expected_file', default=None,
                        help='per-diagonal mean profile for --oe-target (default picks raw/bal file)')
    parser.add_argument('--bad-bins-file', dest='bad_bins_file', default=None,
                        help='npz of bad bins to mask (default picks raw/bal file)')
''')])

edit("src/models/training/module.py", [(
'''            loop_file=getattr(args, 'loop_file', None),
        )
''',
'''            loop_file=getattr(args, 'loop_file', None),
            balance=getattr(args, 'balance', False),
            matrix_scale=getattr(args, 'matrix_scale', 1.0),
            expected_file=getattr(args, 'expected_file', None),
            bad_bins_file=getattr(args, 'bad_bins_file', None),
        )
''')])

edit("src/models/evaluation/evaluation.py", [
('''    p.add_argument('--loop-file', dest='loop_file', default=None)
''',
 '''    p.add_argument('--loop-file', dest='loop_file', default=None)
    p.add_argument('--balance', dest='balance', action='store_true',
                   help='checkpoint was trained on ICE-balanced data (auto-detected from hparams if present)')
    p.add_argument('--matrix-scale', dest='matrix_scale', type=float, default=None,
                   help='scale used in training (auto-detected from hparams if present)')
    p.add_argument('--expected-file', dest='expected_file', default=None)
    p.add_argument('--bad-bins-file', dest='bad_bins_file', default=None)
'''),
('''    if args.trunk != 'corigami' and getattr(ck, 'oe_target', False):
        args.oe_target = True                      # auto-detect from checkpoint hparams

    expected_2d = None
    if args.oe_target:
        prof = np.load("data/expected_log_800.npy")
''',
 '''    if args.trunk != 'corigami' and getattr(ck, 'oe_target', False):
        args.oe_target = True                      # auto-detect from checkpoint hparams
    if args.trunk != 'corigami' and getattr(ck, 'balance', False):
        args.balance = True                        # auto-detect ICE training from checkpoint hparams
        if args.matrix_scale is None:
            args.matrix_scale = float(getattr(ck, 'matrix_scale', 1.0))
    if args.matrix_scale is None:
        args.matrix_scale = 1.0
    if args.expected_file is None:
        args.expected_file = "data/expected_log_800_bal.npy" if args.balance else "data/expected_log_800.npy"
    print(f"[eval] balance={args.balance} matrix_scale={args.matrix_scale:g} oe_target={args.oe_target} "
          f"expected={args.expected_file}")

    expected_2d = None
    if args.oe_target:
        prof = np.load(args.expected_file)
'''),
('''            flank=None if args.trunk != 'dcnn' else args.flank,
        )
''',
 '''            flank=None if args.trunk != 'dcnn' else args.flank,
            balance=args.balance,                   # observed map in the same (balanced) space as training
            matrix_scale=args.matrix_scale,         # NOTE: oe_target is NOT passed: prediction gets expected added back
            bad_bins_file=args.bad_bins_file,
        )
'''),
('''        dump_pred, dump_obs, dump_chr, dump_start, dump_end = [], [], [], [], []
''',
 '''        dump_pred, dump_obs, dump_chr, dump_start, dump_end, dump_mask = [], [], [], [], [], []
'''),
('''                out = out.squeeze()
                true = true.squeeze()

                if args.dump_matrices:
                    dump_pred.append(out.numpy().astype(np.float16))
                    dump_obs.append(true.numpy().astype(np.float16))
''',
 '''                out = out.squeeze()
                true = true.squeeze()
                keep = batch["mask"][k].cpu().squeeze().numpy() > 0   # False on bad (unmappable) bins

                if args.dump_matrices:
                    dump_pred.append(out.numpy().astype(np.float16))
                    dump_obs.append(true.numpy().astype(np.float16))
                    dump_mask.append(keep)
'''),
('''                l_mse = mse(out, true)
''',
 '''                l_mse = mse(out.numpy()[keep], true.numpy()[keep])   # masked MSE (identical to before when no bad bins)
'''),
('''                pred=P, obs=O,
                chrom=np.array(dump_chr),
''',
 '''                pred=P, obs=O, mask=np.stack(dump_mask),
                chrom=np.array(dump_chr),
''')])
print("ALL PATCHES APPLIED")
