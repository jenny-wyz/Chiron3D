import sys
def edit(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        if new.strip() in s: print(f"already patched: {path}"); return
        if old not in s: sys.exit(f"ABORT: expected text not found in {path}:\n{old}")
        s = s.replace(old, new, 1)
    open(path, "w").write(s); print("patched", path)

edit("src/models/dataset/augments.py", [(
'''def gaussian_noise(inputs, std=0.1):''',
'''def reverse_complement_sample(output, chance=0.5):
    """Correct reverse-complement augmentation for Chiron3D samples (v5a).
    sequence is (4, L) in A,C,G,T order: reverse along L and swap A<->T, C<->G.
    matrix, mask and loop_hm are (N, N): flipping the genome flips both axes."""
    if torch.rand(1).item() >= chance:
        return output
    seq = torch.flip(output["sequence"], dims=[1])[[3, 2, 1, 0], :]
    output = dict(output)
    output["sequence"] = seq
    for k in ("matrix", "mask", "loop_hm"):
        if k in output and torch.is_tensor(output[k]):
            output[k] = torch.flip(output[k], dims=[0, 1])
    return output


def gaussian_noise(inputs, std=0.1):''')])

edit("src/models/dataset/genomic_dataset.py", [
('''from src.models.dataset.augments import shift_aug, reverse_complement, gaussian_noise''',
 '''from src.models.dataset.augments import shift_aug, reverse_complement, gaussian_noise, reverse_complement_sample'''),
('''        output = self.build_output(chrom, output)
''',
 '''        output = self.build_output(chrom, output)
        if self.mode == "train" and self.use_aug:
            output = reverse_complement_sample(output, chance=0.5)
''')])

edit("src/models/training/module.py", [(
'''            loop_file=getattr(args, 'loop_file', None),
''',
'''            loop_file=getattr(args, 'loop_file', None),
            use_aug=bool(getattr(args, 'use_aug', False)) and mode == 'train',
''')])

edit("src/models/training/train.py", [(
'''    parser.add_argument('--oe-target', dest='oe_target', action='store_true',
''',
'''    parser.add_argument('--use-aug', dest='use_aug', action='store_true',
                        help='v5a: random reverse-complement of sequence + map (train set only)')
    parser.add_argument('--oe-target', dest='oe_target', action='store_true',
''')])
print("ALL PATCHES APPLIED")
