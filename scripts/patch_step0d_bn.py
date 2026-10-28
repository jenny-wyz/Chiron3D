#!/usr/bin/env python3
"""Step 0d: patch src/models/evaluation/evaluation.py so that, when the environment variable
BN_TRAIN=1 is set, the trunk's BatchNorm layers use batch statistics at evaluation time
(momentum 0, so the stored running stats are not modified). Everything else unchanged.

Run once from the repo root:   python3 scripts/patch_step0d_bn.py
Then evaluate twice, into two different dump dirs, and compare with eval_bands_masked.py:
    python3 -m src.models.evaluation.evaluation ... --dump-matrices dump_v4c_s80_chrX_bneval
    BN_TRAIN=1 python3 -m src.models.evaluation.evaluation ... --dump-matrices dump_v4c_s80_chrX_bntrain
"""
import re, sys

path = "src/models/evaluation/evaluation.py"
src = open(path).read()
if "BN_TRAIN" in src:
    print("already patched"); sys.exit(0)

needle = "    model.eval()\n"
assert src.count(needle) == 1, "expected exactly one 'model.eval()' line in evaluation.py main()"
addition = needle + (
    "    if os.environ.get('BN_TRAIN') == '1':            # step 0d: batch-stat BatchNorm in the trunk\n"
    "        n_bn = 0\n"
    "        for mod in model.modules():\n"
    "            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):\n"
    "                mod.train(); mod.momentum = 0.0; n_bn += 1\n"
    "        print(f'[step0d] BN_TRAIN=1: {n_bn} BatchNorm layers use batch statistics (running stats frozen)')\n"
)
src = src.replace(needle, addition)
open(path, "w").write(src)
print("patched", path)