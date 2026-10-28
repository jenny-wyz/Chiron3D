#!/usr/bin/env python3
"""Fix the logged validation insulation correlation (pearson_corr_val / spearman_corr_val).

Problem: for --oe-target runs the metric was computed on log(O/E) values, where the insulation
ratio (mean(left/right square) / mean(cross square), both ~0) is undefined and flips sign at random.
That is why pearson_corr_val swings between +0.3 and -0.3 from epoch to epoch in the CSV logs.

Fix: add the expected profile back (like evaluation.py does), clamp at 0, and score the
interior insulation correlation at fly scale (25 kb radius), i.e. the same number evaluation.py prints.
Only affects logging; checkpoint selection and early stopping still use val_loss.
Run once from the repo root:   python3 scripts/patch_valmetric.py
"""
import sys

P = "src/models/training/module.py"
src = open(P).read()
if "val_expected_2d" in src:
    print("already patched"); sys.exit(0)


def rep(old, new):
    global src
    assert src.count(old) == 1, f"expected exactly one occurrence of:\n{old}"
    src = src.replace(old, new)


rep("from src.models.evaluation.metrics import mse, insulation_corr\n",
    "from src.models.evaluation.metrics import mse, insulation_corr, insulation_corr_interior\n")

rep("        self._last_preview_epoch = -1\n",
    "        self._last_preview_epoch = -1\n"
    "        # validation insulation metric: expected profile to add back for --oe-target runs\n"
    "        self.register_buffer('val_expected_2d', None, persistent=False)\n"
    "        if getattr(args, 'oe_target', False):\n"
    "            ef = getattr(args, 'expected_file', None) or (\n"
    "                'data/expected_log_800_bal.npy' if getattr(args, 'balance', False) else 'data/expected_log_800.npy')\n"
    "            prof = np.load(ef)\n"
    "            idx = np.abs(np.arange(args.n_bins)[:, None] - np.arange(args.n_bins)[None, :])\n"
    "            self.register_buffer('val_expected_2d', torch.tensor(prof[idx], dtype=torch.float32), persistent=False)\n"
    "            print(f'[TrainModule] val insulation metric adds expected back from {ef}')\n")

rep("        with torch.no_grad():\n"
    "            if not getattr(self.args, 'oe_target', False):\n"
    "                outputs = torch.clamp(outputs, min=0)\n"
    "            for out, true in zip(outputs, mats):\n"
    "                out_c = out.detach().to(torch.float32).cpu()\n"
    "                true_c = true.detach().to(torch.float32).cpu()\n"
    "\n"
    "                r_pearson, r_spearman = insulation_corr(out_c, true_c, res=self.args.resolution)\n",
    "        with torch.no_grad():\n"
    "            if self.val_expected_2d is not None:              # O/E run: back to log-count space\n"
    "                e = self.val_expected_2d.to(outputs.device, outputs.dtype)\n"
    "                outputs = outputs + e\n"
    "                mats = mats + e\n"
    "            outputs = torch.clamp(outputs, min=0)\n"
    "            for out, true in zip(outputs, mats):\n"
    "                out_c = out.detach().to(torch.float32).cpu()\n"
    "                true_c = true.detach().to(torch.float32).cpu()\n"
    "\n"
    "                r_pearson, r_spearman = insulation_corr_interior(out_c, true_c, res=self.args.resolution,\n"
    "                                                                 radius=25000)\n")
open(P, "w").write(src)
print("patched", P)