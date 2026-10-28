#!/usr/bin/env python3
"""Step 1 (v6a): add a --pool {mean,max,meanmax} option to Chiron3D_DCNN.

  mean     : exactly the current model (1x1 projector, then average over the 400 trunk positions of a bin)
  max      : GELU on the trunk output, MAX over the 400 positions, then 1x1 projector 256->128
  meanmax  : GELU, then both mean and max (concatenated, 512 channels), then 1x1 projector 512->128

Edits three files in place (src/models/model/chiron_model.py, src/models/training/utils.py,
src/models/training/train.py). Old checkpoints keep working (missing 'pool' hparam = 'mean').
Run once from the repo root:   python3 scripts/patch_v6a.py
"""
import sys


def patch(path, old, new, count=1):
    src = open(path).read()
    n = src.count(old)
    assert n == count, f"{path}: expected {count} occurrence(s) of\n{old}\nfound {n}"
    open(path, "w").write(src.replace(old, new))
    print(f"patched {path}")


# ---------------- chiron_model.py ----------------
P = "src/models/model/chiron_model.py"
if "pool_mode" in open(P).read():
    print("chiron_model.py already patched"); sys.exit(0)

patch(P,
      "    def __init__(self, mid_hidden=128, resolution=400, n_bins=256,\n"
      "                 flank=1024, asap_ckpt=None, dropout=0.1, loop_head=False):\n"
      "        super().__init__()\n",
      "    def __init__(self, mid_hidden=128, resolution=400, n_bins=256,\n"
      "                 flank=1024, asap_ckpt=None, dropout=0.1, loop_head=False, pool_mode='mean'):\n"
      "        super().__init__()\n"
      "        assert pool_mode in ('mean', 'max', 'meanmax'), pool_mode\n"
      "        self.pool_mode = pool_mode\n")

patch(P,
      "        print(f\"[Chiron3D_DCNN] resolution={self.resolution} n_bins={self.n_bins} \"\n"
      "              f\"span={self.target_span} flank={self.flank} input={self.input_width} \"\n"
      "              f\"pool_factor={self.pool_factor}\")\n",
      "        print(f\"[Chiron3D_DCNN] resolution={self.resolution} n_bins={self.n_bins} \"\n"
      "              f\"span={self.target_span} flank={self.flank} input={self.input_width} \"\n"
      "              f\"pool_factor={self.pool_factor} pool_mode={self.pool_mode}\")\n")

patch(P,
      "        self.activation = nn.ReLU()\n"
      "        self.projector = nn.Conv1d(self.trunk.out_channels, mid_hidden, kernel_size=1, bias=True)\n"
      "\n"
      "        self.attn = blocks.AttnModuleSmall(hidden=mid_hidden, record_attn=False)\n"
      "        self.loop_head = bool(loop_head)\n",
      "        self.activation = nn.GELU()\n"
      "        proj_in = self.trunk.out_channels * (2 if self.pool_mode == 'meanmax' else 1)\n"
      "        self.projector = nn.Conv1d(proj_in, mid_hidden, kernel_size=1, bias=True)\n"
      "\n"
      "        self.attn = blocks.AttnModuleSmall(hidden=mid_hidden, record_attn=False)\n"
      "        self.loop_head = bool(loop_head)\n")

patch(P,
      "        x = x[..., self.trim:x.shape[-1] - self.trim]   # drop flank -> (B, 256, span/2)\n"
      "        x = self.projector(x)\n"
      "        b, c, l = x.shape\n"
      "        x = x.view(b, c, self.n_bins, self.pool_factor).mean(-1)\n",
      "        x = x[..., self.trim:x.shape[-1] - self.trim]   # drop flank -> (B, 256, span/2)\n"
      "        if self.pool_mode == 'mean':                    # v4: project, then average (unchanged)\n"
      "            x = self.projector(x)\n"
      "            b, c, l = x.shape\n"
      "            x = x.view(b, c, self.n_bins, self.pool_factor).mean(-1)\n"
      "        else:                                           # v6a: nonlinearity, pool, then project\n"
      "            x = self.activation(x)\n"
      "            b, c, l = x.shape\n"
      "            x = x.view(b, c, self.n_bins, self.pool_factor)\n"
      "            if self.pool_mode == 'max':\n"
      "                x = x.amax(-1)\n"
      "            else:\n"
      "                x = torch.cat([x.mean(-1), x.amax(-1)], dim=1)   # (B, 512, N)\n"
      "            x = self.projector(x)\n")

# ---------------- utils.py (get_model) ----------------
U = "src/models/training/utils.py"
patch(U,
      "                              flank=args.flank, asap_ckpt=args.asap_ckpt,\n"
      "                              loop_head=bool(getattr(args, 'loop_file', None)))\n",
      "                              flank=args.flank, asap_ckpt=args.asap_ckpt,\n"
      "                              loop_head=bool(getattr(args, 'loop_file', None)),\n"
      "                              pool_mode=getattr(args, 'pool', 'mean'))\n")

# ---------------- train.py (CLI flag) ----------------
T = "src/models/training/train.py"
patch(T,
      "    parser.add_argument('--asap-ckpt', dest='asap_ckpt', default=None)\n",
      "    parser.add_argument('--asap-ckpt', dest='asap_ckpt', default=None)\n"
      "    parser.add_argument('--pool', dest='pool', default='mean', choices=['mean', 'max', 'meanmax'],\n"
      "                        help='v6a: how the 400 trunk positions of a bin are pooled (mean = v4 behaviour)')\n")
print("done. Check with:  grep -n pool_mode src/models/model/chiron_model.py")