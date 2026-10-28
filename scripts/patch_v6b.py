#!/usr/bin/env python3
"""Step 2 (v6b): register the tower trunk as  --trunk tower.

Copies nothing; expects src/models/model/tower_model.py to be in place already.
Edits in place: src/models/training/train.py, src/models/training/utils.py,
src/models/training/module.py, src/models/evaluation/evaluation.py, scripts/step0b_dump_embeddings.py.
Safe to run twice. Works with or without the v6a patch applied.
Run once from the repo root:   python3 scripts/patch_v6b.py
"""
import os, sys

assert os.path.exists("src/models/model/tower_model.py"), \
    "copy tower_model.py to src/models/model/ first"


def patch(path, old, new, count=1):
    src = open(path).read()
    if new in src:
        print(f"{path}: already patched ({old.strip().splitlines()[0][:50]}...)"); return
    n = src.count(old)
    assert n == count, f"{path}: expected {count} occurrence(s) of\n{old}\nfound {n}"
    open(path, "w").write(src.replace(old, new))
    print(f"patched {path}")


# ---------------- train.py ----------------
T = "src/models/training/train.py"
patch(T,
      "                        choices=['borzoi', 'dcnn', 'corigami'], help='Sequence trunk')\n",
      "                        choices=['borzoi', 'dcnn', 'corigami', 'tower'], help='Sequence trunk')\n")
patch(T,
      "    parser.add_argument('--asap-ckpt', dest='asap_ckpt', default=None)\n",
      "    parser.add_argument('--asap-ckpt', dest='asap_ckpt', default=None)\n"
      "    # v6b: tower trunk\n"
      "    parser.add_argument('--tower-blocks', dest='tower_blocks', type=int, default=8,\n"
      "                        help='v6b: number of dilated residual blocks at 32 bp')\n"
      "    parser.add_argument('--tower-dil-mult', dest='tower_dil_mult', type=float, default=2.0,\n"
      "                        help='v6b: dilation growth factor per block (2.0 -> 1,2,4,...)')\n"
      "    parser.add_argument('--tower-pool', dest='tower_pool', default='attn', choices=['attn', 'max', 'meanmax'],\n"
      "                        help='v6b: 32 bp -> 800 bp bin pooling')\n"
      "    parser.add_argument('--tower-channels', dest='tower_channels', default='256,320,384,448,512',\n"
      "                        help='v6b: stem + 4 tower stage widths')\n")

# ---------------- utils.py (get_model) ----------------
U = "src/models/training/utils.py"
patch(U,
      "from src.models.model.chiron_model import Chiron3D, Chiron3D_DCNN\n",
      "from src.models.model.chiron_model import Chiron3D, Chiron3D_DCNN\n"
      "from src.models.model.tower_model import Chiron3D_Tower\n")
patch(U,
      "    if trunk == 'dcnn':\n",
      "    if trunk == 'tower':\n"
      "        ch = tuple(int(c) for c in str(getattr(args, 'tower_channels', '256,320,384,448,512')).split(','))\n"
      "        model = Chiron3D_Tower(mid_hidden=128, resolution=args.resolution, n_bins=args.n_bins,\n"
      "                               flank=args.flank, tower_channels=ch,\n"
      "                               tower_blocks=int(getattr(args, 'tower_blocks', 8)),\n"
      "                               tower_dil_mult=float(getattr(args, 'tower_dil_mult', 2.0)),\n"
      "                               tower_pool=getattr(args, 'tower_pool', 'attn'),\n"
      "                               loop_head=bool(getattr(args, 'loop_file', None)))\n"
      "        replace_bn_with_groupnorm(model.attn)\n"
      "        replace_bn_with_groupnorm(model.decoder)\n"
      "        return model\n"
      "    if trunk == 'dcnn':\n")

# ---------------- module.py (dataset: 4-channel one-hot, own flank) ----------------
M = "src/models/training/module.py"
patch(M,
      "        use_pretrained_backbone = args.trunk in ('borzoi', 'dcnn')   # both want 4-channel one-hot\n",
      "        use_pretrained_backbone = args.trunk in ('borzoi', 'dcnn', 'tower')   # all want 4-channel one-hot\n")

# ---------------- evaluation.py ----------------
E = "src/models/evaluation/evaluation.py"
patch(E,
      "                   choices=['borzoi', 'dcnn', 'corigami'], help='Sequence trunk')\n",
      "                   choices=['borzoi', 'dcnn', 'corigami', 'tower'], help='Sequence trunk')\n")
patch(E,
      "        if args.trunk == 'dcnn':\n"
      "            assert args.flank == ck_flank, f\"CLI flank={args.flank} != checkpoint flank={ck_flank}\"\n",
      "        if args.trunk in ('dcnn', 'tower'):\n"
      "            assert args.flank == ck_flank, f\"CLI flank={args.flank} != checkpoint flank={ck_flank}\"\n")
patch(E,
      "    use_pretrained_backbone = args.trunk in ('borzoi', 'dcnn')\n",
      "    use_pretrained_backbone = args.trunk in ('borzoi', 'dcnn', 'tower')\n")
patch(E,
      "            flank=None if args.trunk != 'dcnn' else args.flank,\n",
      "            flank=None if args.trunk not in ('dcnn', 'tower') else args.flank,\n")

# ---------------- step0b (embedding dump understands the tower model) ----------------
S = "scripts/step0b_dump_embeddings.py"
if os.path.exists(S):
    patch(S,
          "    assert ck.trunk == \"dcnn\"\n",
          "    assert ck.trunk in (\"dcnn\", \"tower\"), ck.trunk\n")
    patch(S,
          "            with torch.no_grad():\n"
          "                if x.shape[-1] != m.input_width:\n",
          "            with torch.no_grad():\n"
          "                if hasattr(m, \"embed\"):                      # v6b tower model\n"
          "                    e_mean = m.embed(x)[0].T; e_max = e_mean\n"
          "                    E_mean.append(e_mean.float().cpu().numpy().astype(np.float16))\n"
          "                    E_max.append(e_max.float().cpu().numpy().astype(np.float16))\n"
          "                    seq = item[\"sequence\"].numpy()[:, ds.flank:ds.flank + N * res]\n"
          "                    KM.append(kmer_features(seq, N, res))\n"
          "                    INS.append(np.asarray(chr_score(item[\"matrix\"].numpy(), res=res, radius=RADIUS), np.float32))\n"
          "                    ST.append(int(item[\"region_start\"]))\n"
          "                    continue\n"
          "                if x.shape[-1] != m.input_width:\n")
else:
    print("scripts/step0b_dump_embeddings.py not found, skipped")
print("done.  Quick check:  python3 -c \"import src.models.model.tower_model\"")