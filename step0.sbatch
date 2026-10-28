#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --job-name=step0
#SBATCH --output=step0.output.txt
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=128G

# ---- EDIT THESE THREE LINES ----
CKPT=checkpoints_dcnn_v4c_s80/models/epoch=39-step=3280.ckpt        # the v4c_s80 best checkpoint
DUMP_X=dump_dcnn_v4c_s80_chrX                                       # existing chrX dump of that checkpoint
COOL=/cluster/work/boeva/Gambetta_collaboration/Micro-C/larval_brain_merge.mcool::resolutions/800
# --------------------------------

set -e
cd /cluster/work/boeva/wangyiz/Chiron3D
source ~/.bashrc
conda activate chiron
mkdir -p results/step0
echo "JOB STARTED $(date) on $(hostname)"

echo "===== 0a: test-time averaging on $DUMP_X ====="
python3 scripts/step0a_tta.py "$DUMP_X" --out results/step0/tta_v4c_s80.json

echo "===== 0b: dump embeddings (chr3R = train, chrX = test) ====="
python3 scripts/step0b_dump_embeddings.py --ckpt "$CKPT" \
  --regions-file data/windows_dm6_C523200_f1024.bed --fasta-dir data/dmel_chromosomes \
  --cool-file "$COOL" --chroms chr3R chrX --out results/step0/emb_v4c_s80

echo "===== 0c: probes ====="
python3 scripts/step0c_probe.py --train results/step0/emb_v4c_s80_chr3R.npz \
  --test results/step0/emb_v4c_s80_chrX.npz --dump "$DUMP_X" | tee results/step0/probe_v4c_s80.txt

echo "===== 0d: BatchNorm batch-stats evaluation ====="
python3 scripts/patch_step0d_bn.py
BN_TRAIN=1 python3 -m src.models.evaluation.evaluation \
  --regions-file data/windows_dm6_C523200_f1024.bed --fasta-dir data/dmel_chromosomes \
  --cool-file "$COOL" --genomic-feature UNUSED --num-genom-feat 0 \
  --ckpt-path "$CKPT" --resolution 800 --n-bins 654 --trunk dcnn --flank 1024 \
  --test-chroms chrX --dump-matrices dump_dcnn_v4c_s80_chrX_bntrain
echo "--- bands, normal eval (reference) ---";   python3 scripts/eval_bands_masked.py "$DUMP_X"
echo "--- bands, BN batch statistics ---";       python3 scripts/eval_bands_masked.py dump_dcnn_v4c_s80_chrX_bntrain

echo "JOB ENDED $(date)"