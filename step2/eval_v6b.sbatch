#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --job-name=eval_v6b
#SBATCH --output=eval_v6b_%j.output.txt
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --mem=128G

# Usage:  sbatch step2/eval_v6b.sbatch std checkpoints_dcnn_v6b_std/models/epoch=XX-step=YYYY.ckpt
ARM=${1:?arm name (std|slim)}
CKPT=${2:?checkpoint path}
COOL=/cluster/work/boeva/Gambetta_collaboration/Micro-C/larval_brain_merge.mcool::resolutions/800
TAG=v6b_${ARM}

cd /cluster/work/boeva/wangyiz/Chiron3D
source ~/.bashrc
conda activate chiron
mkdir -p results/step2
echo "JOB STARTED $(date) arm=$ARM ckpt=$CKPT"

for CHROM in chrX chr3R; do
  python3 -m src.models.evaluation.evaluation \
    --regions-file data/windows_dm6_C523200_f1024.bed --fasta-dir data/dmel_chromosomes \
    --cool-file "$COOL" --genomic-feature UNUSED --num-genom-feat 0 \
    --ckpt-path "$CKPT" --resolution 800 --n-bins 654 --trunk tower --flank 1024 \
    --test-chroms $CHROM --dump-matrices dump_dcnn_${TAG}_${CHROM}
  echo "--- bands $TAG $CHROM ---"; python3 scripts/eval_bands_masked.py dump_dcnn_${TAG}_${CHROM}
done
echo "--- bands v4c_s80 chrX (reference) ---"; python3 scripts/eval_bands_masked.py dump_dcnn_v4c_s80_chrX

echo "===== TTA ====="
python3 scripts/step0a_tta.py dump_dcnn_${TAG}_chrX --out results/step2/tta_${TAG}.json

echo "===== probes (fingerprint R2 and insulation probe on the new embeddings) ====="
python3 scripts/step0b_dump_embeddings.py --ckpt "$CKPT" \
  --regions-file data/windows_dm6_C523200_f1024.bed --fasta-dir data/dmel_chromosomes \
  --cool-file "$COOL" --chroms chr3R chrX --out results/step2/emb_${TAG}
python3 scripts/step0c_probe.py --train results/step2/emb_${TAG}_chr3R.npz \
  --test results/step2/emb_${TAG}_chrX.npz --dump dump_dcnn_${TAG}_chrX | tee results/step2/probe_${TAG}.txt
echo "JOB ENDED $(date)"