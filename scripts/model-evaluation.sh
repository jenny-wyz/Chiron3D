#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:2
#SBATCH --job-name=eval800654
#SBATCH --output=eval_r800_n654_nb8_chrX2L.output.txt
#SBATCH --time=04:00:00
#SBATCH --mem=128G

echo "JOB STARTED at: $(date)"
echo "Running on node: $(hostname)"
echo "JOB ID: $SLURM_JOB_ID"

cd /cluster/work/boeva/wangyiz/Chiron3D

source ~/.bashrc
conda activate chiron


# check cuda
echo "CUDA_VISIBLE_DEVICES=[$CUDA_VISIBLE_DEVICES]"
nvidia-smi || echo "nvidia-smi FAILED"
scontrol show job $SLURM_JOB_ID | grep -i -E "gres|tres|nodelist"
python3 -c "import torch;print(torch.cuda.is_available(), torch.cuda.device_count())"


# rez 800, nbins 654 ---------------------------------------------

python3 -m src.models.evaluation.evaluation \
  --regions-file data/windows_dm6_C523200.bed \
  --fasta-dir data/dmel_chromosomes \
  --cool-file data/lbm.800.cool \
  --genomic-feature UNUSED --num-genom-feat 0 \
  --ckpt-path checkpoints_r800_N654_nb8_2L/models/epoch=23-step=504.ckpt \
  --resolution 800 --n-bins 654 \
  --test-chroms chrX chr2L \
  --borzoi


echo "JOB ENDED at: $(date)"