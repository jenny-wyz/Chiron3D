#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:4
#SBATCH --job-name=train800654
#SBATCH --output=train_r800_n654_nb8_2L.output.txt
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G

echo "JOB STARTED at: $(date)"
echo "Running on node: $(hostname)"
echo "JOB ID: $SLURM_JOB_ID"

cd /cluster/work/boeva/wangyiz/Chiron3D

source ~/.bashrc
conda activate chiron


# rez 800, nbins 654 ---------------------------------------------

python3 -m src.models.training.train \
  --seed 2077 \
  --save_path checkpoints_r800_N654_nb8_2L \
  --regions-file data/windows_dm6_C523200.bed \
  --fasta-dir data/dmel_chromosomes \
  --cool-file data/lbm.800.cool \
  --resolution 800 \
  --n-bins 654 \
  --num-genom-feat 0 \
  --patience 7 --max-epochs 25 --save-top-n 1 \
  --num-gpu 4 --batch-size 4 --ddp-disabled --num-workers 16 \
  --borzoi


echo "JOB ENDED at: $(date)"