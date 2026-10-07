#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:4
#SBATCH --job-name=dcnn_v4a_tr
#SBATCH --output=dcnn_v4a_tr.output.txt
#SBATCH --cpus-per-task=16
#SBATCH --time=0-20:00:00
#SBATCH --mem=128G

echo "JOB STARTED at: $(date)"
echo "Running on node: $(hostname)"
echo "JOB ID: $SLURM_JOB_ID"

cd /cluster/work/boeva/wangyiz/Chiron3D

source ~/.bashrc
conda activate chiron


# dcnn: rez 800, nbins 654 ------------------------------ v4a ---------------

python3 -m src.models.training.train \
  --seed 2077 \
  --save_path checkpoints_dcnn_v4a_ice \
  --regions-file data/windows_dm6_C523200_f1024.bed \
  --fasta-dir data/dmel_chromosomes \
  --cool-file /cluster/work/boeva/Gambetta_collaboration/Micro-C/larval_brain_merge.mcool::resolutions/800 \
  --resolution 800 --n-bins 654 \
  --trunk dcnn --flank 1024 \
  --num-genom-feat 0 --accum 2 \
  --patience 25 --max-epochs 150 --save-top-n 1 \
  --num-gpu 4 --batch-size 4 --ddp-disabled --num-workers 16 \
  --balance --matrix-scale 2621.2

# dcnn: rez 800, nbins 654 ------------------------------ v4b ---------------

# python3 -m src.models.training.train \
#   --seed 2077 \
#   --save_path checkpoints_dcnn_v4b_ice_oe \
#   --regions-file data/windows_dm6_C523200_f1024.bed \
#   --fasta-dir data/dmel_chromosomes \
#   --cool-file /cluster/work/boeva/Gambetta_collaboration/Micro-C/larval_brain_merge.mcool::resolutions/800 \
#   --resolution 800 --n-bins 654 \
#   --trunk dcnn --flank 1024 \
#   --num-genom-feat 0 --accum 2 \
#   --patience 25 --max-epochs 150 --save-top-n 1 \
#   --num-gpu 4 --batch-size 4 --ddp-disabled --num-workers 16 \
#   --balance --matrix-scale 2621.2 \
#   --oe-target


echo "JOB ENDED at: $(date)"