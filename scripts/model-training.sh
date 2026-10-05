#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:4
#SBATCH --job-name=dcnn_v3b2_tr
#SBATCH --output=dcnn_v3b2_tr.output.txt
#SBATCH --cpus-per-task=16
#SBATCH --time=0-20:00:00
#SBATCH --mem=128G

echo "JOB STARTED at: $(date)"
echo "Running on node: $(hostname)"
echo "JOB ID: $SLURM_JOB_ID"

cd /cluster/work/boeva/wangyiz/Chiron3D

source ~/.bashrc
conda activate chiron


# dcnn: rez 800, nbins 654 ------------------------------ v3a ---------------

# python3 -m src.models.training.train \
#   --seed 2077 \
#   --save_path checkpoints_dcnn_v3a \
#   --regions-file data/windows_dm6_C523200_f1024.bed \
#   --fasta-dir data/dmel_chromosomes \
#   --cool-file data/lbm.800.cool \
#   --resolution 800 --n-bins 654 \
#   --trunk dcnn --flank 1024 \
#   --num-genom-feat 0 --accum 2 \
#   --patience 25 --max-epochs 150 --save-top-n 1 \
#   --num-gpu 4 --batch-size 4 --ddp-disabled --num-workers 16 \
#   --oe-target


# dcnn: rez 800, nbins 654 ------------------------------ v3b ---------------

python3 -m src.models.training.train \
  --seed 2077 \
  --save_path checkpoints_dcnn_v3b2 \
  --regions-file data/windows_dm6_C523200_f1024.bed \
  --fasta-dir data/dmel_chromosomes \
  --cool-file data/lbm.800.cool \
  --resolution 800 --n-bins 654 \
  --trunk dcnn --flank 1024 \
  --num-genom-feat 0 --accum 2 \
  --loop-file /cluster/work/boeva/Gambetta_collaboration/Loops/all_loops.tsv \
  --loop-alpha 0.5 \
  --loop-weight 0.87 \
  --patience 25 --max-epochs 150 --save-top-n 1 \
  --num-gpu 4 --batch-size 4 --ddp-disabled --num-workers 16



# control

# python3 -m src.models.training.train \
#   --seed 2077 \
#   --save_path tmp/v3b_control \
#   --regions-file data/windows_dm6_C523200_f1024.bed \
#   --fasta-dir data/dmel_chromosomes \
#   --cool-file data/lbm.800.cool \
#   --resolution 800 --n-bins 654 \
#   --trunk dcnn --flank 1024 \
#   --num-genom-feat 0 --accum 2 \
#   --patience 25 --max-epochs 1 --save-top-n 1 \
#   --num-gpu 4 --batch-size 4 --ddp-disabled --num-workers 16


echo "JOB ENDED at: $(date)"