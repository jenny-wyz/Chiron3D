#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:4
#SBATCH --job-name=dcnn_v6a
#SBATCH --output=dcnn_v6a_%x_%j.output.txt
#SBATCH --cpus-per-task=16
#SBATCH --time=0-24:00:00
#SBATCH --mem=128G

# Usage:   sbatch --job-name=v6a_max     step1/train_v6a.sbatch max
#          sbatch --job-name=v6a_meanmax step1/train_v6a.sbatch meanmax
POOL=${1:?give the pool mode: max or meanmax}

cd /cluster/work/boeva/wangyiz/Chiron3D
source ~/.bashrc
conda activate chiron
echo "JOB STARTED $(date) on $(hostname), pool=$POOL"

# identical to v4c_s80 except --pool and --save_path
python3 -m src.models.training.train \
  --seed 2077 \
  --save_path checkpoints_dcnn_v6a_${POOL} \
  --regions-file data/windows_dm6_C523200_f1024.bed \
  --fasta-dir data/dmel_chromosomes \
  --cool-file /cluster/work/boeva/Gambetta_collaboration/Micro-C/larval_brain_merge.mcool::resolutions/800 \
  --resolution 800 --n-bins 654 \
  --trunk dcnn --flank 1024 \
  --num-genom-feat 0 --accum 2 \
  --patience 25 --max-epochs 150 --save-top-n 1 \
  --num-gpu 4 --batch-size 4 --ddp-disabled --num-workers 16 \
  --balance --matrix-scale 2621.2 \
  --oe-target \
  --dist-weight-file data/dist_weights_800_band54_s80.npy \
  --pool ${POOL}

echo "JOB ENDED $(date)"