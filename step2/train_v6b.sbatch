#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:4
#SBATCH --job-name=dcnn_v6b
#SBATCH --output=dcnn_v6b_%x_%j.output.txt
#SBATCH --cpus-per-task=16
#SBATCH --time=0-24:00:00
#SBATCH --mem=128G

# Usage:   sbatch --job-name=v6b_std  step2/train_v6b.sbatch std     # tower 256->512, 8 dilated blocks, attention pool
#          sbatch --job-name=v6b_slim step2/train_v6b.sbatch slim    # narrower tower (fewer parameters)
ARM=${1:?give the arm name: std or slim}
case "$ARM" in
  std)  CH=256,320,384,448,512 ;;
  slim) CH=256,256,256,384,384 ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac

cd /cluster/work/boeva/wangyiz/Chiron3D
source ~/.bashrc
conda activate chiron
echo "JOB STARTED $(date) on $(hostname), arm=$ARM channels=$CH"

# identical to v4c_s80 except the trunk (and --save_path)
python3 -m src.models.training.train \
  --seed 2077 \
  --save_path checkpoints_dcnn_v6b_${ARM} \
  --regions-file data/windows_dm6_C523200_f1024.bed \
  --fasta-dir data/dmel_chromosomes \
  --cool-file /cluster/work/boeva/Gambetta_collaboration/Micro-C/larval_brain_merge.mcool::resolutions/800 \
  --resolution 800 --n-bins 654 \
  --trunk tower --flank 1024 \
  --tower-channels ${CH} --tower-blocks 8 --tower-dil-mult 2.0 --tower-pool attn \
  --num-genom-feat 0 --accum 2 \
  --patience 25 --max-epochs 150 --save-top-n 1 \
  --num-gpu 4 --batch-size 4 --ddp-disabled --num-workers 16 \
  --balance --matrix-scale 2621.2 \
  --oe-target \
  --dist-weight-file data/dist_weights_800_band54_s80.npy

echo "JOB ENDED $(date)"