#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:4
#SBATCH --job-name=BORZOI
#SBATCH --output=training.output.txt
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G

# Activate conda
source ~/.bashrc
conda activate chiron

echo "JOB STARTED at: $(date)"

# Check env
echo
echo "which python"
which python

cd /cluster/work/boeva/wangyiz/Chiron3D
#pip install -e .

SEED=2077
FLAG=$(echo "$1" | sed 's/^--//')
echo "Using flag: $FLAG"

# Save path
SAVE_PATH="checkpoints"

REGIONS_FILE="data/windows_hg19.bed"
COOL_FILE="data/A673_WT_CTCF_5000.cool"
GENOME_FEAT_PATH="data/ctcf"
FASTA_DIR_HG19="data/chromosomes"

# Model parameters
NUM_GENOM_FEAT=0

# Training Parameters
PATIENCE=7
MAX_EPOCHS=25
SAVE_TOP_N=25
NUM_GPU=4

# Dataloader Parameters
BATCH_SIZE=4
DDP_DISABLED="--ddp-disabled"
NUM_WORKERS=16

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Run the Python script with the arguments
python3 -m src.models.training.train \
  --seed $SEED \
  --save_path $SAVE_PATH \
  --regions-file $REGIONS_FILE \
  --fasta-dir $FASTA_DIR_HG19 \
  --cool-file $COOL_FILE \
  --genom-feat-path $GENOME_FEAT_PATH \
  --num-genom-feat $NUM_GENOM_FEAT \
  --patience $PATIENCE \
  --max-epochs $MAX_EPOCHS \
  --save-top-n $SAVE_TOP_N \
  --num-gpu $NUM_GPU \
  --batch-size $BATCH_SIZE \
  $DDP_DISABLED \
  --num-workers $NUM_WORKERS \
  --borzoi

echo "JOB ENDED at: $(date)"