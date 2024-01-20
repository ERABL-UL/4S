#!/bin/bash

#SBATCH --account=rrg-ergui19
#SBATCH --cpus-per-task=24 
#SBATCH --mem=32000M
#SBATCH --gres=gpu:v100l:1 
#SBATCH --time=01-10:00 # DD-HH:MM:SS
#SBATCH --mail-user=william.guimont-martin.1@ulaval.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MAKEFLAGS="-j$(nproc)"

module load nixpkgs/16.09
module load python/3.7
module load gcc/7.3.0
module load cuda/10.2

source ~/segcont/bin/activate

# Start training
cd $SLURM_TMPDIR
mkdir Data
cd Data
mkdir  SemanticKITTI
cd SemanticKITTI
tar -x --use-compress-program=pigz -f ~/scratch/Phase3/Data/SemanticKITTI/dataset.tar
cd
cd scratch/Phase3/4S_WGM

# Start training
filename=$(basename -- $0)
EXP_NAME="${filename%.*}"
python3 contrastive_train.py --vicreg --batch-size 16 --feature-size 128 --lr 0.1 \
    --num-workers 8 --dataset-name SemanticKITTI --data-dir $SLURM_TMPDIR/Data/SemanticKITTI \
    --epochs 200 --num-points 20000 --use-cuda --use-intensity --segment-contrast --checkpoint $EXP_NAME
