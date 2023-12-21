# VICRegL + Point Clouds

## Installation
Setup devcontainer
```bash
cp .devcontainer/devcontainer.json.turing .devcontainer/devcontainer.json

# In container
pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps

# Run
python3 contrastive_train.py --vicreg --batch-size 64 --feature-size 128 --num-workers 8 \
    --dataset-name SemanticKITTI \
    --data-dir ./Datasets/SemanticKITTI \
    --epochs 200 \
    --lr 0.12 \
    --num-points 20000 \
    --use-cuda \
    --use-intensity \
    --segment-contrast \
    --checkpoint segcontrast

# Tensorboard
tensorboard --logdir lightning_logs
```

## Training
```bash
# TODO make script to run training without devcontainer
docker build -t vicregl -f docker/Dockerfile .

CUDA_VISIBLE_DEVICES=all  # or `0,1` for specific GPUs, will be automatically set by SLURM

# TODO try -v host:container:ro,delegated for volumes
# TODO need to run the following command in the container
pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps

docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm -it \
  --mount type=bind,source=/dev/shm,target=/dev/shm \
  --mount type=bind,source=$(pwd),target=/app/ \
  --mount type=bind,source=/home/william/Datasets,target=/app/Datasets \
  vicregl python3 contrastive_train.py --use-cuda --use-intensity --segment-contrast --checkpoint segcontrast --vicreg
```

On Compute Canada:
```bash
#!/bin/bash

#SBATCH --account=rrg-ergui19
#SBATCH --cpus-per-task=24 
#SBATCH --mem=32000M
#SBATCH --gres=gpu:v100l:1 
#SBATCH --time=01-10:00     # DD-HH:MM:SS
#SBATCH --mail-user=reza.mahmoudi-kouhi.1@ulaval.ca
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
# python contrastive_train.py --data-dir $SLURM_TMPDIR/Data/SemanticKITTI --vicreg --batch-size 32 --feature-size 128 --num-workers 8

python3 contrastive_train.py --vicreg --batch-size 64 --feature-size 128 --num-workers 8     --dataset-name SemanticKITTI     --data-dir $SLURM_TMPDIR/Data/SemanticKITTI     --epochs 200     --lr 0.12     --num-points 20000     --use-cuda     --use-intensity     --segment-contrast     --checkpoint segcontrast
```