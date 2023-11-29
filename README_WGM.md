# VICRegL + Point Clouds

## Installation
Setup devcontainer
```bash
cp .devcontainer/devcontainer.json.turing .devcontainer/devcontainer.json

# In container
pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps
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
  vicregl python3 contrastive_train.py --use-cuda --use-intensity --segment-contrast --checkpoint segcontrast
```
