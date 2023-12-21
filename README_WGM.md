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

# Trying to make it work on Exx

```bash
pip install -U torch torchvision setuptools
pip3 install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps

pip install -U setuptools
cd /workspaces
git clone https://github.com/NVIDIA/MinkowskiEngine
cd MinkowskiEngine
# Apply https://github.com/NVIDIA/MinkowskiEngine/issues/543
# Add header to src/convolution_kernel.cuh
echo "#include <thrust/execution_policy.h>" | cat - src/convolution_kernel.cuh > temp && mv temp src/convolution_kernel.cuh
# Add header to src/coordinate_map_gpu.cu
echo "#include <thrust/unique.h>" | cat - src/coordinate_map_gpu.cu > temp && mv temp src/coordinate_map_gpu.cu
echo "#include <thrust/remove.h>" | cat - src/coordinate_map_gpu.cu > temp && mv temp src/coordinate_map_gpu.cu
# Add header to src/spmm.cu
echo "#include <thrust/execution_policy.h>" | cat - src/spmm.cu > temp && mv temp src/spmm.cu
echo "#include <thrust/reduce.h>" | cat - src/spmm.cu > temp && mv temp src/spmm.cu
echo "#include <thrust/sort.h>" | cat - src/spmm.cu > temp && mv temp src/spmm.cu
# Add header to src/3rdparty/src/spmmconcurrent_unordered_map.cuh
echo "#include <thrust/execution_policy.h>" | cat - src/3rdparty/src/spmmconcurrent_unordered_map.cuh > temp && mv temp src/3rdparty/src/spmmconcurrent_unordered_map.cuh

python3 setup.py install
```
