# VICRegL + Point Clouds

Setup devcontainer
```bash
cp .devcontainer/devcontainer.json.linux .devcontainer/devcontainer.json
```

Build docker image
```bash
docker build -t vicregl -f docker/Dockerfile .
```

Dataset
```bash
ln -s Datasets/SemanticKITTI Datasets/SemanticKITTI
```

Running
```bash
pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps
```

Notes
```bash
./Datasets/SemanticKITTI/dataset/sequences/00/velodyne


pip install -U torch
pip install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps
pip install -U setuptools
```

