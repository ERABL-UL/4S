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

Notes
```bash
pip install -U torch
pip install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps
pip install -U setuptools
```

