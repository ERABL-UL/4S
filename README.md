# 4S: Semi-supervised self-supervised semantic segmentation for 3D LiDAR point clouds


Installing pre-requisites:

`sudo apt install build-essential python3-dev libopenblas-dev`

`pip3 install -r requirements.txt`

`pip3 install torch ninja`

Installing MinkowskiEngine with CUDA support:

`pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps`

# SegContrast with Docker

Inside the `docker/` directory there is a `Dockerfile` to build an image to run SegContrast. You can build the image from scratch or download the image from docker hub by:

```
docker pull nuneslu/segcontrast:minkunet
```

Then start the container with:

```
docker run --gpus all -it --rm -v /PATH/TO/SEGCONTRAST:/home/segcontrast segcontrast /bin/zsh
```

# Data Preparation

Download [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) inside the directory ```./Datasets/SemanticKITTI/datasets```. The directory structure should be:

```
./
└── Datasets/
    └── SemanticKITTI
        └── dataset
          └── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	├── 000000.bin
            |   |	├── 000001.bin
            |   |	└── ...
            │   └── labels/ 
            |       ├── 000000.label
            |       ├── 000001.label
            |       └── ...
            ├── 08/ # for validation
            ├── 11/ # 11-21 for testing
            └── 21/
                └── ...
```
Download [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/user_login.php) inside the directory ```./Datasets/KITTI-360/dataset```. The directory structure should be:
```
./
└── Datasets/
    └── KITTI360
        └── train
          └── sequences
            ├── 00/           
            │   ├── {start_frame:0>10}_{end_frame:0>10}.ply
            |   └── ...
        └── validation
            └── sequences
                ├── 00/
                │   ├── {start_frame:0>10}_{end_frame:0>10}.ply
                |   └── ...
        └── test
            └── sequences
                ├── 08/
                │   ├── {start_frame:0>10}_{end_frame:0>10}.ply
                |   └── ...
                ├── 18/
                │   ├── {start_frame:0>10}_{end_frame:0>10}.ply
                |   └── ...
```


# Pretrained Weights
- SegContrast pretraining [weights](https://www.ipb.uni-bonn.de/html/projects/segcontrast/segcontrast_pretrain.zip)
- Fine-tuned semantic segmentation
    - 0.1% labels [weights](https://www.ipb.uni-bonn.de/html/projects/segcontrast/semantic_segmentation_weights/semseg_finetune_0p001.zip)
    - 1% labels [weights](https://www.ipb.uni-bonn.de/html/projects/segcontrast/semantic_segmentation_weights/semseg_finetune_0p01.zip)
    - 10% labels [weights](https://www.ipb.uni-bonn.de/html/projects/segcontrast/semantic_segmentation_weights/semseg_finetune_0p1.zip)
    - 50% labels [weights](https://www.ipb.uni-bonn.de/html/projects/segcontrast/semantic_segmentation_weights/semseg_finetune_0p5.zip)
    - 100% labels [weights](https://www.ipb.uni-bonn.de/html/projects/segcontrast/semantic_segmentation_weights/semseg_finetune_1p0.zip)

# Reproducing the results

Run the following to start the pre-training:

```
python3 contrastive_train.py --use-cuda --use-intensity --segment-contrast --checkpoint segcontrast
```

The default parameters, e.g., learning rate, batch size and epochs are already the same as the paper.

After pre-training you can run the downstream fine-tuning with:

```
python3 downstream_train.py --use-cuda --use-intensity --checkpoint \
        segment_contrast --contrastive --load-checkpoint --batch-size 2 \
        --sparse-model MinkUNet --epochs 15
```

We provide in `tools` the `contrastive_train.sh` and `downstream_train.sh` scripts to reproduce the results pre-training and fine-tuning with the different label percentages shown on the paper:

For pre-training:

```
./tools/contrastive_train.sh
```

Then for fine-tuning:

```
./tools/downstream_train.sh
```

Finally, to compute the IoU metrics use:

```
./tools/eval_train.sh
```


