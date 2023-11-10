import numpy as np
import torch
import torchvision
from PIL import Image
from math import ceil, floor
import argparse
from numpy import inf
import MinkowskiEngine as ME
import os
from utils import *
from data_utils.collations import numpy_to_sparse_tensor
import open3d as o3d
from tqdm import tqdm
from data_utils.data_map import color_map, labels, labels_poss
from data_utils.ioueval import iouEval
from trainer.kitti360_trainer import validation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SparseSimCLR')

    parser.add_argument('--dataset-name', type=str, default='SemanticKITTI',
                        help='Name of dataset (default: SemanticKITTI')
    parser.add_argument('--data-dir', type=str, default='/home/reza/PHD/Data/KITTI360/fps_knn',
                        help='Path to dataset (default: ./Datasets/ModelNet/modelnet40_normal_resampled')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='using cuda (default: True')
    parser.add_argument('--device-id', type=int, default=0,
                        help='GPU device id (default: 0')
    parser.add_argument('--feature-size', type=int, default=128,
                        help='Feature output size (default: 128')
    parser.add_argument('--sparse-resolution', type=float, default=0.05,
                        help='Sparse tensor resolution (default: 0.01')
    parser.add_argument('--sparse-model', type=str, default='MinkUNet',
                        help='Sparse model to be used (default: MinkUNet')
    parser.add_argument('--use-normals', action='store_true', default=False,
                        help='use points normals (default: False')
    parser.add_argument('--log-dir', type=str, default='checkpoint/contrastive/teacher',
                        help='logging directory (default: checkpoint/downstream_task)')
    # parser.add_argument('--percentage', type=str, default='lastepoch14',
    #                     help='best loss or accuracy over training (default: lastepoch19)')
    parser.add_argument('--checkpoint', type=str, default='lastepoch14',
                        help='model checkpoint (default: lastepoch14)')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input inference batch-size')
    parser.add_argument('--epoch', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--visualize-pcd', action='store_true', default=False,
                        help='visualize inference point cloud (default: False')
    parser.add_argument('--use-intensity', action='store_true', default=False,
                        help='use intensity channel (default: False')

    args = parser.parse_args()
    
    # if args.use_cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    print('GPU')
    
    best_epoch = "lastepoch19"
    checkpoints = ["segment_contrast_0p5_lastepoch199"]
    set_deterministic()


# define backbone architecture
    resnet = get_model(args, dtype)


    classifier = get_classifier_head(args, dtype)


    model_filename = f'{args.checkpoint}_model.pt'
    classifier_filename = f'{args.checkpoint}_model_head.pt'
    print(model_filename, classifier_filename)
    # load pretained weights
    if os.path.isfile(f'{args.log_dir}/{model_filename}') and os.path.isfile(f'{args.log_dir}/{classifier_filename}'):
        checkpoint = torch.load(f'{args.log_dir}/{model_filename}')
        resnet.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']

        checkpoint = torch.load(f'{args.log_dir}/{classifier_filename}')
        classifier.load_state_dict(checkpoint['model'])

        print(f'Loading model: {args.checkpoint}, from epoch: {epoch}')
    else:
        print('Trained model not found!')
        import sys
        sys.exit()
    
    

    model = {'model': resnet.cuda(), 'classifier': classifier.cuda()}
    
    data_val = data_loaders[args.dataset_name](root=args.data_dir, split='validation',
                                                intensity_channel=args.use_intensity, pre_training=False, resolution=args.sparse_resolution)

    # create the data loader for train and validation data
    val_loader = torch.utils.data.DataLoader(
        data_val,
        batch_size=args.batch_size,
        collate_fn=SparseCollation(args.sparse_resolution, inf),
        shuffle=True,
    )

    # retrieve validation loss
    model_acc, model_miou, model_class_iou = validation(model, val_loader, args)
    print(f'\nModel Acc.: {model_acc}\tModel mIoU: {model_miou}\n\n- Per Class mIoU:')
    for class_ in range(model_class_iou.shape[0]):
        print(f'\t{labels[class_]}: {model_class_iou[class_].item()}')

