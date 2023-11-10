from trainer.semantic_kitti_contrastive_trainer import SemanticKITTIContrastiveTrainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from utils import *
from losses.contrastive import ContrastiveLoss
import argparse
from numpy import inf
import MinkowskiEngine as ME

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SparseSimCLR')

    parser.add_argument('--dataset-name', type=str, default='SemanticKITTI',
                        help='Name of dataset (default: SemanticKITTI')
    parser.add_argument('--data-dir', type=str, default='/home/reza/PHD/Data/KITTI360/fps_knn',
                        help='Path to dataset (default: ./Datasets/SemanticKITTI')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input training batch-size')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of training epochs (default: 200)')
    parser.add_argument('--val-epoch', type=int, default=10, metavar='N',
                        help='number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.12,
                        help='learning rate (default: 2.4e-1')
    parser.add_argument("--decay-lr", default=1e-4, action="store", type=float,
                        help='Learning rate decay (default: 1e-4')
    parser.add_argument('--tau', default=0.1, type=float,
                        help='Tau temperature smoothing (default 0.1)')
    parser.add_argument('--log-dir', type=str, default='checkpoint/contrastive',
                        help='logging directory (default: checkpoint/contrastive)')
    parser.add_argument('--checkpoint', type=str, default='lastepoch119',
                        help='model checkpoint (default: contrastive_checkpoint)')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='using cuda (default: False')
    parser.add_argument('--feature-size', type=int, default=128,
                        help='Feature output size (default: 128')
    parser.add_argument('--device-id', type=int, default=0,
                        help='GPU device id (default: 0')
    parser.add_argument('--num-points', type=int, default=20000,
                        help='Number of points sampled from point clouds (default: 20000')
    parser.add_argument('--sparse-resolution', type=float, default=0.05,
                        help='Sparse tensor resolution (default: 0.05')
    parser.add_argument('--sparse-model', type=str, default='MinkUNet',
                        help='Sparse model to be used (default: MinkUNet')
    parser.add_argument('--use-intensity', action='store_true', default=False,
                        help='use points intensity (default: False')
    parser.add_argument('--load-checkpoint', action='store_true', default=True,
                        help='load checkpoint (default: True')
    parser.add_argument('--accum-steps', type=int, default=1,
                        help='Number steps to accumulate gradient')
    parser.add_argument('--segment-contrast', action='store_true', default=True,
                        help='Use segments patches for contrastive learning (default: False')
    parser.add_argument('--num-classes', type=int, default=7,
                        help='number of classes')
    parser.add_argument('--lambda-c', type=int, default=0.1,
                        help='lamba for contrastive loss')
    parser.add_argument('--linear-eval', action='store_true', default=False,
                        help='Fine-tune or linear evaluation (default: False')
    parser.add_argument('--pure', action='store_true', default=False,
                        help='Performs segment purification(default: False)')
    parser.add_argument('--psl-sup', default="None",
                        help='none,psl,psl+seg')

    
    args = parser.parse_args()

    if args.use_cuda:
        dtype = torch.cuda.FloatTensor
        device = torch.device("cuda")
        print('GPU')
    else:
        dtype = torch.FloatTensor
        device = torch.device("cpu")

    data_train, data_test = get_dataset(args)
    train_loader, validation_loader = get_data_loader(data_train, data_test, args)
    criterion_sup = nn.CrossEntropyLoss(ignore_index=0)
    criterion_us = nn.CrossEntropyLoss()


    model = get_moco_model(args, dtype)
    

    # model = {'moco': resnet.cuda(), 'classifier': classifier.cuda()}    
    
    # model_classifier = get_classifier_head(args, dtype)
    # if torch.cuda.device_count() > 1:
    #     model_sem_kitti = SemanticKITTIContrastiveTrainer(model, criterion, train_loader, args)
    #     trainer = Trainer(gpus=-1, accelerator='ddp', max_epochs=args.epochs, accumulate_grad_batches=args.accum_steps)
    #     trainer.fit(model_sem_kitti)

    # else:
    model_sem_kitti = SemanticKITTIContrastiveTrainer(model, criterion_sup, criterion_us, train_loader, validation_loader, args)
    trainer = Trainer(gpus=[0], check_val_every_n_epoch=args.val_epoch, max_epochs=args.epochs, accumulate_grad_batches=args.accum_steps)
    trainer.fit(model_sem_kitti)

    # (xi_coord, xi_feats, si), (xj_coord, xj_feats, sj) = next(iter(train_loader))
    
    # xi, xj = collate_points_to_sparse_tensor(xi_coord, xi_feats, xj_coord, xj_feats)
    # model.cuda()
    # out_seg, tgt_seg = model(xi, xj, [si, sj])





