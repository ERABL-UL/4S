from trainer.semantic_kitti_contrastive_trainer import SemanticKITTIContrastiveTrainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from utils import *
from losses.contrastive import ContrastiveLoss
import argparse
from numpy import inf
import MinkowskiEngine as ME
import pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SparseSimCLR')

    parser.add_argument('--dataset-name', type=str, default='SemanticKITTI',
                        help='Name of dataset (default: SemanticKITTI')
    parser.add_argument('--data-dir', type=str, default='./Datasets/SemanticKITTI',
                        help='Path to dataset (default: ./Datasets/SemanticKITTI')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input training batch-size')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=2.4e-1,
                        help='learning rate (default: 2.4e-1')
    parser.add_argument("--decay-lr", default=1e-4, action="store", type=float,
                        help='Learning rate decay (default: 1e-4')
    parser.add_argument('--tau', default=0.1, type=float,
                        help='Tau temperature smoothing (default 0.1)')
    parser.add_argument('--log-dir', type=str, default='checkpoint/contrastive',
                        help='logging directory (default: checkpoint/contrastive)')
    parser.add_argument('--checkpoint', type=str, default='contrastive_checkpoint',
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
    parser.add_argument('--load-checkpoint', action='store_true', default=False,
                        help='load checkpoint (default: True')
    parser.add_argument('--accum-steps', type=int, default=1,
                        help='Number steps to accumulate gradient')
    parser.add_argument('--segment-contrast', action='store_true', default=False,
                        help='Use segments patches for contrastive learning (default: False')
    parser.add_argument('--shuffle', action='store_true', default=False, help='Shuffle dataset (default: False)')
    
    # VICReg arguments and added arguments
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for data loader')

    parser.add_argument('--vicreg', action='store_true', default=False, help='Use VICReg')
    parser.add_argument('--vicreg-alpha', type=float, default=0.75,
                        help='VICRegL alpha that controls the importance put on learning global vs local features.'
                        'alpha = 1 -> only global features, alpha = 0 -> only local features (default: 0.1')
    parser.add_argument('--vicreg-num-global', type=int, default=20, help='Number of nearest neighbors for global features (gamma_1 in the paper)')
    parser.add_argument('--vicreg-num-local', type=int, default=4, help='Number of nearest neighbors for local features (gamma 2 in the paper)')
    parser.add_argument('--vicreg-inv-coeff', type=float, default=25, help='Invariance coefficient for VICReg loss (lambda in the paper)')
    parser.add_argument('--vicreg-var-coeff', type=float, default=25, help='Variance coefficient for VICReg loss (mu in the paper))')
    parser.add_argument('--vicreg-cov-coeff', type=float, default=1, help='Covariance coefficient for VICReg loss (nu in the paper)')
    parser.add_argument('--vicreg-eps', type=float, default=1e-4, help='Epsilon for VICReg loss')
    parser.add_argument('--vicreg-gamma', type=float, default=1, help='Target variance for VICReg loss (gamma in VICReg paper)')
    # TODO add mlp dimensions for VICReg
    #      mlp = 8192-8192-9192 for global features
    #      mlp = 512-512-512 for local features
    # parser.add_argument('--vicreg-local-dim', type=int, default=32, help='Dimension of local features')
    # parser.add_argument('--vicreg-global-dim', type=int, default=128, help='Dimension of global features')

    args = parser.parse_args()

    # Create log directory
    logdir = pathlib.Path(args.log_dir)
    logdir.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    if args.use_cuda:
        dtype = torch.cuda.FloatTensor
        device = torch.device("cuda")
        print('GPU')
    else:
        dtype = torch.FloatTensor
        device = torch.device("cpu")

    data_train, data_test = get_dataset(args)
    train_loader, _ = get_data_loader(data_train, data_test, args)
    criterion = nn.CrossEntropyLoss().cuda() if not args.vicreg else None

    model = get_moco_model(args, dtype)

    if torch.cuda.device_count() > 1:
        model_sem_kitti = SemanticKITTIContrastiveTrainer(model, criterion, train_loader, args)
        trainer = Trainer(gpus=-1, accelerator='ddp', max_epochs=args.epochs, accumulate_grad_batches=args.accum_steps)
        trainer.fit(model_sem_kitti)

    else:
        model_sem_kitti = SemanticKITTIContrastiveTrainer(model, criterion, train_loader, args)
        trainer = Trainer(gpus=[0], max_epochs=args.epochs, accumulate_grad_batches=args.accum_steps)
        trainer.fit(model_sem_kitti)
