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

def validation(model, data, args):
    eval = iouEval(n_classes=len(content.keys()), ignore=0)
    model['model'].eval()
    model['classifier'].eval()
    for iter_n, (x_coord, x_feats, x_label) in enumerate(tqdm(data)):
        x, y = numpy_to_sparse_tensor(x_coord, x_feats, x_label)

        if 'UNet' in args.sparse_model:
            y = y[:, 0]
        else:
            y = torch.from_numpy(np.asarray(y))
            y = y[:, 0]

        h = model['model'](x)
        z = model['classifier'](h)
        loss = self.criterion(z, y.long())
        losses.append(loss.cpu().item())
        y = y.cuda() if args.use_cuda else y

        # accumulate accuracy
        pred = z.max(dim=1)[1]
        eval.addBatch(pred.long().cpu().numpy(), y.long().cpu().numpy())


    model_acc = eval.getacc()
    model_miou, model_class_iou = eval.getIoU()
    # return the epoch mean loss
    print(f'\nModel Acc.: {model_acc}\tModel mIoU: {model_miou}\n\n- Per Class mIoU:')
    for class_ in range(model_class_iou.shape[0]):
        print(f'\t{labels[class_]}: {model_class_iou[class_].item()}')




def training_step(model, data, args):
    eval = iouEval(n_classes=len(content.keys()), ignore=0)
    model['model'].train()
    model['classifier'].train()
    for iter_n, (x_coord, x_feats, x_label) in enumerate(tqdm(data)):
        x, y = numpy_to_sparse_tensor(x_coord, x_feats, x_label)

        if 'UNet' in args.sparse_model:
            y = y[:, 0]
        else:
            y = torch.from_numpy(np.asarray(y))
            y = y[:, 0]

        h = model['model'](x)
        z = model['classifier'](h)

        y = y.cuda() if args.use_cuda else y
        loss = self.criterion(z, y.long())
        losses.append(loss.cpu().item())
        # accumulate accuracy
        pred = z.max(dim=1)[1]
        eval.addBatch(pred.long().cpu().numpy(), y.long().cpu().numpy())


    acc = eval.getacc()
    mean_iou, class_iou = eval.getIoU()
    # return the epoch mean loss
    return acc, mean_iou, class_iou



def train(model, train_loader, args):
    model.train()
    accs = []
    mean_ious = []
    class_ious = []
    for epoch in args.epoch:
        acc, mean_iou, class_iou = training_step(model, train_loader, args)
        accs.append(acc)
        mean_ious.append(mean_iou)
        class_ious.append(class_iou)
    return accs, mean_iou, class_iou









