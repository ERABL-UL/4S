import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import iou
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.distributed as dist
from data_utils.data_map import labels, content
from data_utils.ioueval import iouEval
from data_utils.collations import *
from numpy import inf, pi, cos, mean
from functools import partial
import OSToolBox as ost
import torch.nn as nn
class SemanticKITTIContrastiveTrainer(pl.LightningModule):
    def __init__(self, model, criterion_sup, criterion_us, train_loader, val_loader, params):
        super().__init__()
        self.moco_model = model        
        self.criterion_sup = criterion_sup
        self.criterion_us = criterion_us        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.params = params
        self.writer = SummaryWriter(f'runs/{params.checkpoint}')
        self.iter_log = 10
        self.loss_eval = []
        self.loss_c_eval = []
        self.loss_s_eval = []
        self.train_step = 0
        self.val_step = 0
        self.evaluator = iouEval(n_classes=self.params.num_classes, ignore=0)
        if self.params.load_checkpoint:
            self.load_checkpoint()

    ############################################################################################################################################
    # FORWARD                                                                                                                                  #
    ############################################################################################################################################

    def forward(self, xi, xj, s_i, s_j, inv_i, inv_j):
        return self.moco_model(xi, xj, s_i, s_j, inv_i, inv_j, self.train_step, self.current_epoch) 
    


    ############################################################################################################################################

    ############################################################################################################################################
    # TRAINING                                                                                                                                 #
    ############################################################################################################################################
    

                
    def teacher_student_step(self, batch, batch_nb):
        (pci_coord, pci_feats, s_pci, pci_inverse), (pcj_coord, pcj_feats, s_pcj, pcj_inverse) = batch
        
        #Supervised Loss
        pci, pcj = collate_points_to_sparse_tensor(pci_coord, pci_feats, pcj_coord, pcj_feats)
        batch_idx = []
        batch_idx.append(0)
        for i in range(len(pci_coord)):
            batch_idx.append(np.argwhere(pci.C.cpu().numpy()[:,0]==i)[-1][0]+1)
            
        out_seg, tgt_seg, pred_q, pred_k = self.forward(pci, pcj, s_pci, s_pcj, pci_inverse, pcj_inverse)
        ps_l = pred_q.max(dim=1)[1]
        loss_s = self.criterion_sup(pred_k, ps_l)
        
        pred = pred_k.max(dim=1)[1]
        correct = pred.eq(ps_l).sum().item()
        correct /= ps_l.size(0)
        batch_acc = (correct * 100.)

        #Contrastive Loss
        loss_c = self.criterion_us(out_seg, tgt_seg)
        #Total Loss
        loss = loss_s + self.params.lambda_c*loss_c

        
        self.downstream_iter_callback(loss.item(), loss_c.item(), loss_s.item(), batch_acc, pred, ps_l, True)
        return {'loss': loss, 'acc': batch_acc}


    def training_step(self, batch, batch_nb):
        self.train_step += 1
        torch.cuda.empty_cache()
        return self.teacher_student_step(batch, batch_nb)

    def training_epoch_end(self, outputs):
        avg_loss = torch.FloatTensor([ x['loss'] for x in outputs ]).mean()
        epoch_dict = {'avg_loss': avg_loss}

        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            self.checkpoint_callback()

    ############################################################################################################################################

    ############################################################################################################################################
    # VALIDATION                                                                                                                               #
    ############################################################################################################################################


    def validation_step(self, batch, batch_nb):
        # validation step for downstream task
        self.val_step += 1
        x_coord, x_feats, x_label = batch
        x, y = numpy_to_sparse_tensor(x_coord, x_feats, x_label)
        y = y[:,0]

        h = self.moco_model.model_q(x)
        z = self.moco_model.classifier_q(h)

        loss = self.criterion_sup(z, y.long())
        pred = z.max(dim=1)[1]
        correct = pred.eq(y).sum().item()
        correct /= y.size(0)
        batch_acc = (correct * 100.)

        self.downstream_iter_callback_val(loss.item(), batch_acc, pred, y, x.C, False)

        return {'loss': loss.item(), 'acc': batch_acc}

    def validation_epoch_end(self, outputs):
        # at the end of each validation epoch will call our implemented callback to save the checkpoint
        avg_loss = torch.FloatTensor([ x['loss'] for x in outputs ]).mean()
        avg_acc = torch.FloatTensor([ x['acc'] for x in outputs ]).mean()
        epoch_dict = {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc}
        self.validation_epoch_callback(avg_loss, avg_acc)

        torch.cuda.empty_cache()
        return epoch_dict



    ############################################################################################################################################

    ############################################################################################################################################
    # CALLBACKS                                                                                                                                #
    ############################################################################################################################################

    def checkpoint_callback(self):
        if self.current_epoch % 10 == 0:
            self.save_checkpoint(f'epoch{self.current_epoch}')

        if self.current_epoch == self.params.epochs - 1:
            self.save_checkpoint(f'lastepoch{self.current_epoch}')

    def contrastive_iter_callback(self, batch_loss, batch_pcd_loss=None, batch_segment_loss=None):
        # after each iteration we log the losses on tensorboard
        self.loss_eval.append(batch_loss)

        if self.train_step % self.iter_log == 0:
            self.write_summary(
                'training/learning_rate',
                self.scheduler.get_lr()[0],
                self.train_step,
            )

            # loss
            self.write_summary(
                'training/loss',
                mean(self.loss_eval),
                self.train_step,
            )

            self.loss_eval = []
    def downstream_iter_callback(self, batch_loss, batch_loss_c, batch_loss_s, batch_acc, pred, target, is_train):
        # after each iteration we log the losses on tensorboard
        self.evaluator.addBatch(pred.long().cpu().numpy(), target.long().cpu().numpy())
        self.evaluator.addLoss(batch_loss)
        self.loss_c_eval.append(batch_loss_c)
        self.loss_s_eval.append(batch_loss_s)
        
        if self.train_step % self.iter_log == 0 or not is_train:
            if is_train:
                self.write_summary(
                    'training/learning_rate',
                    self.scheduler.get_lr()[0],
                    self.train_step,
                )

            # loss
            self.write_summary(
                'training/loss' if is_train else 'validation/loss',
                self.evaluator.getloss(),
                self.train_step if is_train else self.val_step,
            )

            # Contrastive loss
            self.write_summary(
                'training/loss_c',
                mean(self.loss_c_eval),
                self.train_step,
            )
            
            # Supervised loss
            self.write_summary(
                'training/loss_s',
                mean(self.loss_s_eval),
                self.train_step,
            )

            # accuracy
            self.write_summary(
                'training/acc' if is_train else 'validation/acc',
                self.evaluator.getacc(),
                self.train_step if is_train else self.val_step,
            )

            # mean iou
            mean_iou, class_iou = self.evaluator.getIoU()
            self.write_summary(
                'training/miou' if is_train else 'validation/miou',
                mean_iou.item(),
                self.train_step if is_train else self.val_step,
            )

            # per class iou
            for class_num in range(class_iou.shape[0]):
                self.write_summary(
                    f'training/per_class_iou/{labels[class_num]}' if is_train else f'validation/per_class_iou/{labels[class_num]}',
                    class_iou[class_num].item(),
                    self.train_step if is_train else self.val_step,
                )

            self.evaluator.reset()
            
            
    def downstream_iter_callback_val(self, batch_loss, batch_acc, pred, target, coord, is_train):
        # after each iteration we log the losses on tensorboard
        self.evaluator.addBatch(pred.long().cpu().numpy(), target.long().cpu().numpy())
        self.evaluator.addLoss(batch_loss)

        if self.train_step % self.iter_log == 0 or not is_train:
            if is_train:
                self.write_summary(
                    'training/learning_rate',
                    self.scheduler.get_lr()[0],
                    self.train_step,
                )

            # loss
            self.write_summary(
                'training/loss' if is_train else 'validation/loss',
                self.evaluator.getloss(),
                self.train_step if is_train else self.val_step,
            )

            # accuracy
            self.write_summary(
                'training/acc' if is_train else 'validation/acc',
                self.evaluator.getacc(),
                self.train_step if is_train else self.val_step,
            )

            # mean iou
            mean_iou, class_iou = self.evaluator.getIoU()
            self.write_summary(
                'training/miou' if is_train else 'validation/miou',
                mean_iou.item(),
                self.train_step if is_train else self.val_step,
            )

            # per class iou
            for class_num in range(class_iou.shape[0]):
                self.write_summary(
                    f'training/per_class_iou/{labels[class_num]}' if is_train else f'validation/per_class_iou/{labels[class_num]}',
                    class_iou[class_num].item(),
                    self.train_step if is_train else self.val_step,
                )

            self.evaluator.reset()

            
            
    def validation_epoch_callback(self, curr_loss, curr_acc):
        self.save_checkpoint(f'epoch{self.current_epoch}')

        if self.current_epoch == self.params.epochs - 1:
            self.save_checkpoint(f'lastepoch{self.current_epoch}')

    ############################################################################################################################################

    ############################################################################################################################################
    # SUMMARY WRITERS                                                                                                                          #
    ############################################################################################################################################

    def write_summary(self, summary_id, report, iter):
        self.writer.add_scalar(summary_id, report, iter)

    def contrastive_mesh_writer(self):
        val_iterator = iter(self.train_loader)

        # get just the first iteration(BxNxM) validation set point clouds
        x, y = next(val_iterator)
        z = self.forward(x)
        for i in range(self.params.batch_size):
            points = x.C.cpu().numpy()
            labels = z.max(dim=1)[1].cpu().numpy()

            batch_ind = points[:, 0] == i
            points = expand_dims(points[batch_ind][:, 1:], 0) * self.params.sparse_resolution
            colors = array([ color_map[lbl][::-1] for lbl in labels[batch_ind] ])
            colors = expand_dims(colors, 0)

            point_size_config = {
                'material': {
                    'cls': 'PointsMaterial',
                    'size': 0.3
                }
            }
        
            self.writer.add_mesh(
                f'validation_vis_{i}/point_cloud',
                torch.from_numpy(points),
                torch.from_numpy(colors),
                config_dict=point_size_config,
                global_step=self.current_epoch,
            )

        del val_iterator

    ############################################################################################################################################

    ############################################################################################################################################
    # CHECKPOINT HANDLERS                                                                                                                      #
    ############################################################################################################################################

    def load_checkpoint(self):
        self.configure_optimizers()

        # load model, best loss and optimizer
        file_name = f'{self.params.log_dir}/teacher/lastepoch14_model.pt'
        checkpoint = torch.load(file_name)
        self.moco_model.model_q.load_state_dict(checkpoint['model'])
        print(f'Contrastive {file_name} loaded from epoch {checkpoint["epoch"]}')
        # load model head
        file_name = f'{self.params.log_dir}/teacher/lastepoch199_model_projector.pt'
        checkpoint = torch.load(file_name)
        self.moco_model.head_q.load_state_dict(checkpoint['model'])

        # load model classifier
        file_name = f'{self.params.log_dir}/teacher/lastepoch14_model_head.pt'
        checkpoint = torch.load(file_name)
        self.moco_model.classifier_q.load_state_dict(checkpoint['model'])

    def save_checkpoint(self, checkpoint_id):
        # save the best loss checkpoint
        print(f'Writing model checkpoint for {checkpoint_id}')
        state = {
            'model': self.moco_model.model_q.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
        }
        file_name = f'{self.params.log_dir}/{checkpoint_id}_teacher_{self.params.checkpoint}.pt'
        torch.save(state, file_name)
        
        state = {
            'model': self.moco_model.model_k.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
        }
        file_name = f'{self.params.log_dir}/{checkpoint_id}_student_{self.params.checkpoint}.pt'
        torch.save(state, file_name)
        
        state = {
            'model': self.moco_model.classifier_q.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
        }
        file_name = f'{self.params.log_dir}/{checkpoint_id}_teacher_head_{self.params.checkpoint}.pt'
        torch.save(state, file_name)
        
        state = {
            'model': self.moco_model.classifier_k.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
        }
        file_name = f'{self.params.log_dir}/{checkpoint_id}_student_head_{self.params.checkpoint}.pt'
        torch.save(state, file_name)
        
        state = {
            'model': self.moco_model.head_q.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
        }
        file_name = f'{self.params.log_dir}/{checkpoint_id}_teacher_projector_{self.params.checkpoint}.pt'
        torch.save(state, file_name)
        
        state = {
            'model': self.moco_model.head_k.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
        }
        file_name = f'{self.params.log_dir}/{checkpoint_id}_student_projector_{self.params.checkpoint}.pt'
        torch.save(state, file_name)

        torch.save(self.state_dict(), f'checkpoint/contrastive/{checkpoint_id}_full_model_{self.params.checkpoint}.pt')

    ############################################################################################################################################

    ############################################################################################################################################
    # OPTIMIZER CONFIG                                                                                                                         #
    ############################################################################################################################################

    def configure_optimizers(self):
        # define optimizers
        optimizer = torch.optim.SGD(self.moco_model.parameters(), lr=self.params.lr, momentum=0.9, weight_decay=self.params.decay_lr, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.params.epochs, eta_min=self.params.lr / 1000)

        self.optimizer = optimizer
        self.scheduler = scheduler

        return [optimizer], [scheduler]


    # def configure_optimizers(self):
    #     # define optimizers
    #     if not self.params.linear_eval:
    #         print('Fine-tuning!')
    #         optim_params = list(self.moco_model.model_q.parameters()) + list(self.moco_model.head_q.parameters() + list(self.moco_model.classifier_q.parameters())
    #     else:
    #         print('Linear eval!')
    #         optim_params = list(self.model_head.parameters())
    #         self.model.eval()

    #     #optimizer = torch.optim.Adam(optim_params, lr=self.params.lr, weight_decay=self.params.decay_lr)
    #     optimizer = torch.optim.SGD(
    #         optim_params, lr=self.params.lr, momentum=0.9, weight_decay=self.params.decay_lr, nesterov=True
    #     )

        # def cosine_schedule_with_warmup(k, num_epochs, batch_size, dataset_size):
        #     iter_per_epoch = (dataset_size + batch_size - 1) // batch_size
        #     return 0.5 * (1 + cos(pi * k /
        #                             (num_epochs * iter_per_epoch)))

        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, lr_lambda=partial(
        #         cosine_schedule_with_warmup,
        #         num_epochs=self.params.epochs,
        #         batch_size=self.params.batch_size,
        #         dataset_size=len(self.train_loader) * self.params.batch_size,
        #     )
        # )

        # #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.params.lr, eta_min=self.params.lr / 1000)

        # self.optimizer = optimizer
        # self.scheduler = scheduler

        # return [optimizer], [scheduler]


    ############################################################################################################################################

    #@pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    #@pl.data_loader
    def val_dataloader(self):
        return self.val_loader

    #@pl.data_loader
    def test_dataloader(self):
        pass
