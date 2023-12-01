import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils.collations import *

latent_features = {
    'SparseResNet14': 512,
    'SparseResNet18': 1024,
    'SparseResNet34': 2048,
    'SparseResNet50': 2048,
    'SparseResNet101': 2048,
    'MinkUNet': 96,
    'MinkUNetSMLP': 96,
    'MinkUNet14': 96,
    'MinkUNet18': 1024,
    'MinkUNet34': 2048,
    'MinkUNet50': 2048,
    'MinkUNet101': 2048,
}

class MoCoVICReg(nn.Module):
    def __init__(self, model, model_head, dtype, args, K=65536, m=0.999, T=0.1):
        super(MoCoVICReg, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.model_q = model(in_channels=4 if args.use_intensity else 3, out_channels=latent_features[args.sparse_model])#.type(dtype)
        self.head_q = model_head(in_channels=latent_features[args.sparse_model], out_channels=args.feature_size)#.type(dtype)

        self.model_k = model(in_channels=4 if args.use_intensity else 3, out_channels=latent_features[args.sparse_model])#.type(dtype)
        self.head_k = model_head(in_channels=latent_features[args.sparse_model], out_channels=args.feature_size)#.type(dtype)

        # initialize model k and q
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # initialize headection head k and q
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer('queue_pcd', torch.randn(args.feature_size,K))
        self.queue_pcd = nn.functional.normalize(self.queue_pcd, dim=0)

        self.register_buffer('queue_seg', torch.randn(args.feature_size,K))
        self.queue_seg = nn.functional.normalize(self.queue_seg, dim=0)

        self.register_buffer("queue_pcd_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_seg_ptr", torch.zeros(1, dtype=torch.long))

        if torch.cuda.device_count() > 1:
            self.model_q = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model_q)
            self.head_q = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.head_q)

            self.model_k = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model_k)
            self.head_k = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.head_k)

        self.args = args

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size = []

        # sparse tensor should be decomposed
        c, f = x.decomposed_coordinates_and_features

        # each pcd has different size, get the biggest size as default
        newx = list(zip(c, f))
        for bidx in newx:
            batch_size.append(len(bidx[0]))
        all_size = concat_all_gather(torch.tensor(batch_size).cuda())
        max_size = torch.max(all_size)

        # create a tensor with shape (batch_size, max_size)
        # copy each sparse tensor data to the begining of the biggest sized tensor
        shuffle_c = []
        shuffle_f = []
        for bidx in range(len(newx)):
            shuffle_c.append(torch.ones((max_size, newx[bidx][0].shape[-1])).cuda())
            shuffle_c[bidx][:len(newx[bidx][0]),:] = newx[bidx][0]

            shuffle_f.append(torch.ones((max_size, newx[bidx][1].shape[-1])).cuda())
            shuffle_f[bidx][:len(newx[bidx][1]),:] = newx[bidx][1]

        batch_size_this = len(newx)

        shuffle_c = torch.stack(shuffle_c)
        shuffle_f = torch.stack(shuffle_f)

        # gather all the ddp batches pcds
        c_gather = concat_all_gather(shuffle_c)
        f_gather = concat_all_gather(shuffle_f)

        batch_size_all = c_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        c_this = []
        f_this = []
        batch_id = []

        # after shuffling we get only the actual information of each tensor
        # :actual_size is the information, actual_size:biggest_size are just ones (ignore)
        for idx in range(len(idx_this)):
            c_this.append(c_gather[idx_this[idx]][:all_size[idx_this[idx]],:].cpu().numpy())
            f_this.append(f_gather[idx_this[idx]][:all_size[idx_this[idx]],:].cpu().numpy())

        # final shuffled coordinates and features, build back the sparse tensor
        c_this = np.array(c_this)
        f_this = np.array(f_this)
        x_this = numpy_to_sparse_tensor(c_this, f_this)

        return x_this, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size = []

        # sparse tensor should be decomposed
        c, f = x.decomposed_coordinates_and_features

        # each pcd has different size, get the biggest size as default
        newx = list(zip(c, f))
        for bidx in newx:
            batch_size.append(len(bidx[0]))
        all_size = concat_all_gather(torch.tensor(batch_size).cuda())
        max_size = torch.max(all_size)

        # create a tensor with shape (batch_size, max_size)
        # copy each sparse tensor data to the begining of the biggest sized tensor
        shuffle_c = []
        shuffle_f = []
        for bidx in range(len(newx)):
            shuffle_c.append(torch.ones((max_size, newx[bidx][0].shape[-1])).cuda())
            shuffle_c[bidx][:len(newx[bidx][0]),:] = newx[bidx][0]

            shuffle_f.append(torch.ones((max_size, newx[bidx][1].shape[-1])).cuda())
            shuffle_f[bidx][:len(newx[bidx][1]),:] = newx[bidx][1]

        batch_size_this = len(newx)

        shuffle_c = torch.stack(shuffle_c)
        shuffle_f = torch.stack(shuffle_f)

        # gather all the ddp batches pcds
        c_gather = concat_all_gather(shuffle_c)
        f_gather = concat_all_gather(shuffle_f)

        batch_size_all = c_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        c_this = []
        f_this = []
        batch_id = []

        # after unshuffling we get only the actual information of each tensor
        # :actual_size is the information, actual_size:biggest_size are just ones (ignore)
        for idx in range(len(idx_this)):
            c_this.append(c_gather[idx_this[idx]][:all_size[idx_this[idx]],:].cpu().numpy())
            f_this.append(f_gather[idx_this[idx]][:all_size[idx_this[idx]],:].cpu().numpy())

        # final unshuffled coordinates and features, build back the sparse tensor
        c_this = np.array(c_this)
        f_this = np.array(f_this)
        x_this = numpy_to_sparse_tensor(c_this, f_this)

        return x_this

    def forward(self, pcd_q, pcd_k, segments=None):
        """
        Input:
            pcd_q: a batch of query pcds
            pcd_k: a batch of key pcds
        Output:
            logits, targets
        """
        assert segments is not None

        # compute query features
        h_q = self.model_q(pcd_q)  # queries: NxC

        if segments is None:
            z_q = self.head_q(h_q)
            q_pcd = nn.functional.normalize(z_q, dim=1)
        else:
            # coord and feat in the shape N*SxPx3 and N*SxPxF
            # where N is the batch size and S is the number of segments in each scan

            h_qs = list_segments_points(h_q.C, h_q.F, segments[0])

            z_qs = self.head_q(h_qs)
            q_seg = nn.functional.normalize(z_qs, dim=1)

        # compute key features
        # shuffle for making use of BN
        #if torch.cuda.device_count() > 1:
        #    pcd_k, idx_unshuffle = self._batch_shuffle_ddp(pcd_k)

        # many Batch Normalization operations we shuffle before it to have shuffle BN
        h_k = self.model_k(pcd_k)  # keys: NxC

        # IMPORTANT: in the projection head we dont have batch normalization, so unshuffling before
        # passing over the proj head will maintain the shuffle BN chracteristics
        #if torch.cuda.device_count() > 1:
        #    h_k = self._batch_unshuffle_ddp(h_k, idx_unshuffle)

        if segments is None:
            z_k = self.head_k(h_k)
            k_pcd = nn.functional.normalize(z_k, dim=1)
        else:
            # coord and feat in the shape N*SxPx3 and N*SxPxF
            # where N is the batch size and S is the number of segments in each scan
            h_ks = list_segments_points(h_k.C, h_k.F, segments[1])

            z_ks = self.head_k(h_ks)
            k_seg = nn.functional.normalize(z_ks, dim=1)

        if segments is None:
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos_pcd = torch.einsum('nc,nc->n', [q_pcd, k_pcd]).unsqueeze(-1)
            # negative logits: NxK
            l_neg_pcd = torch.einsum('nc,ck->nk', [q_pcd, self.queue_pcd.clone().detach()])

            # logits: Nx(1+K)
            logits_pcd = torch.cat([l_pos_pcd, l_neg_pcd], dim=1)

            # apply temperature
            logits_pcd /= self.T

            # labels: positive key indicators
            labels_pcd = torch.zeros(logits_pcd.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            self._dequeue_and_enqueue_pcd(k_pcd)

            return logits_pcd, labels_pcd
        else:
            return self.vicreg_loss(q_seg, k_seg)
        
    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def vicreg_loss(self, x, y):
        # Variance
        diff_x = x - torch.mean(x, dim=0)
        diff_y = y - torch.mean(y, dim=0)
        std_x = torch.sqrt(diff_x.var(dim=0) + self.args.vicreg_eps)
        std_y = torch.sqrt(diff_y.var(dim=0) + self.args.vicreg_eps)
        var_loss_x = torch.mean(F.relu(self.args.vicreg_gamma - std_x))
        var_loss_y = torch.mean(F.relu(self.args.vicreg_gamma - std_y))
        var_loss = (var_loss_x + var_loss_y) / 2

        # Invariance
        inv_loss = F.mse_loss(x, y)

        # Covariance
        cov_x = (diff_x.T @ diff_x) / (x.shape[0] - 1)
        cov_y = (diff_y.T @ diff_y) / (y.shape[0] - 1)
        cov_loss_x = self.off_diagonal(cov_x).pow_(2).sum().div(x.shape[1])
        cov_loss_y = self.off_diagonal(cov_y).pow_(2).sum().div(y.shape[1])
        cov_loss = (cov_loss_x + cov_loss_y) / 2

        # Total loss
        loss = self.args.vicreg_var_coeff * var_loss + \
               self.args.vicreg_inv_coeff * inv_loss + \
               self.args.vicreg_cov_coeff * cov_loss

        return loss

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
