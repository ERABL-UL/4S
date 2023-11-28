import torch
import torch.nn as nn
from data_utils.collations import *
from pcd_utils.segment_process import pc_to_segment_pure

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
#
class MoCo(nn.Module):
    def __init__(self, model, model_head, model_classifier, dtype, args, K=65536, T=0.1):
        super(MoCo, self).__init__()
        self.args = args
        self.K = K
        self.T = T
        self.num_points = self.args.num_points
        self.resolution = self.args.sparse_resolution
        self.model_q = model(in_channels=4 if self.args.use_intensity else 3, out_channels=latent_features[self.args.sparse_model])#.type(dtype)
        self.head_q = model_head(in_channels=latent_features[self.args.sparse_model], out_channels=self.args.feature_size)#.type(dtype)
        self.classifier_q = model_classifier(in_channels=latent_features[self.args.sparse_model], out_channels=self.args.num_classes)#.type(dtype)
        
        self.model_k = model(in_channels=4 if self.args.use_intensity else 3, out_channels=latent_features[self.args.sparse_model])#.type(dtype)
        self.head_k = model_head(in_channels=latent_features[self.args.sparse_model], out_channels=self.args.feature_size)#.type(dtype)
        self.classifier_k = model_classifier(in_channels=latent_features[self.args.sparse_model], out_channels=self.args.num_classes)#.type(dtype)

        # initialize model k and q
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)
            # param_k.requires_grad = True
            param_q.requires_grad = False

        # initialize headection head k and q
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)
            # param_k.requires_grad = True
            param_q.requires_grad = False

        # initialize classifier head k and q
        for param_q, param_k in zip(self.classifier_q.parameters(), self.classifier_k.parameters()):
            param_k.data.copy_(param_q.data)
            # param_k.requires_grad = True
            param_q.requires_grad = False

        self.register_buffer('queue_pcd', torch.randn(self.args.feature_size,K))
        self.queue_pcd = nn.functional.normalize(self.queue_pcd, dim=0)

        self.register_buffer('queue_seg', torch.randn(self.args.feature_size,K))
        self.queue_seg = nn.functional.normalize(self.queue_seg, dim=0)

        self.register_buffer('queue_seg_psl', torch.tensor(np.ones((1,K))*(-1)))


        self.register_buffer("queue_pcd_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_seg_ptr", torch.zeros(1, dtype=torch.long))

        if torch.cuda.device_count() > 1:
            self.model_q = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model_q)
            self.head_q = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.head_q)
            self.classifier_q = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.classifier_q)

            self.model_k = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model_k)
            self.head_k = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.head_k)
            self.classifier_k = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.classifier_k)


    @torch.no_grad()
    def _dequeue_and_enqueue_pcd(self, keys):
        # gather keys before updating queue
        if torch.cuda.device_count() > 1:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_pcd_ptr)
        #assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.K:
            self.queue_pcd[:, ptr:ptr + batch_size] = keys.T
        else:
            tail_size = self.K - ptr
            head_size = batch_size - tail_size
            self.queue_pcd[:, ptr:self.K] = keys.T[:, :tail_size]
            self.queue_pcd[:, :head_size] = keys.T[:, tail_size:]

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_pcd_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_seg(self, keys, keys_psl=None):

        # gather keys before updating queue
        if torch.cuda.device_count() > 1:
            # similar to shuffling, since for each gpu the number of segments may not be the same
            # we create a aux variable keys_gather of size (1, MAX_SEG_BATCH, 128)
            # add the current seg batch to [0,:CURR_SEG_BATCH, 128] gather them all in
            # [NUM_GPUS,MAX_SEG_BATCH,128] and concatenate only the filled seg batches
            seg_size = torch.from_numpy(np.array([keys.shape[0]])).cuda()
            all_seg_size = concat_all_gather(seg_size)

            keys_gather = torch.ones((1, all_seg_size.max(), keys.shape[-1])).cuda()
            keys_gather[0, :keys.shape[0],:] = keys[:,:]

            all_keys = concat_all_gather(keys_gather)
            gather_keys = None

            for k in range(len(all_seg_size)):
                if gather_keys is None:
                    gather_keys = all_keys[k][:all_seg_size[k],:]
                else:
                    gather_keys = torch.cat((gather_keys, all_keys[k][:all_seg_size[k],:]))


            keys = gather_keys

        batch_size = keys.shape[0]

        ptr = int(self.queue_seg_ptr)
        #assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if keys_psl == None:
            if ptr + batch_size <= self.K:
                self.queue_seg[:, ptr:ptr + batch_size] = keys.T
            else:
                tail_size = self.K - ptr
                head_size = batch_size - tail_size
                self.queue_seg[:, ptr:self.K] = keys.T[:, :tail_size]
                self.queue_seg[:, :head_size] = keys.T[:, tail_size:]
        else:
            if ptr + batch_size <= self.K:
                self.queue_seg[:, ptr:ptr + batch_size] = keys.T
                self.queue_seg_psl[:, ptr:ptr + batch_size] = keys_psl
            else:
                tail_size = self.K - ptr
                head_size = batch_size - tail_size
                self.queue_seg[:, ptr:self.K] = keys.T[:, :tail_size]
                self.queue_seg[:, :head_size] = keys.T[:, tail_size:]
                self.queue_seg_psl[:, ptr:self.K] = keys_psl[:tail_size]
                self.queue_seg_psl[:, :head_size] = keys_psl[tail_size:]





        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_seg_ptr[0] = ptr

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


    @torch.no_grad()
    def _EMA_teacher_update(self, alpha=0.99999):
        """
        Momentum update of the teacher
        """
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_q.data = param_q.data * alpha + param_k.data * (1. - alpha)

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_q.data = param_q.data * alpha + param_k.data * (1. - alpha)

        for param_q, param_k in zip(self.classifier_q.parameters(), self.classifier_k.parameters()):
            param_q.data = param_q.data * alpha + param_k.data * (1. - alpha)
            
    @torch.no_grad()
    def psl_selection(self, ps_l_conf, ps_l):
        conf = [0, 0.959, 0.844, 0.672, 0.933, 0.409, 0.879]
        ind = []
        for _class in np.unique(ps_l):
            class_ind = ps_l == _class
            _class_ind = np.argwhere(class_ind==True)
            ind_true = ps_l_conf[class_ind][:,_class] > conf[_class]
            ind.append((_class_ind[ind_true])[:,0])
        ind = np.hstack(ind)
        np.sort(ind)
   
        return ind
    
    @torch.no_grad()
    def segment_purification(self,  x_coord, x_feats, s, idx, ps_l):
        ps_l_idx = np.c_[np.asarray(ps_l.cpu()),np.asarray(range(ps_l.shape[0]))]

        coord_t = []
        feats_t = []
        s_t = []
        psl = []
        for pc_idx in range(x_coord.shape[0]):
            ps_l_i = ps_l_idx[np.int32(idx[pc_idx])]
            pc = np.c_[x_coord[pc_idx], x_feats[pc_idx], ps_l_i,s[pc_idx]]          
            pc = np.c_[pc[:,:7],pc[:,8:]]
            unique, count = np.unique(pc[:,7], return_counts=True)
            pc1 = pc[pc[:, 7].argsort()]
            a = count[0]
            pc3 = []
            pc3.append(pc1[:count[0]])
            for i in range(1,unique.shape[0]):
                unique_seg, count_seg = np.unique(pc1[a:count[i]+a,6], return_counts=True)
                if self.params.psl_sup != "psl":
                    if unique_seg.shape[0] != 1:
                        pc2 = pc1[a:count[i]+a,:]
                        pc3.append(pc2[np.where(pc2[:,6] == unique_seg[np.argmax(count_seg)])])
                        psl.append(np.int32(unique_seg[np.argmax(count_seg)]))
                        del pc2
                    else:
                        psl.append(np.int32(unique_seg[0]))
                        pc3.append(pc1[a:count[i]+a,:])
                elif self.params.psl_sup == "psl":
                    psl.append(np.int32(unique_seg[0]))
                    pc3.append(pc1[a:count[i]+a,:])
                a += count[i]
            
            pc3 = np.random.permutation(np.vstack(pc3))
            coord_t.append(pc3[:,:3])
            feats_t.append(pc3[:,3:6])
            s_t.append(pc3[:,7])
            
        return coord_t, feats_t, s_t, psl

    def forward(self, pcd_q, pcd_k, s_pci, s_pcj, inv_i, inv_j, train_step, current_epoch):
        """
        Input:
            pcd_q: a batch of query pcds
            pcd_k: a batch of key pcds
        Output:
            logits, targets
        """

        
        #updating the teacher using EMA, except the first itteration of the first epoch
        if train_step != 1 or current_epoch != 0:
            self._EMA_teacher_update()
        # training teacher and generating pseudo-labels
        ps_l = []
        with torch.no_grad():
            h_q = self.model_q(pcd_q)  # queries: NxC
            pred_q = self.classifier_q(h_q)
            ps_l.append((pred_q.max(dim=1)[1]).cpu())

        # training student and generating predictions
        h_k = self.model_k(pcd_k)  # keys: NxC
        pred_k = self.classifier_k(h_k)
        ps_l.append((pred_k.max(dim=1)[1]).cpu())
        
        #pseudo-label selection
        ps_l_conf = torch.softmax(pred_q, dim=1)
        ind = self.psl_selection(ps_l_conf.cpu().numpy(), ps_l[0].numpy())
        
        #getting the selected pseudo-labels for each batch with their inds from the original batches
        selected_inds = []
        ps_l_true = []
        _len = 0
        for batch_num in range(s_pci.shape[0]):
            batch_ind = h_q.C[:,0].cpu().numpy() == batch_num
            selected_inds.append(batch_ind[ind])
            selected_inds[batch_num] = ind[selected_inds[batch_num]]
            selected_inds[batch_num] = selected_inds[batch_num]-_len
            _len += len(np.argwhere(batch_ind==True))
            ps_l_true.append(ps_l[0][selected_inds[batch_num]].numpy())
        
        #getting the coords, feats and segments for the projection head
        if self.args.pure == False and self.args.psl_sup == 'none':
            c_coord_i = h_q.C
            c_feats_i = h_q.F
            segment_i = s_pci
            
            c_coord_j = h_k.C
            c_feats_j = h_k.F
            segment_j = s_pcj

        else: 
            #segment purification and false negative pair removal
            (segment_i, pi_ind, psl_i), (segment_j, pj_ind, psl_j) = pc_to_segment_pure(self.args, s_pci, s_pcj, selected_inds, ps_l_true, self.num_points, self.resolution)
            
            c_coord_i = []
            c_feats_i = []
            
            c_coord_j = []
            c_feats_j = []
            
            for batch_num in range(pi_ind.shape[0]):
                batch_ind_i = h_q.C[:,0] == batch_num
                c_coord_i.append(h_q.C[batch_ind_i][pi_ind[batch_num]])
                c_feats_i.append(h_q.F[batch_ind_i][pi_ind[batch_num]])
                
                batch_ind_j = h_k.C[:,0] == batch_num
                c_coord_j.append(h_k.C[batch_ind_j][pj_ind[batch_num]])
                c_feats_j.append(h_k.F[batch_ind_j][pj_ind[batch_num]])
                
            c_coord_i = torch.vstack(c_coord_i)
            c_feats_i = torch.vstack(c_feats_i)
            segment_i = np.asarray(segment_i)
            
            c_coord_j = torch.vstack(c_coord_j)
            c_feats_j = torch.vstack(c_feats_j)
            segment_j = np.asarray(segment_j)          
        
        #training the projection head for teacher and student to get the positive and negative embadings for the Contrastive loss   
        with torch.no_grad():
           
            h_qs = list_segments_points(c_coord_i, c_feats_i , segment_i)
            z_qs = self.head_q(h_qs)
            q_seg = nn.functional.normalize(z_qs, dim=1)
            
        h_ks = list_segments_points(c_coord_j, c_feats_j , segment_j)
        z_ks = self.head_k(h_ks)
        k_seg = nn.functional.normalize(z_ks, dim=1)
        
        #Positive logits
        l_pos_seg = torch.einsum('nc,nc->n', [q_seg, k_seg]).unsqueeze(-1)

        if self.args.psl_sup != "none":
            #getting Negative logits using class-wise memory banks
            if len(np.unique(psl_i)) > 1:
                l_neg_seg = []
                l_pos_seg_i = []
                for i in np.unique(psl_i):
                    neg_bank_ci = self.queue_seg.clone().detach()[:,np.argwhere((self.queue_seg_psl!=i)[0].cpu().numpy()).squeeze()]
                    q_seg_ci = q_seg[np.argwhere(psl_i==i).squeeze(),:]
                    l_pos_seg_i.append(l_pos_seg[np.argwhere(psl_i==i).squeeze(),:])
                    if len(q_seg_ci.shape) > 1:
                        l_neg_seg.append(torch.einsum('nc,ck->nk', [q_seg_ci, neg_bank_ci]))
                    else:
                        l_neg_seg.append(torch.einsum('nc,ck->nk', [torch.unsqueeze(q_seg_ci,0), neg_bank_ci]))
            
                min_shape = min(arr.shape[1] for arr in l_neg_seg)
                for i, tensor in enumerate(l_neg_seg):
                    l_neg_seg[i] = tensor[:,:min_shape]
                l_neg_seg = torch.vstack(l_neg_seg)
                l_pos_seg = torch.vstack(l_pos_seg_i)
            
            else:
                l_neg_seg = torch.einsum('nc,ck->nk', [q_seg, self.queue_seg.clone().detach()])
            # dequeue and enqueue
            self._dequeue_and_enqueue_seg(k_seg, torch.tensor(psl_j))
        else:
            #Negative logits using a memory bank
            l_neg_seg = torch.einsum('nc,ck->nk', [q_seg, self.queue_seg.clone().detach()])
            # dequeue and enqueue
            self._dequeue_and_enqueue_seg(k_seg)

#######################################################################        
        
        # logits: Nx(1+K)
        logits_seg = torch.cat([l_pos_seg, l_neg_seg], dim=1)
        # apply temperature
        logits_seg /= self.T
        # labels: positive key indicators
        labels_seg = torch.zeros(logits_seg.shape[0], dtype=torch.long).cuda()
        # return logits_seg, labels_seg
        return logits_seg, labels_seg, pred_q, pred_k

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
