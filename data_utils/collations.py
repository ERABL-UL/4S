import numpy as np
import MinkowskiEngine as ME
import torch
from pcd_utils.pcd_preprocess import overlap_clusters

def array_to_sequence(batch_data):
        return [ row for row in batch_data ]

def array_to_torch_sequence(batch_data):
    return [ torch.from_numpy(row).float() for row in batch_data ]

def list_segments_points(p_coord, p_feats, labels):
    c_coord = []
    c_feats = []
    seg_batch_count = 0

    for batch_num in range(labels.shape[0]):
        for segment_lbl in np.unique(labels[batch_num]):
            if segment_lbl == -1:
                continue

            batch_ind = p_coord[:,0] == batch_num
            segment_ind = labels[batch_num] == segment_lbl

            # we are listing from sparse tensor, the first column is the batch index, which we drop
            segment_coord = p_coord[batch_ind][segment_ind][:,:]
            segment_coord[:,0] = seg_batch_count
            seg_batch_count += 1

            segment_feats = p_feats[batch_ind][segment_ind]

            c_coord.append(segment_coord)
            c_feats.append(segment_feats)
    
    seg_coord = torch.vstack(c_coord)
    seg_feats = torch.vstack(c_feats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return ME.SparseTensor(
                features=seg_feats,
                coordinates=seg_coord,
                device=device,
            )

def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(array_to_sequence(p_coord), dtype=torch.float32)
    p_feats = ME.utils.batched_coordinates(array_to_torch_sequence(p_feats), dtype=torch.float32)[:, 1:]
    if p_label is not None:
        p_label = ME.utils.batched_coordinates(array_to_torch_sequence(p_label), dtype=torch.float32)[:, 1:]
        return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            ), p_label.cuda()
    
    return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            )


def point_set_to_coord_feats_idx(point_set, labels, idx, resolution, num_points, deterministic=False):
    p_feats = point_set.copy()
    p_coord = np.round(point_set[:, :3] / resolution)
    p_coord -= p_coord.min(0, keepdims=1)

    _, mapping = ME.utils.sparse_quantize(coordinates=p_coord, return_index=True)
    if len(mapping) > num_points:
        if deterministic:
            # for reproducibility we set the seed
            np.random.seed(42)
        mapping = np.random.choice(mapping, num_points, replace=False)
    return p_coord[mapping], p_feats[mapping], labels[mapping], idx[mapping]

def point_set_to_coord_feats(point_set, labels, resolution, num_points, deterministic=False):
    p_feats = point_set.copy()
    p_coord = np.round(point_set[:, :3] / resolution)
    p_coord -= p_coord.min(0, keepdims=1)

    _, mapping, inverse = ME.utils.sparse_quantize(coordinates=p_coord, return_index=True, return_inverse=True)
    if len(mapping) > num_points:
        if deterministic:
            # for reproducibility we set the seed
            np.random.seed(42)
        mapping = np.random.choice(mapping, num_points, replace=False)
    return p_coord[mapping], p_feats[mapping], labels[mapping], inverse.numpy()

def collate_points_to_sparse_tensor(pi_coord, pi_feats, pj_coord, pj_feats):
    # voxelize on a sparse tensor
    points_i = numpy_to_sparse_tensor(pi_coord, pi_feats)
    points_j = numpy_to_sparse_tensor(pj_coord, pj_feats)

    return points_i, points_j


class SparseAugmentedCollation:
    def __init__(self, resolution, num_points=80000):
        self.resolution = resolution
        self.num_points = num_points

    def __call__(self, list_data):
        # points_i, points_j, pc_i, pc_j = list(zip(*list_data))
        pc_i, pc_j = list(zip(*list_data))
        pc_i = np.asarray(pc_i)
        pc_j = np.asarray(pc_j)

        pci_coord = []
        pci_feats = []
        pci_inverse = []
        pci_cluster = []
        
        pcj_coord = []
        pcj_feats = []
        pcj_inverse = []
        pcj_cluster = []
        
        for pi, pj in zip(pc_i, pc_j):
            # pi[:,:-1] will be the points and intensity values, and the labels will be the cluster ids
            coord_pi, feats_pi, cluster_pi, inverse_pi = point_set_to_coord_feats(pi[:,:3], pi[:,3], self.resolution, pi.shape[0])
            pci_coord.append(coord_pi)
            pci_feats.append(coord_pi)
            pci_inverse.append(inverse_pi)
            # pj[:,:-1] will be the points and intensity values, and the labels will be the cluster ids
            coord_pj, feats_pj, cluster_pj, inverse_pj = point_set_to_coord_feats(pj[:,:3], pj[:,3], self.resolution, pi.shape[0])
            pcj_coord.append(coord_pj)
            pcj_feats.append(coord_pj)
            pcj_inverse.append(inverse_pj)
            
            cluster_pi, cluster_pj = overlap_clusters(cluster_pi, cluster_pj)

            pci_cluster.append(cluster_pi)
            pcj_cluster.append(cluster_pj)
            
        pci_coord = np.asarray(pci_coord)
        pci_feats = np.asarray(pci_feats)
        pci_inverse = np.asarray(pci_inverse)
        
        pcj_coord = np.asarray(pcj_coord)
        pcj_feats = np.asarray(pcj_feats)
        pcj_inverse = np.asarray(pcj_inverse)
        
        segment_pci = np.asarray(pci_cluster)
        segment_pcj = np.asarray(pcj_cluster)
        
        return (pci_coord, pci_feats, segment_pci, pci_inverse), (pcj_coord, pcj_feats, segment_pcj, pcj_inverse)

class SparseCollation:
    def __init__(self, resolution, num_points=80000):
        self.resolution = resolution
        self.num_points = num_points

    def __call__(self, list_data):
        points_set, labels = list(zip(*list_data))

        points_set = np.asarray(points_set)
        labels = np.asarray(labels)

        p_feats = []
        p_coord = []
        p_label = []
        for points, label in zip(points_set, labels):
            coord, feats, label_, _ = point_set_to_coord_feats(points, label, self.resolution, self.num_points, True)
            p_feats.append(feats)                         
            p_coord.append(coord)
            p_label.append(label_)

        p_feats = np.asarray(p_feats)
        p_coord = np.asarray(p_coord)
        p_label = np.asarray(p_label)

        # if we directly map coords and feats to SparseTensor it will loose the map over the coordinates
        # if the mapping between point and voxels are necessary, please use TensorField
        # as in https://nvidia.github.io/MinkowskiEngine/demo/segmentation.html?highlight=segmentation
        # we first create TensorFields and from it we create the sparse tensors, so we can map the coordinate
        # features across different SparseTensors, i.e. output prediction and target labels

        return p_coord, p_feats, p_label
