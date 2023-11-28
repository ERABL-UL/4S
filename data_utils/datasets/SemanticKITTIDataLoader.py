import numpy as np
import warnings
import os
from torch.utils.data import Dataset
from data_utils.data_map import *
from pcd_utils.pcd_preprocess import *
from pcd_utils.pcd_transforms import *
import MinkowskiEngine as ME
import torch
import json
import OSToolBox as ost

warnings.filterwarnings('ignore')

class SemanticKITTIDataLoader(Dataset):
    def __init__(self, root,  split='train', segment_augment=False, resolution=0.05, intensity_channel=False):
        self.root = root
        self.augmented_dir = 'augmented_views'
        self.n_clusters = 50
        self.segment_augment = segment_augment
        if not os.path.isdir(os.path.join(self.root, self.augmented_dir)):
            os.makedirs(os.path.join(self.root, self.augmented_dir))
        self.resolution = resolution
        self.intensity_channel = intensity_channel

        self.seq_ids = {}
        self.seq_ids['train'] = [ '00', '02', '03', '04', '05', '06', '07', '09', '10']
        self.seq_ids['validation'] = ['00', '02', '03', '04', '05', '06', '07', '09', '10']

        self.split = split

        assert (split == 'train' or split == 'validation')
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list(split)

        print('The size of %s data is %d'%(split,len(self.points_datapath)))


    def datapath_list(self, split):
        self.points_datapath = []
        # self.labels_datapath = []

        for seq in self.seq_ids[split]:
            point_seq_path = os.path.join(self.root, split, 'sequences', seq)
            point_seq_bin = os.listdir(point_seq_path)
            point_seq_bin.sort()
            self.points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]

            # try:
            #     label_seq_path = os.path.join(self.root, split, 'sequences', seq)
            #     point_seq_label = os.listdir(label_seq_path)
            #     point_seq_label.sort()
            #     self.labels_datapath += [ os.path.join(label_seq_path, label_file) for label_file in point_seq_label ]
            # except:
            #     pass


    def transforms_segment(self, points, drop=False):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])
        points[:,:,:3] = jitter_point_cloud(points[:,:,:3])
        if drop == True:
            points = random_drop_n_cuboids(points)

        return np.squeeze(points, axis=0)
    
    
    def transforms(self, points, drop=False):
        if self.split == 'train':
            theta = torch.FloatTensor(1,1).uniform_(0, 2*np.pi).item()
            scale_factor = torch.FloatTensor(1,1).uniform_(0.95, 1.05).item()
            rot_mat = np.array([[np.cos(theta),
                                    -np.sin(theta), 0],
                                [np.sin(theta),
                                    np.cos(theta), 0], [0, 0, 1]])

            points[:, :3] = np.dot(points[:, :3], rot_mat) * scale_factor
            return points
        else:
            return points


    def __len__(self):
        return len(self.points_datapath) #if self.split =="validation" else 21
        # return 
    def read_ply_seg(self, file_name):
        data = ost.read_ply(file_name)
        cloud_x = data['x']
        cloud_y = data['y']
        cloud_z = data['z']
        labels = (data['c']).astype(np.int32)
        labels = labels.reshape(len(labels), 1)
        seg = (data['seg']).astype(np.int32)
        seg = seg.reshape(len(seg), 1)
        return(np.c_[cloud_x,cloud_y,cloud_z,labels,seg])

    def read_ply(self, file_name):
        data = ost.read_ply(file_name)
        cloud_x = data['x']
        cloud_y = data['y']
        cloud_z = data['z']
        labels = (data['c']).astype(np.int32)
        labels = labels.reshape(len(labels), 1)
        return(np.c_[cloud_x,cloud_y,cloud_z],labels)



    def _get_augmented_item(self, index):
        # we need to preprocess the data to get the cuboids and guarantee overlapping points
        # so if we never have done this we do and save this preprocessing
        cluster_path = os.path.join(self.root, self.augmented_dir, f'{index}.ply')
        # print(cluster_path)
        # if os.path.isfile(cluster_path):
            # if the preprocessing is done and saved already we simply load it
        pc = self.read_ply_seg(cluster_path)
        pc = points_set = np.delete(pc, 3, 1)
        points_set = np.c_[points_set, np.asarray(range(points_set.shape[0]))]

        
        # np.random.shuffle(points_set)
            # Px5 -> [x, y, z, i, c] where i is the intesity and c the Cluster associated to the point
        # points_i = random_cuboid_point_cloud(points_set.copy())
        # points_i = self.transforms_segment(points_i, drop=True)
        # points_j = random_cuboid_point_cloud(points_set.copy())
        # points_j = self.transforms_segment(points_j, drop=True)        
        pc_i = self.transforms(pc)
        pc_j = self.transforms(pc)
        # if not self.intensity_channel:
        #     points_i = points_i[:, :3]
        #     points_j = points_j[:, :3]
        # now the point set returns [x,y,z,i,c] always
        del pc, points_set
        # return points_i, points_j, pc_i, pc_j
        return pc_i, pc_j

    def _get_item(self, index):
        points_set,label = self.read_ply(self.points_datapath[index])
        #remap labels to learning values
        labels = np.vectorize(learning_map.get)(label)
        # labels = np.expand_dims(labels, axis=-1)
        unlabeled = labels[:,0] == 0

        # remove unlabeled points
        labels = np.delete(labels, unlabeled, axis=0)
        points_set = np.delete(points_set, unlabeled, axis=0)
        points_set[:, :3] = self.transforms(points_set[:, :3])

        if not self.intensity_channel:
            points_set = points_set[:, :3]

        # now the point set return [x,y,z,i] always
        return points_set, labels.astype(np.int32)

    def __getitem__(self, index):
        return self._get_augmented_item(index) if self.segment_augment else self._get_item(index)
