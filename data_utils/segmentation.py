from pcd_utils.pcd_preprocess import *
import numpy as np
import OSToolBox as ost
import argparse


def read_ply(file_name):
    data = ost.read_ply(file_name)
    cloud_x = data['x']
    cloud_y = data['y']
    cloud_z = data['z']
    return(np.c_[cloud_x,cloud_y,cloud_z])

def datapath_list(root, seq_ids, split):
    points_datapath = []
    # self.labels_datapath = []
    for seq in seq_ids:
        point_seq_path = os.path.join(root, split, 'sequences', seq)
        point_seq_bin = os.listdir(point_seq_path)
        point_seq_bin.sort()
        points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]
        
    return points_datapath

def generate_segmented_pc(index, root, dataset, save_path, points_datapath, n_clusters):
    # we need to preprocess the data to get the cuboids and guarantee overlapping points
    # so if we never have done this we do and save this preprocessing
    cluster_path = os.path.join(root, save_path, f'{index}.ply')
    if dataset == "KITTI360":
        points_set = read_ply(points_datapath[index])
        points_set = clusterize_pcd(points_set, n_clusters)
        ost.write_ply(cluster_path, points_set, ['x','y','z','seg'])
    elif dataset == "SemanticKITTI":
        points_set = np.fromfile(points_datapath[index], dtype=np.float32)
        points_set = points_set.reshape((-1, 4))
    
        # remove ground and get clusters from point cloud
        points_set = clusterize_pcd(points_set, n_clusters)

        ost.write_ply(cluster_path, points_set, ['x','y','z','seg'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SparseSimCLR')
    parser.add_argument('--dataset', type=str, 
                        help='Name_of_the_dataset')
    parser.add_argument('--split', type=str, default='train',
                        help='split set (default: train')
    parser.add_argument('--root', type=str, default='path_to_the__dataset_folder',
                        help='Path to the folder of dataset')
    parser.add_argument('--save-path', type=str, default='augmented_views',
                        help='Path to the folder that you want to save the generated files')
    parser.add_argument('--n-clusters', type=int, default=50 ,
                        help='size of each block')
    parser.add_argument('--seq-ids', type=list, default=[0,2,3,4,5,6,7,9,10] ,
                        help='list of sequences #numbers')
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.root, args.save_path)):
        os.makedirs(os.path.join(args.root, args.save_path))
        
    points_datapath = datapath_list(args.root, args.seq_ids, args.split)
    generate_segmented_pc(index, args.root, args.dataset, args.save_path, points_datapath, args.n_clusters)