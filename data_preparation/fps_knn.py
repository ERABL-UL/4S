import numpy as np
import OSToolBox as ost
from dgl.geometry import farthest_point_sampler as FPS
import torch
import open3d as o3d
import os

def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def fps_knn_blcoks(pc0, num_seeds, knn, seq, f, split, save_folder_path):
    pc0 = pc0[0,:,:]
    t_pc = torch.unsqueeze(torch.Tensor(pc0),0)
    fps_pts = FPS(t_pc, num_seeds)
    #fps_pts = FPS(t_pc, (pc0.shape[0]*3)/8192*16) #if you want to choose different number of seeds per cloud
    pcd0 = make_open3d_point_cloud(pc0[:,:3])
    pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd0)
    fps_pts = pc0[fps_pts][0]
    k = 0
    for i, point in enumerate(fps_pts):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point[:3], knn)
        # [_, idx, _] = pcd_tree.search_radius_vector_3d(point[:3], 10) #if you want radius search
        if len(idx) > 10:
            pc = pc0[idx]
            ost.write_ply(save_path + "/" + split + "/sequences/" +
                          "{0:0=2d}".format(seq) + "/" + str(f) +
                          str(k) + ".ply", pc, ['x','y','z','c'])
            k += 1
    del pc0, pcd0, fps_pts, pcd_tree, t_pc, pc



