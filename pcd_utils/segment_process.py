from pcd_utils.pcd_preprocess import *
import MinkowskiEngine as ME
import torch
import numpy as np



def pc_to_segment_pure(params, s_pci, s_pcj, ind, ps_l, num_points, resolution):
    
    pi_cluster = []
    pi_ind = []
    psl_i = []
    
    pj_cluster = []
    pj_ind = []
    psl_j = []
    
    i = 0
    for i in range(s_pci.shape[0]):

        cluster_pi, cluster_pj = overlap_clusters(s_pci[i][ind[i]], s_pcj[i][ind[i]])

        s_pure_i, ind_pure_i, psl_pure_i = segment_purification(params, cluster_pi, ind[i], ps_l[i])
         
        pi_ind.append(ind_pure_i)
        psl_i.append(psl_pure_i)
        
        s_pure_j, ind_pure_j, psl_pure_j = segment_purification(params, cluster_pj, ind[i], ps_l[i])
        
        pj_ind.append(ind_pure_j)
        psl_j.append(psl_pure_j)
        
    
        pi_cluster.append(s_pure_i)
        pj_cluster.append(s_pure_j)
        i += 1
    
    
    segment_i = np.asarray(pi_cluster)
    segment_j = np.asarray(pj_cluster)
    psl_i = np.hstack(psl_i)
    psl_j = np.hstack(psl_j)
    pi_ind = np.asarray(pi_ind)
    pj_ind = np.asarray(pj_ind)
    
    return (segment_i, pi_ind, psl_i), (segment_j, pj_ind, psl_j)

def segment_purification(params, s, ind, ps_l):
    
    psl = []
    pc = np.c_[ps_l, s, ind]  
    unique, count = np.unique(pc[:,1], return_counts=True)
    pc = pc[pc[:, 1].argsort()]
    pc = pc[count[0]:]
    pc_segments = []
    unique, count = np.unique(pc[:,1], return_counts=True)
    for i in range(unique.shape[0]):
        unique_class, count_class = np.unique(pc[sum(count[:i]) : count[i]+sum(count[:i]),0], return_counts=True)
        if params.psl_sup != "psl":
            if unique_class.shape[0] != 1:
                pc1 = pc[sum(count[:i]) : count[i]+sum(count[:i]), :]
                pc_segments.append(pc1[np.where(pc1[:,0] == unique_class[np.argmax(count_class)])])
                psl.append(np.int32(unique_class[np.argmax(count_class)]))
                del pc1
            else:
                pc_segments.append(pc[sum(count[:i]) : count[i]+sum(count[:i]), :])
                psl.append(np.int32(unique_class[0]))
        elif params.psl_sup == "psl":
            pc_segments.append(pc[sum(count[:i]) : count[i]+sum(count[:i]), :])
            psl.append(np.int32(unique_class[0]))
    if len(pc_segments) > 0:
        pc_segments = np.random.permutation(np.vstack(pc_segments))
        return pc_segments[:,1], pc_segments[:,2], psl
    else:
        return s, ind, psl
    
