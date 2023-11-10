import numpy as np
from kde_func import dens_kde
import gc
import h5py


def create_h5_labeled_ply(data,label, dens, h5_filename):
    d = h5py.File(h5_filename, 'w')
    d.create_dataset('data', data = data)
    d.create_dataset('label', data = label)
    d.create_dataset('density', data = dens)
    d.close()


def density_comp(data_pc, file, k):
    data_ds = data_pc[0,:,:]
    data_ds = data_ds[::32]
    values = data_ds[:,:3].T
    density = dens_kde(values)
    density_norm = density/(density.max()/100)
    density_norm = density_norm[:, np.newaxis]

    h5_filename = "/home/reza/PHD/Sum21/test 18/Data"  + "/" + "{0:0=2d}".format(file) + "/" + str(k) + ".h5"
    create_h5_labeled_ply(data_ds[:,:3],data_ds[:,3:4], density_norm, h5_filename)
    del data_ds, values, density_norm, density, data_pc
    gc.collect()

# def split_h5_labeled_KITTI(filename, block_size, overlap, knn, downsampling_rate, high_density_rate, low_density_rate, file):

#     data, label = load_h5_labeled(filename)
#     print(filename)
#     pc = np.append(data, label, axis=2)
#     del data, label
#     delet_classes = [10, 11, 26, 27, 28, 29, 30, 31, 32, 33]
#     for i in delet_classes:
#         pc = np.delete(pc, np.where(pc[:,:,3]==i), axis=1)

#     gc.collect()
#     deltaX = pc[0, :, 0].max() - pc[0, :, 0].min()
#     deltaY = pc[0, :, 1].max() - pc[0, :, 1].min()
#     deltaZ = pc[0, :, 2].max() - pc[0, :, 2].min()
#     first_block = np.array([[pc[0, :, 0].min(), pc[0, :, 1].min(), pc[0, :, 2].min()]])
#     #Generating X, Y, Z for the next blocks
#     x = np.zeros((int(deltaX/block_size)+1),)
#     for i in range(int(deltaX/block_size)+1):
#         x[i] = first_block[0,0]+i*(block_size)
    
#     y = np.zeros((int(deltaY/(block_size-overlap))+1),)
#     for j in range(int(deltaY/(block_size-overlap))+1):
#         y[j] = first_block[0,1]+j*(block_size-overlap)
    
#     z = first_block[0,2]
    
#     block_coordinates = np.zeros((x.shape[0]*y.shape[0]*z.size,3))
    
#     for i in range(0,x.shape[0]):
#         block_coordinates[i*y.shape[0]*z.size:y.shape[0]*z.size+i*y.shape[0]*z.size,0] = x[i]
#     for n in range(int(block_coordinates.shape[0]/y.shape[0])):
#         for j in range(0,y.shape[0]):   
#             block_coordinates[j*z.size+n*z.size*y.shape[0]:(j+1)*z.size+n*z.size*y.shape[0],1] = y[j]
#     for k in range(x.shape[0]*y.shape[0]):
#         block_coordinates[:,2] = z
#     block_coordinates = block_coordinates[np.newaxis ,:, :]
#     #Feeding the point into the blocks           
#     point_cloud_blocks = [0] * block_coordinates.shape[1]
#     for i in range(block_coordinates.shape[1]):
#         point_cloud_blocks[i] = pc[:, (pc[0, :, 0] > block_coordinates[0, i, 0]) &
#                                     (pc[0, :, 0] <= block_coordinates[0, i, 0]+block_size) &
#                                     (pc[0, :, 1] > block_coordinates[0, i, 1]) &
#                                     (pc[0, :, 1] <= block_coordinates[0, i, 1]+block_size)]
#                                     # (pc[0, :, 2] > block_coordinates[0, i, 2]) &
#                                     # (pc[0, :, 2] <= block_coordinates[0, i, 2]+deltaZ)]
#         if point_cloud_blocks[i].shape[1]>1000:
#             # if point_cloud_blocks[i].shape[1]>=1000:
#             #     cores = mp.cpu_count()
#             #     pool = mp.Pool(processes = cores)
                
#             #     data_pc = point_cloud_blocks[i]
#             #     data_ds = data_pc[0,:,:]
                
                
#             #     values = data_ds[:,:3].T
#             #     torun = np.array_split(values, cores, axis=1)
#             #     r = pool.map(dens_kde, torun)
#             #     density = np.concatenate(r)
                
#             #     density_norm = density/(density.max()/100)
#             #     density_norm = density_norm[:, np.newaxis]
#             #     pool.terminate()
#             #     pool.join() 
#             #     del torun, r, density
#             # else:
#             data_pc = point_cloud_blocks[i]
#             data_ds = data_pc[0,:,:]
            
          
#             data_ds = data_ds[::32]
#             values = data_ds[:,:3].T
#             density = dens_kde(values)
#             density_norm = density/(density.max()/100)
#             density_norm = density_norm[:, np.newaxis]
                
#             # for j in range(density_norm.shape[0]):
#             #     if density_norm[j,0] >= high_density_rate:
#             #         high += 1
#             #     if density_norm[j,0] <= low_density_rate:
#             #         low += 1
#             h5_filename = "/home/reza/PHD/Sum21/test 18/Data"  + "/" + "{0:0=2d}".format(file) + "/" + str(i) + ".h5"
#             create_h5_labeled_ply(data_ds[:,:3],data_ds[:,3:4], density_norm, h5_filename)
#             del data_pc, data_ds, values, density_norm
#         print((i+1)/int(block_coordinates.shape[1])*100, " file: ", file)
#         point_cloud_blocks[i] = []
#     # if knn > 1000:
#     #     point_cloud_blocks = list(filter(lambda a: a.shape[1] > knn, point_cloud_blocks))
#     # else:
#     #     point_cloud_blocks = list(filter(lambda a: a.shape[1] > 1000, point_cloud_blocks))
#     del pc, deltaX, deltaY, deltaZ, first_block, x, y, z, block_coordinates
#     gc.collect() 


# def load_h5_dens(h5_filename):
#     f = h5py.File(h5_filename)
#     data = f['data'][:]
#     label = f['label'][:]
#     dens = f['density'][:]
#     del f
#     return (data,label,dens)

# import os
# files = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
# for file in files:
#     all_data = []
#     sequence_path = "/home/reza/PHD/Sum21/test 18/Data"  + "/" + "{0:0=2d}".format(file)
#     # data_path = "/home/reza/PHD/Data/PL3D/DB/sequences/00/"
#     for filename in (os.scandir(sequence_path)):
#         pred_path = os.path.join(sequence_path, filename)
#         # pred = np.load(pred_path)
#         # true_path = data_path + str(0) + filename.name[:(len(filename.name)-4)]
#         data, label , dens = load_h5_dens(pred_path)
        
#         pc1 = np.c_[data, label , dens]
#         all_data.append(pc1)
    
    
#     all_pcs = np.concatenate(all_data,axis=0)
    
    
    
#     high = 0
#     low = 0
#     for j in range(all_pcs.shape[0]):
#         if all_pcs[j,4] >= 70:
#             high += 1
#         # if all_pcs[j,4] < 30:
#         #     low += 1
#     print(high*100/all_pcs.shape[0])


# data, label , dens = load_h5_dens(h5_filename)
# pc1 = np.c_[data, label , dens]

# from ply import write_ply
# field_names = ['x', 'y', 'z','label','density' ]
# write_ply('04.ply', all_pcs, field_names)

