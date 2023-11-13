import numpy as np
import gc

def blocking(pc, block_size, overlap):
    pc = pc[np.newaxis, :, :]
    deltaX = pc[0, :, 0].max() - pc[0, :, 0].min()
    deltaY = pc[0, :, 1].max() - pc[0, :, 1].min()
    # deltaZ = pc[0, :, 2].max() - pc[0, :, 2].min()
    first_block = np.array([[pc[0, :, 0].min(), pc[0, :, 1].min(), pc[0, :, 2].min()]])
    #Generating X, Y, Z for the next blocks
    x = np.zeros((int(deltaX/block_size)+1),)
    for i in range(int(deltaX/block_size)+1):
        x[i] = first_block[0,0]+i*(block_size)
    
    y = np.zeros((int(deltaY/(block_size-overlap))+1),)
    for j in range(int(deltaY/(block_size-overlap))+1):
        y[j] = first_block[0,1]+j*(block_size-overlap)
    
    z = first_block[0,2]
    
    block_coordinates = np.zeros((x.shape[0]*y.shape[0]*z.size,3))
    
    for i in range(0,x.shape[0]):
        block_coordinates[i*y.shape[0]*z.size:y.shape[0]*z.size+i*y.shape[0]*z.size,0] = x[i]
    for n in range(int(block_coordinates.shape[0]/y.shape[0])):
        for j in range(0,y.shape[0]):   
            block_coordinates[j*z.size+n*z.size*y.shape[0]:(j+1)*z.size+n*z.size*y.shape[0],1] = y[j]
    for k in range(x.shape[0]*y.shape[0]):
        block_coordinates[:,2] = z
    block_coordinates = block_coordinates[np.newaxis ,:, :]
    #Feeding the point into the blocks           
    point_cloud_blocks = [0] * block_coordinates.shape[1]
    for i in range(block_coordinates.shape[1]):
        point_cloud_blocks[i] = pc[:, (pc[0, :, 0] > block_coordinates[0, i, 0]) &
                                    (pc[0, :, 0] <= block_coordinates[0, i, 0]+block_size) &
                                    (pc[0, :, 1] > block_coordinates[0, i, 1]) &
                                    (pc[0, :, 1] <= block_coordinates[0, i, 1]+block_size)]
                                    # (pc[0, :, 2] > block_coordinates[0, i, 2]) &
                                    # (pc[0, :, 2] <= block_coordinates[0, i, 2]+deltaZ)]
        print((i+1)/int(block_coordinates.shape[1])*100)
    point_cloud_blocks = list(filter(lambda a: a.shape[1] > 1000, point_cloud_blocks)) #remove blocks with less than 1000 points
    del pc
    gc.collect()
    return point_cloud_blocks
