from fps_knn import fps_knn_blcoks
import time
import numpy as np
from blocks import blocking
import argparse
import threading
import OSToolBox as ost

def read_ply(ply_path):
    data0 = ost.read_ply(ply_path)
    cloud_x = data0['x']
    cloud_y = data0['y']
    cloud_z = data0['z']
    labels = data0['scalar_c']
    pc0 = np.c_[cloud_x, cloud_y, cloud_z, labels]
    return pc0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SparseSimCLR')
    parser.add_argument('--sequences', type=list, default=[0,2,3,4,5,6,7,9,10] ,
                        help='list of sequences #numbers')
    parser.add_argument('--path', type=str, default='path_to_dataset',
                        help='Path to dataset')
    parser.add_argument('--split', type=str, default='train',
                        help='split set (default: train')
    parser.add_argument('--save-path', type=str, default='path_to save_folder',
                        help='Path to the folder that you want to save the generated files')
    parser.add_argument('--block-size', type=int, default=100 ,
                        help='size of each block')
    parser.add_argument('--overlap', type=int, default=20 ,
                        help='overlap between the blocks')
    parser.add_argument('--num-seeds', type=int, default=60 ,
                        help='number of seed points')
    parser.add_argument('--knn', type=int, default=16*8192 ,
                        help='number of points in each generated group')
    args = parser.parse_args()

    for seq in args.sequences:
        seq_path = args.path + "{0:0=2d}".format(seq) + "/file_name.ply"
        pc = read_ply(seq_path)
        point_cloud_blocks = blocking(pc, block_size, overlap)
        f = 0
        length = len(point_cloud_blocks)
        k = 0
        while k < length:
            start = time.perf_counter()
            threads = []
            for pc0 in point_cloud_blocks[k:k+48]:
                t = threading.Thread(target=fps_knn_blcoks, args=[pc0, args.num_seeds, args.knn, seq, f, args.split, args.save_path])
                t.start()
                threads.append(t)
                f += 1
                k += 1
                print("File: ", f, "Seq: ", seq)
            j = 0
            for thread in threads:
                thread.join()
                j += 1
            finish = time.perf_counter()
            print(f'Finished in {round(finish-start, 2)} seconds') 

    
    