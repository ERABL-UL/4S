import numpy as np
import os
import json
import OSToolBox as ost

train_seqs = [ '00', '02', '03', '04', '05', '06', '07', '09', '10']
#train_seqs = [ '01' ]
percentiles = [0.75, 0.5, 0.1, 0.01, 0.001]
# percentiles = [0.001]
full_content = {
  0: False,
  1: False,
  2: False,
  3: False,
  4: False,
  5: False,
  6: False
}

learning_map = {
  0: 0,      # "unlabeled"               mapped to "void" --------------------------mapped
  1: 0,      # "ego vehicle"             mapped to "void" --------------------------mapped
  2: 0,      # "rectification border"    mapped to "void" --------------------------mapped
  3: 0,      # "out of roi"              mapped to "void" --------------------------mapped
  4: 0,      # "static"                  mapped to "void" --------------------------mapped
  5: 0,      # "dynamic"                 mapped to "void" --------------------------mapped
  6: 0,      # "ground"                  mapped to "void" --------------------------mapped
  7: 1,      # "road"                    mapped to "flat" --------------------------mapped
  8: 1,      # "sidewalk"                mapped to "flat" --------------------------mapped
  9: 1,      # "parking"                 mapped to "flat" --------------------------mapped
  10: 1,     # "rail track"              mapped to "flat" --------------------------mapped
  11: 2,     # "building"                mapped to "construction" ------------------mapped
  12: 2,     # "wall"                    mapped to "construction" ------------------mapped
  13: 2,     # "fence"                   mapped to "construction" ------------------mapped
  14: 2,     # "guard rail"              mapped to "construction" ------------------mapped
  15: 2,     # "bridge"                  mapped to "construction" ------------------mapped
  16: 2,     # "tunnel"                  mapped to "construction" ------------------mapped
  17: 3,     # "pole"                    mapped to "object" ------------------------mapped
  18: 3,     # "polegroup"               mapped to "object" ------------------------mapped
  19: 3,     # "traffic light"           mapped to "object" ------------------------mapped
  20: 3,     # "traffic sign"            mapped to "object" ------------------------mapped
  21: 4,     # "vegetation"              mapped to "nature" ------------------------mapped
  22: 4,     # "terrain"                 mapped to "nature" ------------------------mapped
  23: 0,     # "sky"                     mapped to "sky" ---------------------------mapped
  24: 5,     # "person"                  mapped to "human" -------------------------mapped
  25: 5,     # "rider"                   mapped to "human" -------------------------mapped
  26: 6,     # "car"                     mapped to "vehicle" -----------------------mapped
  27: 6,     # "truck"                   mapped to "vehicle" -----------------------mapped
  28: 6,     # "bus"                     mapped to "vehicle" -----------------------mapped
  29: 6,     # "caravan"                 mapped to "vehicle" -----------------------mapped
  30: 6,     # "trailer"                 mapped to "vehicle" -----------------------mapped
  31: 6,     # "train"                   mapped to "vehicle" -----------------------mapped
  32: 6,     # "motorcycle"              mapped to "vehicle" -----------------------mapped
  33: 6,     # "bicycle"                 mapped to "vehicle" -----------------------mapped
  34: 2,     # "garage"                  mapped to "construction" ------------------mapped
  35: 2,     # "gate"                    mapped to "construction" ------------------mapped
  36: 2,     # "stop"                    mapped to "construction"-------------------mapped
  37: 3,     # "smallpole"               mapped to "object"-------------------------mapped
  38: 3,     # "lamp"                    mapped to "object"-------------------------mapped
  39: 3,     # "trash bin"               mapped to "object"-------------------------mapped
  40: 3,     # "vending machine"         mapped to "object"-------------------------mapped
  41: 3,     # "box"                     mapped to "object"-------------------------mapped
  42: 0,     # "unknown construction"    mapped to "void"---------------------------mapped
  43: 0,     # "unknown vehicle"         mapped to "void"---------------------------mapped
  44: 0,     # "unknown object"          mapped to "void"---------------------------mapped
  45: 6      # "license plate"           mapped to "vehicle" -----------------------mapped
}

root = 'Data/KITTI360/fps_knn/train'
present_classes = []

data_datapath = {}

def read_ply(file_name):
    data = ost.read_ply(file_name)
    cloud_x = data['x']
    cloud_y = data['y']
    cloud_z = data['z']
    labels = (data['c']).astype(np.int32)
    labels = labels.reshape(len(labels), 1)
    return(np.c_[cloud_x,cloud_y,cloud_z],labels)


def evaluate_percentile(label_files, seq_content):
    for class_ in present_classes:
        seq_content[class_] = False

    percentile_content = full_content.copy()
    for file_ in label_files:
        _, labels = read_ply(file_)

        #remap labels to learning values
        labels = np.vectorize(learning_map.get)(labels)

        for class_ in np.unique(labels):
            percentile_content[class_] = True

    
    #print(list(percentile_content.values()), list(seq_content.values()))
    return np.all(np.equal(list(percentile_content.values()), list(seq_content.values())))


if __name__ == '__main__':
    for seq in train_seqs:
        print('Listing Sequence ', seq)
        seq_content = full_content.copy()
        data_datapath[seq] = {'data': [], 'seq_content': None}
        data_seq_path = os.path.join(root, 'sequences', seq)
        data_seq_ply = os.listdir(data_seq_path)
        data_seq_ply.sort()
        data_datapath[seq]['data'] += [ os.path.join(data_seq_path, data_file) for data_file in data_seq_ply]

        for label_file in data_datapath[seq]['data']:
            _, labels = read_ply(label_file)
            #remap labels to learning values
            labels = np.vectorize(learning_map.get)(labels)

            for class_ in np.unique(labels):
                seq_content[class_] = True
            
        data_datapath[seq]['seq_content'] = seq_content
    data_datapath[seq]['seq_content'][0] = False
    percentiles_paths = {}

    for percentile in percentiles:
        percentiles_paths[percentile] = {}
        print('PERCENTILE :', percentile)
        for seq in train_seqs:
            print('SEQUENCE: ', seq)
            percentiles_paths[percentile][seq] = {'data': []}
            seq_percent = max(1,int(len(data_datapath[seq]['data']) * percentile))
            print(len(data_datapath[seq]['data']), ' -> ', seq_percent)

            labels_n_index = np.array(list(enumerate(data_datapath[seq]['data'])))[:,0]
            percentile_ind = np.random.choice(labels_n_index, seq_percent, replace=False).astype(int)
            percentile_seq = [ data_datapath[seq]['data'][i] for i in percentile_ind ]
            tries = 0
            while not evaluate_percentile(percentile_seq, data_datapath[seq]['seq_content']) and tries < 1000:
                labels_n_index = np.array(list(enumerate(data_datapath[seq]['data'])))[:,0]
                percentile_ind = np.random.choice(labels_n_index, seq_percent, replace=False).astype(int)
                percentile_seq = [ data_datapath[seq]['data'][i] for i in percentile_ind ]
                tries += 1
                print(tries)
            for class_ in data_datapath[seq]['seq_content']:
                if class_ not in present_classes:
                    present_classes.append(class_)

            for i in percentile_ind:
                percentiles_paths[percentile][seq]['data'] += [ data_datapath[seq]['data'][i] ]

    with open('percentiles_split.json', 'w+') as f:
        json.dump(percentiles_paths, f)

    


