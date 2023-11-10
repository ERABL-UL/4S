import json
import numpy as np
import OSToolBox as ost

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

def read_ply(file_name):
    data = ost.read_ply(file_name)
    cloud_x = data['x']
    cloud_y = data['y']
    cloud_z = data['z']
    labels = (data['c']).astype(np.int32)
    labels = labels.reshape(len(labels), 1)
    return(np.c_[cloud_x,cloud_y,cloud_z],labels)

def class_contents(labels_files, lbl_count):
    for file_ in labels_files['data']:
        _, labels = read_ply(file_)
        #remap labels to learning values
        labels = np.vectorize(learning_map.get)(labels)
        classes, counts = np.unique(labels, return_counts=True)

        for class_, count in zip(classes, counts):
            lbl_count[class_] += count

    return lbl_count


splits = None
with open('percentiles_split.json', 'r') as f:
    splits = json.load(f)

for percentile in splits:
    print(f'PERCENT: {percentile}')
    lbl_count = [ 0 for _ in range(20) ]
    for seq in splits[percentile]:
        lbl_count = class_contents(splits[percentile][seq], lbl_count)


    lbl_count = np.array(lbl_count)
    class_dist = lbl_count / np.sum(lbl_count)
    for class_ in range(20):
        print(f'{class_}: {round(class_dist[class_],5)}')
    print(f'\t- CLASS DIST: {class_dist}')
