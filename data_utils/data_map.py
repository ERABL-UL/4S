# This file is covered by the LICENSE file in the root of this project.
labels = {
  0 : "void",
  1 : "flat",
  2: "construction",
  3: "object",
  4: "nature",
  5: "human",
  6: "vehicle",
  7: "road",
  8: "sidewalk",
  9: "parking",
  10: "rail track",
  11: "building",
  12: "wall",
  13: "fence",
  14: "guard rail",
  15: "bridge",
  16: "tunnel",
  17: "pole",
  18: "polegroup",
  19: "traffic light",
  20: "traffic sign",
  21: "vegetation",
  22: "terrain",
  23: "sky",
  24: "person",
  25: "rider",
  26: "car",
  27: "truck",
  28: "bus",
  29: "caravan",
  30: "trailer",
  31: "train",
  32: "motorcycle",
  33: "bicycle",
  34: "garage",
  35: "gate",
  36: "stop",
  37: "smallpole",
  38: "lamp",
  39: "trash bin",
  40: "vending machine",
  41: "box",
  42: "unknown construction",
  43: "unknown vehicle",
  44: "unknown object",
  45: "license plate"
}
# classes that are indistinguishable from single scan or inconsistent in
# ground truth are mapped to their closest equivalent
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
learning_map_inv = { # inverse of previous map
  0: 0,      # "unlabeled"               mapped to "void" --------------------------mapped
  0: 1,      # "ego vehicle"             mapped to "void" --------------------------mapped
  0: 2,      # "rectification border"    mapped to "void" --------------------------mapped
  0: 3,      # "out of roi"              mapped to "void" --------------------------mapped
  0: 4,      # "static"                  mapped to "void" --------------------------mapped
  0: 5,      # "dynamic"                 mapped to "void" --------------------------mapped
  0: 6,      # "ground"                  mapped to "void" --------------------------mapped
  1: 7,      # "road"                    mapped to "flat" --------------------------mapped
  1: 8,      # "sidewalk"                mapped to "flat" --------------------------mapped
  1: 9,      # "parking"                 mapped to "flat" --------------------------mapped
  1: 10,     # "rail track"              mapped to "flat" --------------------------mapped
  2: 11,     # "building"                mapped to "construction" ------------------mapped
  2: 12,     # "wall"                    mapped to "construction" ------------------mapped
  2: 13,     # "fence"                   mapped to "construction" ------------------mapped
  2: 14,     # "guard rail"              mapped to "construction" ------------------mapped
  2: 15,     # "bridge"                  mapped to "construction" ------------------mapped
  2: 16,     # "tunnel"                  mapped to "construction" ------------------mapped
  3: 17,     # "pole"                    mapped to "object" ------------------------mapped
  3: 18,     # "polegroup"               mapped to "object" ------------------------mapped
  3: 19,     # "traffic light"           mapped to "object" ------------------------mapped
  3: 20,     # "traffic sign"            mapped to "object" ------------------------mapped
  4: 21,     # "vegetation"              mapped to "nature" ------------------------mapped
  4: 22,     # "terrain"                 mapped to "nature" ------------------------mapped
  0: 23,     # "sky"                     mapped to "sky" ---------------------------mapped
  5: 24,     # "person"                  mapped to "human" -------------------------mapped
  5: 25,     # "rider"                   mapped to "human" -------------------------mapped
  6: 26,     # "car"                     mapped to "vehicle" -----------------------mapped
  6: 27,     # "truck"                   mapped to "vehicle" -----------------------mapped
  6: 28,     # "bus"                     mapped to "vehicle" -----------------------mapped
  6: 29,     # "caravan"                 mapped to "vehicle" -----------------------mapped
  6: 30,     # "trailer"                 mapped to "vehicle" -----------------------mapped
  6: 31,     # "train"                   mapped to "vehicle" -----------------------mapped
  6: 32,     # "motorcycle"              mapped to "vehicle" -----------------------mapped
  6: 33,     # "bicycle"                 mapped to "vehicle" -----------------------mapped
  2: 34,     # "garage"                  mapped to "construction" ------------------mapped
  2: 35,     # "gate"                    mapped to "construction" ------------------mapped
  2: 36,     # "stop"                    mapped to "construction"-------------------mapped
  3: 37,     # "smallpole"               mapped to "object"-------------------------mapped
  3: 38,     # "lamp"                    mapped to "object"-------------------------mapped
  3: 39,     # "trash bin"               mapped to "object"-------------------------mapped
  3: 40,     # "vending machine"         mapped to "object"-------------------------mapped
  3: 41,     # "box"                     mapped to "object"-------------------------mapped
  0: 42,     # "unknown construction"    mapped to "void"---------------------------mapped
  0: 43,     # "unknown vehicle"         mapped to "void"---------------------------mapped
  0: 44,     # "unknown object"          mapped to "void"---------------------------mapped
  6: 45      # "license plate"           mapped to "vehicle" -----------------------mapped
}
learning_ignore = { # Ignore classes
  0: True,       # "void"
  1: False,      # "flat"
  2: False,      # "construction"
  3: False,      # "object"
  4: False,      # "nature"
  5: False,      # "human"
  6: False      # "vehicle"
}

color_map = {
  0: [0, 0, 0],
  1: [255, 0, 255],
  2: [0, 200, 255],
  3: [0, 60, 135],
  4: [0, 175, 0],
  5: [30, 30, 255],
  6: [245, 150, 100]
}

content = { # as a ratio with the total number of points
  0: 0.01848,
  1: 0.36919,
  2: 0.17464,
  3: 0.00482,
  4: 0.39541,
  5: 0.00026,
  6: 0.0372
}
content_indoor = {
  0: 0.18111755628849344,
  1: 0.15350115272756307,
  2: 0.264323444618407,
  3: 0.017095487624768667,
  4: 0.02018415055214108,
  5: 0.025684283218171625,
  6: 0.05237503359636922,
  7: 0.03495118545614923,
  8: 0.04252921527371275,
  9: 0.004767541066020183,
  10: 0.06899976905686542,
  11: 0.012345517150886037,
  12: 0.12212566337045223,
}

poss_map = {
  0: 0, # 'void'
  1: 1, # 'flat'
  2: 2, # 'construction'
  3: 3, # 'object'
  4: 4, # 'nature'
  5: 5, # 'human'
  6: 6 # 'vehicle'
}


labels_poss = {
  0: 'void',
  1: 'flat',
  2: 'construction',
  3: 'object',
  4: 'nature',
  5: 'human',
  6: 'vehicle'
}
