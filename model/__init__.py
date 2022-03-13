import logging
from . import blocks
from .nets import ResGCN
from .modules import ResGCN_Module, AttGCN_Module
from .attentions import *
from .process import *

__model = {
    'resgcn': ResGCN,
}

__attention = {
    'pa': Part_Att,
    'ca': Channel_Att,
    'fa': Frame_Att,
    'ja': Joint_Att,
    'pca': Part_Conv_Att,
    'psa': Part_Share_Att,
}

__structure = {
    'b15': {'structure': [1, 2, 2, 2], 'block': 'Basic'},
    'b19': {'structure': [1, 2, 3, 3], 'block': 'Basic'},
    'b23': {'structure': [1, 3, 4, 3], 'block': 'Basic'},
    'b29': {'structure': [1, 3, 6, 4], 'block': 'Basic'},
    'n39': {'structure': [1, 2, 2, 2], 'block': 'Bottleneck'},
    'n51': {'structure': [1, 2, 3, 3], 'block': 'Bottleneck'},
    'n57': {'structure': [1, 3, 4, 3], 'block': 'Bottleneck'},
    'n75': {'structure': [1, 3, 6, 4], 'block': 'Bottleneck'},
}

__reduction = {
    'r1': {'reduction': 1},
    'r2': {'reduction': 2},
    'r4': {'reduction': 4},
    'r8': {'reduction': 8},
}

# 骨骼点分组
parts = [
    np.array([3, 4, 5]) - 1,  # left_arm
    np.array([6, 7, 8]) - 1,  # right_arm
    np.array([9, 10, 11]) - 1,  # left_leg
    np.array([12, 13, 14]) - 1,  # right_leg
    np.array([1, 2, 15]) - 1  # torso
]

# 相邻骨骼点
neighbor_1base = [
    (1, 15), (2, 15), (3, 15), (4, 3), (5, 4),
    (6, 15), (7, 6), (8, 7), (9, 1), (10, 9),
    (11, 10), (12, 1), (13, 12), (14, 13)
]

neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
self_link = [(i, i) for i in range(15)]
edge = neighbor_link + self_link
connect_joint = np.array([15, 15, 15, 3, 4, 15, 6, 7, 1, 9, 10, 1, 12, 13, 15]) - 1

interpreter = {
    0: 'drink water', 1: 'eat meal/snack', 2: 'brushing teeth', 3: 'brushing hair', 4: 'drop', 5: 'pickup',
    6: 'throw', 7: 'sitting down', 8: 'standing up', 9: 'clapping', 10: 'reading', 11: 'writing',
    12: 'tear up paper', 13: 'wear jacket', 14: 'take off jacket', 15: 'wear a shoe', 16: 'take off a shoe',
    17: 'wear on glasses', 18: 'take off glasses', 19: 'put on a hat/cap', 20: 'take off a hat/cap', 21: 'cheer up',
    22: 'hand waving', 23: 'kicking something', 24: 'put/take out sth', 25: 'hopping', 26: 'jump up',
    27: 'make a phone call', 28: 'playing with a phone', 29: 'typing on a keyboard',
    30: 'pointing to sth with finger', 31: 'taking a selfie', 32: 'check time (from watch)',
    33: 'rub two hands together', 34: 'nod head/bow', 35: 'shake head', 36: 'wipe face', 37: 'salute',
    38: 'put the palms together', 39: 'cross hands in front', 40: 'sneeze/cough', 41: 'staggering', 42: 'falling',
    43: 'touch head', 44: 'touch chest', 45: 'touch back', 46: 'touch neck', 47: 'nausea or vomiting condition',
    48: 'use a fan', 49: 'punching', 50: 'kicking other person', 51: 'pushing other person',
    52: 'pat on back of other person', 53: 'point finger at the other person', 54: 'hugging other person',
    55: 'fighting', 56: 'touching other person pocket', 57: 'handshaking',
    58: 'walking towards each other', 59: 'walking apart from each other'
}


def create(model_type, **kwargs):
    model_split = model_type.split('-')
    if model_split[0] in __attention.keys():
        kwargs.update({'module': AttGCN_Module, 'attention': __attention[model_split[0]]})
        del (model_split[0])
    else:
        kwargs.update({'module': ResGCN_Module, 'attention': None})
    try:
        [model, structure, reduction] = model_split
    except:
        [model, structure], reduction = model_split, 'r1'
    if not (model in __model.keys() and structure in __structure.keys() and reduction in __reduction.keys()):
        logging.info('')
        logging.error('Error: Do NOT exist this model_type: {}!'.format(model_type))
        raise ValueError()
    return __model[model](**(__structure[structure]), **(__reduction[reduction]), **kwargs)
