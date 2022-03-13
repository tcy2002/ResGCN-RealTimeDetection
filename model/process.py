import model as md
import numpy as np
import pyzed.sl as sl
import cv_viewer as cvv
import torch
import os


def data_process(f, data, objects):
    # 修改data，保持75帧，删除第1帧，在第75帧前插入一帧
    new_data = np.zeros((3, 300, 15, 2))
    new_data[:, :f - 1, :, :] = data[:, 1:f, :, :]
    # 更新第75帧
    for n, obj in enumerate(objects):
        if n <= 1:
            for i in range(14):
                new_data[:, f - 1, i + 1, n] = obj.keypoint[cvv.key_points[i].value]
            spine_base = obj.keypoint[sl.BODY_PARTS.RIGHT_HIP.value] + obj.keypoint[sl.BODY_PARTS.LEFT_HIP.value]
            new_data[:, f - 1, 0, n] = spine_base / 2
    return new_data


def get_hop_distance(V):
    A = np.zeros((V, V))
    for i, j in md.edge:
        A[j, i] = 1
        A[i, j] = 1
    hop_dis = np.zeros((V, V)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(3 + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(3, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def get_adjacency(V):
    hop_dis = get_hop_distance(V)
    valid_hop = range(0, 3 + 1, 1)
    adjacency = np.zeros((V, V))
    for hop in valid_hop:
        adjacency[hop_dis == hop] = 1
    normalize_adjacency = normalize_digraph(adjacency)
    A = np.zeros((len(valid_hop), V, V))
    for i, hop in enumerate(valid_hop):
        A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def init_model():
    kwargs = {
        'kernel_size': [9, 2],
        'data_shape': [3, 6, 300, 15, 2],
        'num_class': 60,
        'A': torch.Tensor(get_adjacency(15)),
        'parts': [torch.Tensor(part).long() for part in md.parts]
    }

    model = md.create(model_type='resgcn-n51-r4', **kwargs)
    checkpoint = torch.load(os.getcwd() + '/checkpoint.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def multi_input(data, conn):
    C, T, V, M = data.shape
    data_new = np.zeros((3, C * 2, T, V, M))
    data_new[0, :C, :, :, :] = data
    for i in range(V):
        data_new[0, C:, :, i, :] = data[:, :, i, :] - data[:, :, 1, :]
    for i in range(T - 2):
        data_new[1, :C, i, :, :] = data[:, i + 1, :, :] - data[:, i, :, :]
        data_new[1, C:, i, :, :] = data[:, i + 2, :, :] - data[:, i, :, :]
    for i in range(len(conn)):
        data_new[2, :C, :, i, :] = data[:, :, i, :] - data[:, :, conn[i], :]
    bone_length = 0
    for i in range(C):
        bone_length += np.power(data_new[2, i, :, :, :], 2)
    bone_length = np.sqrt(bone_length) + 0.0001
    for i in range(C):
        data_new[2, C + i, :, :, :] = np.arccos(data_new[2, i, :, :, :] / bone_length)
    return data_new
