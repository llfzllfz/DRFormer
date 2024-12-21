import pandas as pd
import os
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
import math

def get_clip_data(filepath, filename, k_fold):
    train_data = pd.DataFrame({})
    test_data = pd.DataFrame({})
    folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
    for idx, fold in enumerate(folds):
        path = os.path.join(os.path.join(filepath, fold), filename + '.csv')
        data = pd.read_csv(path, header = None, skiprows=1, names = ['Typeloc', 'Seq', 'Str', 'Label', 'filename'])
        if idx + 1 != k_fold:
            train_data = pd.concat([train_data, data], axis=0, ignore_index=True)
        else:
            test_data = pd.concat([test_data, data], axis=0, ignore_index=True)
    return train_data, test_data

def get_RBP24_data(filepath, filename):
    train = pd.read_csv(os.path.join(filepath, '{}_train.csv'.format(filename)))
    test = pd.read_csv(os.path.join(filepath, '{}_test.csv'.format(filename)))
    return train, test

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_pe(Embed_dim, length):
    pe = torch.zeros(length, Embed_dim)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, Embed_dim, 2).float() * (-math.log(10000.0) / Embed_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def get_one_hot(seq):
    onehot = np.zeros((4, len(seq)))
    onehot[0, np.where(np.array(list(seq)) == 'A')] = 1
    onehot[1, np.where(np.array(list(seq)) == 'C')] = 1
    onehot[2, np.where(np.array(list(seq)) == 'T')] = 1
    onehot[3, np.where(np.array(list(seq)) == 'G')] = 1
    return onehot

from scipy.spatial import distance
def hamming_distance(seq1, seq2):
    return distance.hamming(seq1, seq2)

def calculate_distance_matrix(seq):
    distance_matrix = np.zeros((len(seq), len(seq)))
    def get_one_hot_distance(c):
        if c == 'A':
            return [1, 0, 0, 0]
        if c == 'C':
            return [0, 1, 0, 0]
        if c == 'T':
            return [0, 0, 1, 0]
        if c == 'G':
            return [0, 0, 0, 1]
        return [0.25, 0.25, 0.25, 0.25]
    for i in range(len(seq)):
        for j in range(i, len(seq)):
            dist = hamming_distance(get_one_hot_distance(seq[i]), get_one_hot_distance(seq[j]))
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix

def get_MLDP(seq):
    def check_paired(x, y):
        if x == 'A' and y == 'T':
            return 1
        if x == 'T' and y == 'A':
            return 1
        if x == 'C' and y == 'G':
            return 1
        if x == 'G' and y == 'C':
            return 1
        if x == 'G' and y == 'T':
            return 1
        if x == 'T' and y == 'G':
            return 1
        return 0
    dp = np.zeros((len(seq), len(seq) + 1))
    for idx in range(1, dp.shape[0]):
        for idy in range(dp.shape[1]-2, idx, -1):
            if check_paired(seq[idx], seq[idy]) == 1:
                dp[idx, idy] = dp[idx-1, idy+1] + check_paired(seq[idx], seq[idy])
    dp = dp[:, :-1]
    for idx in range(dp.shape[0]-2, -1, -1):
        for idy in range(1, dp.shape[1]):
            if dp[idx, idy] != 0:
                dp[idx, idy] = max(dp[idx, idy], dp[idx+1, idy-1])
    dp_T = np.transpose(dp)
    dp = dp_T + dp
    return dp

def get_Pair(one_hot):
    pair = []
    for idx in range(4):
        for idy in range(4):
            tmp = (np.expand_dims(one_hot[idx, :], axis=1) * np.expand_dims(one_hot[idy, :], axis=0))
            pair.append(tmp)
    return np.array(pair)

def get_CDP_w(x, y):
    if x == 'A' and y == 'T':
        return 2
    if x == 'T' and y == 'A':
        return 2
    if x == 'G' and y == 'C':
        return 3
    if x == 'C' and y == 'G':
        return 3
    if x == 'G' and y == 'T':
        return 0.8
    if x == 'T' and y == 'G':
        return 0.8
    return 0

def CDP_feature(RNA_seq):
    feature = np.zeros((len(RNA_seq), len(RNA_seq)))
    for idx in range(len(RNA_seq)):
        for idy in range(len(RNA_seq)):
            x = RNA_seq[idx]
            y = RNA_seq[idy]
            feature[idx][idy] = get_CDP_w(x, y)
            if get_CDP_w(x, y) != 0:
                TA = 0
                TB = 0
                a = 1
                b = 1
                while(idx - a >= 0 and idx - a < len(RNA_seq) and idy + a >= 0 and idy + a < len(RNA_seq) and get_CDP_w(RNA_seq[idx - a], RNA_seq[idy + a]) != 0):
                    TA = TA + get_CDP_w(RNA_seq[idx-a], RNA_seq[idy+a]) * math.exp(-a**2 / 2)
                    a = a + 1
                while(idx + b >= 0 and idx + b < len(RNA_seq) and idy - b >= 0 and idy - b < len(RNA_seq) and get_CDP_w(RNA_seq[idx + b], RNA_seq[idy - b]) != 0):
                    TB = TB + get_CDP_w(RNA_seq[idx+b], RNA_seq[idy-b]) * math.exp(-b**2 / 2 )
                    b = b + 1
                feature[idx][idy] = feature[idx][idy] + TA + TB
    feature = feature.reshape(1, len(RNA_seq), len(RNA_seq))
    return feature