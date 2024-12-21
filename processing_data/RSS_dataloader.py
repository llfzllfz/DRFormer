import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from tools.utils import get_one_hot, get_Pair
import sys
sys.path.append('../bin')
sys.path.append('bin')
sys.path.append('DNABERT/src')
import Entropy
import numpy as np
from transformers import DNATokenizer

class RSS_dataset(Dataset):
    def __init__(self, path, feature = 'DNAV',
                 MLDP = 1, DIS = 1, CDP = 1, SPTIAL_DIS = 1, UFOLD = 1, UNPAIR = 1, REPEAT = 1, UFOLD_ADD_UNPAIR = 0, SWIN_cross = 0):
        super().__init__()
        self.path = path
        self.filenames = self.get_filenames()
        # self.drop_duplicates()
        self.MLDP = MLDP
        self.DIS = DIS
        self.CDP = CDP
        self.SPTIAL_DIS = SPTIAL_DIS
        self.UFOLD = UFOLD
        self.UNPAIR = UNPAIR
        self.REPEAT = REPEAT
        self.UFOLD_ADD_UNPAIR = UFOLD_ADD_UNPAIR
        self.SWIN_cross = SWIN_cross
        self.feature = feature
        self.tokenizer = DNATokenizer.from_pretrained('DNABERT3')

    def get_filenames(self):
        with open(self.path, 'r') as f:
            filenames = f.readlines()
        f.close()
        result = []
        for filename in filenames:
            result.append(filename.replace('\n', ''))
        return result
    
    def process_data(self, idx, pad = 16, length = 112):
        root_path = self.path.split('/')
        path = ''
        for _, __ in enumerate(root_path):
            if _ == len(root_path) - 2:
                break
            path = path + __ + '/'
        with open(os.path.join(path, self.filenames[idx]), 'r') as f:
            data = f.readlines()
        f.close()
        seq = ""
        start, end = [], []
        for _ in data:
            _ = _.replace('  ', ' ')
            __ = _.split(' ')
            seq = seq + __[1]
            start.append(int(__[0]))
            end.append(int(__[2]))
        seq = seq.replace('U', 'T')
        seq = seq.upper()
        seq_length = len(seq)
        if pad != 0 and len(seq) % pad != 0:
            matrix_seq = seq + 'N' * (pad - len(seq) % pad)
        else:
            matrix_seq = seq
        if len(matrix_seq) < length:
            matrix_seq = matrix_seq + 'N' * (length - len(matrix_seq))
        label = self.get_label(start, end, len(matrix_seq))
        return seq, label, seq_length, matrix_seq
    
    def get_label(self, start, end, length):
        label = torch.zeros((length, length))
        for idx in range(len(start)):
            if end[idx] == 0:
                continue
            label[start[idx] - 1, end[idx] - 1] = 1
            label[end[idx] - 1, start[idx] - 1] = 1
        return label

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if self.feature == 'DNAV':
            seq, label, seq_length, matrix_seq = self.process_data(idx, 16, 112)
            # _, _, _, seq_tokens = self.process_data(idx, 16, 510)
            # if len(seq) < 112:
            #     seq_tokens = seq + 'N' * (112 - len(seq))
            
            RNA_onehot = get_one_hot(matrix_seq)
            matrix = torch.tensor(Entropy.GET_ALL_CHANNEL(matrix_seq, self.MLDP, self.DIS, self.CDP, self.SPTIAL_DIS, self.UFOLD, self.UNPAIR, self.REPEAT, self.UFOLD_ADD_UNPAIR))
            mask = torch.zeros(matrix.size())
            mask[:, :seq_length, :seq_length] = 1
            matrix = matrix * mask

            tokens = self.get_3_mer(matrix_seq)
            tokens = ['CLS'] + tokens + ['SEP']
            attention_mask = [1] * len(tokens)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            token_type_ids = [0] * len(tokens)
            tokens = torch.IntTensor(tokens)
            attention_mask = torch.IntTensor(attention_mask)


        elif self.feature == 'UFOLD':
            from tools.ufold_utils import creatmat
            seq, label, seq_length, matrix_seq = self.process_data(idx, 16)
            CDP = creatmat(matrix_seq)
            CDP = torch.Tensor(CDP)
            RNA_onehot = get_one_hot(matrix_seq)
            pair = get_Pair(RNA_onehot)
            matrix = torch.cat([torch.Tensor(pair), CDP.unsqueeze(0)], dim = 0)
        
        mask = torch.zeros((len(matrix_seq), len(matrix_seq)))
        mask[:seq_length, :seq_length] = 1
        # print(tokens.shape)
        result = {
            'label': label,
            'matrix': matrix,
            'length': seq_length,
            'one_hot': RNA_onehot.transpose(-1, -2),
            'mask': mask,
            'seq': seq,
            'tokens': tokens,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }
        return result
    
    def drop_duplicates(self):
        filename_dict = {}
        lists = []
        for idx, filename in enumerate(self.filenames):
            seq, _, _, _ = self.process_data(idx)
            if seq not in filename_dict:
                filename_dict[seq] = filename
                lists.append(filename)
            else:
                print(filename, filename_dict[seq])
        self.filenames = lists
    
    def get_3_mer(self, seq):
        assert len(seq) > 3
        tokens = []
        for idx in range(2, len(seq)):
            tmp = seq[idx - 2 : idx + 1]
            if 'N' in tmp:
                tokens.append('UNK')
            else:
                tokens.append(tmp)
        return tokens

def RSS_dataloader(path, feature_mode = 'DNAV', 
                   batch_size = 32, shuffle = True, num_workers = 1,
                   MLDP = 1, DIS = 1, CDP = 1, SPTIAL_DIS = 1, UFOLD = 1, UNPAIR = 1, REPEAT = 1, UFOLD_ADD_UNPAIR = 0, SWIN_cross = 0):
    dataset = RSS_dataset(path, feature_mode, MLDP, DIS, CDP, SPTIAL_DIS, UFOLD, UNPAIR, REPEAT, UFOLD_ADD_UNPAIR, SWIN_cross = SWIN_cross)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return dataloader
