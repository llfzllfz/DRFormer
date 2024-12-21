import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from tqdm import tqdm
import sys
sys.path.append('../bin')
sys.path.append('bin')
import Entropy
sys.path.append('DNABERT/src')
from transformers import DNATokenizer
import numpy as np
import time

class GUE_dataset(Dataset):
    def __init__(self, path, mode = 'train',
                 MLDP = 1, DIS = 1, CDP = 1, SPTIAL_DIS = 1, UFOLD = 1, UNPAIR = 1, REPEAT = 1, UFOLD_ADD_UNPAIR = 0,
                 pad_length = 112):
        super().__init__()
        self.path = path
        self.mode = mode
        data = self.get_data()
        self.data = data.sample(frac=1)
        # self.data = data
        self.tokenizer = DNATokenizer.from_pretrained('DNABERT3')
        self.MLDP = MLDP
        self.DIS = DIS
        self.CDP = CDP
        self.SPTIAL_DIS = SPTIAL_DIS
        self.UFOLD = UFOLD
        self.UNPAIR = UNPAIR
        self.REPEAT = REPEAT
        self.UFOLD_ADD_UNPAIR = UFOLD_ADD_UNPAIR
        self.pad_length = pad_length

    def get_data(self):
        data = pd.read_csv(os.path.join(self.path, '{}.csv'.format(self.mode)))
        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        seq = self.data['sequence'].iloc[idx]
        # assert len(seq) == 500, f'Please make sure the length is 500, the length is {len(seq)}, {seq}'
        if len(seq) != 500:
            seq = seq + 'N' * (500 - len(seq))
        pre_seq = seq
        seq = seq.replace('U', 'T')
        seq = seq.upper()
        seq_length = len(seq)

        label = self.data['label'].iloc[idx]
        label = torch.FloatTensor([label])

        if len(seq) != self.pad_length:
            matrix_seq = seq + 'N' * (self.pad_length - len(seq))
        else:
            matrix_seq = seq
        
        # matrix_seq = matrix_seq[:112]
        matrix_seq = matrix_seq[:self.pad_length]

        matrix = torch.tensor(Entropy.GET_ALL_CHANNEL(matrix_seq, self.MLDP, self.DIS, self.CDP, self.SPTIAL_DIS, self.UFOLD, self.UNPAIR, self.REPEAT, self.UFOLD_ADD_UNPAIR))
        mask = torch.zeros(matrix.size())
        mask[:, :seq_length, :seq_length] = 1
        matrix = matrix * mask


        tokens = self.get_3_mer(seq)
        # attention_mask = [1] * len(tokens)
        # if len(tokens) % 16 != 14:
        #     tokens.extend(['PAD'] * ((14 - (len(tokens) % 16) + 16) % 16))
        tokens = ['CLS'] + tokens + ['SEP']
        attention_mask = [1] * len(tokens)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        # attention_mask = attention_mask + [0] * (len(tokens) - len(attention_mask))
        token_type_ids = [0] * len(tokens)

        tokens = torch.IntTensor(tokens)
        attention_mask = torch.IntTensor(attention_mask)
        # print(seq)
        # print(matrix.shape, tokens.shape, attention_mask.shape)
        result = {
            'label': label,
            'matrix': matrix,
            'tokens': tokens,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'seq': pre_seq
        }
        return result
    
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

def GUE_dataloader(mode, path,
                   batch_size = 32, shuffle = True, num_workers = 1,
                   MLDP = 1, DIS = 1, CDP = 1, SPTIAL_DIS = 1, UFOLD = 1, UNPAIR = 1, REPEAT = 1, UFOLD_ADD_UNPAIR = 0,
                   pad_length = 112):
    dataset = GUE_dataset(path, mode, MLDP, DIS, CDP, SPTIAL_DIS, UFOLD, UNPAIR, REPEAT, UFOLD_ADD_UNPAIR, pad_length)
    # if mode != 'train':
    #     dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    # else:
    #     sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #     batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last = False)
    #     dataloader = torch.utils.data.DataLoader(dataset,
    #         batch_sampler=batch_sampler, pin_memory=True, num_workers=num_workers)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return dataloader
