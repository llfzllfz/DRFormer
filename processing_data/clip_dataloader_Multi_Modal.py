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

class CLIP_dataset(Dataset):
    def __init__(self, data,
                 MLDP = 1, DIS = 1, CDP = 1, SPTIAL_DIS = 1, UFOLD = 1, UNPAIR = 1, REPEAT = 1, UFOLD_ADD_UNPAIR = 0):
        super().__init__()
        self.data = data.sample(frac=1)
        self.tokenizer = DNATokenizer.from_pretrained('DNABERT3')
        self.MLDP = MLDP
        self.DIS = DIS
        self.CDP = CDP
        self.SPTIAL_DIS = SPTIAL_DIS
        self.UFOLD = UFOLD
        self.UNPAIR = UNPAIR
        self.REPEAT = REPEAT
        self.UFOLD_ADD_UNPAIR = UFOLD_ADD_UNPAIR

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        seq = self.data['Seq'].iloc[idx]
        seq = seq.replace('U', 'T')
        seq = seq.upper()
        seq_length = len(seq)

        label = self.data['Label'].iloc[idx]
        label = torch.FloatTensor([label])

        if len(seq) != 112:
            matrix_seq = seq + 'N' * (112 - len(seq))
        else:
            matrix_seq = seq
        matrix = torch.tensor(Entropy.GET_ALL_CHANNEL(matrix_seq, self.MLDP, self.DIS, self.CDP, self.SPTIAL_DIS, self.UFOLD, self.UNPAIR, self.REPEAT, self.UFOLD_ADD_UNPAIR))
        mask = torch.zeros(matrix.size())
        mask[:, :seq_length, :seq_length] = 1
        matrix = matrix * mask

        tokens = self.get_3_mer(seq)
        # attention_mask = [1] * len(tokens)
        # if len(tokens) % 16 != 14:
        #     tokens.extend(['PAD'] * ((14 - (len(tokens) % 16) + 16) % 16))
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        attention_mask = [1] * len(tokens)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        # attention_mask = attention_mask + [0] * (len(tokens) - len(attention_mask))
        token_type_ids = [0] * len(tokens)

        tokens = torch.IntTensor(tokens)
        attention_mask = torch.IntTensor(attention_mask)

        result = {
            'label': label,
            'matrix': matrix,
            'tokens': tokens,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
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

def CLIP_dataloader(data,
                   batch_size = 32, shuffle = True, num_workers = 1,
                   MLDP = 1, DIS = 1, CDP = 1, SPTIAL_DIS = 1, UFOLD = 1, UNPAIR = 1, REPEAT = 1, UFOLD_ADD_UNPAIR = 0):
    dataset = CLIP_dataset(data, MLDP, DIS, CDP, SPTIAL_DIS, UFOLD, UNPAIR, REPEAT, UFOLD_ADD_UNPAIR)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return dataloader
