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
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder

class CLS_dataset(Dataset):
    def __init__(self, path, mode = 'train',
                 MLDP = 1, DIS = 1, CDP = 1, SPTIAL_DIS = 1, UFOLD = 1, UNPAIR = 1, REPEAT = 1, UFOLD_ADD_UNPAIR = 0,
                 pad_length = 112, vision = 1, text = 1):
        super().__init__()
        self.text = text
        self.vision = vision
        self.path = path
        self.mode = mode
        self.pad_length = pad_length
        data, label = self.get_data()
        self.data = data
        le = LabelEncoder()
        self.label = le.fit_transform(label)
        self.le = le
        self.shuffle()
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
        

    def get_data(self):
        # data = pd.read_csv(os.path.join(self.path, '{}.csv'.format(self.mode)))
        data = []
        label = []
        for seq in SeqIO.parse(os.path.join(self.path, '{}.fa'.format(self.mode)), "fasta"):
            # if len(str(seq.seq)) > self.pad_length:
            #     continue
            data.append(str(seq.seq))
            label.append(seq.description.split(' ')[-1])
        # print(set(label))
        return data, label

    def __len__(self):
        # return 30
        return len(self.data)

    def shuffle(self):
        indicates = np.arange(0, len(self.data))
        np.random.shuffle(indicates)
        self.data = np.array(self.data)
        self.label = np.array(self.label)
        self.data = self.data[indicates]
        self.label = self.label[indicates]
        self.data = self.data.tolist()
        self.label = self.label.tolist()

    def __getitem__(self, idx):
        seq = self.data[idx]
        seq = seq[:self.pad_length]
        pre_seq = seq
        seq = seq.replace('U', 'T')
        seq = seq.upper()
        seq_length = len(seq)

        label = self.label[idx]
        label = torch.LongTensor([label])

        if len(seq) < self.pad_length:
            matrix_seq = seq + 'N' * (self.pad_length - len(seq))
        else:
            matrix_seq = seq
        # if len(seq) < 510:
        #     tokens_seq = seq + 'N' * (510 - len(seq))
        # else:
        #     tokens_seq = seq

        result = {'label': label,}

        if self.vision == 1:
            matrix = torch.tensor(Entropy.GET_ALL_CHANNEL(matrix_seq, self.MLDP, self.DIS, self.CDP, self.SPTIAL_DIS, self.UFOLD, self.UNPAIR, self.REPEAT, self.UFOLD_ADD_UNPAIR))
            mask = torch.zeros(matrix.size())
            mask[:, :seq_length, :seq_length] = 1
            matrix = matrix * mask
            result['matrix'] = matrix

        if self.text == 1:
            tokens = self.get_3_mer(matrix_seq)
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
            result['tokens'] = tokens
            result['attention_mask'] = attention_mask
            result['token_type_ids'] = token_type_ids
        result['seq'] = pre_seq
        result['le'] = self.get_le_dict()
        # print(self.get_le_dict())
        # print(seq)
        # print(tokens.shape)
        # print(label.shape, matrix.shape, tokens.shape, attention_mask.shape)
        
        return result
    
    def get_le_dict(self):
        dicts = {}
        for idx in range(13):
            dicts[idx] = self.le.inverse_transform([idx])[0]
        return dicts

    
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

def CLS_dataloader(mode, path,
                   batch_size = 32, shuffle = True, num_workers = 1,
                   MLDP = 1, DIS = 1, CDP = 1, SPTIAL_DIS = 1, UFOLD = 1, UNPAIR = 1, REPEAT = 1, UFOLD_ADD_UNPAIR = 0,
                   pad_length = 112, multi_gpu = 0,
                   vision = 1, text = 1):
    dataset = CLS_dataset(path, mode, MLDP, DIS, CDP, SPTIAL_DIS, UFOLD, UNPAIR, REPEAT, UFOLD_ADD_UNPAIR, pad_length, vision, text)
    if mode != 'train':
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    else:
        if multi_gpu > 0:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last = False)
            dataloader = torch.utils.data.DataLoader(dataset,
                batch_sampler=batch_sampler, pin_memory=True, num_workers=num_workers)
        else:
            dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    # dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return dataloader



# from Bio import SeqIO
# max_seq = 0
# lengths = 0
# labels = []
# for seq in SeqIO.parse("../../RNAErnie/data/ft/seq_cls/nRC/train.fa", "fasta"):
#     lengths = lengths + 1
#     max_seq = max(max_seq, len(seq))
#     # print(len(seq))
#     print(seq.description)
#     print(seq.seq)
#     print(type(seq.seq))
#     print(str(seq.seq))
#     print(type(str(seq.seq)))
#     # print(seq)
#     # labels.append(seq.description.split(' ')[-1])
#     break

# # print(max_seq, lengths)
# # from sklearn.preprocessing import LabelEncoder
# # le = LabelEncoder()
# # new_labels = le.fit_transform(labels)
# # print(new_labels)
# # print(set(new_labels))
