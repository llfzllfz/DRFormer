import pandas as pd
import numpy as np
from tools.utils import set_seed, get_clip_data
from tools.config import get_config
import torch
import torch.nn as nn
import logging
import os
from tqdm import tqdm
from tools.select import select_dataset, select_model, select_criterion_optimizer, cal_loss, get_metric, predict
from tools.metric import Init_metric, update_metric, cal_metric, pd_to_metric
from tools.utils import get_one_hot, get_Pair
import sys
sys.path.append('../bin')
sys.path.append('bin')
sys.path.append('DNABERT/src')
import Entropy
from transformers import DNATokenizer
class Predict():
    def __init__(self, config, test_dataloader = None):
        self.config = config
        self.test_dataloader = test_dataloader
        self.device = 'cuda:{}'.format(config.gpu) if torch.cuda.is_available() and config.gpu >= 0 else 'cpu'
        self.model = torch.load(self.config.save_path, map_location=self.device)

        self.criterion, self.optimizer = select_criterion_optimizer(config.command, self.device, self.model, config)
        logging.basicConfig(filename=config.log_path, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        self.log = logging.getLogger(config.filename)
        self.log.info(config)
    
    def Predict(self):
        self.model.eval()
        all_loss = 0
        metric = pd.DataFrame({})
        preds = []
        label = []
        RSS = pd.DataFrame({})
        with torch.no_grad():
            for tmp in tqdm(self.test_dataloader):
                result, result_metric, label = predict(self.config.command, tmp, self.device, self.model, self.criterion, self.config)
                pre_seq = tmp['seq'][0]
                result_metric['seq'] = pre_seq
                result_metric['RSS'] = result
                result_metric['label'] = label
                RSS = pd.concat([RSS, pd.DataFrame(result_metric)])
        RSS.to_csv(self.config.output, index=False)
    
    def Predict_single(self):
        self.model.eval()
        seq = self.config.seq
        msg = self.get_msg(seq, length = 112)
        with torch.no_grad():
            result = predict(self.config.command, msg, self.device, self.model, self.criterion, self.config, 0)
        # print(seq)
        # print(result)
        self.log.info(seq)
        self.log.info(result)
        with open('RSS_predict.fasta', 'w') as f:
            f.write(f'>{seq}\n{result}')
        f.close()

    def get_msg(self, seq, pad = 16, length = 112):
        seq_length = len(seq)
        if pad != 0 and len(seq) % pad != 0:
            matrix_seq = seq + 'N' * (pad - len(seq) % pad)
        else:
            matrix_seq = seq
        if len(matrix_seq) < length:
            matrix_seq = matrix_seq + 'N' * (length - len(matrix_seq))
        RNA_onehot = get_one_hot(matrix_seq)
        matrix = torch.tensor(Entropy.GET_ALL_CHANNEL(matrix_seq, self.config.MLDP, self.config.DIS, self.config.CDP, self.config.SPTIAL_DIS, self.config.UFOLD, self.config.UNPAIR, self.config.REPEAT, self.config.UFOLD_ADD_UNPAIR))
        mask = torch.zeros(matrix.size())
        mask[:, :seq_length, :seq_length] = 1
        matrix = matrix * mask

        tokens = self.get_3_mer(matrix_seq)
        tokens = ['CLS'] + tokens + ['SEP']
        attention_mask = [1] * len(tokens)
        tokenizer = DNATokenizer.from_pretrained('DNABERT3')
        tokens = tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0] * len(tokens)
        tokens = torch.IntTensor(tokens)
        attention_mask = torch.IntTensor(attention_mask)
        mask = torch.zeros((len(matrix_seq), len(matrix_seq)))
        mask[:seq_length, :seq_length] = 1
        # print(tokens.shape)
        result = {
            'matrix': matrix.unsqueeze(0),
            'length': seq_length,
            'one_hot': torch.tensor(RNA_onehot.transpose(-1, -2)).unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'seq': seq,
            'tokens': tokens.unsqueeze(0),
            'attention_mask': attention_mask.unsqueeze(0),
            'token_type_ids': token_type_ids,
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



if __name__ == '__main__':
    config = get_config()
    print(config)
    set_seed(config.seed)
    if config.seq == '':
        test_dataloader = select_dataset(config.dataset, config, 'test')
    if config.seq == '':
        run = Predict(config, test_dataloader)
        run.Predict()
    else:
        run = Predict(config)
        run.Predict_single()

