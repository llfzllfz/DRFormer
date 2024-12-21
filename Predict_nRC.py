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
                result, label = predict(self.config.command, tmp, self.device, self.model, self.criterion, self.config)
                pre_seq = tmp['seq'][0]
                le = tmp['le']
                # print(result, label)
                # print(le)
                result_metric = pd.DataFrame({'seq': pre_seq, 'preds': le[result[0]], 'label': le[label[0]]})
                RSS = pd.concat([RSS, pd.DataFrame(result_metric)])
        RSS.to_csv(self.config.output, index=False)
    
    
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
        run.Predict()
