import pandas as pd
import numpy as np
from tools.utils import set_seed, get_clip_data
from tools.config import get_config
import torch
import torch.nn as nn
import logging
import os
from tqdm import tqdm
from tools.select import select_dataset, select_model, select_criterion_optimizer, cal_loss, get_metric
from tools.metric import Init_metric, update_metric, cal_metric, pd_to_metric


class Evaluate():
    def __init__(self, config, test_dataloader):
        self.config = config
        self.test_dataloader = test_dataloader
        self.device = 'cuda:{}'.format(config.gpu) if torch.cuda.is_available() and config.gpu >= 0 else 'cpu'
        self.model = torch.load(self.config.save_path, map_location=self.device)

        self.criterion, self.optimizer = select_criterion_optimizer(config.command, self.device, self.model, config)
        logging.basicConfig(filename=config.log_path, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        self.log = logging.getLogger(config.filename)
        self.log.info(config)
    
    def Evaluate(self):
        self.model.eval()
        # self.model.train()
        all_loss = 0
        metric = pd.DataFrame({})
        preds = []
        label = []
        with torch.no_grad():
            for tmp in tqdm(self.test_dataloader):
                if self.config.command in ['PrismNet', 'PrismNet_Str', 'Multi_Modal']:
                    loss, _, __ = get_metric(self.config.command, tmp, self.device, self.model, self.criterion, self.config)
                    preds.extend(_)
                    label.extend(__)
                else:
                    loss, result = get_metric(self.config.command, tmp, self.device, self.model, self.criterion, self.config)
                    metric = pd.concat([metric, pd.DataFrame(result)])
                
                all_loss = all_loss + loss
                
        if self.config.command in ['PrismNet', 'PrismNet_Str', 'Multi_Modal']:
            preds = np.array(preds)
            label = np.array(label)
            result = cal_metric(preds, label, all_loss / len(self.test_dataloader))
        else:
            result = pd_to_metric(metric, all_loss / len(self.test_dataloader))

        print(result)
        self.log.info('Loss: {:.6f}\t{}'.format(all_loss, result))

if __name__ == '__main__':
    config = get_config()
    print(config)
    set_seed(config.seed)
    test_dataloader = select_dataset(config.dataset, config, 'test')
    run = Evaluate(config, test_dataloader)
    run.Evaluate()
