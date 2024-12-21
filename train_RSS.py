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
# from model.Multi_Modal import Multi_Modal


class train():
    def __init__(self, config, train_dataloader, test_dataloader):
        print('Start process')
        self.config = config
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = 'cuda:{}'.format(config.gpu) if torch.cuda.is_available() and config.gpu >= 0 else 'cpu'

        self.model = select_model(config.command, self.device, self.config)
        self.criterion, self.optimizer = select_criterion_optimizer(config.command, self.device, self.model, self.config)
        if self.config.command == 'Multi_Modal' and self.config.text_vision:
            self.optimizer1, self.optimizer2 = self.optimizer[0], self.optimizer[1]
            self.optimizer = self.optimizer1

        if self.config.pretrain == True:
            self.load_model(self.config.pretrain_path)
        self.epochs = config.epochs
        logging.basicConfig(filename=config.log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.log = logging.getLogger(config.filename)
        self.log.info(config)
        self.step = 0
        if self.config.command in ['Multi_Modal']:
            self.eval_step = config.eval_step
            self.max_step = config.max_step
        else:
            self.eval_step = -1

    def train_one_loop(self):
        self.model.train()
        all_loss = 0
        for tmp in tqdm(self.train_dataloader):
            
            self.step = self.step + 1
            if self.eval_step != -1 and self.step > self.max_step and self.max_step != -1:
                break
            loss = cal_loss(self.config.command, tmp, self.device, self.model, self.criterion, self.config)

            all_loss = all_loss + loss.item()
            if self.config.command == 'Multi_Modal' and self.config.text_vision:
                self.optimizer2.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.config.command == 'Multi_Modal' and self.config.text_vision:
                self.optimizer2.step()

            if self.config.command in ['Multi_Modal'] and self.step % self.eval_step == 0 and self.eval_step != -1:
                self.eval()
                self.model.train()
            

        print(all_loss)
        return all_loss
    
    def train(self):
        self.best_metric = Init_metric()
        early_stop = self.config.early_stop
        start_step = 0
        for epoch in range(self.epochs):
            if self.config.command in ['Multi_Modal'] and self.step > self.max_step and self.eval_step != -1 and self.max_step != -1:
                break
            loss = self.train_one_loop()
            if self.eval_step == -1:
                now_metric = self.evaluate()
                if now_metric[self.config.metric] > self.best_metric[self.config.metric] and self.config.metric != 'LOSS':
                    update_metric(now_metric, self.best_metric)
                    early_stop = self.config.early_stop
                    self.save()
                elif now_metric[self.config.metric] <= self.best_metric[self.config.metric] and self.config.metric == 'LOSS':
                    update_metric(now_metric, self.best_metric)
                    early_stop = self.config.early_stop
                    self.save()
                else:
                    early_stop = early_stop - 1

                self.log.info('Epoch: {}\nNOW METRIC: {}\nBEST METRIC: {}'.format(epoch, now_metric, self.best_metric))

                if early_stop == 0:
                    break
        if self.eval_step != -1:
            self.eval()

    def eval(self):
        now_metric = self.evaluate()
        if now_metric[self.config.metric] > self.best_metric[self.config.metric] and self.config.metric != 'LOSS':
            update_metric(now_metric, self.best_metric)
            early_stop = self.config.early_stop
            self.save()
        elif now_metric[self.config.metric] <= self.best_metric[self.config.metric] and self.config.metric == 'LOSS':
            update_metric(now_metric, self.best_metric)
            early_stop = self.config.early_stop
            self.save()
        self.log.info('Step: {}\nNOW METRIC: {}\nBEST METRIC: {}'.format(self.step, now_metric, self.best_metric))
        
    def save(self):
        save_path = self.config.save_path
        import dill
        torch.save(self.model, save_path, pickle_module=dill)
        print('finish save model {}'.format(save_path))

    def load(self):
        load_path = self.config.load_model
        load_model_state = torch.load(load_path, map_location=self.device).state_dict()
        model_state = self.model.state_dict()
        for name in load_model_state.keys():
            if name in model_state:
                model_state[name] = load_model_state[name]
                model_state[name].requires_grad = False
        self.model.load_state_dict(model_state)
        for name, param in self.model.named_parameters():
            if name[:6] == 'stage3' or name[:6] == 'stage4' or name[:5] == 'class':
                continue
            if name in load_model_state.keys():
                param.requires_grad = False

    def evaluate(self):
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
        # self.log.info('Loss: {:.6f}\t{}'.format(all_loss, result))
        return result
    
    def load_model(self, path):
        loads = torch.load(path, map_location=self.device)
        self.model.load_state_dict(loads.state_dict())
        for param in self.model.parameters():
            param.requires_grad_(True)
        self.model.to(self.device)
        print('finish load')


if __name__ == '__main__':
    config = get_config()
    print(config)
    set_seed(config.seed)
    train_dataloader, test_dataloader = select_dataset(config.dataset, config)
    run = train(config, train_dataloader, test_dataloader)
    run.train()
