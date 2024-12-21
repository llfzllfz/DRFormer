from tools.utils import get_clip_data
import os
import torch
import torch.nn as nn
from tools.ufold_postprocess import postprocess_new as postprocess
from tools.ufold_utils import evaluate_exact_new
import numpy as np
import pandas as pd
from tools.metric import cal_metric, list_to_metric
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

RSS_LIST = ['UFold', 'SWIN_UNET']
RBP_LIST = ['PrismNet', 'PrismNet_Str', 'Multi_Modal']
GUE_LIST = ['Multi_Modal']

def select_dataset(dataset, parameters = None, mode = 'train'):
    if dataset == 'clip':
        if parameters.command == 'Multi_Modal':
            from processing_data.clip_dataloader_Multi_Modal import CLIP_dataloader
            train_dataset, test_dataset = get_clip_data(parameters.k_fold_data_path, parameters.filename, parameters.kfold_k)
            train_dataloader = CLIP_dataloader(train_dataset,
                                            batch_size = parameters.batch_size,
                                            shuffle = True,
                                                num_workers = parameters.dataloader_num_workers,
                                                MLDP = parameters.MLDP,
                                                DIS = parameters.DIS,
                                                CDP = parameters.CDP,
                                                SPTIAL_DIS = parameters.SPTIAL_DIS,
                                                UFOLD = parameters.UFOLD,
                                                UNPAIR = parameters.UNPAIR,
                                                REPEAT = parameters.REPEAT,
                                                UFOLD_ADD_UNPAIR = parameters.UFOLD_ADD_UNPAIR
                                            )
            test_dataloader = CLIP_dataloader(test_dataset,
                                            shuffle = True,
                                            batch_size = parameters.batch_size,
                                                num_workers = parameters.dataloader_num_workers,
                                                MLDP = parameters.MLDP,
                                                DIS = parameters.DIS,
                                                CDP = parameters.CDP,
                                                SPTIAL_DIS = parameters.SPTIAL_DIS,
                                                UFOLD = parameters.UFOLD,
                                                UNPAIR = parameters.UNPAIR,
                                                REPEAT = parameters.REPEAT,
                                                UFOLD_ADD_UNPAIR = parameters.UFOLD_ADD_UNPAIR
                                            )
            if mode == 'test':
                return test_dataloader
        elif parameters.command == 'PrismNet' or parameters.command == 'PrismNet_Str':
            from  processing_data.clip_dataloader_prismnet import CLIP_dataloader
            train_dataset, test_dataset = get_clip_data(parameters.k_fold_data_path, parameters.filename, parameters.kfold_k)
            train_dataloader = CLIP_dataloader(train_dataset,
                                            batch_size = parameters.batch_size,
                                            shuffle = True,
                                                num_workers = parameters.dataloader_num_workers
                                            )
            test_dataloader = CLIP_dataloader(test_dataset,
                                            shuffle = True,
                                            batch_size = parameters.batch_size,
                                                num_workers = parameters.dataloader_num_workers
                                            )
            if mode == 'test':
                return test_dataloader
    elif dataset == 'GUE':
        from processing_data.GUE_dataloader_Multi_Modal import GUE_dataloader
        train_dataloader = GUE_dataloader('train', parameters.k_fold_data_path,
                                                shuffle = True,
                                                batch_size = parameters.batch_size,
                                                num_workers = parameters.dataloader_num_workers,
                                                MLDP = parameters.MLDP,
                                                DIS = parameters.DIS,
                                                CDP = parameters.CDP,
                                                SPTIAL_DIS = parameters.SPTIAL_DIS,
                                                UFOLD = parameters.UFOLD,
                                                UNPAIR = parameters.UNPAIR,
                                                REPEAT = parameters.REPEAT,
                                                UFOLD_ADD_UNPAIR = parameters.UFOLD_ADD_UNPAIR,
                                                pad_length = parameters.pad_length
                                              )
        test_dataloader = GUE_dataloader('test', parameters.k_fold_data_path,
                                                shuffle = True,
                                                batch_size = parameters.batch_size,
                                                num_workers = parameters.dataloader_num_workers,
                                                MLDP = parameters.MLDP,
                                                DIS = parameters.DIS,
                                                CDP = parameters.CDP,
                                                SPTIAL_DIS = parameters.SPTIAL_DIS,
                                                UFOLD = parameters.UFOLD,
                                                UNPAIR = parameters.UNPAIR,
                                                REPEAT = parameters.REPEAT,
                                                UFOLD_ADD_UNPAIR = parameters.UFOLD_ADD_UNPAIR,
                                                pad_length = parameters.pad_length
                                                )
        if mode == 'test':
            test_dataloader = GUE_dataloader('test', parameters.k_fold_data_path,
                                                shuffle = True,
                                                batch_size = parameters.batch_size,
                                                num_workers = parameters.dataloader_num_workers,
                                                MLDP = parameters.MLDP,
                                                DIS = parameters.DIS,
                                                CDP = parameters.CDP,
                                                SPTIAL_DIS = parameters.SPTIAL_DIS,
                                                UFOLD = parameters.UFOLD,
                                                UNPAIR = parameters.UNPAIR,
                                                REPEAT = parameters.REPEAT,
                                                UFOLD_ADD_UNPAIR = parameters.UFOLD_ADD_UNPAIR,
                                                pad_length = parameters.pad_length)
            return test_dataloader
    elif dataset == 'CLS':
        from processing_data.CLS_dataloader import CLS_dataloader
        train_dataloader = CLS_dataloader('train', parameters.k_fold_data_path,
                                                shuffle = True,
                                                batch_size = parameters.batch_size,
                                                num_workers = parameters.dataloader_num_workers,
                                                MLDP = parameters.MLDP,
                                                DIS = parameters.DIS,
                                                CDP = parameters.CDP,
                                                SPTIAL_DIS = parameters.SPTIAL_DIS,
                                                UFOLD = parameters.UFOLD,
                                                UNPAIR = parameters.UNPAIR,
                                                REPEAT = parameters.REPEAT,
                                                UFOLD_ADD_UNPAIR = parameters.UFOLD_ADD_UNPAIR,
                                                pad_length = parameters.pad_length, multi_gpu = parameters.multi_gpu,
                                                vision = (parameters.vision_only | parameters.text_vision ), text = (parameters.text_only | parameters.text_vision))
        test_dataloader = CLS_dataloader('test', parameters.k_fold_data_path,
                                                shuffle = True,
                                                batch_size = parameters.batch_size,
                                                num_workers = parameters.dataloader_num_workers,
                                                MLDP = parameters.MLDP,
                                                DIS = parameters.DIS,
                                                CDP = parameters.CDP,
                                                SPTIAL_DIS = parameters.SPTIAL_DIS,
                                                UFOLD = parameters.UFOLD,
                                                UNPAIR = parameters.UNPAIR,
                                                REPEAT = parameters.REPEAT,
                                                UFOLD_ADD_UNPAIR = parameters.UFOLD_ADD_UNPAIR,
                                                pad_length = parameters.pad_length, multi_gpu = parameters.multi_gpu,
                                                vision = (parameters.vision_only | parameters.text_vision ), text = (parameters.text_only | parameters.text_vision))
        if mode == 'test':
            return test_dataloader
    elif dataset == 'RSS':
        from processing_data.RSS_dataloader import RSS_dataloader
        path = os.path.join(parameters.data_path, parameters.dataset, parameters.filename)
        train_dataloader = RSS_dataloader(path, parameters.feature_mode,
                                        parameters.batch_size,
                                        shuffle = False,
                                        num_workers = parameters.dataloader_num_workers,
                                        MLDP = parameters.MLDP,
                                        DIS = parameters.DIS,
                                        CDP = parameters.CDP,
                                        SPTIAL_DIS = parameters.SPTIAL_DIS,
                                        UFOLD = parameters.UFOLD,
                                        UNPAIR = parameters.UNPAIR,
                                        REPEAT = parameters.REPEAT,
                                        UFOLD_ADD_UNPAIR = parameters.UFOLD_ADD_UNPAIR
                                    )
        path = os.path.join(parameters.data_path, parameters.dataset, parameters.test_filename)
        test_dataloader = RSS_dataloader(path, parameters.feature_mode,
                                        parameters.batch_size,
                                        shuffle = False,
                                        num_workers = parameters.dataloader_num_workers,
                                        MLDP = parameters.MLDP,
                                        DIS = parameters.DIS,
                                        CDP = parameters.CDP,
                                        SPTIAL_DIS = parameters.SPTIAL_DIS,
                                        UFOLD = parameters.UFOLD,
                                        UNPAIR = parameters.UNPAIR,
                                        REPEAT = parameters.REPEAT,
                                        UFOLD_ADD_UNPAIR = parameters.UFOLD_ADD_UNPAIR
                                    )
        if mode == 'test':
            return test_dataloader
    return train_dataloader, test_dataloader

def cal_feature_dim(MLDP = 1, DIS = 1, CDP = 1, SPTIAL_DIS = 1, UFOLD = 1, UNPAIR = 1, REPEAT = 1):
    dim = 0
    if MLDP == 1:
        dim = dim + 1
    if DIS == 1:
        dim = dim + 1
    if SPTIAL_DIS == 1:
        dim = dim + 1
    if CDP == 1:
        dim = dim + 1
    if UFOLD == 1:
        dim = dim + 10
    if UNPAIR == 1:
        dim = dim + 5
    if REPEAT == 1:
        dim = dim + 5
    return dim

def select_model(command, device, parameters = None):
    if command == 'PrismNet':
        from model.PrismNet.PrismNet import PrismNet
        model = PrismNet(device, 'seq').to(device)
    elif command == 'Multi_Modal':
        from model.Multi_Modal import Multi_Modal
        assert parameters.text_only + parameters.vision_only + parameters.text_vision <= 1
        dim = dim = cal_feature_dim(MLDP = parameters.MLDP,
                                  DIS = parameters.DIS,
                                  SPTIAL_DIS = parameters.SPTIAL_DIS,
                                  CDP = parameters.CDP,
                                  UFOLD = parameters.UFOLD,
                                  UNPAIR = parameters.UNPAIR,
                                  REPEAT = parameters.REPEAT)
        classes =  1
        if parameters.dataset == 'CLS':
            classes = 13
        model = Multi_Modal(device = device, input_channel = dim,
                            text_only = parameters.text_only, vision_only = parameters.vision_only,
                            text_vision = parameters.text_vision, kl = 1 if parameters.distribution != 0 else 0,
                            direction = parameters.direction, cross = parameters.cross,
                            cross_attention_num_layers = parameters.cross_attention_num_layers,
                            dropout = parameters.dropout, pretrain = parameters.pretrain, pretrain_module = parameters.pretrain_module, SWIT_pretrain_path = parameters.SWIT_pretrain_path,
                            pad_length = parameters.pad_length, num_classes=classes
                            ).to(device)
        # model = DDP(model, )
    elif command == 'PrismNet_Str':
        from model.PrismNet.PrismNet import PrismNet
        model = PrismNet(device, 'pu').to(device)
    elif command == 'UFold':
        if parameters.FTL == 1:
            from model.UFold_Network2 import U_Net
        else:
            from model.UFold_Network import U_Net
        if parameters.feature_mode == 'UFOLD':
            dim = 17
        if parameters.feature_mode == 'DNAV':
            dim = cal_feature_dim(MLDP = parameters.MLDP,
                                  DIS = parameters.DIS,
                                  SPTIAL_DIS = parameters.SPTIAL_DIS,
                                  CDP = parameters.CDP,
                                  UFOLD = parameters.UFOLD,
                                  UNPAIR = parameters.UNPAIR,
                                  REPEAT = parameters.REPEAT)
        model = U_Net(dim, 1).to(device)
    elif command == 'SWIN_UNET':
        from model.SWIN_UNET import Swin_Unet
        if parameters.feature_mode == 'UFOLD':
            dim = 17
        if parameters.feature_mode == 'DNAV':
            dim = cal_feature_dim(MLDP = parameters.MLDP,
                                  DIS = parameters.DIS,
                                  SPTIAL_DIS = parameters.SPTIAL_DIS,
                                  CDP = parameters.CDP,
                                  UFOLD = parameters.UFOLD,
                                  UNPAIR = parameters.UNPAIR,
                                  REPEAT = parameters.REPEAT)
        # model = Swin_Unet(dim, 1).to(device)
        from model.RSS_SWIN import RSS_SWIN
        model = RSS_SWIN(device).to(device)
        from model.SC_UNet import SC_UNet
        model = SC_UNet(device, dim, parameters.SWIT_pretrain_path).to(device)
        if parameters.pretrain_path is not None:
            model = torch.load(parameters.pretrain_path, map_location=device)
    if parameters.multi_gpu > 0:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
        print('Init Multi gpu model')
    return model

def select_criterion_optimizer(command, device, model, parameters = None):
    if command == 'UFold' or command == 'SWIN_UNET':
        pos_weight = torch.Tensor([200]).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight = pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-5, betas=(0.9, 0.95), weight_decay=0.05)
    elif command == 'Multi_Modal':
        if parameters.dataset == 'CLS':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))
        # criterion = nn.BCEWithLogitsLoss()
        if parameters.text_vision:
            # optimizer = torch.optim.Adam(model.parameters(), lr = parameters.lr, betas=(0.9, 0.999), weight_decay=1e-3)
            optimizer1 = torch.optim.AdamW(model.parameters(), lr = parameters.lr, betas=(0.9, 0.95), weight_decay=0.05)
            if parameters.multi_gpu > 0:
                optimizer2 = torch.optim.AdamW(model.module.vision.parameters(), lr = parameters.Vision_lr, betas = (0.9, 0.95), weight_decay=0.05)
            else:
                optimizer2 = torch.optim.AdamW(model.vision.parameters(), lr = parameters.Vision_lr, betas=(0.9, 0.95), weight_decay=0.05)
            optimizer = [optimizer1, optimizer2]
        if parameters.text_only:
            optimizer = torch.optim.AdamW(model.parameters(), lr = parameters.lr, betas=(0.9, 0.95), weight_decay=0.05)
            # optimizer = torch.optim.Adam(model.parameters(), lr = parameters.lr, betas=(0.9, 0.999), weight_decay=3e-5)
        if parameters.vision_only:
            # optimizer = torch.optim.Adam(model.parameters(), lr = parameters.lr, betas=(0.9, 0.999), weight_decay=5e-5)
            optimizer = torch.optim.AdamW(model.parameters(), lr = parameters.lr, betas=(0.9, 0.95), weight_decay=0.05)
    elif command == 'PrismNet' or command == 'PrismNet_Str':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))
        optimizer = torch.optim.Adam(model.parameters(), lr = parameters.lr, betas=(0.9, 0.999), weight_decay=1e-6)
    return criterion, optimizer

def cal_loss(command, data, device, model, criterion, parameters = None):
    if command == 'PrismNet':
        one_hot = data['one_hot'].to(device)
        out = model(one_hot)
        loss = criterion(out.cpu(), data['label'])
    elif command == 'PrismNet_Str':
        pu = data['pu'].to(device)
        out = model(pu)
        loss = criterion(out.cpu(), data['label'])
    elif command == 'Multi_Modal':
        if parameters.text_vision or parameters.text_only:
            tokens = data['tokens'].to(device)
            attention_mask = data['attention_mask'].to(device)
            vision = None
        if parameters.text_vision or parameters.vision_only:
            tokens = None
            attention_mask = None
            vision = data['matrix'].to(device)
        if parameters.text_vision:
            tokens = data['tokens'].to(device)
            vision = data['matrix'].to(device)
            attention_mask = data['attention_mask'].to(device)
        if parameters.distribution != 0:
            out, _ = model(tokens, attention_mask, vision)
            loss = criterion(out.cpu(), data['label']) + parameters.distribution * _
        else:
            out = model(tokens, attention_mask, vision)
            if parameters.dataset == 'CLS':
                loss = criterion(out.cpu(), data['label'].view(-1))
            else:
                loss = criterion(out.cpu(), data['label'])
    elif command in RSS_LIST:
        matrix = data['matrix'].to(device)
        seq_lens = data['length']
        if command == 'UFold':
            tokens = None
            attention_mask = None
        else:
            tokens = data['tokens'].to(device)
            attention_mask = data['attention_mask'].to(device)
        
        out = model(matrix, tokens, attention_mask)
        mask = data['mask']
        loss = criterion(out.to(device) * mask.to(device), data['label'].to(device))
    return loss

def get_metric(command, data, device, model, criterion, parameters = None):
    if command in RSS_LIST:
        matrix = data['matrix'].to(device)
        seq_lens = data['length']
        if command == 'UFold':
            tokens = None
            attention_mask = None
        else:
            tokens = data['tokens'].to(device)
            attention_mask = data['attention_mask'].to(device)
        with torch.no_grad():
            out = model(matrix, tokens, attention_mask)
        # mask = torch.zeros_like(out)
        # mask[:, :seq_lens[0], :seq_lens[0]] = 1
        mask = data['mask']
        loss = criterion(out.to(device) * mask.to(device), data['label'].to(device))
        # print(data['label'].shape, data['one_hot'].shape)
        seq_ori = data['one_hot']

        u_no_train = postprocess((out.to(device) * mask.to(device)).cpu(),
            seq_ori, 0.01, 0.1, 100, 1.6, True,1.5) ## 1.6
        map_no_train = (u_no_train > 0.5).float()
        result_no_train_tmp = list(map(lambda i: evaluate_exact_new(map_no_train.cpu()[i][:seq_lens[i], :seq_lens[i]],
            data['label'].cpu()[i][:seq_lens[i], :seq_lens[i]]), range(data['label'].shape[0])))
        nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train_tmp)
        result = {
            'F1': np.array(nt_exact_f1),
            'PRECISION': np.array(nt_exact_p),
            'RECALL': np.array(nt_exact_r)
        }
    elif command == 'Multi_Modal':
        if parameters.text_vision or parameters.text_only:
            tokens = data['tokens'].to(device)
            attention_mask = data['attention_mask'].to(device)
            vision = None
        if parameters.text_vision or parameters.vision_only:
            tokens = None
            attention_mask = None
            vision = data['matrix'].to(device)
        if parameters.text_vision:
            tokens = data['tokens'].to(device)
            vision = data['matrix'].to(device)
        # with torch.no_grad():
        if parameters.distribution != 0:
            out, _ = model(tokens, attention_mask, vision)
            loss = criterion(out.cpu(), data['label']) + parameters.distribution * _
        else:
            out = model(tokens, attention_mask, vision)
            if parameters.dataset == 'CLS':
                loss = criterion(out.cpu(), data['label'].view(-1))
            else:
                loss = criterion(out.cpu(), data['label'])
        
        # print(out.view(-1))

        if parameters.dataset != 'CLS':
            out = torch.sigmoid(out)
        else:
            out = torch.max(out, 1).indices.cpu()
        # print(out)
        preds = out.view(-1).tolist()
        label = data['label'].view(-1).tolist()
        return loss.item(), preds, label
        result = [cal_metric(np.array([preds[i]]), np.array([label[i]])) for i in range(len(label))]
        result = list_to_metric(result)

        # result = cal_metric(np.array(preds), np.array(label))
    elif command == 'PrismNet':
        one_hot = data['one_hot'].to(device)
        out = model(one_hot)
        out = torch.sigmoid(out)
        preds = out.view(-1).tolist()
        label = data['label'].view(-1).tolist()
        loss = criterion(out.cpu(), data['label'])
        return loss.item(), preds, label

    elif command == 'PrismNet_Str':
        pu = data['pu'].to(device)
        out = model(pu)
        out = torch.sigmoid(out)
        preds = out.view(-1).tolist()
        label = data['label'].view(-1).tolist()
        loss = criterion(out.cpu(), data['label'])
        return loss.item(), preds, label
        
    return loss.item(), result

def predict(command, data, device, model, criterion, parameters = None, label_need = 1):
    if command in RSS_LIST:
        matrix = data['matrix'].to(device)
        B, _, _, _ = matrix.size()
        assert B == 1, 'Please make sure the batch size is 1'
        seq_lens = data['length']
        tokens = data['tokens'].to(device)
        attention_mask = data['attention_mask'].to(device)
        with torch.no_grad():
            out = model(matrix, tokens, attention_mask)
        # mask = torch.zeros_like(out)
        # mask[:, :seq_lens[0], :seq_lens[0]] = 1
        mask = data['mask']
        # loss = criterion(out.to(device) * mask.to(device), data['label'].to(device))
        # print(data['label'].shape, data['one_hot'].shape)
        seq_ori = data['one_hot']

        u_no_train = postprocess((out.to(device) * mask.to(device)).cpu(),
            seq_ori, 0.01, 0.1, 100, 1.6, True,1.5) ## 1.6
        map_no_train = (u_no_train > 0.5).float()
        result = ['.'] * seq_lens
        for idx in range(seq_lens):
            for idy in range(idx + 1, seq_lens):
                if map_no_train[0, idx, idy] == 1:
                    result[idx] = '('
                    result[idy] = ')'
        if label_need == 1:
            result_no_train_tmp = list(map(lambda i: evaluate_exact_new(map_no_train.cpu()[i][:seq_lens[i], :seq_lens[i]],
                data['label'].cpu()[i][:seq_lens[i], :seq_lens[i]]), range(data['label'].shape[0])))
            nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train_tmp)
            result_metric = {
                'F1': np.array(nt_exact_f1),
                'PRECISION': np.array(nt_exact_p),
                'RECALL': np.array(nt_exact_r)
            }
            label = ['.'] * seq_lens
            for idx in range(seq_lens):
                for idy in range(idx, seq_lens):
                    if data['label'][0, idx, idy] == 1:
                        label[idx] = '('
                        label[idy] = ')'
            return ''.join(result), result_metric, ''.join(label)
        else:
            return ''.join(result)
    if command == 'Multi_Modal':
        if parameters.text_vision or parameters.text_only:
            tokens = data['tokens'].to(device)
            attention_mask = data['attention_mask'].to(device)
            vision = None
        if parameters.text_vision or parameters.vision_only:
            tokens = None
            attention_mask = None
            vision = data['matrix'].to(device)
        if parameters.text_vision:
            tokens = data['tokens'].to(device)
            vision = data['matrix'].to(device)
        # with torch.no_grad():
        if parameters.distribution != 0:
            out, _ = model(tokens, attention_mask, vision)
            loss = criterion(out.cpu(), data['label']) + parameters.distribution * _
        else:
            out = model(tokens, attention_mask, vision)
            if parameters.dataset == 'CLS':
                loss = criterion(out.cpu(), data['label'].view(-1))
            else:
                loss = criterion(out.cpu(), data['label'])
        
        # print(out.view(-1))

        if parameters.dataset != 'CLS':
            out = torch.sigmoid(out)
        else:
            out = torch.max(out, 1).indices.cpu()
        # print(out)
        preds = out.view(-1).tolist()
        label = data['label'].view(-1).tolist()
        return preds, label