import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
import os
from functools import partial
from torch.nn import init
sys.path.append('../MAE-Lite/projects/mae_lite')
sys.path.append('DNABERT/src')
from models_swit_mae import green_mim_swin_base_patch4_dec512b1, SwinTransformer, MaskedAutoencoder
import numpy as np
from transformers import BertConfig, BertForSequenceClassification

class DNABERT(nn.Module):
    def __init__(self, device, pretrain = True):
        super().__init__()
        config = BertConfig.from_pretrained('DNABERT3', finetuning_task = 'dnaprom', num_labels = 2)
        if pretrain == True:
            self.text_emb = BertForSequenceClassification.from_pretrained('DNABERT3', from_tf=bool(".ckpt" in 'DNABERT3'), config=config)
        else:
            self.text_emb = BertForSequenceClassification(config = config)
        self.text_emb.classifier = torch.nn.Identity()
        self.text_emb.bert.pooler = torch.nn.Identity()
        for param in self.text_emb.parameters():
            param.requires_grad = True
        self.text_emb.to(device)

    def forward(self, x1, x2):
        x = self.text_emb(input_ids = x1, attention_mask = x2)
        return x[0]

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, is_last = False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
        self.is_last = is_last

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out

class FFN(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.linear = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(0.3)
        # self.linear = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim * 2),
        #     nn.Linear(hid_dim * 2, hid_dim)
        # )
        # self.norm = nn.LayerNorm(hid_dim)
        # self.relu = nn.SELU()

    def forward(self, x):
        out = self.linear(x)
        out = self.dropout(out)
        # out = self.relu(self.norm(out) + x)
        return out

class Cross_Attention(nn.Module):
    def __init__(self, emb_dim = 768, num_heads = 1, dropout = 0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=dropout,batch_first=True)
        self.ffn = FFN(emb_dim)

    def forward(self, x1, x2):
        '''
        x1, x2: B, C, L
        '''
        x, _ = self.attn(x1, x2, x2)
        x = self.ffn(x)
        # return x1
        return x

class Transpose_conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, last = False, ca_kernel_size = 3):
        super().__init__()
        self.last = last
        self.up = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding)
        if last != True:
            self.res_block = BasicBlock(input_channels, output_channels, ca_kernel_size, last)
        else:
            self.res_block = BasicBlock(output_channels, output_channels, ca_kernel_size, last)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.last == False:
            # x2 = self.cross_block(x1, x2)
            x1 = torch.cat([x1, x2], dim = 1)
            x = self.res_block(x1)
        else:
            x = self.res_block(x1)
        return x

class SC_UNet(nn.Module):
    def __init__(self, device, input_channel = 24, swin_pretrain_path = None):
        super().__init__()
        self.swin_pretrain_path = swin_pretrain_path
        self.input_channel = input_channel
        self.device = device
        self.model = self.load_model()
        self.up1 = Transpose_conv(768, 384, 4, 2, 1, ca_kernel_size = 7)
        self.up2 = Transpose_conv(384, 192, 4, 2, 1, ca_kernel_size = 7)
        self.up3 = Transpose_conv(192, 96, 4, 2, 1, ca_kernel_size = 7)
        self.up4 = Transpose_conv(96, 24, 4, 2, 1, True, ca_kernel_size = 7)
        self.text = DNABERT(device, True)
        self.ca = Cross_Attention()
        self.classifier = nn.Conv2d(24, 1, 3, 1, 1)
        self.capture = ActivationCapture()
        self.get_hook()
        self.dropout = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.1)

    def load_model(self):
        model = green_mim_swin_base_patch4_dec512b1()
        model.encoder = SwinTransformer(
                img_size=112,
                patch_size=2,
                in_chans=self.input_channel,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0,
                drop_path_rate=0,
                ape=False,
                patch_norm=True,
                use_checkpoint=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # SWIN-T-PRETRAIN
        # model_path = '../MAE-Lite/outputs/mae_lite/SWIN2/last_epoch_ckpt.pth.tar'
        # model_path = '../MAE-Lite/outputs/mae_lite/SWIN2_model_store/30.pth.tar'
        if self.swin_pretrain_path != None:
            model_path = self.swin_pretrain_path
            print('load pretrain', model_path)
            state_dict = torch.load(model_path, map_location=self.device)['model']
            dicts = {}
            model_state_dicts = model.state_dict()
            for key in state_dict.keys():
                dicts[key.replace('module.model.', '')] = state_dict[key]
                if key.replace('module.model.', '') in model_state_dicts:
                    if model_state_dicts[key.replace('module.model.', '')].shape == state_dict[key].shape:
                        model_state_dicts[key.replace('module.model.', '')] = state_dict[key]
                        # print(key.replace('module.model.', ''))
                    else:
                        print(key.replace('module.model.', ''), ' not suit')
                else:
                    print(key.replace('module.model.', ''), ' not in')
        


        # model.load_state_dict(dicts)
        model.decoder_embed = torch.nn.Identity()
        model.decoder_blocks = torch.nn.Identity()
        model.decoder_norm = torch.nn.Identity()
        model.decoder_pred = torch.nn.Identity()
        for param in model.parameters():
            param.requires_grad = True
        return model

    def get_hook(self):
        for name, module in self.model.named_modules():
            if 'identity' in name and 'attn.identity' not in name:
                print(name)
                module.register_forward_hook(self.capture.get_activation(name))

    def change_shape(self, x):
        B, C, L = x.size()
        x = x.view(B, np.sqrt(C).astype(np.int64), np.sqrt(C).astype(np.int64), L).permute(0, 3, 1, 2)
        # x = x + x.transpose(-1, -2)
        return x

    def forward(self, x, tokens, attention_mask):
        x1, _, _ = self.model.forward_encoder(x, 0)
        d1 = self.capture.activation['encoder.layers.0.identity']
        d2 = self.capture.activation['encoder.layers.1.identity']
        d3 = self.capture.activation['encoder.layers.2.identity']
        d4 = self.capture.activation['encoder.layers.3.identity']

        text_emb = self.text(tokens, attention_mask)
        d4 = self.ca(d4, text_emb)

        d1 = self.change_shape(d1)
        d1 = self.dropout(d1)
        d2 = self.change_shape(d2)
        d2 = self.dropout(d2)
        d3 = self.change_shape(d3)
        d3 = self.dropout(d3)
        d4 = self.change_shape(d4)
        d4 = self.dropout(d4)
        

        # print(d1.shape, d2.shape, d3.shape, d4.shape)
        d3 = self.up1(d4, d3)
        d3 = self.dropout2(d3)
        d2 = self.up2(d3, d2)
        d2 = self.dropout2(d2)
        d1 = self.up3(d2, d1)
        d1 = self.dropout2(d1)
        x = self.up4(d1, x)


        # x1 = x1 + d4
        # x1 = self.up1(x1)
        # x1 = x1 + d3
        # x1 = self.up2(x1)
        # x1 = x1 + d2
        # x1 = self.up3(x1)
        # x1 = x1 + d1
        # x1 = self.up4(x1)
        out = self.classifier(x)


        # B, C, L = x1.size()
        # x1 = x1.view(B, np.sqrt(C).astype(np.int64), np.sqrt(C).astype(np.int64), L).permute(0, 3, 1, 2)
        # x1 = self.transpose(x1)

        # wap = self.wap(x)

        # out = wap + x1

        return (out * out.transpose(-1, -2)).squeeze(1)


class ActivationCapture:
    def __init__(self):
        self.activation = {}

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook