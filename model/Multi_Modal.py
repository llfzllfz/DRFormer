import torch
import torch.nn as nn
import sys
sys.path.append('DNABERT/src')
from transformers import BertConfig, BertForSequenceClassification
import torch.nn.functional as F
import torch
import os
import sys

class DNABERT(nn.Module):
    def __init__(self, device, pretrain = True):
        super().__init__()
        config = BertConfig.from_pretrained('DNABERT3', finetuning_task = 'dnaprom', num_labels = 2)
        # config.attention_probs_dropout_prob = 0.3
        # config.hidden_dropout_prob = 0.3
        # config.rnn_dropout = 0.3
        if pretrain == True:
            self.text_emb = BertForSequenceClassification.from_pretrained('DNABERT3', from_tf=bool(".ckpt" in 'DNABERT3'), config=config)
        else:
            self.text_emb = BertForSequenceClassification(config = config)
        self.text_emb.classifier = torch.nn.Identity()
        self.text_emb.bert.pooler = torch.nn.Identity()
        for param in self.text_emb.parameters():
            param.requires_grad = True
        self.text_emb.to(device)
        # print(self.text_emb.classifier.parameters())
        # for name, param in self.text_emb.classifier.parameters():
        #     print(name, param)
    
    def forward(self, x1, x2):
        x = self.text_emb(input_ids = x1, attention_mask = x2)
        return x[0]
        return x

class VIT(nn.Module):
    def __init__(self, input_channel, device, pretrain = True, SWIT_pretrain_path = 'models/swin.pth.tar', pad_length = 112):
        super().__init__()
        self.input_channel = input_channel
        self.pad_length = pad_length
        self.device = device
        self.pretrain = pretrain
        self.pic_emb = self.load_model(SWIT_pretrain_path)
        for param in self.pic_emb.parameters():
            param.requires_grad = True
        self.pic_emb.to(device)
        # self.identity = nn.Identity()
    
    def forward(self, x):
        latent, _, __ = self.pic_emb.forward_encoder(x, 0)
        # latent = self.identity(latent)
        # print(latent.shape)
        return latent
    
    def load_model(self, SWIT_pretrain_path):
        import sys
        from functools import partial
        from .SWIN import green_mim_swin_base_patch4_dec512b1, SwinTransformer, MaskedAutoencoder, green_mim_swin_large_patch4_dec512b1

        model = green_mim_swin_base_patch4_dec512b1()
        if self.pad_length != 112:
            model_encoder = SwinTransformer(
                img_size=self.pad_length,
                patch_size=self.pad_length // 112 * 2,
                in_chans=24,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.1,
                attn_drop_rate=0.1,
                drop_path_rate=0.1,
                ape=False,
                patch_norm=True,
                use_checkpoint=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-6))
            model.encoder = model_encoder
        if self.pretrain == True:
            # SWIN-T-PRETRAIN
            model_path = SWIT_pretrain_path
            state_dict = torch.load(model_path, map_location=self.device)['model']
            dicts = {}
            model_state_dicts = model.state_dict()
            for key in state_dict.keys():
                dicts[key.replace('module.model.', '')] = state_dict[key]
                if key.replace('module.model.', '') in model_state_dicts:
                    if model_state_dicts[key.replace('module.model.', '')].shape == state_dict[key].shape:
                        model_state_dicts[key.replace('module.model.', '')] = state_dict[key]
                        # print(key.replace('module.model.', ''))
                        pass
                    else:
                        print(key.replace('module.model.', ''), ' not suit')
                        pass
                else:
                    print(key.replace('module.model.', ''), ' not in')
                    pass
            
            model.load_state_dict(model_state_dicts)
            print('load model {}'.format(model_path))
        model.decoder_embed = torch.nn.Identity()
        model.decoder_blocks = torch.nn.Identity()
        model.decoder_norm = torch.nn.Identity()
        model.decoder_pred = torch.nn.Identity()
        return model

class FFN(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.linear = nn.Linear(hid_dim, hid_dim)
        self.norm = nn.LayerNorm(hid_dim)
        self.relu = nn.SELU()

    def forward(self, x):
        out = self.linear(x)
        out = self.relu(self.norm(out) + x)
        return out

class Cross_Attention(nn.Module):
    def __init__(self, dropout, direction = 0, cross = 0, kl = 0):
        super().__init__()
        self.direction = direction
        self.cross = cross
        self.kl = kl
        if self.direction == 1:
            self.attention_direction_1 = nn.MultiheadAttention(embed_dim = 768, num_heads = 6, dropout = dropout, batch_first=True)
            self.attention_direction_2 = nn.MultiheadAttention(embed_dim = 768, num_heads = 6, dropout = dropout, batch_first=True)
        if self.cross == 1:
            self.attention_corss_1 = nn.MultiheadAttention(embed_dim = 768, num_heads = 6, dropout = dropout, batch_first=True)
            self.attention_cross_2 = nn.MultiheadAttention(embed_dim = 768, num_heads = 6, dropout = dropout, batch_first=True)
        self.ffn_1 = FFN(768)
        self.ffn_2 = FFN(768)
    
    def forward(self, x1, x2):
        l1 = torch.zeros_like(x1)
        l2 = torch.zeros_like(x2)
        if self.direction == 1:
            x1_self, _ = self.attention_direction_1(x1, x1, x1)
            # print(x1.shape, x1_self.shape, x2.shape)
            x2_self, _ = self.attention_direction_2(x2, x2, x2)
            l1 = l1 + x1_self
            l2 = l2 + x2_self
        if self.cross == 1:
            x1_cross, _ = self.attention_corss_1(x1, x2, x2)
            x2_cross, _ = self.attention_cross_2(x2, x1, x1)
            l1 = l1 + x1_cross
            l2 = l2 + x2_cross
        x1 = self.ffn_1(l1)
        x2 = self.ffn_2(l2)
        if self.kl == 1:
            return x1, x2, x2_cross
        return x1, x2

class Multi_Modal(nn.Module):
    def __init__(self, input_channel = 17,
                 text_only = 0, vision_only = 0, text_vision = 1, kl = 0,
                 direction = 0, cross = 0,
                 cross_attention_num_layers = 2, dropout = 0.1, device = 'cpu', pretrain = False,
                 SWIT_pretrain_path = 'models/swin.pth.tar',
                 pretrain_module = True,
                 pad_length = 112, num_classes = 13):
        super().__init__()
        self.text_only = text_only
        self.vision_only = vision_only
        self.text_vision = text_vision
        self.kl = kl
        if self.text_only or self.text_vision:
            self.text = DNABERT(device, pretrain_module)
        if self.vision_only or self.text_vision:
            self.vision = VIT(input_channel, device, pretrain_module, SWIT_pretrain_path, pad_length)
        if self.text_vision:
            self.cross_attention_layers = nn.ModuleList([Cross_Attention(dropout, direction = direction, cross = cross, kl = self.kl) for _ in range(cross_attention_num_layers)])
        self.classifier = nn.Linear(768, num_classes)
        # self.classifier1 = nn.Linear(768, seq_length + seq_length // 16)
        # self.output = nn.Linear(seq_length + seq_length // 16, num_classes)
        # self.classifier = nn.Linear(68 + 49, 1)
        self.pretrain = pretrain
        
    def forward(self, tokens, attention_mask, vision):
        if self.text_only or self.text_vision:
            text = self.text(tokens, attention_mask) # B, L, C
        if self.text_only == 1:
            text = torch.mean(text, dim = 1)
            if self.kl == 1:
                return self.classifier(text), 0
            return self.classifier(text)
        
        if self.vision_only or self.text_vision:
            vision = self.vision(vision) # B, H, W, C
        if self.vision_only == 1:
            vision = torch.mean(vision, dim = 1)
            if self.kl == 1:
                return self.classifier(vision), 0
            return self.classifier(vision)
        
        residule_text, residule_vision = text, vision

        for layer in self.cross_attention_layers:
            if self.kl == 1:
                text, vision, _ = layer(text, vision)
            else:
                text, vision = layer(text, vision)

        if self.kl == 1:
            kl = KL_div(_, residule_vision)
        
        # print(text.shape, vision.shape)
        cat = torch.cat([text, vision], dim = 1)
        cat = torch.mean(cat, dim = 1)
        # cat = text[:, 0, :]
        result = self.classifier(cat)
        # result = torch.sigmoid(result)
        if self.pretrain == True:
            return cat, result, kl
        if self.kl == 1:
            return result, kl
        return result

def KL_div(p, q):
    p = torch.softmax(p, -1)
    q = torch.softmax(q, -1)
    kl = torch.nn.functional.kl_div(torch.log(p), q, reduction='batchmean')
    return kl

def cal_JS_Loss(p, q):
    m = 0.5 * (p + q)
    m = torch.softmax(m, -1)
    p = torch.softmax(p, -1)
    q = torch.softmax(q, -1)
    kl_pm = torch.nn.functional.kl_div(torch.log(p), m, reduction='batchmean')
    kl_qm = torch.nn.functional.kl_div(torch.log(q), m, reduction='batchmean')
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd

if __name__ == '__main__':
    model = DNABERT(device = 'cpu')
    x = torch.ones(1,10).int()
    print(model(x, x))
