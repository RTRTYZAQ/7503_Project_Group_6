import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import math

class GAU_Attention(nn.Module):
    def __init__(self, config, vis):
        super(GAU_Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def forward(self, hidden_states):

        ## TO DO:
        ## 实现相应attention的forward
        ## 参照__init__中的变量名完成实现, VIT需要按照 Wq,Wk,Wv,Wo 的名称读取预训练权重
        ## 需要添加一行 weights = attention_probs if self.vis else None, attention_probs是softmax的结果(变量名无所谓), 
        ##             在return时 一同返回 output和weights，便于后续注意力可视化展示
        ## 可以参考下我乱写的 random_attention.py

        pass