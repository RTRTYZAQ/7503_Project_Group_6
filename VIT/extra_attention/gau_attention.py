import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import math

# class GAU_Attention(nn.Module):
#     def __init__(self, config, vis):
#         super(GAU_Attention, self).__init__()
#         self.vis = vis
#         self.num_attention_heads = config.transformer["num_heads"]
#         self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = Linear(config.hidden_size, self.all_head_size)
#         self.key = Linear(config.hidden_size, self.all_head_size)
#         self.value = Linear(config.hidden_size, self.all_head_size)

#         self.out = Linear(config.hidden_size, config.hidden_size)
#         self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
#         self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

#         self.softmax = Softmax(dim=-1)

#     def forward(self, hidden_states):

#         ## TO DO:
#         ## 实现相应attention的forward
#         ## 参照__init__中的变量名完成实现, VIT需要按照 Wq,Wk,Wv,Wo 的名称读取预训练权重
#         ## 需要添加一行 weights = attention_probs if self.vis else None, attention_probs是softmax的结果(变量名无所谓), 
#         ##             在return时 一同返回 output和weights，便于后续注意力可视化展示
#         ## 可以参考下我乱写的 random_attention.py

#         pass



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

        # GAU-specific parameters
        self.s = 128  # 可调整的超参数
        self.expansion_factor = 2
        self.e = int(self.attention_head_size * self.expansion_factor)
        self.gamma = nn.Parameter(torch.randn(2, self.s))
        self.beta = nn.Parameter(torch.randn(2, self.s))
        self.a = nn.Parameter(torch.randn(1, self.s))
        self.b_param = nn.Parameter(torch.randn(1, self.s))  # 避免命名冲突

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def rope(self, x, dim):
        shape = x.shape
        spatial_shape = (shape[dim],)
        position = torch.arange(spatial_shape[0], dtype=torch.float, device=x.device).reshape(spatial_shape)
        for _ in range(dim + 1, len(shape) - 1):
            position = position.unsqueeze(-1)
        half_size = shape[-1] // 2
        freq_seq = -torch.arange(half_size, dtype=torch.float, device=x.device) / half_size
        inv_freq = 10000 ** freq_seq
        sinusoid = torch.einsum("...,d->...d", position, inv_freq)
        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def rel_pos_bias(self, seq_len):
        a = self.rope(self.a.repeat(seq_len, 1), dim=0)
        b = self.rope(self.b_param.repeat(seq_len, 1), dim=0)
        return torch.einsum("mk,nk->mn", a, b)

    def forward(self, hidden_states):
        # Project inputs to Q/K/V
        mixed_query = self.query(hidden_states)
        mixed_key = self.key(hidden_states)
        mixed_value = self.value(hidden_states)

        # Reshape to multi-head format
        query = self.transpose_for_scores(mixed_query)
        key = self.transpose_for_scores(mixed_key)
        value = self.transpose_for_scores(mixed_value)

        # # Apply RoPE to Q and K
        # query = self.rope(query, dim=2)
        # key = self.rope(key, dim=2)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Add relative position bias
        seq_length = hidden_states.size(1)
        rel_bias = self.rel_pos_bias(seq_length).unsqueeze(0).unsqueeze(0)
        attention_scores += rel_bias

        # GAU gating mechanism
        attention_weights = torch.square(F.relu(attention_scores / seq_length))
        attention_probs = self.softmax(attention_weights)
        attention_probs = self.attn_dropout(attention_probs)

        # Context layer
        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*context_shape)

        # Output projection
        attention_output = self.out(context)
        attention_output = self.proj_dropout(attention_output)
        
        weights = attention_probs if self.vis else None
        return attention_output, weights