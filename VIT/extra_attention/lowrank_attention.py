import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from scipy import ndimage

import math

class LowRank_Attention(nn.Module):
    def __init__(self, config, vis):
        super(LowRank_Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 线性投影层
        self.query = Linear(config.hidden_size, self.all_head_size)  # Wq
        self.key = Linear(config.hidden_size, self.all_head_size)    # Wk
        self.value = Linear(config.hidden_size, self.all_head_size)  # Wv
        self.out = Linear(config.hidden_size, config.hidden_size)    # Wo

        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)
        
        # lowrank参数
        self.projection_dim = min(256, self.attention_head_size)  # 随机特征的维度
        self.eps = 1e-6  # 数值稳定性
        self.scale = 1 / math.sqrt(self.attention_head_size)
        
        # 初始化随机投影矩阵
        projection_matrix = torch.randn(
            self.num_attention_heads, self.attention_head_size, self.projection_dim
        ) * self.scale
        self.register_buffer('projection_matrix', projection_matrix)

    def transpose_for_scores(self, x):
        """将输入重塑为多头注意力格式"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]

    def forward(self, hidden_states):
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
        
        # 线性变换
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # 调整形状为多头格式
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # 投影到随机特征空间
        query_projection = torch.einsum('bhnd,hdm->bhnm', query_layer, self.projection_matrix)
        key_projection = torch.einsum('bhnd,hdm->bhnm', key_layer, self.projection_matrix)
        
        # 为数值稳定性减去最大值
        query_projection = query_projection - query_projection.max(dim=-1, keepdim=True)[0]
        key_projection = key_projection - key_projection.max(dim=-1, keepdim=True)[0]
        
        # 应用exp变换
        query_projection = torch.exp(query_projection)
        key_projection = torch.exp(key_projection)
        
        # 计算低秩近似
        kv = torch.einsum('bhnd,bhne->bhde', key_projection, value_layer)
        
        # 计算Q' * (K' * V)
        qkv = torch.einsum('bhnd,bhde->bhne', query_projection, kv)
        
        # 计算归一化因子
        k_sum = torch.sum(key_projection, dim=2)  # [batch, heads, projection_dim]
        normalizer = torch.einsum('bhnd,bhd->bhn', query_projection, k_sum).unsqueeze(-1)
        
        # 归一化输出
        context_layer = qkv / (normalizer + self.eps)
        
        # 重塑输出
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # 输出投影
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        
        # 计算用于可视化的注意力权重
        if self.vis:
            with torch.no_grad():
                attention_probs = torch.matmul(query_layer, key_layer.transpose(-1, -2)) * self.scale
                attention_probs = self.softmax(attention_probs)
        else:
            attention_probs = None
            
        weights = attention_probs if self.vis else None
        
        return attention_output, weights