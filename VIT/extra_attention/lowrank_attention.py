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
        
        # Linformer 参数
        # 低秩投影的目标维度，越小效率越高
        self.k = min(128, self.attention_head_size)
        self.scale = 1 / math.sqrt(self.attention_head_size)
        
        # 一个标准 ViT 中，序列长度 = 1 (CLS token) + patches数量
        # 对于 224x224 图像和 16x16 patch，有 (224/16)² = 14² = 196 个 patch
        # 序列长度 = 1 + 196 = 197
        seq_len = 197  # CLS token + patches
        
        # 初始化 Linformer 的 E_k 和 E_v 投影矩阵
        # 这些矩阵将序列长度从 seq_len 降到 k
        self.E_k = nn.Parameter(torch.Tensor(seq_len, self.k))
        self.E_v = nn.Parameter(torch.Tensor(seq_len, self.k))
        
        # 以正态分布初始化参数
        nn.init.normal_(self.E_k, mean=0, std=0.02)
        nn.init.normal_(self.E_v, mean=0, std=0.02)

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
        query_layer = self.transpose_for_scores(mixed_query_layer)  # [batch, heads, seq_len, head_dim]
        key_layer = self.transpose_for_scores(mixed_key_layer)      # [batch, heads, seq_len, head_dim]
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [batch, heads, seq_len, head_dim]
        
        # Linformer 投影 - 降低序列长度维度
        # key_layer: [batch, heads, seq_len, head_dim]
        # E_k: [seq_len, k]
        # 结果: [batch, heads, k, head_dim]
        projected_keys = torch.matmul(self.E_k.t().unsqueeze(0).unsqueeze(0), key_layer)
        
        # value_layer: [batch, heads, seq_len, head_dim]
        # E_v: [seq_len, k]
        # 结果: [batch, heads, k, head_dim]
        projected_values = torch.matmul(self.E_v.t().unsqueeze(0).unsqueeze(0), value_layer)
        
        # 计算注意力分数
        # query_layer: [batch, heads, seq_len, head_dim]
        # projected_keys: [batch, heads, k, head_dim]
        # 结果: [batch, heads, seq_len, k]
        attention_scores = torch.matmul(query_layer, projected_keys.transpose(-1, -2))
        attention_scores = attention_scores * self.scale
        
        # 应用 Softmax 获得注意力权重
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        
        # 将注意力权重应用于投影后的值
        # attention_probs: [batch, heads, seq_len, k]
        # projected_values: [batch, heads, k, head_dim]
        # 结果: [batch, heads, seq_len, head_dim]
        context_layer = torch.matmul(attention_probs, projected_values)
        
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
                # 这里计算的是原始的注意力矩阵，用于可视化目的
                orig_attention_probs = torch.matmul(query_layer, key_layer.transpose(-1, -2)) * self.scale
                orig_attention_probs = self.softmax(orig_attention_probs)
        else:
            orig_attention_probs = None
            
        weights = orig_attention_probs if self.vis else None
        
        return attention_output, weights