import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import math

class Global_SlidingWindow_Attention(nn.Module):
    def __init__(self, config, vis):
        super(Global_SlidingWindow_Attention, self).__init__()
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

        # Window size for sliding window attention
        self._window_mask_cache = {}
        self.window_size = config.window_size if hasattr(config, 'window_size') else 128

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def create_sliding_window_mask(self, seq_length):

        if seq_length not in self._window_mask_cache:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mask = torch.zeros((seq_length, seq_length), dtype=torch.bool, device=device)

            # 基础滑动窗口
            for i in range(seq_length):
                start = max(0, i - self.window_size // 2)
                end = min(seq_length, i + self.window_size // 2 + 1)
                mask[i, start:end] = True

            # 分层窗口增强
            if seq_length > 64:
                stride = 64
                for i in range(0, seq_length, stride):
                    center = min(i + stride // 2, seq_length - 1)
                    mask[center, :] = True  # 中心节点全局可见

                    # 连接相邻中心点
                    if i > 0:
                        prev_center = max(0, i - stride // 2)
                        mask[center, prev_center] = True
                        mask[prev_center, center] = True

            self._window_mask_cache[seq_length] = mask

        return self._window_mask_cache[seq_length]

    def forward(self, hidden_states):
        device = hidden_states.device

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Create sliding window mask
        seq_length = hidden_states.size(1)
        window_mask = self.create_sliding_window_mask(seq_length).to(device)
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

        # Apply mask - set masked positions to -infinity
        attention_scores = attention_scores.masked_fill(window_mask == 0, -1e9)

        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights