# 只是一个标准多头注意力，方便对照
import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from scipy import ndimage

import math

class Normal_Attention(nn.Module):
    def __init__(self, config, vis):
        super(Normal_Attention, self).__init__()
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
        batch_size, seq_length = hidden_states.size(0), hidden_states.size(1)
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # [B, seq_len, hidden] -> [B, seq_len, num_heads, head_size] -> [B, num_heads, seq_len, head_size]
        query_layer = mixed_query_layer.view(batch_size, seq_length, self.num_attention_heads, 
                                            self.attention_head_size).permute(0, 2, 1, 3)
        key_layer = mixed_key_layer.view(batch_size, seq_length, self.num_attention_heads, 
                                        self.attention_head_size).permute(0, 2, 1, 3)
        value_layer = mixed_value_layer.view(batch_size, seq_length, self.num_attention_heads, 
                                            self.attention_head_size).permute(0, 2, 1, 3)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        attention_probs = self.softmax(attention_scores)
        
        weights = attention_probs if self.vis else None
        
        attention_probs = self.attn_dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # reshape: [B, num_heads, seq_len, head_size] -> [B, seq_len, num_heads, head_size] -> [B, seq_len, hidden_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        
        return attention_output, weights