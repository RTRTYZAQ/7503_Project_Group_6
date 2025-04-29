import torch
import torch.nn as nn
import math

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

class Random_Attention(nn.Module):
    def __init__(self, config, vis):
        super(Random_Attention, self).__init__()
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

        # 使用更安全的属性存储方式
        self._random_mask_cache = {}
        self._current_seq_length = None

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def create_random_mask(self, seq_length, sparsity=0.5):
        """创建随机注意力掩码"""
        device = self.query.weight.device
        # 生成随机矩阵并保留top-k（控制稀疏度）
        rand_matrix = torch.rand(seq_length, seq_length, device=device)
        rand_matrix.fill_diagonal_(1)  # 确保自注意力
        
        # 计算要保留的元素数量
        k = int(seq_length * (1 - sparsity))
        topk = torch.topk(rand_matrix, k=k, dim=1)
        
        # 创建稀疏掩码
        mask = torch.zeros_like(rand_matrix)
        mask.scatter_(1, topk.indices, 1)
        mask.fill_diagonal_(1)  # 再次确保对角线为1
        
        return mask.bool()

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 获取或创建随机掩码
        seq_length = hidden_states.size(1)
        if seq_length not in self._random_mask_cache:
            self._random_mask_cache[seq_length] = self.create_random_mask(seq_length)
        
        mask = self._random_mask_cache[seq_length]
        mask = mask.unsqueeze(0).unsqueeze(0)  # 增加batch和head维度
        
        # 应用掩码
        attention_scores = attention_scores.masked_fill(~mask, -1e9)

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