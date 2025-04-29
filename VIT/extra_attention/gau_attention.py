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


# class GAU_Attention(nn.Module):
#     def __init__(self, config, vis):
#         super(GAU_Attention, self).__init__()
#         self.vis = vis
#         self.hidden_size = config.hidden_size
        
#         # 强制单头注意力
#         self.num_attention_heads = 1
#         # self.attention_head_size = int(config.hidden_size * 2)  # 扩展维度e=2d
#         self.attention_head_size = int(config.hidden_size)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         # 定义投影矩阵（保持原始参数名称以兼容预训练权重）
#         self.query = Linear(config.hidden_size, self.all_head_size)  # W_u
#         self.key = Linear(self.all_head_size, self.all_head_size)    # W_q
#         self.value = Linear(config.hidden_size, self.all_head_size)  # W_v
        
#         # 门控和输出投影
#         self.gate = Linear(config.hidden_size, config.hidden_size)
#         nn.init.xavier_uniform_(self.gate.weight)
        
#         self.out = Linear(self.all_head_size, config.hidden_size)    # W_o
        
#         # Dropout层
#         self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
#         self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        
        
#         self.softmax = Softmax(dim=-1)

#     def forward(self, hidden_states):
#         # Step 1: 生成U和V（带扩展维度）
#         U = torch.nn.functional.silu(self.query(hidden_states))  # [B, L, 2d]
#         V = torch.nn.functional.silu(self.value(hidden_states))  # [B, L, 2d]

#         # Step 2: 生成Q和K
#         Q = self.key(U)  # [B, L, 2d]
#         K = U            # [B, L, 2d]

#         # Step 3: 计算注意力分数
#         attention_scores = torch.matmul(Q, K.transpose(-1, -2))  # [B, L, L]
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
#         # Step 4: 应用Sigmoid门控
#         attention_probs = torch.sigmoid(attention_scores)
#         attention_probs = self.attn_dropout(attention_probs)

#         # Step 5: 值聚合
#         context = torch.matmul(attention_probs, V)  # [B, L, 2d]
        
#         # Step 6: 门控输出
#         output = self.out(context)  # [B, L, d]
#         gate = torch.sigmoid(self.gate(hidden_states))  # [B, L, d]
#         output = output * gate
        
#         # Step 7: 最终处理
#         output = self.proj_dropout(output)
#         weights = attention_probs if self.vis else None
        
#         return output, weights


# class GAU_Attention(nn.Module):
#     def __init__(self, config, vis):
#         super(GAU_Attention, self).__init__()
#         self.vis = vis
#         self.hidden_size = config.hidden_size
#         # self.extended_size = 2 * config.hidden_size  # 扩展维度（例如2倍）
#         self.extended_size = config.hidden_size  # 扩展维度（例如2倍）

#         # 保持与ViT原始结构一致的线性层（用于加载预训练权重）
#         self.query = nn.Linear(config.hidden_size, config.hidden_size)
#         self.key = nn.Linear(config.hidden_size, config.hidden_size)
#         self.value = nn.Linear(config.hidden_size, config.hidden_size)
#         # self.out = nn.Linear(config.hidden_size, config.hidden_size)

#         # GAU特有扩展层（新增参数，不加载预训练权重）
#         self.W_u = nn.Linear(config.hidden_size, self.extended_size)
#         self.W_v = nn.Linear(config.hidden_size, self.extended_size)
#         self.W_gate = nn.Linear(config.hidden_size, config.hidden_size)
#         self.out = nn.Linear(self.extended_size, config.hidden_size)
#         # self.W_u = self.query
#         # self.W_v = self.k

#         # 初始化扩展层权重
#         nn.init.xavier_uniform_(self.W_u.weight)
#         nn.init.xavier_uniform_(self.W_v.weight)
#         nn.init.xavier_uniform_(self.W_gate.weight)
#         nn.init.xavier_uniform_(self.out.weight)

#         # Dropout
#         self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
#         self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])

#     def forward(self, hidden_states):
#         # 使用原始query/key/value层（仅用于加载预训练权重，实际计算不使用）
#         _ = self.query(hidden_states)  # 保持参数加载但不参与计算
#         _ = self.key(hidden_states)
#         _ = self.value(hidden_states)

#         # GAU实际计算流程
#         U = torch.nn.functional.silu(self.W_u(hidden_states))  # [B, L, 2d]
#         V = torch.nn.functional.silu(self.W_v(hidden_states))  # [B, L, 2d]

#         Q = self.key(U)  # 使用扩展后的维度计算Q
#         K = U            # 直接使用U作为K

#         # 计算注意力分数（Sigmoid门控）
#         attention_scores = torch.matmul(Q, K.transpose(-1, -2))  # [B, L, L]
#         attention_scores = attention_scores / math.sqrt(self.extended_size)
#         attention_probs = torch.sigmoid(attention_scores)
#         attention_probs = self.attn_dropout(attention_probs)

#         # 值聚合
#         context = torch.matmul(attention_probs, V)  # [B, L, 2d]
#         output = self.out(context)  # [B, L, d]

#         # 门控机制
#         gate = torch.sigmoid(self.W_gate(hidden_states))  # [B, L, d]
#         output = output * gate

#         # 最终处理
#         output = self.proj_dropout(output)
#         weights = attention_probs if self.vis else None

#         return output, weights



# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Linear, LayerNorm

# class GAU_Attention(nn.Module):
#     def __init__(self, config, vis):
#         super().__init__()
#         self.vis = vis
#         self.hidden_size = config.hidden_size
#         self.max_seq_length = config.max_position_embeddings
#         self.expansion_factor = 2  # 可配置参数
        
#         # 保持原始ViT结构用于加载预训练权重
#         self.query = Linear(config.hidden_size, config.hidden_size)
#         self.key = Linear(config.hidden_size, config.hidden_size)
#         self.value = Linear(config.hidden_size, config.hidden_size)
#         self.out = Linear(config.hidden_size, config.hidden_size)
        
#         # GAU扩展参数
#         self.e = int(config.hidden_size * self.expansion_factor)
#         self.s = 128  # 可配置参数
        
#         # 门控扩展层
#         self.W_u = Linear(config.hidden_size, self.e)
#         self.W_v = Linear(config.hidden_size, self.e)
#         self.W_gate = Linear(config.hidden_size, self.s)
        
#         # 位置编码参数
#         self.gamma = nn.Parameter(torch.randn(2, self.s))
#         self.beta = nn.Parameter(torch.randn(2, self.s))
#         self.register_buffer('a', torch.randn(1, self.s))
#         self.register_buffer('b', torch.randn(1, self.s))
        
#         # 初始化
#         nn.init.xavier_uniform_(self.W_u.weight)
#         nn.init.xavier_uniform_(self.W_v.weight)
#         nn.init.xavier_uniform_(self.W_gate.weight)
        
#         # 标准化层
#         self.norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
#         # Dropout
#         self.attn_dropout = nn.Dropout(config.attention_dropout_rate)
#         self.proj_dropout = nn.Dropout(config.hidden_dropout_prob)

#     def rope(self, x, dim=-1):
#         """旋转位置编码实现"""
#         shape = x.shape
#         half_dim = shape[dim] // 2
#         freqs = torch.arange(half_dim, dtype=torch.float, device=x.device) / half_dim
#         inv_freq = 10000 ** -freqs
        
#         position = torch.arange(shape[1], dtype=torch.float, device=x.device).unsqueeze(-1)
#         sinusoid = position * inv_freq.unsqueeze(0)
        
#         sin = torch.sin(sinusoid)
#         cos = torch.cos(sinusoid)
        
#         x1, x2 = x.chunk(2, dim=dim)
#         return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=dim)

#     def rel_pos_bias(self, seq_len):
#         """相对位置偏置计算"""
#         a = self.rope(self.a.expand(seq_len, -1))
#         b = self.rope(self.b.expand(seq_len, -1))
#         return torch.einsum('mk,nk->mn', a, b)

#     def forward(self, hidden_states, attention_mask=None):
#         # 保持原始层调用以加载预训练权重
#         _ = self.query(hidden_states)
#         _ = self.key(hidden_states)
#         _ = self.value(hidden_states)
        
#         # 标准化
#         x = self.norm(hidden_states)
#         B, L, _ = x.shape
        
#         # 生成扩展特征
#         u = F.silu(self.W_u(x))
#         v = F.silu(self.W_v(x))
#         gate = self.W_gate(x)
        
#         # 位置编码
#         base = torch.einsum('...r,hr->...hr', gate, self.gamma) + self.beta
#         base = self.rope(base, dim=1)
#         q, k = base.unbind(dim=-2)
        
#         # 注意力计算
#         attn_scores = torch.einsum('bnd,bmd->bnm', q, k) / math.sqrt(self.s)
#         attn_scores += self.rel_pos_bias(L)[:L, :L]
        
#         # 掩码处理
#         if attention_mask is not None:
#             attn_scores += attention_mask.squeeze(1)
        
#         # 门控注意力
#         attn_weights = torch.sigmoid(attn_scores)
#         attn_weights = self.attn_dropout(attn_weights)
        
#         # 值聚合
#         context = torch.einsum('bnm,bme->bne', attn_weights, v)
        
#         # 输出投影（复用原始out层权重）
#         output_part1 = self.out(context[..., :self.hidden_size])
#         output_part2 = self.out(context[..., self.hidden_size:])
#         output = output_part1 + output_part2
        
#         # 残差连接
#         output = self.proj_dropout(output) + hidden_states
        
#         # 可视化权重
#         weights = attn_weights if self.vis else None
#         return output, weights

#     def load_pretrained(self, query_w, key_w, value_w, out_w):
#         """加载预训练权重"""
#         self.query.weight.data.copy_(query_w)
#         self.key.weight.data.copy_(key_w)
#         self.value.weight.data.copy_(value_w)
#         self.out.weight.data.copy_(out_w)




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