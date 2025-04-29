import math
import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention


class LowRankAttention(BertSelfAttention):
    """基于Linformer的低秩自注意力模块"""

    def __init__(self, config, projection_dim=64):
        super().__init__(config)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 原始BERT的query, key, value投影矩阵
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Linformer的核心：序列长度降维投影矩阵
        self.projection_dim = projection_dim
        
        # 投影矩阵E和F分别用于key和value
        # 这些矩阵将序列长度维度从seq_length降至projection_dim
        # 注意：这些投影矩阵会在forward中适应实际序列长度进行初始化
        self.E = None
        self.F = None
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def _init_projection_matrices(self, seq_length, device):
        """根据序列长度初始化低秩投影矩阵"""
        if self.E is None or self.E.size(0) != seq_length:
            self.E = nn.Parameter(torch.Tensor(seq_length, self.projection_dim)).to(device)
            self.F = nn.Parameter(torch.Tensor(seq_length, self.projection_dim)).to(device)
            
            # 使用Xavier初始化
            nn.init.xavier_uniform_(self.E)
            nn.init.xavier_uniform_(self.F)

    def transpose_for_scores(self, x):
        """将输入tensor重塑为多头形式"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        # Linformer目前不支持encoder_hidden_states(即不支持cross-attention)
        if encoder_hidden_states is not None:
            raise ValueError("LinformerAttention does not support cross-attention yet")

        batch_size, seq_length, _ = hidden_states.shape
        device = hidden_states.device
        
        # 初始化或更新投影矩阵
        self._init_projection_matrices(seq_length, device)

        # 计算query, key, value
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [batch, heads, seq_len, head_dim]
        key_layer = self.transpose_for_scores(mixed_key_layer)      # [batch, heads, seq_len, head_dim]
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [batch, heads, seq_len, head_dim]

        # Linformer核心：应用低秩投影到key和value的序列维度
        # 原本: key_layer [batch, heads, seq_len, head_dim]
        # 投影后: key_layer [batch, heads, projection_dim, head_dim]
        projected_key = torch.matmul(key_layer.permute(0, 1, 3, 2), self.E).permute(0, 1, 3, 2)
        projected_value = torch.matmul(value_layer.permute(0, 1, 3, 2), self.F).permute(0, 1, 3, 2)

        # 计算注意力分数: Q * K^T / sqrt(d_k)
        # 这里key已经是低秩投影后的，所以维度是[batch, heads, projection_dim, head_dim]
        attention_scores = torch.matmul(query_layer, projected_key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 应用注意力mask
        if attention_mask is not None:
            # 注意：因为key被投影了，我们需要调整mask或分数的应用方式
            # 这里假设注意力掩码对应于序列中每个token，我们扩展它以匹配query的长度
            attention_scores = attention_scores + attention_mask

        # 归一化注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # 应用head mask(如果有)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 与投影后的value相乘获取输出
        context_layer = torch.matmul(attention_probs, projected_value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # 恢复原始形状
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 与原始BERT保持一致的返回格式
        outputs = (context_layer,)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs