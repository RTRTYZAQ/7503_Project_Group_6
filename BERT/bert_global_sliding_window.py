import torch
import torch.nn as nn
import math


class LongformerAttention(nn.Module):
    def __init__(self, config, vis=False):
        super(LongformerAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.proj_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)
        self.vis = vis

        # Longformer specific
        self.window_size = 64  # 可配置为config.window_size
        self._window_mask_cache = {}

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def create_window_mask(self, seq_length):
        """创建滑动窗口注意力掩码（完全保持RandomAttention的缓存机制）"""
        if seq_length not in self._window_mask_cache:
            device = self.query.weight.device
            mask = torch.zeros((seq_length, seq_length), dtype=torch.bool, device=device)
            for i in range(seq_length):
                start = max(0, i - self.window_size // 2)
                end = min(seq_length, i + self.window_size // 2 + 1)
                mask[i, start:end] = 1
            self._window_mask_cache[seq_length] = mask
        return self._window_mask_cache[seq_length]

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
        # 完全保持与RandomAttention相同的投影逻辑
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算注意力分数（保持原始矩阵乘法）
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 仅将随机掩码替换为滑动窗口掩码（其他逻辑完全不变）
        seq_length = hidden_states.size(1)
        window_mask = self.create_window_mask(seq_length)
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # 保持与RandomAttention相同的维度
        attention_scores = attention_scores.masked_fill(~window_mask, -1e9)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        # 完全保持原始的形状变换逻辑
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        outputs = (attention_output,)
        if output_attentions:
            outputs += (weights,)

        return outputs