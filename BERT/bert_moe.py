import math
import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention


class MoEAttentionExpert(nn.Module):
    """单个Attention专家"""

    def __init__(self, config):
        super().__init__()
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

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return (context_layer,)


class MoEAttention(BertSelfAttention):
    """混合专家Attention层"""

    def __init__(self, config, num_experts=4, top_k=2):
        super().__init__(config)
        self.num_experts = num_experts
        self.top_k = top_k

        # 创建专家池
        self.experts = nn.ModuleList([MoEAttentionExpert(config) for _ in range(num_experts)])

        # 门控网络
        self.gate = nn.Linear(config.hidden_size, num_experts)
        self.softmax = nn.Softmax(dim=-1)

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
        # MoE Attention目前不支持encoder_hidden_states(即不支持cross-attention)
        if encoder_hidden_states is not None:
            raise ValueError("MoEAttention does not support cross-attention")

        # 计算门控权重 - 使用[CLS] token或平均池化
        gate_input = hidden_states[:, 0, :]  # 使用[CLS] token
        # 或者: gate_input = hidden_states.mean(dim=1)  # 使用平均池化

        gate_logits = self.gate(gate_input)  # [batch_size, num_experts]
        gate_probs = self.softmax(gate_logits)

        # 选择top-k专家
        top_k_gate_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_gate_probs = top_k_gate_probs / top_k_gate_probs.sum(dim=-1, keepdim=True)

        # 初始化输出
        batch_size, seq_length, _ = hidden_states.shape
        context_layer = torch.zeros(
            (batch_size, seq_length, self.all_head_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        # 如果需要输出attention权重
        all_attentions = () if output_attentions else None

        # 计算各专家的输出并加权组合
        for i, expert in enumerate(self.experts):
            # 创建当前专家的mask
            expert_mask = (top_k_indices == i).any(dim=1).float()  # [batch_size]

            if expert_mask.sum() == 0:
                continue

            # 计算当前专家输出
            expert_output = expert(
                hidden_states=hidden_states,
                attention_mask=attention_mask
            )[0]  # 取第一个输出(忽略可能的attention probs)

            # 计算权重 (batch_size, 1, 1)
            weights = (top_k_gate_probs * (top_k_indices == i).float()).sum(dim=1)
            weights = weights.view(-1, 1, 1)

            # 只对选中的batch应用该专家的输出
            context_layer += expert_output * weights * expert_mask.view(-1, 1, 1)

        # 处理head mask(如果需要)
        if head_mask is not None:
            context_layer = context_layer * head_mask

        # 返回格式与原始BERT一致
        outputs = (context_layer,)
        if output_attentions:
            outputs += (all_attentions,)

        return outputs