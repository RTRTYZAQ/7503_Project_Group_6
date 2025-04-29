import torch
import torch.nn as nn
import math
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertSelfAttention

class GAUAttention(nn.Module):
    def __init__(self, config, vis=False):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        self.vis = vis
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 严格保持BERT参数命名
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)  # 原始BERT命名
        
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.proj_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

        # 修正门控参数维度
        self.gating_factor = nn.Parameter(torch.tensor([0.5]))
        self.gating_bias = nn.Parameter(torch.randn(1))  # 标量维度

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
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
        # 投影Q/K/V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 注意力分数计算
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 计算门控信号（关键修正）
        gating = torch.sigmoid(
            self.gating_factor * hidden_states.mean(dim=-1) + self.gating_bias  # [batch, seq_len]
        ).unsqueeze(1).unsqueeze(-1)  # 重塑为 [batch, 1, seq_len, 1]

        # 应用门控
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights * gating  # 广播到 [batch, heads, seq_len, seq_len]

        attention_probs = self.atten_dropout(attention_weights)
        
        # 上下文聚合
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 输出投影
        attention_output = self.out(context_layer)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs