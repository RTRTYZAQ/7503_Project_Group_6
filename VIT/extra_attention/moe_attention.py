import torch
import torch.nn as nn
import math


class MoEAttention(nn.Module):
    def __init__(self, config, vis, num_experts=4, top_k=2):
        super(MoEAttention, self).__init__()
        self.vis = vis
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Query, Key, Value
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Experts
        self.experts = nn.ModuleList([nn.Linear(self.all_head_size, self.all_head_size)
                                      for _ in range(num_experts)])
        self.gate = nn.Linear(self.all_head_size, num_experts)
        self.softmax = nn.Softmax(dim=-1)

        # Expert initialization
        for expert in self.experts:
            nn.init.xavier_uniform_(expert.weight)
            nn.init.zeros_(expert.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        # Output layers
        self.out = nn.Linear(self.all_head_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # Compute Q, K, V - 保持原始attention计算不变
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        # Context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 改进的MoE门控机制
        # 使用所有token的平均作为门控输入，而非仅[CLS]
        gate_input = context_layer.mean(dim=1)  # [batch_size, all_head_size]
        gate_logits = self.gate(gate_input)  # [batch_size, num_experts]

        # Top-k专家选择
        gate_probs = self.softmax(gate_logits)
        top_k_gate_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)  # [batch_size, top_k]

        # 高效计算专家输出
        batch_size = context_layer.size(0)
        seq_len = context_layer.size(1)

        # 展平处理以便高效计算
        flat_context = context_layer.view(-1, self.all_head_size)  # [batch*seq_len, all_head_size]

        # 为每个token选择top-k专家 (复制门控选择到所有token)
        expanded_top_k_indices = top_k_indices.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1,
                                                                                            self.top_k)  # [batch*seq_len, top_k]
        expanded_top_k_gate_probs = top_k_gate_probs.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1,
                                                                                                  self.top_k)  # [batch*seq_len, top_k]

        # 初始化输出
        moe_output = torch.zeros_like(flat_context)

        # 只计算被选中的专家
        for expert_idx in range(self.num_experts):
            # 找出需要使用当前专家的样本
            expert_mask = (expanded_top_k_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            # 计算这些样本的专家输出
            expert_input = flat_context[expert_mask]
            expert_out = self.experts[expert_idx](expert_input)

            # 计算这些样本的权重
            sample_weights = (expanded_top_k_indices[expert_mask] == expert_idx).float() * \
                             expanded_top_k_gate_probs[expert_mask]
            sample_weights = sample_weights.sum(dim=-1, keepdim=True)  # [num_selected, 1]

            # 加权累加到输出
            moe_output[expert_mask] += expert_out * sample_weights

        # 恢复原始形状
        moe_output = moe_output.view(batch_size, seq_len, self.all_head_size)

        # Final output
        attention_output = self.out(moe_output)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, attention_probs if self.vis else None