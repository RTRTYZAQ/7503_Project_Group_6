import torch
import torch.nn as nn
import math

class RandomAttention(nn.Module):
    def __init__(self, config, vis=False):
        super(RandomAttention, self).__init__()
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

        # 缓存随机掩码
        self._random_mask_cache = {}

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def create_random_mask(self, seq_length, sparsity=0.5):
        """创建随机注意力掩码"""
        device = self.query.weight.device
        rand_matrix = torch.rand(seq_length, seq_length, device=device)
        rand_matrix.fill_diagonal_(1)  # 确保自注意力

        k = int(seq_length * (1 - sparsity))
        topk = torch.topk(rand_matrix, k=k, dim=1)

        mask = torch.zeros_like(rand_matrix)
        mask.scatter_(1, topk.indices, 1)
        mask.fill_diagonal_(1)  # 确保对角线为1

        return mask.bool()

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
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        seq_length = hidden_states.size(1)
        if seq_length not in self._random_mask_cache:
            self._random_mask_cache[seq_length] = self.create_random_mask(seq_length)

        random_mask = self._random_mask_cache[seq_length].unsqueeze(0).unsqueeze(0)
        attention_scores = attention_scores.masked_fill(~random_mask, -1e9)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        outputs = (attention_output,)
        if output_attentions:
            outputs += (weights,)

        return outputs