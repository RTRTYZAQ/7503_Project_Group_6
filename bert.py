from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertSelfAttention
from bert_moe import MoEAttention, MoEAttentionExpert
import torch
import torch.nn as nn

class BertAttentionEnhancedSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, enhanced_attention="None"):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.enhanced_attention = enhanced_attention
        print("Enhanced Attention is",self.enhanced_attention)
        # 原始BERT模型（不包含分类头）
        self.bert = BertModel(config)

        # 替换所有attention层为MoE Attention
        for layer in self.bert.encoder.layer:
            if self.enhanced_attention == "MoE":
                # 替换为MoE Attention, 并且随机初始化参数
                layer.attention.self = MoEAttention(config, num_experts=4, top_k=2)
            else:
                # 替换为普通BERT Attention
                layer.attention.self = BertSelfAttention(config)

        # 分类器
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 通过BERT模型获取输出
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取[CLS] token的隐藏状态
        pooled_output = outputs[1]

        # 分类头
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


