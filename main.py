from datasets import load_dataset
from transformers import BertForSequenceClassification
from train import train_model
from bert import BertAttentionEnhancedSequenceClassification
from transformers import BertPreTrainedModel, BertModel, BertConfig,BertForPreTraining
import torch
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR

# 加载SST-2数据集
dataset = load_dataset("glue", "sst2")
random_seed = 42
torch.manual_seed(random_seed)

# 查看数据集结构
print(dataset)
print("\n样例:")
print(dataset["train"][0])
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# 预处理数据集
encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


config = BertConfig.from_pretrained('bert-base-uncased')

# 训练原始BERT模型
print("=== 训练原始BERT模型 ===")
original_bert = BertAttentionEnhancedSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
)
for name, param in original_bert.named_parameters():
    if 'attention' in name and param.data.dim() >=2:
        # 如果参数属于attention部分，则重新随机初始化
        print(name)
        param.data = torch.nn.init.xavier_uniform_(param.data)
        original_bert.state_dict()[name] = param.data

optimizer = AdamW(original_bert.parameters(), lr=1e-5)
lr_scheduler = None

train_model(
    original_bert,
    encoded_dataset["train"],
    encoded_dataset["validation"],
    "Original BERT",
    num_epochs=3,
    train_size=5000,
    val_size=len(encoded_dataset["validation"]),
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    batch_size=16
)


# 训练自定义MoE Attention BERT模型
print("\n=== 训练自定义MoE Attention BERT模型 ===")
moe_bert = BertAttentionEnhancedSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    enhanced_attention="MoE", # 使用MoE Attention
)

optimizer = AdamW(moe_bert.parameters(), lr=1e-5)
lr_scheduler = None

train_model(
    moe_bert,
    encoded_dataset["train"],
    encoded_dataset["validation"],
    "MoE BERT",
    num_epochs=3,
    train_size=5000,
    val_size=len(encoded_dataset["validation"]),
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    batch_size=16
)