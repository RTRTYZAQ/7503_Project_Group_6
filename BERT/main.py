from datasets import load_dataset, load_from_disk
from transformers import BertForSequenceClassification
from train import train_model
from bert import BertAttentionEnhancedSequenceClassification
from transformers import BertPreTrainedModel, BertModel, BertConfig,BertForPreTraining
import torch
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
from transformers import BertTokenizer
import os

# 设置随机种子
random_seed = 42
torch.manual_seed(random_seed)

# 定义数据集列表（名称，子集，文本字段）
dataset_list = [
    ("glue", "sst2", "sentence"),
    ("glue", "cola", "sentence"),
    ("glue", "mrpc", ["sentence1", "sentence2"]),
    ("glue", "stsb", ["sentence1", "sentence2"]),
    ("glue", "qqp", ["question1", "question2"]),
    ("glue", "mnli", ["premise", "hypothesis"]),
    ("glue", "qnli", ["question", "sentence"]),
    ("glue", "rte", ["sentence1", "sentence2"]),
    ("glue", "wnli", ["sentence1", "sentence2"]),
]

def load_all_datasets(base_dir="processed_datasets"):
    datasets = {}
    for dataset_dir in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_dir)
        if os.path.isdir(dataset_path):
            datasets[dataset_dir] = {
                "train": load_from_disk(os.path.join(dataset_path, "train")),
                "validation": load_from_disk(os.path.join(dataset_path, "validation")) if os.path.exists(os.path.join(dataset_path, "validation")) else None,
                "test": load_from_disk(os.path.join(dataset_path, "test")) if os.path.exists(os.path.join(dataset_path, "test")) else None
            }
    return datasets

# 加载所有数据集
all_datasets = load_all_datasets()

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
    all_datasets["glue_sst2"]["train"],
    all_datasets["glue_sst2"]["validation"],
    "Original BERT",
    num_epochs=3,
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
    all_datasets["glue_sst2"]["train"],
    all_datasets["glue_sst2"]["validation"],
    "MoE BERT",
    num_epochs=3,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    batch_size=16
)