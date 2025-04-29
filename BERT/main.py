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
import pandas as pd
import matplotlib.pyplot as plt

# 设置随机种子
random_seed = 42
torch.manual_seed(random_seed)

# 定义数据集列表（名称，子集，文本字段）
dataset_config = {
    "glue_sst2": 2,
    # "glue_cola": 2, # 过于依赖预训练模型，效果极差
    "glue_mrpc": 2,
    # "glue_qqp": 2, # 同上
    # "glue_qnli": 2,
    # "glue_rte": 2,
    # "glue_wnli": 2,
}

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

'''# 训练原始BERT模型
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
)'''

# 初始化结果存储
results = []
num_epochs = 5
batch_size = 16

# 遍历数据集配置
for dataset_name, num_labels in dataset_config.items():
    print(f"\n=== 开始训练数据集: {dataset_name} ===")

    # 加载数据集
    dataset = all_datasets[dataset_name]
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # 配置BERT模型
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=num_labels)

    # 训练随机注意力BERT模型
    print(f"\n=== 训练随机注意力BERT模型: {dataset_name} ===")
    random_bert = BertAttentionEnhancedSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        enhanced_attention="Random",
    )
    for name, param in random_bert.named_parameters():
        if 'attention' in name and param.data.dim() >= 2:
            param.data = torch.nn.init.xavier_uniform_(param.data)
            random_bert.state_dict()[name] = param.data
    optimizer = AdamW(random_bert.parameters(), lr=2e-5, weight_decay=0.01)
    # optimizer = Adam(random_bert.parameters(), lr=2e-5, betas=(0.9, 0.999))
    model, metrics = train_model(
        random_bert,
        train_dataset.select(range(min(5000, len(train_dataset)))),
        val_dataset,
        test_dataset,
        f"Random BERT - {dataset_name}",
        num_epochs=num_epochs,
        optimizer=optimizer,
        batch_size=batch_size,
        dataset_name=dataset_name.split("_")[1] if "_" in dataset_name else dataset_name
    )
    results.append({
        "Model": "Random BERT",
        "Dataset": dataset_name,
        **metrics
    })
    del random_bert
    torch.cuda.empty_cache()


    # 训练原始BERT模型
    print(f"\n=== 训练原始BERT模型: {dataset_name} ===")
    original_bert = BertAttentionEnhancedSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
    )
    for name, param in original_bert.named_parameters():
        if 'attention' in name and param.data.dim() >= 2:
            param.data = torch.nn.init.xavier_uniform_(param.data)
            original_bert.state_dict()[name] = param.data

    optimizer = AdamW(original_bert.parameters(), lr=2e-5, weight_decay=0.01)
    # optimizer = Adam(original_bert.parameters(), lr=2e-5, betas=(0.9, 0.999))
    model, metrics = train_model(
        original_bert,
        train_dataset.select(range(min(5000,len(train_dataset)))),
        val_dataset,
        test_dataset,
        f"Original BERT - {dataset_name}",
        num_epochs=num_epochs,
        optimizer=optimizer,
        batch_size=batch_size,
        dataset_name=dataset_name.split("_")[1] if "_" in dataset_name else dataset_name,
    )
    results.append({
        "Model": "Original BERT",
        "Dataset": dataset_name,
        **metrics
    })
    del original_bert
    torch.cuda.empty_cache()

    # 训练自定义MoE Attention BERT模型
    print(f"\n=== 训练自定义MoE Attention BERT模型: {dataset_name} ===")
    moe_bert = BertAttentionEnhancedSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        enhanced_attention="MoE",
    )
    for name, param in moe_bert.named_parameters():
        if 'attention' in name and param.data.dim() >= 2:
            param.data = torch.nn.init.xavier_uniform_(param.data)
            moe_bert.state_dict()[name] = param.data

    optimizer = AdamW(moe_bert.parameters(), lr=2e-5, weight_decay=0.01)
    # optimizer = Adam(moe_bert.parameters(), lr=2e-5, betas=(0.9, 0.999))
    model, metrics = train_model(
        moe_bert,
        train_dataset.select(range(min(5000,len(train_dataset)))),
        val_dataset,
        test_dataset,
        f"MoE BERT - {dataset_name}",
        num_epochs=num_epochs,
        optimizer=optimizer,
        batch_size=batch_size,
        dataset_name=dataset_name.split("_")[1] if "_" in dataset_name else dataset_name
    )
    results.append({
        "Model": "MoE BERT",
        "Dataset": dataset_name,
        **metrics
    })
    del moe_bert
    torch.cuda.empty_cache()


    # 训练低秩注意力BERT模型
    print(f"\n=== 训练低秩注意力BERT模型: {dataset_name} ===")
    lowrank_bert = BertAttentionEnhancedSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        enhanced_attention="LowRank",
    )
    for name, param in lowrank_bert.named_parameters():
        if 'attention' in name and param.data.dim() >= 2:
            param.data = torch.nn.init.xavier_uniform_(param.data)
            lowrank_bert.state_dict()[name] = param.data
    optimizer = AdamW(lowrank_bert.parameters(), lr=2e-5, weight_decay=0.01)
    # optimizer = Adam(lowrank_bert.parameters(), lr=2e-5, betas=(0.9, 0.999))
    model, metrics = train_model(
        lowrank_bert,
        train_dataset.select(range(min(5000,len(train_dataset)))),
        val_dataset,
        test_dataset,
        f"LowRank BERT - {dataset_name}",
        num_epochs=num_epochs,
        optimizer=optimizer,
        batch_size=batch_size,
        dataset_name=dataset_name.split("_")[1] if "_" in dataset_name else dataset_name
    )
    results.append({
        "Model": "LowRank BERT",
        "Dataset": dataset_name,
        **metrics
    })
    del lowrank_bert
    torch.cuda.empty_cache()


    # 训练自定义BigBird Attention BERT模型
    print(f"\n=== 训练自定义BigBird Attention BERT模型: {dataset_name} ===")
    bigbird_bert = BertAttentionEnhancedSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        enhanced_attention="BigBird",
    )
    for name, param in bigbird_bert.named_parameters():
        if 'attention' in name and param.data.dim() >= 2:
            param.data = torch.nn.init.xavier_uniform_(param.data)
            bigbird_bert.state_dict()[name] = param.data

    optimizer = AdamW(bigbird_bert.parameters(), lr=2e-5, weight_decay=0.01)
    # optimizer = Adam(bigbird_bert.parameters(), lr=2e-5, betas=(0.9, 0.999))
    model, metrics = train_model(
        bigbird_bert,
        train_dataset.select(range(min(5000,len(train_dataset)))),
        val_dataset,
        f"BigBird BERT - {dataset_name}",
        num_epochs=3,
        optimizer=optimizer,
        batch_size=16,
        dataset_name=dataset_name.split("_")[1] if "_" in dataset_name else dataset_name
    )
    results.append({
        "Model": "BigBird BERT",
        "Dataset": dataset_name,
        **metrics
    })
    del bigbird_bert
    torch.cuda.empty_cache()


# 将结果转换为DataFrame
results_df = pd.DataFrame(results)

# 打印表格
print("\n=== 训练结果 ===")
print(results_df)

# 绘制图表
for dataset_name in dataset_config.keys():
    dataset_results = results_df[results_df["Dataset"] == dataset_name]
    plt.figure(figsize=(12, 6))

    # 绘制 accuracy
    plt.subplot(1, 2, 1)
    for model_name in dataset_results["Model"].unique():
        model_results = dataset_results[dataset_results["Model"] == model_name]
        plt.bar(model_name, model_results["accuracy"].values[0], label=model_name)
    plt.title(f"Accuracy for {dataset_name}")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    # 绘制 f1 分数
    plt.subplot(1, 2, 2)
    for model_name in dataset_results["Model"].unique():
        model_results = dataset_results[dataset_results["Model"] == model_name]
        f1_score = model_results["f1"].values[0]
        if not pd.isna(f1_score):  # 检查是否有 f1 分数
            plt.bar(model_name, f1_score, label=model_name)
    plt.title(f"F1 Score for {dataset_name}")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()