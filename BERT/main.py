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

# 初始化结果存储
results = []
num_epochs = 1
batch_size = 16

# 定义模型配置
model_configs = [
    {"name": "Random BERT", "attention": "Random"},
    {"name": "Original BERT", "attention": "None"},
    {"name": "MoE BERT", "attention": "MoE"},
    {"name": "GSW BERT", "attention": "Longformer"},
    {"name": "LowRank BERT", "attention": "LowRank"},
    {"name": "GAU BERT", "attention": "GAU"},
    {"name": "BigBird BERT", "attention": "BigBird"},
]

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

    # 遍历模型配置
    for model_config in model_configs:
        model_name = model_config["name"]
        enhanced_attention = model_config["attention"]

        print(f"\n=== 训练{model_name}: {dataset_name} ===")
        model = BertAttentionEnhancedSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels,
            enhanced_attention=enhanced_attention,
        )

        # 初始化注意力层参数
        for name, param in model.named_parameters():
            if 'attention' in name and param.data.dim() >= 2:
                param.data = torch.nn.init.xavier_uniform_(param.data)
                model.state_dict()[name] = param.data

        # 配置优化器
        ft_parameters = []
        for encoder_layer in model.bert.encoder.layer:
            ft_parameters.extend(encoder_layer.attention.parameters())
        ft_parameters.extend(model.classifier.parameters())
        optimizer = AdamW(ft_parameters, lr=2e-5, weight_decay=0.01)
        # optimizer = SGD(ft_parameters, lr=3e-2, momentum=0.9)
        # 训练模型
        metrics = train_model(
            model,
            train_dataset.select(range(min(5000, len(train_dataset)))),
            val_dataset,
            test_dataset,
            f"{model_name} - {dataset_name}",
            num_epochs=num_epochs,
            optimizer=optimizer,
            batch_size=batch_size,
            dataset_name=dataset_name.split("_")[1] if "_" in dataset_name else dataset_name,
        )

        # 保存结果
        results.append({
            "Model": model_name,
            "Dataset": dataset_name,
            **metrics
        })

        # 清理显存
        del model
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