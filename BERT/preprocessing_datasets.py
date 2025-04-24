from datasets import load_dataset
from transformers import BertTokenizer
import torch
from tqdm import tqdm
import random
import os

# 设置随机种子保证可复现性
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

# 定义数据集列表（名称，子集，文本字段）
dataset_list = [
    ("glue", "sst2", "sentence"),
    ("glue", "cola", "sentence"),
    ("glue", "mrpc", ["sentence1", "sentence2"]),
    ("glue", "stsb", ["sentence1", "sentence2"]),
    ("glue", "qqp", ["question1", "question2"]),
    ("glue", "qnli", ["question", "sentence"]),
    ("glue", "rte", ["sentence1", "sentence2"]),
    ("glue", "wnli", ["sentence1", "sentence2"]),
]

# 初始化tokenizer和存储字典
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
processed_datasets = {}  # 存储所有处理好的数据集

# 定义数据量限制
MAX_TRAIN_SAMPLES = 20000
MAX_VAL_SAMPLES = 5000
MAX_TEST_SAMPLES = 20000

# 定义输出目录
OUTPUT_DIR = "./processed_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def limit_dataset_size(dataset, max_samples):
    """如果数据集超过max_samples，随机抽样"""
    if len(dataset) > max_samples:
        print(f"  Randomly sampling {max_samples} from {len(dataset)} samples")
        return dataset.shuffle(seed=random_seed).select(range(max_samples))
    return dataset

def preprocess_function(examples, text_fields):
    """统一处理单句/双句任务的预处理"""
    if isinstance(text_fields, str):
        return tokenizer(
            examples[text_fields],
            truncation=True,
            padding="max_length",
            max_length=256
        )
    else:
        return tokenizer(
            examples[text_fields[0]],
            examples[text_fields[1]],
            truncation=True,
            padding="max_length",
            max_length=256
        )

def save_dataset_to_disk(dataset, save_path):
    """将数据集保存到磁盘"""
    if dataset is not None:
        dataset.save_to_disk(save_path)
        print(f"  Saved to {save_path}")

# 加载并预处理所有数据集
for dataset_name, subset, text_fields in tqdm(dataset_list, desc="Processing Datasets"):
    try:
        print(f"\n{'='*50}")
        print(f"Loading dataset: {dataset_name}/{subset}")

        # 1. 加载原始数据集
        raw_dataset = load_dataset(dataset_name, subset)
        print(f"\nOriginal dataset sizes:")
        print(f"  Train: {len(raw_dataset['train'])}")
        if 'validation' in raw_dataset:
            print(f"  Validation: {len(raw_dataset['validation'])}")
        if 'test' in raw_dataset:
            print(f"  Test: {len(raw_dataset['test'])}")

        # 2. 应用数据量限制
        limited_dataset = {}
        limited_dataset["train"] = limit_dataset_size(raw_dataset["train"], MAX_TRAIN_SAMPLES)

        if 'validation' in raw_dataset:
            limited_dataset["validation"] = limit_dataset_size(
                raw_dataset["validation"],
                MAX_VAL_SAMPLES
            )
        else:
            limited_dataset["validation"] = None

        if 'test' in raw_dataset:
            limited_dataset["test"] = limit_dataset_size(
                raw_dataset["test"],
                MAX_TEST_SAMPLES
            )
        else:
            limited_dataset["test"] = None

        # 3. 预处理数据集
        encoded_dataset = {}
        encoded_dataset["train"] = limited_dataset["train"].map(
            lambda x: preprocess_function(x, text_fields),
            batched=True
        )

        # 对验证集和测试集同样处理（如果存在）
        if limited_dataset["validation"] is not None:
            encoded_dataset["validation"] = limited_dataset["validation"].map(
                lambda x: preprocess_function(x, text_fields),
                batched=True
            )

        if limited_dataset["test"] is not None:
            encoded_dataset["test"] = limited_dataset["test"].map(
                lambda x: preprocess_function(x, text_fields),
                batched=True
            )

        # 4. 统一标签列名
        if "label" in encoded_dataset["train"].column_names:
            if encoded_dataset["train"] is not None:
                encoded_dataset["train"] = encoded_dataset["train"].rename_column("label", "labels")
            if encoded_dataset["validation"] is not None:
                encoded_dataset["validation"] = encoded_dataset["validation"].rename_column("label", "labels")
            if encoded_dataset["test"] is not None:
                encoded_dataset["test"] = encoded_dataset["test"].rename_column("label", "labels")

        # 5. 设置PyTorch格式
        required_columns = ["input_ids", "attention_mask"]
        if encoded_dataset["train"] is not None and "labels" in encoded_dataset["train"].column_names:
            required_columns.append("labels")

        if encoded_dataset["train"] is not None:
            encoded_dataset["train"].set_format("torch", columns=required_columns)
        if encoded_dataset["validation"] is not None:
            encoded_dataset["validation"].set_format("torch", columns=required_columns)
        if encoded_dataset["test"] is not None:
            encoded_dataset["test"].set_format("torch", columns=required_columns)

        # 6. 存储到字典 (使用组合键名)
        key = f"{dataset_name}_{subset}"
        processed_datasets[key] = {
            "train": encoded_dataset["train"],
            "validation": encoded_dataset["validation"],
            "test": encoded_dataset["test"],
            "text_fields": text_fields  # 保留原始文本字段信息
        }

        print(f"\nFinal dataset sizes after sampling:")
        print(f"  Train: {len(processed_datasets[key]['train'])}")
        if processed_datasets[key]["validation"] is not None:
            print(f"  Validation: {len(processed_datasets[key]['validation'])}")
        if processed_datasets[key]["test"] is not None:
            print(f"  Test: {len(processed_datasets[key]['test'])}")

        # 7. 保存到文件
        dataset_dir = os.path.join(OUTPUT_DIR, key)
        os.makedirs(dataset_dir, exist_ok=True)

        print(f"\nSaving dataset to disk...")
        save_dataset_to_disk(processed_datasets[key]["train"], os.path.join(dataset_dir, "train"))
        save_dataset_to_disk(processed_datasets[key]["validation"], os.path.join(dataset_dir, "validation"))
        save_dataset_to_disk(processed_datasets[key]["test"], os.path.join(dataset_dir, "test"))

    except Exception as e:
        print(f"\nError processing {dataset_name}/{subset}: {str(e)}")
        continue

# 最终输出汇总信息
print("\n" + "="*50)
print(f"Completed processing. Saved {len(processed_datasets)} datasets:")
print(f"All datasets saved to directory: {OUTPUT_DIR}")
print("-"*50)
for key in processed_datasets:
    print(f"{key}:")
    print(f"  Train samples: {len(processed_datasets[key]['train'])}")
    if processed_datasets[key]["validation"] is not None:
        print(f"  Val samples: {len(processed_datasets[key]['validation'])}")
    if processed_datasets[key]["test"] is not None:
        print(f"  Test samples: {len(processed_datasets[key]['test'])}")