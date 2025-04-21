import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm


def train_model(model, train_dataset, val_dataset, model_name="model", num_epochs=3):
    # 准备DataLoader
    train_dataloader = DataLoader(
        train_dataset.select(range(5000)),  # 使用前5000个样本作为小数据集
        batch_size=16,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset.select(range(872)),  # 使用前1000个样本验证
        batch_size=16
    )

    # 优化器和学习率调度
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / len(train_dataloader)

        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            val_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += len(batch["labels"])

        val_accuracy = correct / total
        avg_val_loss = val_loss / len(val_dataloader)

        print(f"{model_name} - Epoch {epoch + 1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Acc: {val_accuracy:.4f}\n")

    return model