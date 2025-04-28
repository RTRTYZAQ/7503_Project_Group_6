import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef
from evaluate import load

def train_model(model,
                train_dataset,
                val_dataset,
                test_dataset,
                model_name="model",
                num_epochs=3,
                optimizer=None,
                lr_scheduler=None,
                batch_size=16,
                dataset_name="sst-2"):
    # 准备DataLoader

    train_dataloader = DataLoader(
        # random sample train_size
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    # 优化器和学习率调度
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=1e-5)
    num_training_steps = num_epochs * len(train_dataloader)
    if lr_scheduler is None:
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_dataloader)//2,
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
        all_predictions = []
        all_labels = []

        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            val_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += len(batch["labels"])
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

        glue_metric = load("glue", dataset_name)
        results = glue_metric.compute(predictions=all_predictions, references=all_labels)
        print(results)

    if dataset_name != "glue_sst-2":
        test_loss = 0
        test_correct = 0
        test_total = 0
        test_all_predictions = []
        test_all_labels = []

        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            test_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            test_correct += (predictions == batch["labels"]).sum().item()
            test_total += len(batch["labels"])
            test_all_predictions.extend(predictions.cpu().numpy())
            test_all_labels.extend(batch["labels"].cpu().numpy())
        glue_metric = load("glue", dataset_name)
        test_results = glue_metric.compute(predictions=test_all_predictions, references=test_all_labels)
        print(test_results)

        return model, test_results
    else:
        return model, results