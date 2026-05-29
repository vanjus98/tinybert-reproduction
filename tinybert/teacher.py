"""
Fine-tune BERT-base on a GLUE task to create the teacher model.

Supports SST-2, MRPC, MNLI, QQP, QNLI, RTE, CoLA, STS-B.
The fine-tuned model is saved and later used as the teacher for distillation.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from torch.optim import AdamW

from .glue_utils import TASK_CONFIG, preprocess_dataset, compute_metrics


@dataclass
class TeacherConfig:
    task: str = "sst2"
    model_name: str = "bert-base-uncased"
    output_dir: str = "checkpoints/teacher"
    max_seq_len: int = 128
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42


def train_teacher(cfg: TeacherConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    task_cfg = TASK_CONFIG[cfg.task]
    raw = load_dataset("glue", cfg.task)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds = preprocess_dataset(raw["train"], tokenizer, task_cfg, cfg.max_seq_len)
    val_ds = preprocess_dataset(raw["validation"], tokenizer, task_cfg, cfg.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=task_cfg["num_labels"]
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_metric = -1.0
    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        metrics = evaluate_model(model, val_loader, device, task_cfg)
        print(f"Epoch {epoch+1}/{cfg.num_epochs}  loss={avg_loss:.4f}  {metrics}")

        score = list(metrics.values())[0]
        if score > best_metric:
            best_metric = score
            os.makedirs(cfg.output_dir, exist_ok=True)
            model.save_pretrained(cfg.output_dir)
            tokenizer.save_pretrained(cfg.output_dir)
            print(f"  Saved best teacher to {cfg.output_dir}")

    print(f"Teacher training complete. Best validation score: {best_metric:.4f}")
    return cfg.output_dir


def evaluate_model(model, loader, device, task_cfg) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1) if task_cfg["num_labels"] > 1 else outputs.logits.squeeze(-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return compute_metrics(cfg=task_cfg, preds=all_preds, labels=all_labels)


def load_and_evaluate(model_dir: str, task: str, max_seq_len: int = 128) -> dict:
    """Convenience: load a saved model and evaluate it on the GLUE validation split."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from .glue_utils import TASK_CONFIG, preprocess_dataset

    task_cfg = TASK_CONFIG[task]
    raw = load_dataset("glue", task)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    val_ds = preprocess_dataset(raw["validation"], tokenizer, task_cfg, max_seq_len)
    val_loader = DataLoader(val_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    return evaluate_model(model, val_loader, device, task_cfg)
