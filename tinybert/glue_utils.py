"""
Shared utilities for GLUE task preprocessing and metrics.
"""

from typing import Any
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


TASK_CONFIG: dict[str, dict] = {
    "sst2": {
        "sentence_keys": ("sentence", None),
        "num_labels": 2,
        "metric": "accuracy",
        "is_regression": False,
    },
    "mrpc": {
        "sentence_keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "metric": "f1",
        "is_regression": False,
    },
    "qqp": {
        "sentence_keys": ("question1", "question2"),
        "num_labels": 2,
        "metric": "f1",
        "is_regression": False,
    },
    "qnli": {
        "sentence_keys": ("question", "sentence"),
        "num_labels": 2,
        "metric": "accuracy",
        "is_regression": False,
    },
    "rte": {
        "sentence_keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "metric": "accuracy",
        "is_regression": False,
    },
    "cola": {
        "sentence_keys": ("sentence", None),
        "num_labels": 2,
        "metric": "mcc",
        "is_regression": False,
    },
    "stsb": {
        "sentence_keys": ("sentence1", "sentence2"),
        "num_labels": 1,
        "metric": "pearson",
        "is_regression": True,
    },
    "mnli": {
        "sentence_keys": ("premise", "hypothesis"),
        "num_labels": 3,
        "metric": "accuracy",
        "is_regression": False,
    },
}


class GLUEDataset(Dataset):
    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float if isinstance(self.labels[idx], float) else torch.long)
        return item


def preprocess_dataset(
    dataset,
    tokenizer: PreTrainedTokenizer,
    task_cfg: dict,
    max_len: int,
    augmented_sentences: list[str] | None = None,
    augmented_labels: list | None = None,
) -> GLUEDataset:
    key1, key2 = task_cfg["sentence_keys"]

    if augmented_sentences is not None:
        sentences1 = list(augmented_sentences)
        sentences2 = None if key2 is None else list(augmented_sentences)
        labels = list(augmented_labels)
    else:
        sentences1 = list(dataset[key1])
        sentences2 = list(dataset[key2]) if key2 else None
        labels = list(dataset["label"])

    if sentences2 is not None:
        enc = tokenizer(
            sentences1, sentences2,
            truncation=True, padding="max_length", max_length=max_len,
            return_tensors=None,
        )
    else:
        enc = tokenizer(
            sentences1,
            truncation=True, padding="max_length", max_length=max_len,
            return_tensors=None,
        )

    return GLUEDataset(enc, labels)


def compute_metrics(cfg: dict, preds: list, labels: list) -> dict[str, float]:
    metric = cfg["metric"]
    preds_arr = np.array(preds)
    labels_arr = np.array(labels)

    if metric == "accuracy":
        return {"accuracy": float((preds_arr == labels_arr).mean())}
    elif metric == "f1":
        return {"f1": float(f1_score(labels_arr, preds_arr))}
    elif metric == "mcc":
        return {"mcc": float(matthews_corrcoef(labels_arr, preds_arr))}
    elif metric == "pearson":
        r, _ = pearsonr(preds_arr, labels_arr)
        rho, _ = spearmanr(preds_arr, labels_arr)
        return {"pearson": float(r), "spearman": float(rho)}
    else:
        return {"accuracy": float((preds_arr == labels_arr).mean())}
