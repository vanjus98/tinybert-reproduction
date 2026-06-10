"""
Post-training dynamic quantization extension for the TinyBERT reproduction.

This script evaluates whether the already distilled TinyBERT student can be
compressed further using INT8 dynamic quantization.

For each selected GLUE task, it loads the TinyBERT student model from:
  - `checkpoints/<task>/student`

The experiment:
  1. Evaluates the original TinyBERT student on the GLUE validation set
  2. Applies PyTorch dynamic quantization to Linear layers using INT8 weights
  3. Evaluates the quantized TinyBERT model on the same validation set
  4. Compares original vs quantized TinyBERT using:
       - task metric: accuracy for SST-2, F1 for MRPC
       - parameter count
       - latency per batch
       - throughput
       - metric retention %
       - quantized model speedup

Note:
  Dynamic quantization is performed on CPU because PyTorch dynamic quantization
  is intended for CPU inference.

Results are written to:
  - `evaluation/quantization_results/quantization_sst2.json`
  - `evaluation/quantization_results/quantization_mrpc.json`
  - `evaluation/quantization_results/quantization_summary.json`
"""

import os
import json
import time
import torch

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tinybert.glue_utils import TASK_CONFIG, preprocess_dataset
from tinybert.teacher import evaluate_model


CHECKPOINTS_DIR = "checkpoints"
OUTPUT_DIR = "evaluation/quantization_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")  # dynamic quantization works on CPU


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_latency(model, sample_batch, n_batches=100, n_warmup=10):
    model.eval()
    batch = {k: v.to(DEVICE) for k, v in sample_batch.items() if k != "labels"}

    with torch.no_grad():
        for _ in range(n_warmup):
            model(**batch)

        start = time.perf_counter()
        for _ in range(n_batches):
            model(**batch)
        elapsed = time.perf_counter() - start

    return elapsed / n_batches


def evaluate_quantization(task, batch_size=32):
    task_cfg = TASK_CONFIG[task]
    metric_key = task_cfg["metric"]
    max_seq = 64 if task_cfg["sentence_keys"][1] is None else 128

    student_dir = os.path.join(CHECKPOINTS_DIR, task, "student")

    print(f"\n{'='*60}")
    print(f"Quantization experiment: {task.upper()} | metric: {metric_key}")
    print(f"{'='*60}")

    raw = load_dataset("nyu-mll/glue", task)
    tokenizer = AutoTokenizer.from_pretrained(student_dir)

    val_ds = preprocess_dataset(raw["validation"], tokenizer, task_cfg, max_seq)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    sample_batch = next(iter(val_loader))

    # Original TinyBERT
    student = AutoModelForSequenceClassification.from_pretrained(student_dir)
    student.to(DEVICE)
    student.eval()

    original_metrics = evaluate_model(student, val_loader, DEVICE, task_cfg)
    original_latency = measure_latency(student, sample_batch)
    original_params = count_params(student)

    # INT8 dynamic quantization
    quantized_student = torch.quantization.quantize_dynamic(
        student,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    quantized_student.eval()

    quantized_metrics = evaluate_model(quantized_student, val_loader, DEVICE, task_cfg)
    quantized_latency = measure_latency(quantized_student, sample_batch)
    quantized_params = count_params(quantized_student)

    result = {
        "task": task,
        "metric": metric_key,
        "device": "cpu",
        "original_tinybert": {
            metric_key: original_metrics[metric_key],
            "params": original_params,
            "latency_ms_per_batch": original_latency * 1000,
            "throughput_examples_per_s": batch_size / original_latency
        },
        "quantized_tinybert_int8": {
            metric_key: quantized_metrics[metric_key],
            "params": quantized_params,
            "latency_ms_per_batch": quantized_latency * 1000,
            "throughput_examples_per_s": batch_size / quantized_latency
        },
        "metric_retention_pct": (
            quantized_metrics[metric_key] / original_metrics[metric_key] * 100
            if original_metrics[metric_key] else None
        ),
        "speedup": original_latency / quantized_latency
    }

    print(f"\n{'':22}{'Original':>14}{'INT8 Quantized':>18}")
    print(f"{metric_key:<22}{original_metrics[metric_key]:>14.4f}{quantized_metrics[metric_key]:>18.4f}")
    print(f"{'params':<22}{original_params/1e6:>12.2f}M{quantized_params/1e6:>16.2f}M")
    print(f"{'ms/batch':<22}{original_latency*1000:>14.2f}{quantized_latency*1000:>18.2f}")
    print(f"{'throughput ex/s':<22}{batch_size/original_latency:>14.2f}{batch_size/quantized_latency:>18.2f}")
    print(f"\nMetric retention: {result['metric_retention_pct']:.2f}%")
    print(f"INT8 speedup:     {result['speedup']:.2f}x")

    out_path = os.path.join(OUTPUT_DIR, f"quantization_{task}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("Saved:", out_path)
    return result


def main():
    all_results = {}
    for task in ["sst2", "mrpc"]:
        all_results[task] = evaluate_quantization(task)

    out_path = os.path.join(OUTPUT_DIR, "quantization_summary.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved summary:", out_path)


if __name__ == "__main__":
    main()