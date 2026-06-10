"""
Evaluation entry point for the TinyBERT reproduction.

For a given GLUE task it loads the distilled student and its BERT-base teacher
from `checkpoints/<task>/{student,teacher}` and reports everything needed for the
comparison-to-paper analysis in one run:

  1. Dev-set metric for teacher and student (accuracy / F1 / MCC / Pearson)
  2. Retention %  = student / teacher
  3. Parameter count of each model + compression ratio
  4. Measured inference speed (latency + throughput) and student-vs-teacher speedup

Results are written to `evaluation/results_<task>.json`.
"""

import argparse
import json
import os
import sys
import time

# Make the `tinybert` package importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tinybert.glue_utils import TASK_CONFIG, preprocess_dataset
from tinybert.teacher import evaluate_model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate distilled TinyBERT checkpoints.")
    p.add_argument("--task", action="append", choices=list(TASK_CONFIG.keys()),
                   help="GLUE task; repeat for several (e.g. --task sst2 --task mrpc). "
                        "Defaults to sst2 + mrpc.")
    p.add_argument("--checkpoints_dir", default="checkpoints",
                   help="Root folder holding <task>/teacher and <task>/student.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"],
                   help="Force a device. Default: cuda if available, else cpu.")
    p.add_argument("--latency_batches", type=int, default=50,
                   help="Number of timed batches for the speed benchmark.")
    return p.parse_args()


def pick_device(arg):
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_latency(model, sample_batch, device, n_batches: int, n_warmup: int = 5) -> float:
    """Return mean seconds per batch over `n_batches` timed forward passes."""
    model.eval()
    batch = {k: v.to(device) for k, v in sample_batch.items() if k != "labels"}

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(n_warmup):          # warm up kernels / caches
            model(**batch)
        sync()
        start = time.perf_counter()
        for _ in range(n_batches):
            model(**batch)
        sync()
        elapsed = time.perf_counter() - start
    return elapsed / n_batches


def load_model(model_dir: str, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    return model


def evaluate_task(task: str, ckpt_root: str, batch_size: int, device, latency_batches: int) -> dict:
    task_cfg = TASK_CONFIG[task]
    metric_key = task_cfg["metric"]  # "accuracy" | "f1" | "mcc" | "pearson"
    # Match the training-time sequence length (64 for single-sentence, 128 for pairs).
    max_seq = 64 if task_cfg["sentence_keys"][1] is None else 128

    teacher_dir = os.path.join(ckpt_root, task, "teacher")
    student_dir = os.path.join(ckpt_root, task, "student")
    for d in (teacher_dir, student_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"Missing checkpoint: {d}\n"
                f"Unzip the shared checkpoints so that '{ckpt_root}/{task}/teacher' and "
                f"'{ckpt_root}/{task}/student' exist."
            )

    print(f"\n{'='*64}\nTask: {task.upper()}   (metric: {metric_key}, device: {device})\n{'='*64}")

    # Dev set
    raw = load_dataset("nyu-mll/glue", task)
    tokenizer = AutoTokenizer.from_pretrained(student_dir)
    val_ds = preprocess_dataset(raw["validation"], tokenizer, task_cfg, max_seq)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    sample_batch = next(iter(val_loader))

    # Teacher
    teacher = load_model(teacher_dir, device)
    teacher_metrics = evaluate_model(teacher, val_loader, device, task_cfg)
    teacher_params = count_params(teacher)
    teacher_lat = measure_latency(teacher, sample_batch, device, latency_batches)

    # Student
    student = load_model(student_dir, device)
    student_metrics = evaluate_model(student, val_loader, device, task_cfg)
    student_params = count_params(student)
    student_lat = measure_latency(student, sample_batch, device, latency_batches)

    # Derived numbers 
    t_score = teacher_metrics[metric_key]
    s_score = student_metrics[metric_key]
    retention = (s_score / t_score * 100) if t_score else float("nan")
    speedup = teacher_lat / student_lat if student_lat else float("nan")
    bs = sample_batch["input_ids"].shape[0]

    result = {
        "task": task,
        "metric": metric_key,
        "device": str(device),
        "teacher": {
            metric_key: t_score,
            "params": teacher_params,
            "latency_ms_per_batch": teacher_lat * 1000,
            "throughput_examples_per_s": bs / teacher_lat,
        },
        "student": {
            metric_key: s_score,
            "params": student_params,
            "latency_ms_per_batch": student_lat * 1000,
            "throughput_examples_per_s": bs / student_lat,
        },
        "retention_pct": retention,
        "param_ratio_pct": student_params / teacher_params * 100,
        "speedup": speedup,
    }

    # ── Pretty print ──────────────────────────────────────────────────────
    print(f"\n{'':14}{'Teacher':>14}{'Student':>14}")
    print(f"{metric_key:<14}{t_score:>14.4f}{s_score:>14.4f}")
    print(f"{'params':<14}{teacher_params/1e6:>12.1f}M{student_params/1e6:>12.1f}M")
    print(f"{'ms/batch':<14}{teacher_lat*1000:>14.1f}{student_lat*1000:>14.1f}")
    print(f"\n  Retention:       {retention:6.1f}%   (student {metric_key} / teacher {metric_key})")
    print(f"  Param ratio:     {result['param_ratio_pct']:6.1f}%   (student / teacher)")
    print(f"  Speedup:         {speedup:6.2f}x   (teacher latency / student latency)")
    return result


def main():
    args = parse_args()
    tasks = args.task or ["sst2", "mrpc"]
    device = pick_device(args.device)

    all_results = {}
    for task in tasks:
        res = evaluate_task(task, args.checkpoints_dir, args.batch_size, device, args.latency_batches)
        all_results[task] = res
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{task}.json")
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n  Saved -> {out_path}")

    # ── Combined summary table ────────────────────────────────────────────
    print(f"\n{'='*64}\nSUMMARY\n{'='*64}")
    print(f"{'task':<8}{'metric':<10}{'teacher':>9}{'student':>9}{'retain%':>9}{'speedup':>9}")
    for task, r in all_results.items():
        print(f"{task:<8}{r['metric']:<10}"
              f"{r['teacher'][r['metric']]:>9.4f}{r['student'][r['metric']]:>9.4f}"
              f"{r['retention_pct']:>8.1f}%{r['speedup']:>8.2f}x")


if __name__ == "__main__":
    main()
