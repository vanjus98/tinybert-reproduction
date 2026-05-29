"""
Full TinyBERT reproduction pipeline.

Steps:
  1. Fine-tune BERT-base teacher on chosen GLUE task.
  2. Run data augmentation on the training set.
  3. Run task-specific distillation (intermediate + prediction phases).
  4. Evaluate teacher and student side-by-side.

Usage:
    python run_experiment.py --task sst2
    python run_experiment.py --task mrpc --skip_teacher   # if teacher already trained
    python run_experiment.py --task sst2 --fast           # fewer epochs for quick testing
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from tinybert.glue_utils import TASK_CONFIG, preprocess_dataset, compute_metrics
from tinybert.teacher import TeacherConfig, train_teacher, evaluate_model
from tinybert.distillation import DistillConfig, TinyBERTDistiller
from tinybert.augmentation import TinyBERTAugmenter

from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="sst2", choices=list(TASK_CONFIG.keys()))
    p.add_argument("--teacher_dir", default=None, help="Path to fine-tuned teacher (skips training)")
    p.add_argument("--student_init", default="huawei-noah/TinyBERT_General_4L_312D",
                   help="HuggingFace model ID or path for student initialisation")
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--glove_path", default=None, help="Optional path to GloVe .txt file")
    p.add_argument("--fast", action="store_true",
                   help="Reduce epochs/augmentation for quick smoke-test")
    p.add_argument("--skip_augmentation", action="store_true",
                   help="Skip data augmentation (use original training data only)")
    p.add_argument("--na", type=int, default=20, help="Augmented samples per example (paper: 20)")
    p.add_argument("--subset", type=float, default=None,
                   help="Use a random fraction of training data, e.g. 0.1 for 10%%. "
                        "Useful to demonstrate the method when compute is limited.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def maybe_subset(dataset, fraction: float | None, seed: int):
    """Return a random subset of a HuggingFace dataset split."""
    if fraction is None or fraction >= 1.0:
        return dataset
    n = max(1, int(len(dataset) * fraction))
    return dataset.shuffle(seed=seed).select(range(n))


def main():
    args = parse_args()
    task = args.task
    task_cfg = TASK_CONFIG[task]
    teacher_dir = args.teacher_dir or os.path.join(args.output_dir, task, "teacher")
    student_dir = os.path.join(args.output_dir, task, "student")

    # ── 1. Teacher fine-tuning ─────────────────────────────────────────────
    if args.teacher_dir is None and not os.path.isdir(teacher_dir):
        print(f"\n{'='*60}")
        print(f"Step 1: Fine-tuning BERT-base teacher on {task.upper()}")
        print(f"{'='*60}")
        t_cfg = TeacherConfig(
            task=task,
            output_dir=teacher_dir,
            num_epochs=2 if args.fast else 3,
            seed=args.seed,
        )
        teacher_dir = train_teacher(t_cfg)
    else:
        print(f"\nStep 1: Using existing teacher at {teacher_dir}")

    # ── 2. Load dataset ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 2: Loading GLUE/{task} dataset")
    print(f"{'='*60}")
    raw = load_dataset("glue", task)
    tokenizer = AutoTokenizer.from_pretrained(teacher_dir)

    max_seq = 64 if task_cfg["sentence_keys"][1] is None else 128
    val_ds = preprocess_dataset(raw["validation"], tokenizer, task_cfg, max_seq)

    # Apply subset if requested
    train_split = maybe_subset(raw["train"], args.subset, args.seed)
    if args.subset:
        print(f"  Using {len(train_split)}/{len(raw['train'])} training examples "
              f"({args.subset*100:.0f}% subset)")

    # ── 3. Data augmentation ───────────────────────────────────────────────
    if args.skip_augmentation:
        print("\nStep 3: Skipping augmentation — using original training data")
        train_ds = preprocess_dataset(train_split, tokenizer, task_cfg, max_seq)
    else:
        print(f"\n{'='*60}")
        print(f"Step 3: Data augmentation (Na={args.na if not args.fast else 5})")
        print(f"{'='*60}")
        na = 5 if args.fast else args.na
        aug_device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        augmenter = TinyBERTAugmenter(
            bert_model_name="bert-base-uncased",
            glove_path=args.glove_path,
            na=na,
            device=aug_device,
        )

        key1, key2 = task_cfg["sentence_keys"]
        orig_sentences = train_split[key1]
        orig_labels = train_split["label"]

        print(f"  Augmenting {len(orig_sentences)} training examples ...")
        aug_sentences, aug_labels = [], []
        for sent, lbl in zip(orig_sentences, orig_labels):
            aug_sentences.append(sent)
            aug_labels.append(lbl)
            for aug in augmenter.augment(sent):
                aug_sentences.append(aug)
                aug_labels.append(lbl)

        print(f"  Augmented dataset size: {len(aug_sentences)} (was {len(orig_sentences)})")

        if key2 is None:
            from transformers import AutoTokenizer as _T
            enc = tokenizer(
                aug_sentences,
                truncation=True, padding="max_length", max_length=max_seq
            )
        else:
            # For pair tasks we only augment sentence1; sentence2 repeats with same label.
            # IMPORTANT: source sentence2 from `train_split`, not `raw["train"]`, so the
            # ordering matches `orig_sentences` after `--subset` shuffling.
            orig_s2 = train_split[key2]
            aug_s2 = []
            for s2 in orig_s2:
                aug_s2.append(s2)
                for _ in range(na):
                    aug_s2.append(s2)
            enc = tokenizer(
                aug_sentences, aug_s2,
                truncation=True, padding="max_length", max_length=max_seq
            )

        from tinybert.glue_utils import GLUEDataset
        import torch as _torch
        train_ds = GLUEDataset(enc, aug_labels)

    # ── 4. Task-specific distillation ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 4: Task-specific distillation")
    print(f"{'='*60}")
    d_cfg = DistillConfig(
        task=task,
        teacher_dir=teacher_dir,
        student_init=args.student_init,
        output_dir=student_dir,
        max_seq_len=max_seq,
        intermediate_epochs=3 if args.fast else 20,
        pred_epochs=1 if args.fast else 3,
        seed=args.seed,
    )
    distiller = TinyBERTDistiller(d_cfg)
    student_metrics = distiller.distill(train_ds, val_ds)

    # ── 5. Compare teacher vs student ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 5: Teacher vs Student comparison on {task.upper()} validation")
    print(f"{'='*60}")

    from transformers import AutoModelForSequenceClassification
    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    val_loader = DataLoader(val_ds, batch_size=32)
    teacher_metrics = evaluate_model(teacher_model, val_loader, device, task_cfg)

    print(f"\n  Teacher ({teacher_dir}): {teacher_metrics}")
    print(f"  Student ({student_dir}): {student_metrics}")

    # compute retention %
    for key in teacher_metrics:
        if key in student_metrics:
            retention = student_metrics[key] / teacher_metrics[key] * 100
            print(f"  {key} retention: {retention:.1f}%  (paper target: >96.8%)")

    results = {
        "task": task,
        "teacher": teacher_metrics,
        "student": student_metrics,
    }
    results_path = os.path.join(args.output_dir, task, "results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
