# TinyBERT Reproduction

Reproduction of **TinyBERT: Distilling BERT for Natural Language Understanding**  
(Jiao et al., 2020 — [arXiv:1909.10351](https://arxiv.org/abs/1909.10351))

Group project for the *Deep Learning for NLP* course.

---

## What we reproduce

| Component | Paper Section | Status |
|-----------|--------------|--------|
| Transformer distillation losses (Attn, Hidn, Embd, Pred) | §3.1 | ✅ |
| Two-phase task-specific distillation | §3.2 | ✅ |
| Data augmentation (BERT MLM + GloVe) | §3.2, Alg. 1 | ✅ |
| Uniform layer mapping g(m) = 3m | §3.1 | ✅ |
| GLUE evaluation (SST-2, MRPC) | §4 | ✅ |

We skip **general distillation** (requires training on English Wikipedia, ~2.5B words) and instead initialise the student from the publicly released `huawei-noah/TinyBERT_General_4L_312D` checkpoint, exactly as the authors suggest for downstream fine-tuning.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running experiments

### Full pipeline on SST-2 (sentiment classification)
```bash
python run_experiment.py --task sst2
```

### Fast smoke-test (few epochs, small augmentation)
```bash
python run_experiment.py --task sst2 --fast
```

### Skip augmentation (original training data only)
```bash
python run_experiment.py --task sst2 --skip_augmentation
```

### MRPC (paraphrase detection)
```bash
python run_experiment.py --task mrpc
```

### With GloVe for multi-piece word augmentation
```bash
# Download GloVe: https://nlp.stanford.edu/projects/glove/
python run_experiment.py --task sst2 --glove_path glove.6B.100d.txt
```

### Run on a fraction of the training data (compute-limited)
`--subset <fraction>` applies to both the teacher fine-tuning data and the distillation data.
```bash
python run_experiment.py --task sst2 --subset 0.3        # 30% of train set
python run_experiment.py --task sst2 --subset 0.05 --fast  # smoke test
```

### Available tasks
`sst2`, `mrpc`, `qqp`, `qnli`, `rte`, `cola`, `stsb`, `mnli` — defined in [`tinybert/glue_utils.py`](tinybert/glue_utils.py) with the metric for each (accuracy / F1 / Matthews corr. / Pearson).

---

## Output artifacts

Every run writes to `checkpoints/<task>/`:

```
checkpoints/<task>/
  teacher/          — fine-tuned BERT-base teacher (HuggingFace save_pretrained format)
  student/          — distilled TinyBERT₄ student (same format)
  results.json      — { "task": ..., "teacher": {metric: value}, "student": {metric: value} }
```

To re-evaluate a saved student without retraining:
```python
from tinybert.teacher import load_and_evaluate
print(load_and_evaluate("checkpoints/sst2/student", task="sst2"))
```

If a teacher already exists at `checkpoints/<task>/teacher/`, the pipeline detects this and skips teacher fine-tuning — useful when iterating on the distillation phase.

---

## Project structure

```
tinybert/
  losses.py         — Distillation loss functions (Eq. 7–11)
  distillation.py   — Task-specific distillation trainer
  augmentation.py   — Data augmentation (Algorithm 1)
  teacher.py        — BERT-base fine-tuning
  glue_utils.py     — Dataset preprocessing + metrics

run_experiment.py   — End-to-end pipeline
```

---

## Key results (TinyBERT4 vs BERT-base, from paper Table 1)

| Model | SST-2 | MRPC | Params | Speedup |
|-------|-------|------|--------|---------|
| BERT-base (teacher) | 93.4 | 87.5 | 109M | 1× |
| TinyBERT4 | 92.6 | 86.4 | 14.5M | 9.4× |
| Retention | 99.1% | 98.7% | 13.3% | — |

---

## Original paper code
[https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)
