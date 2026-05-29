# TinyBERT Explained Simply

> **Paper:** TinyBERT: Distilling BERT for Natural Language Understanding  
> **Authors:** Jiao et al. (Huawei Noah's Ark Lab), 2020  
> **arXiv:** 1909.10351

---

## The Problem: BERT is too big

BERT is an extremely powerful language model. It can understand text well enough to answer questions, classify sentiment, detect paraphrases, and more. But it has **110 million parameters** — the trained model file is ~440 MB and doing a single prediction takes significant compute time.

This makes BERT impractical on:
- Mobile phones
- Edge devices (IoT, embedded systems)
- Applications that need to respond in milliseconds
- Services that need to run inference at very high scale cheaply

The goal of TinyBERT is to build a model that is **7.5× smaller** and **9.4× faster** than BERT, while keeping **96.8%+ of BERT's accuracy**.

---

## The Core Idea: Knowledge Distillation

Imagine a student learning from a teacher. Instead of the student re-learning everything from scratch (reading raw textbooks = raw data), the student directly learns from the teacher's explanations — their reasoning process, not just their final answers.

**Knowledge Distillation (KD)** works the same way:

```
Teacher (large BERT, 110M params)
    ↓  "teaches"
Student (TinyBERT, 14.5M params)
```

The student doesn't just try to get the same final answers as the teacher — it tries to **mimic the teacher's internal thinking process** at every layer. This is much richer information than just matching predictions.

The key insight: if the student learns to reproduce the teacher's internal representations, it captures the linguistic knowledge that BERT spent billions of words learning — without needing to re-learn it from scratch.

---

## What Makes TinyBERT Different

Most prior distillation work (e.g. DistilBERT) only copied knowledge from the **final output** of the teacher. TinyBERT goes deeper — it copies knowledge from **every intermediate Transformer layer**.

Concretely, TinyBERT introduces losses at three levels:

### Level 1 — Attention matrices (inside each Transformer layer)

Each BERT layer computes attention weights: for each word, how much should it "attend to" every other word in the sentence?

```
"The cat sat on the mat"
      ↕ ↕ ↕ ↕ ↕ ↕
attention weight matrix (6×6 for 6 words)
```

These attention patterns encode real linguistic knowledge — BERT's heads learn to track subject-verb agreement, coreference, syntactic dependencies, etc.

**TinyBERT loss:** `MSE(student attention matrix, teacher attention matrix)` — make the student produce the same attention patterns as the teacher.

> The paper uses the **raw** (pre-softmax) attention scores, not the softmax probabilities, because this converges faster and performs better empirically.

### Level 2 — Hidden states (the vector for each word after each layer)

After each Transformer layer, every word is represented as a vector. These vectors encode the contextual meaning of the word given the whole sentence.

The student has smaller vectors (312 dimensions) than the teacher (768 dimensions), so a small learnable linear projection W maps student → teacher space before comparing.

**TinyBERT loss:** `MSE(student hidden states × W, teacher hidden states)`

### Level 3 — Embeddings (the initial word representations, before any Transformer layer)

Before any processing, each word is converted to a vector via the embedding table. The same projection trick is applied here.

**TinyBERT loss:** `MSE(student embeddings × W, teacher embeddings)`

### Level 4 — Prediction layer (the final classification output)

The soft cross-entropy between teacher and student outputs. Rather than training against the hard ground-truth label (0 or 1), the student trains against the teacher's probability distribution.

For example if the teacher says "80% positive, 20% negative" for a movie review, the student tries to match that distribution — not just "positive". The distribution carries calibration information that the hard label discards.

**TinyBERT loss:** `cross_entropy(student logits / T, teacher logits / T)` where T is temperature (= 1 in the paper).

### Combined per-layer loss (Equation 11 in the paper)

```
Layer 0 (embedding):        L_embd
Layers 1–M (Transformer):   L_attn + L_hidn
Layer M+1 (prediction):     L_pred
```

---

## The Student Architecture: TinyBERT4

| | BERT-base (teacher) | TinyBERT4 (student) |
|---|---|---|
| Transformer layers | 12 | 4 |
| Hidden size | 768 | 312 |
| FFN size | 3072 | 1200 |
| Attention heads | 12 | 12 |
| Parameters | 109M | 14.5M |
| Inference speedup | 1× | **9.4×** |

The number of attention heads stays the same (12) even though the model is much smaller — this is intentional so the attention matrices can be directly compared for distillation.

### Layer Mapping: which teacher layers does the student learn from?

The student has 4 layers, the teacher has 12. The paper uses a **uniform mapping**: student layer m learns from teacher layer 3m.

```
Student layer 1  →  Teacher layer 3
Student layer 2  →  Teacher layer 6
Student layer 3  →  Teacher layer 9
Student layer 4  →  Teacher layer 12
```

This spreads the student's learning evenly across all levels of the teacher's hierarchy — bottom layers (syntax), middle layers, and top layers (semantics). The paper compares this against "top-only" and "bottom-only" mappings and shows uniform works best.

---

## The Two-Stage Learning Framework

BERT itself is trained in two stages: pre-training on large unlabeled text, then fine-tuning on a specific task. TinyBERT mirrors this.

```
Stage 1: General Distillation
  Teacher = pre-trained BERT (not fine-tuned)
  Data    = English Wikipedia (2.5 billion words)
  Output  = General TinyBERT (good general language understanding)

                      ↓

Stage 2: Task-specific Distillation
  Teacher = BERT fine-tuned on the task (e.g. SST-2 sentiment)
  Data    = Task training set + augmented versions
  Output  = Fine-tuned TinyBERT (ready to use for the task)
```

**Why both stages?** General distillation gives TinyBERT broad language knowledge (grammar, facts, common sense). Task-specific distillation then focuses it on the particular task. Removing either stage hurts performance significantly — see ablation study Table 2 in the paper.

**In our reproduction:** We skip Stage 1 (training on Wikipedia for 3 epochs would take days) and instead use the publicly released General TinyBERT checkpoint from the authors, which is exactly how the original code is structured for downstream use.

---

## Data Augmentation

Before task-specific distillation, the training set is expanded using word substitution. The goal is to give the student more diverse examples to learn from.

**Algorithm (for each training sentence, repeat Na=20 times):**

For each word in the sentence:
1. **Single-piece word** (e.g. "cat"): mask it, ask BERT to predict the top K=15 replacements. With probability pt=0.4, swap the word with a random one from those 15.
2. **Multi-piece word** (e.g. "running" tokenized as "run" + "##ning"): look up the K=15 most similar words in GloVe word vectors. With probability pt=0.4, swap.

Example (SST-2 sentiment):
```
Original:  "The film is a total bore"
Augmented: "The movie is a complete bore"   (film→movie, total→complete)
Augmented: "The film was a total bore"      (is→was)
```

The label is preserved — all augmented versions have the same sentiment as the original.

This is important because the student is a small model that can easily overfit on small datasets. More varied training data improves generalisation.

---

## Results

### Main result (GLUE test set, Table 1)

| Model | Params | Speedup | SST-2 | MRPC | Average |
|---|---|---|---|---|---|
| BERT-base (teacher) | 109M | 1× | 93.4 | 87.5 | 79.5 |
| TinyBERT4 (ours) | **14.5M** | **9.4×** | 92.6 | 86.4 | **77.0** |
| DistilBERT4 (baseline) | 52.2M | 3.0× | 91.4 | 82.4 | 71.9 |
| BERT4-PKD (baseline) | 52.2M | 3.0× | 89.4 | 82.6 | 72.6 |

TinyBERT4 beats the 4-layer baselines by 4.4+ points on average, **while being 28% the size** of DistilBERT4 and PKD. It also matches the much larger 24-layer MobileBERT at 38.7% of the FLOPs.

### Ablation: what matters most?

From Table 2 (removing one component at a time):

| Removed | Average drop |
|---|---|
| None (full TinyBERT4) | 75.6 |
| Remove general distillation | 72.5 (−3.1) |
| Remove task-specific distillation | 68.5 (−7.1) |
| Remove data augmentation | 68.4 (−7.2) |

All three components are important. Task-specific distillation and data augmentation matter most.

From Table 3 (removing one loss type at a time):

| Removed loss | Average drop |
|---|---|
| None (full) | 75.6 |
| No Transformer-layer distillation | 56.3 (−19.3) |
| No attention distillation | 71.0 (−4.6) |
| No hidden state distillation | 72.9 (−2.7) |
| No embedding distillation | 74.1 (−1.5) |
| No prediction distillation | 73.5 (−2.1) |

The **Transformer-layer distillation (attention + hidden)** is by far the most important. Without it, performance collapses. This validates the core hypothesis: teaching the student to mimic the teacher's internal attention patterns is the key to effective BERT compression.

---

## Summary

| Question | Answer |
|---|---|
| What is the problem? | BERT is too large and slow for deployment |
| What is the approach? | Knowledge distillation — small student learns from large teacher |
| What is novel? | Distilling from every Transformer layer (attention + hidden states), not just the output |
| What is the two-stage framework? | General distillation (Wikipedia) → task-specific distillation (task data + augmentation) |
| What are the results? | 7.5× smaller, 9.4× faster, retains 96.8%+ of BERT's performance |
| Why does it work? | Attention matrices encode rich linguistic knowledge; copying them transfers this knowledge efficiently |
