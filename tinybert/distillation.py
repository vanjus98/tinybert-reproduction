"""
Task-specific distillation for TinyBERT (Section 3.2, arXiv:1909.10351).

Two phases per the paper:
  Phase 1 – intermediate layer distillation (Attn + Hidn + Embd) on augmented data
             for `intermediate_epochs` epochs.
  Phase 2 – prediction layer distillation (Pred) on augmented data
             for `pred_epochs` epochs.

Layer mapping: g(m) = 3*m  (uniform strategy, TinyBERT4 → BERT-base-12L).
Student layer m learns from teacher layer g(m).
"""

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

from .losses import attention_loss, hidden_loss, embedding_loss, prediction_loss
from .glue_utils import TASK_CONFIG, compute_metrics


def _register_raw_attention_hooks(model) -> list:
    """
    Stash pre-softmax attention scores on every BertSelfAttention module.
    Paper §3.1 / Eq. 7 uses raw Q·Kᵀ/√d, not the softmaxed probabilities
    that HuggingFace returns via `output_attentions=True`.

    The hook re-derives Q,K from the layer's `query`/`key` projections and
    writes the scores to `module._raw_attn` as a side effect. Returns the
    handles so callers can remove them if needed.

    Note: transformers >=4.49 removed BertSelfAttention.transpose_for_scores
    when BERT was refactored onto the SDPA attention dispatcher. We reshape
    Q/K manually using `num_attention_heads` and `attention_head_size`, which
    are still set in __init__.
    """
    handles = []
    encoder = model.bert.encoder if hasattr(model, "bert") else model.base_model.encoder

    def make_hook(self_attn):
        head_dim = self_attn.attention_head_size
        num_heads = self_attn.num_attention_heads

        def hook(module, inputs, output):
            hidden = inputs[0]
            new_shape = hidden.size()[:-1] + (num_heads, head_dim)
            q = module.query(hidden).view(new_shape).permute(0, 2, 1, 3)
            k = module.key(hidden).view(new_shape).permute(0, 2, 1, 3)
            module._raw_attn = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
        return hook

    for layer in encoder.layer:
        self_attn = layer.attention.self
        handles.append(self_attn.register_forward_hook(make_hook(self_attn)))
    return handles


def _collect_raw_attn(model) -> list[torch.Tensor]:
    encoder = model.bert.encoder if hasattr(model, "bert") else model.base_model.encoder
    return [layer.attention.self._raw_attn for layer in encoder.layer]


# TinyBERT4 architecture from Table 1 of the paper
TINYBERT4_CONFIG = dict(
    num_hidden_layers=4,
    hidden_size=312,
    intermediate_size=1200,
    num_attention_heads=12,
    max_position_embeddings=512,
)

# Layer mapping g(m) = 3*m  (student layer m → teacher layer 3m)
def uniform_mapping(student_layer: int) -> int:
    return 3 * student_layer


@dataclass
class DistillConfig:
    task: str = "sst2"
    teacher_dir: str = "checkpoints/teacher"
    student_init: str = "huawei-noah/TinyBERT_General_4L_312D"  # pre-trained general TinyBERT
    output_dir: str = "checkpoints/student"
    max_seq_len: int = 64          # paper uses 64 for single-sentence tasks
    batch_size: int = 32
    intermediate_lr: float = 5e-5
    pred_lr: float = 3e-5
    intermediate_epochs: int = 20  # paper: 20 epochs (10 for large tasks)
    pred_epochs: int = 3
    temperature: float = 1.0
    seed: int = 42


class ProjectionLayer(nn.Module):
    """Linear W_h or W_e that projects student d' → teacher d."""
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.linear = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class TinyBERTDistiller:
    def __init__(self, cfg: DistillConfig):
        self.cfg = cfg
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Device: {self.device}")
        torch.manual_seed(cfg.seed)

        task_cfg = TASK_CONFIG[cfg.task]
        self.num_labels = task_cfg["num_labels"]
        self.task_cfg = task_cfg

        # ── Teacher ──────────────────────────────────────────────────────────
        self.teacher = AutoModelForSequenceClassification.from_pretrained(cfg.teacher_dir)
        self.teacher.eval()
        self.teacher.to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.teacher_dim = self.teacher.config.hidden_size  # 768 for BERT-base

        # ── Student ───────────────────────────────────────────────────────────
        self.student = AutoModelForSequenceClassification.from_pretrained(
            cfg.student_init, num_labels=self.num_labels, ignore_mismatched_sizes=True
        )
        self.student.to(self.device)
        self.student_dim = self.student.config.hidden_size  # 312 for TinyBERT4

        # Hook every self-attention block to capture raw (pre-softmax) scores.
        _register_raw_attention_hooks(self.teacher)
        _register_raw_attention_hooks(self.student)

        # ── Projection matrices W_h and W_e ──────────────────────────────────
        num_student_layers = self.student.config.num_hidden_layers  # 4
        # one projection per student Transformer layer + one for embedding
        self.projections = nn.ModuleList([
            ProjectionLayer(self.student_dim, self.teacher_dim)
            for _ in range(num_student_layers + 1)  # +1 for embedding layer
        ])
        self.projections.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_dir)

    # ─────────────────────────────────────────────────────────────────────────
    def _teacher_outputs(self, batch: dict) -> dict:
        """Run teacher. Raw attention scores are captured via forward hooks; we
        only need hidden states from the standard output."""
        with torch.no_grad():
            out = self.teacher(**batch, output_hidden_states=True)
        return out

    def _student_outputs(self, batch: dict) -> dict:
        batch_no_labels = {k: v for k, v in batch.items() if k != "labels"}
        return self.student(**batch_no_labels, output_hidden_states=True)

    # ─────────────────────────────────────────────────────────────────────────
    def _intermediate_loss(self, batch: dict) -> torch.Tensor:
        """
        Eq. 11 (m > 0 case): L_hidn + L_attn for each mapped layer pair,
        plus embedding loss (m=0 case): L_embd.

        Uses raw (pre-softmax) attention scores stashed by the forward hooks,
        and masks padded positions out of every MSE.
        """
        t_out = self._teacher_outputs(batch)
        s_out = self._student_outputs(batch)

        attn_mask = batch.get("attention_mask")  # (batch, seq)

        # Hooks have populated `_raw_attn` on every self-attention layer.
        s_raw_attn = _collect_raw_attn(self.student)  # list of (b, h, s, s)
        t_raw_attn = _collect_raw_attn(self.teacher)

        num_student_layers = self.student.config.num_hidden_layers  # 4

        total = torch.tensor(0.0, device=self.device)

        # Embedding layer loss (index 0 in hidden_states is embedding output)
        s_emb = s_out.hidden_states[0]   # (batch, seq, d')
        t_emb = t_out.hidden_states[0]   # (batch, seq, d)
        total = total + embedding_loss(s_emb, t_emb, self.projections[0], attn_mask)

        # Transformer layer losses
        for m in range(1, num_student_layers + 1):
            g_m = uniform_mapping(m)  # teacher layer index

            # ── Attention loss (raw scores, paper §3.1) ─────────────────────
            total = total + attention_loss(
                s_raw_attn[m - 1], t_raw_attn[g_m - 1], attn_mask
            )

            # ── Hidden state loss ───────────────────────────────────────────
            s_hidn = s_out.hidden_states[m]    # (batch, seq, d')
            t_hidn = t_out.hidden_states[g_m]  # (batch, seq, d)
            total = total + hidden_loss(s_hidn, t_hidn, self.projections[m], attn_mask)

        return total

    def _pred_loss(self, batch: dict) -> torch.Tensor:
        t_out = self._teacher_outputs(batch)
        batch_no_labels = {k: v for k, v in batch.items() if k != "labels"}
        s_out = self.student(**batch_no_labels)
        return prediction_loss(s_out.logits, t_out.logits, self.cfg.temperature)

    # ─────────────────────────────────────────────────────────────────────────
    def _train_phase(
        self,
        loader: DataLoader,
        optimizer,
        scheduler,
        loss_fn,
        desc: str,
        num_epochs: int,
    ):
        for epoch in range(num_epochs):
            self.student.train()
            total_loss = 0.0
            pbar = tqdm(loader, desc=f"[{desc}] epoch {epoch+1}/{num_epochs}", leave=False)
            for step, batch in enumerate(pbar, start=1):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = loss_fn(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.student.parameters()) + list(self.projections.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{total_loss/step:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            print(f"  [{desc}] Epoch {epoch+1}/{num_epochs}  loss={total_loss/len(loader):.4f}", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    def distill(self, train_dataset, val_dataset):
        cfg = self.cfg

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

        # ── Phase 1: Intermediate layer distillation ─────────────────────────
        print("\n=== Phase 1: Intermediate layer distillation ===")
        params1 = list(self.student.parameters()) + list(self.projections.parameters())
        opt1 = AdamW(params1, lr=cfg.intermediate_lr)
        steps1 = len(train_loader) * cfg.intermediate_epochs
        sch1 = get_linear_schedule_with_warmup(opt1, int(0.1 * steps1), steps1)
        self._train_phase(train_loader, opt1, sch1, self._intermediate_loss,
                          "Intermediate", cfg.intermediate_epochs)

        # ── Phase 2: Prediction layer distillation ───────────────────────────
        print("\n=== Phase 2: Prediction layer distillation ===")
        opt2 = AdamW(self.student.parameters(), lr=cfg.pred_lr)
        steps2 = len(train_loader) * cfg.pred_epochs
        sch2 = get_linear_schedule_with_warmup(opt2, int(0.1 * steps2), steps2)
        self._train_phase(train_loader, opt2, sch2, self._pred_loss,
                          "Prediction", cfg.pred_epochs)

        # ── Validation ────────────────────────────────────────────────────────
        metrics = self._evaluate(val_loader)
        print(f"\nFinal validation metrics: {metrics}")

        os.makedirs(cfg.output_dir, exist_ok=True)
        self.student.save_pretrained(cfg.output_dir)
        self.tokenizer.save_pretrained(cfg.output_dir)
        print(f"Student saved to {cfg.output_dir}")
        return metrics

    def _evaluate(self, loader: DataLoader) -> dict:
        self.student.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch["labels"]
                out = self.student(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                )
                if self.task_cfg["is_regression"]:
                    preds = out.logits.squeeze(-1)
                else:
                    preds = out.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        return compute_metrics(self.task_cfg, all_preds, all_labels)
