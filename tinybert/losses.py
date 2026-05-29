"""
Distillation loss functions from TinyBERT (arXiv:1909.10351).

Four loss types:
  - Lattn : MSE between raw (pre-softmax) attention matrices, averaged over heads
  - Lhidn : MSE between hidden states after linear projection W_h
  - Lembd : MSE between embedding layers after linear projection W_e
  - Lpred : soft cross-entropy between teacher/student logits (temperature t)

All intermediate losses accept an optional `attention_mask` (batch, seq) so that
padded positions are excluded from the MSE — otherwise the loss is dominated by
matching the model's behaviour on padding tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE reduced only over positions where `mask` is 1. `mask` broadcasts to `pred`."""
    sq = (pred - target).pow(2) * mask
    denom = mask.sum().clamp(min=1.0)
    # average per element of pred along the dims that mask does not span
    extra = pred.numel() / mask.numel()  # e.g. seq for (batch,seq,seq); d for (batch,seq,d)
    return sq.sum() / (denom * extra)


def attention_loss(
    student_attn: torch.Tensor,
    teacher_attn: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Eq. 7: L_attn = (1/h) * sum_i MSE(A_i^S, A_i^T)

    student_attn / teacher_attn: (batch, h, seq, seq) — raw (pre-softmax) scores.
    attention_mask: (batch, seq); rows/cols belonging to padding are zeroed out.
    """
    if attention_mask is None:
        return F.mse_loss(student_attn, teacher_attn)
    # (batch, 1, seq, 1) * (batch, 1, 1, seq) -> (batch, 1, seq, seq)
    m = attention_mask.to(student_attn.dtype)
    pair_mask = m.unsqueeze(1).unsqueeze(-1) * m.unsqueeze(1).unsqueeze(-2)
    return _masked_mse(student_attn, teacher_attn, pair_mask)


def hidden_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    linear: nn.Linear,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Eq. 8: L_hidn = MSE(H^S * W_h, H^T)

    student_hidden: (batch, seq, d')
    teacher_hidden: (batch, seq, d)
    linear: projects d' -> d
    """
    projected = linear(student_hidden)
    if attention_mask is None:
        return F.mse_loss(projected, teacher_hidden)
    m = attention_mask.to(projected.dtype).unsqueeze(-1)  # (batch, seq, 1)
    return _masked_mse(projected, teacher_hidden, m)


def embedding_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    linear: nn.Linear,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Eq. 9: L_embd = MSE(E^S * W_e, E^T)
    """
    return hidden_loss(student_emb, teacher_emb, linear, attention_mask)


def prediction_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                    temperature: float = 1.0) -> torch.Tensor:
    """
    Eq. 10: L_pred = CE(softmax(z^T / t), softmax(z^S / t))

    Soft cross-entropy over the full teacher distribution. We must NOT collapse
    the teacher to argmax: the whole point of distillation is to transfer the
    soft probability mass (dark knowledge) that hard labels discard.
    """
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return -(t_probs * s_log_probs).sum(dim=-1).mean() * (temperature ** 2)
