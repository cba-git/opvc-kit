"""opvc.step3_losses

Training losses for Step3 (SCD + ATC + DAC + QPL).

This file intentionally keeps *training-only* objectives separated from the
inference entry in :mod:`opvc.step3` so that Step3 inference contracts remain
stable and clean.

Implemented (aligned with "missing items" checklist):
- SCD decouple loss (content/style covariance Frobenius norm)
- DAC multi-label BCE loss
- Prototype constraint loss (optional; uses Step3Core.prototypes)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def scd_decouple_loss(z_content: Tensor, z_style: Tensor, eps: float = 1e-12) -> Tensor:
    """Decouple loss encouraging content/style independence.

    We compute the batch cross-covariance between centered z_content and z_style
    and penalize its Frobenius norm.

    Args:
        z_content: [B,ds] or [ds]
        z_style:   [B,ds] or [ds]
    """
    if z_content.ndim == 1:
        zc = z_content.view(1, -1)
    else:
        zc = z_content
    if z_style.ndim == 1:
        zs = z_style.view(1, -1)
    else:
        zs = z_style
    # center
    zc = zc - zc.mean(dim=0, keepdim=True)
    zs = zs - zs.mean(dim=0, keepdim=True)
    B = float(max(zc.shape[0], 1))
    cov = (zc.T @ zs) / (B + eps)  # [ds,ds]
    return (cov ** 2).sum()  # Frobenius norm squared


def dac_multilabel_bce_loss(logits: Tensor, y: Tensor) -> Tensor:
    """Multi-label BCE loss for DAC head.

    Args:
        logits: [B,Ka] (pre-sigmoid)
        y:      [B,Ka] in {0,1}
    """
    return F.binary_cross_entropy_with_logits(logits, y)


def prototype_constraint_loss(
    z_style: Tensor,
    y: Tensor,
    prototypes: Tensor,
    pos_weight: float = 1.0,
    eps: float = 1e-12,
) -> Tensor:
    """Prototype constraint for multi-label setting.

    For each positive label, encourage z_style to be close to its prototype in
    cosine similarity space.

    Args:
        z_style:     [B,ds]
        y:          [B,Ka] in {0,1}
        prototypes:  [Ka,ds]
    """
    if z_style.ndim == 1:
        zs = z_style.view(1, -1)
    else:
        zs = z_style
    if y.ndim == 1:
        yy = y.view(1, -1)
    else:
        yy = y

    zs_n = F.normalize(zs, dim=-1)
    p_n = F.normalize(prototypes, dim=-1)
    # cosine sims: [B,Ka]
    sims = zs_n @ p_n.T

    pos = (yy > 0.5).to(sims.dtype)
    denom = pos.sum().clamp_min(1.0)
    # loss = mean(1 - sim) over positives
    loss = ((1.0 - sims) * pos).sum() / denom
    return loss * float(pos_weight)
