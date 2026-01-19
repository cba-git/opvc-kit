"""opvc.step2_losses

Losses and signals for Step2 aligned to the method final:

- ASD (structure distillation)
- AT-InfoNCE (dynamic-temperature contrastive distillation)
- Utility / Risk signals for temperature & DP adaptation

This file is intentionally math-focused and side-effect free.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def asd_loss(u_s: Tensor, u_t: Tensor, lambda_sample: float = 1.0, lambda_mean: float = 1.0) -> Tensor:
    """ASD: sample-level alignment + mean alignment."""
    if u_s.shape != u_t.shape:
        raise ValueError(f"u_s/u_t shape mismatch: {tuple(u_s.shape)} vs {tuple(u_t.shape)}")
    l_sample = F.mse_loss(u_s, u_t, reduction="mean")
    l_mean = F.mse_loss(u_s.mean(dim=0), u_t.mean(dim=0), reduction="mean")
    return lambda_sample * l_sample + lambda_mean * l_mean


def info_nce_distill(u_s: Tensor, u_t: Tensor, tau: Tensor) -> Tensor:
    """AT-InfoNCE: positive pairs are diagonal in similarity matrix."""
    u_s = F.normalize(u_s, dim=-1)
    u_t = F.normalize(u_t, dim=-1)
    logits = u_s @ u_t.t()  # [B,B]
    if tau.ndim == 0:
        logits = logits / tau.clamp_min(1e-6)
    else:
        logits = logits / tau.view(-1, 1).clamp_min(1e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def entropy_confidence(p: Tensor, eps: float = 1e-8) -> Tensor:
    """Entropy-based concentration confidence in [0,1]."""
    if p.ndim == 1:
        pb = p.view(1, -1)
    elif p.ndim == 2:
        pb = p
    else:
        raise ValueError(f"p must be [K] or [B,K], got {tuple(p.shape)}")
    pb = pb.clamp_min(eps)
    pb = pb / pb.sum(dim=-1, keepdim=True).clamp_min(eps)
    H = -(pb * pb.log()).sum(dim=-1)  # [B]
    K = pb.shape[-1]
    Hn = H / torch.log(torch.tensor(float(K), device=pb.device)).clamp_min(eps)
    return (1.0 - Hn).clamp(0.0, 1.0)


def alpha_confidence(alpha: Tensor) -> Tensor:
    return entropy_confidence(alpha)


def pi_uncertainty(pi: Tensor) -> Tensor:
    """Normalized entropy uncertainty in [0,1] (higher => more uncertain)."""
    return (1.0 - entropy_confidence(pi)).clamp(0.0, 1.0)


def utility_signal(distill_err: Tensor, conf: Tensor) -> Tensor:
    """nu(x) = distill_err * confidence."""
    return (distill_err * conf).clamp_min(0.0)


def mi_risk_estimate_from_logits(logits: Tensor) -> Tensor:
    """A practical MI-risk proxy from prediction confidence (entropy).

    When explicit membership inference labels are unavailable, a standard proxy is to
    treat overly-confident outputs as higher privacy risk.
    Returns a value in [0,1] with higher => higher risk.
    """
    prob = torch.sigmoid(logits)
    eps = 1e-8
    H = -(prob * (prob + eps).log() + (1 - prob) * (1 - prob + eps).log())  # [B,Ka]
    Hn = H / torch.log(torch.tensor(2.0, device=prob.device)).clamp_min(eps)
    risk = (1.0 - Hn).mean(dim=-1)
    return risk.clamp(0.0, 1.0)


def risk_signal(sensitivity: Tensor, mi_est: Optional[Tensor] = None) -> Tensor:
    """P(x) = sensitivity + mi_est (if provided)."""
    r = sensitivity
    if mi_est is not None:
        r = r + mi_est
    return r.clamp_min(0.0)


def dynamic_temperature(
    util: Tensor,
    risk: Tensor,
    round_idx: int,
    total_rounds: int,
    tau_min: float,
    tau_max: float,
    kappa_u: float,
    kappa_p: float,
) -> Tensor:
    """Dynamic temperature tau(x) (direction aligned to method final).

    Higher utility -> smaller tau (stronger alignment).
    Higher risk -> larger tau (more conservative).

    We use:
        tau = tau_min + (tau_max - tau_min) * sigmoid(kappa_p*risk - kappa_u*util + sched(round))
    """
    util = util.detach()
    risk = risk.detach()
    # schedule: later rounds slightly reduce temperature
    if total_rounds <= 1:
        sched = 0.0
    else:
        t = float(round_idx) / float(max(total_rounds - 1, 1))
        sched = -0.2 * t  # later => smaller tau
    score = (kappa_p * risk) - (kappa_u * util) + float(sched)
    g = torch.sigmoid(score)
    tau = tau_min + (tau_max - tau_min) * g
    return tau.clamp(tau_min, tau_max)


def dp_sigma(
    base_sigma: float,
    util: Tensor,
    risk: Tensor,
    sigma_min: float,
    sigma_max: float,
    k_u: float = 0.3,
    k_r: float = 0.7,
) -> Tensor:
    """Risk high => sigma larger; utility high => sigma smaller."""
    mult = 1.0 + (k_r * risk) - (k_u * util)
    sigma = base_sigma * mult
    return sigma.clamp(sigma_min, sigma_max)
