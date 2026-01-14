# OPVC Step2 losses & signals
# Align to 方法定稿：ASD + 动态温度 AT-InfoNCE + (utility/risk)->DP/temperature control

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def asd_loss(u_s: torch.Tensor, u_t: torch.Tensor, w_sample: float = 1.0, w_mean: float = 1.0) -> torch.Tensor:
    """
    ASD: sample-level L2 alignment + mean alignment.
    u_s/u_t: [B, D]
    """
    assert u_s.shape == u_t.shape, (u_s.shape, u_t.shape)
    l_sample = F.mse_loss(u_s, u_t, reduction="mean")
    mu_s = u_s.mean(dim=0)
    mu_t = u_t.mean(dim=0)
    l_mean = F.mse_loss(mu_s, mu_t, reduction="mean")
    return w_sample * l_sample + w_mean * l_mean


def info_nce_distill(u_s: torch.Tensor, u_t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    AT-InfoNCE distillation:
      normalize; sim = u_s @ u_t^T; positive pairs are diagonal; CE over rows.
    tau: scalar tensor or [B] (broadcastable to logits rows)
    """
    u_s = F.normalize(u_s, dim=-1)
    u_t = F.normalize(u_t, dim=-1)
    logits = u_s @ u_t.t()  # [B,B]

    if tau.ndim == 0:
        logits = logits / tau.clamp_min(1e-6)
    else:
        # tau [B] -> divide each row
        logits = logits / tau.view(-1, 1).clamp_min(1e-6)

    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def alpha_confidence(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    From Step1 reliability weights alpha: entropy -> concentration confidence in [0,1].
    alpha: [B,V] or [V]
    """
    if alpha.ndim == 1:
        a = alpha.view(1, -1)
    else:
        a = alpha
    a = a.clamp_min(eps)
    a = a / a.sum(dim=-1, keepdim=True).clamp_min(eps)
    H = -(a * a.log()).sum(dim=-1)  # [B]
    V = a.shape[-1]
    Hn = H / (torch.log(torch.tensor(float(V), device=a.device)).clamp_min(eps))
    conf = (1.0 - Hn).clamp(0.0, 1.0)
    return conf  # [B]


def utility_signal(distill_err: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    """
    方法定稿：样本效用信号 = 蒸馏误差 × 置信（示意，工程可进一步调参/平滑）
    distill_err/conf: [B]
    """
    return (distill_err * conf).clamp_min(0.0)


def risk_signal(sensitivity: torch.Tensor, mi_est: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    方法定稿：风险信号 = 敏感度加权项 +（可选）成员推断风险估计项
    sensitivity: [B] or scalar
    """
    r = sensitivity
    if mi_est is not None:
        r = r + mi_est
    return r.clamp_min(0.0)


def dynamic_temperature(
    util: torch.Tensor,
    risk: torch.Tensor,
    round_idx: int,
    tau_min: float = 0.05,
    tau_max: float = 0.5,
    gamma_u: float = 2.0,
    gamma_r: float = 2.0,
    sched: str = "linear",
    total_rounds: int = 10,
) -> torch.Tensor:
    """
    方法定稿：更有用且风险低 -> tau 更小；风险高/不确定 -> tau 更大，并可随轮次调度
    util/risk: [B]
    """
    util = util.detach()
    risk = risk.detach()

    # score higher -> smaller tau
    score = gamma_u * util - gamma_r * risk
    gate = torch.sigmoid(score)  # [B] in (0,1)

    base = tau_max - (tau_max - tau_min) * gate  # [B]

    if total_rounds <= 1:
        return base.clamp(tau_min, tau_max)

    t = float(round_idx) / float(max(total_rounds - 1, 1))
    if sched == "linear":
        s = 1.0 - 0.3 * t  # later rounds: slightly smaller tau
    elif sched == "cosine":
        import math
        s = 0.85 + 0.15 * math.cos(math.pi * t)
    else:
        s = 1.0

    return (base * s).clamp(tau_min, tau_max)


def dp_noise_multiplier(
    base_sigma: float,
    util: torch.Tensor,
    risk: torch.Tensor,
    k_u: float = 0.3,
    k_r: float = 0.7,
    min_mult: float = 0.5,
    max_mult: float = 3.0,
) -> torch.Tensor:
    """
    方法定稿：风险高 -> 噪声更大；效用高 -> 噪声可适度减小（合规范围内）
    """
    mult = 1.0 + k_r * risk - k_u * util
    mult = mult.clamp(min_mult, max_mult)
    return base_sigma * mult
