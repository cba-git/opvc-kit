"""
OPVC Step3 (method-aligned runnable, contract-aligned)

Inputs:
- Step1: H_seq [T,da] (window evidence), alpha [V] (view weights, optional), pi [Kr] (routing, optional)
- Step2: theta_global (trained student params) -> compute URAS US per window

Core:
- URAS: US_tau = forward_uras(H_tau, pi_tau)
- Score: Mahalanobis distance over US sequence (deterministic, stable)
- ATC: quantile threshold + uncertainty margin -> sigmoid -> p_det_seq
- QPL: tau_x = argmax p_det_seq, J_view = merge(top-K taus)
- I_view: prefer alpha, fallback to pi mapping
- Unknown: explanation weak / flat / near boundary

Contracts:
Step3Config fields: V,T,da,Kr,du,Ka,ds,beta_det
Step3Outputs fields: p_det,tau_x,y_hat,I_view,J_view,flag_unknown,e_score,E_view,r_view
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

import torch

from .contracts import Step3Config, Step3Outputs, Step2Config
from .step2 import Step2Model


def _entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalized entropy (not normalized here; caller can divide by log K)."""
    p = p.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def _merge_to_segments(idxs_1based: List[int], max_gap: int = 0) -> List[Tuple[int, int]]:
    """Merge sorted 1-based indices into contiguous segments."""
    if not idxs_1based:
        return []
    idxs = sorted(set(int(x) for x in idxs_1based))
    segs: List[Tuple[int, int]] = []
    s = e = idxs[0]
    for x in idxs[1:]:
        if x <= e + 1 + max_gap:
            e = x
        else:
            segs.append((s, e))
            s = e = x
    segs.append((s, e))
    return segs


def _normalize_last_dim(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.sum(dim=-1, keepdim=True).clamp_min(eps)


def _build_step2_student(cfg3: Step3Config, theta_global: Dict[str, Any], device: str) -> Step2Model:
    # inference: no noise
    cfg2 = Step2Config(Kr=cfg3.Kr, du=cfg3.du, Cb=1.0, sigma_b=0.0)
    m2 = Step2Model(cfg2, dz=cfg3.da).to(device).eval()
        # theta_global can be packaged: {state_dict: <Step2Model weights>, mu/cov_inv/...}
    if isinstance(theta_global, dict) and 'state_dict' in theta_global:
        sd = theta_global['state_dict']
    else:
        sd = theta_global
    if (not isinstance(sd, dict)) or (not any(str(k).startswith(('student.','heads.')) for k in sd.keys())):
        raise ValueError("theta_global does not contain Step2Model weights (expected keys like student.* / heads.*). "
                         "Re-run scripts/run_step2_train_from_step1_out.py to regenerate a packaged theta_global with 'state_dict'.")
    m2.load_state_dict(sd, strict=True)

    return m2


def _get_quantile_level(beta_det: float) -> float:
    """
    Interpret beta_det:
      - if 0 < beta_det <= 1 : treat as quantile level q (ATC base threshold)
      - if beta_det > 1      : use default q=0.95
    """
    if 0.0 < beta_det <= 1.0:
        return float(beta_det)
    return 0.95


def _get_sigmoid_scale(beta_det: float) -> float:
    """
    Scale for sigmoid mapping:
      - if beta_det > 1 : use as scale (sharper)
      - else            : default 10
    """
    if beta_det > 1.0:
        return float(beta_det)
    return 10.0


def run_step3_from_H(
    cfg3: Step3Config,
    H_seq: torch.Tensor,            # [T,da]
    pi: torch.Tensor,               # [Kr] or [T,Kr]
    alpha: Optional[torch.Tensor],  # [V] or [T,V]
    theta_global: Dict[str, Any],
    device: str = "cpu",
) -> Step3Outputs:
    T = int(cfg3.T)
    da = int(cfg3.da)
    Kr = int(cfg3.Kr)
    V = int(cfg3.V)

    if H_seq.shape != (T, da):
        raise ValueError(f"H_seq shape {tuple(H_seq.shape)} != (T,da)=({T},{da})")

    H_seq = H_seq.to(device).float()

    # ---- pi_seq ----
    if pi.dim() == 1:
        pi_seq = pi.view(1, -1).repeat(T, 1)
    else:
        pi_seq = pi
    if pi_seq.shape != (T, Kr):
        raise ValueError(f"pi_seq shape {tuple(pi_seq.shape)} != (T,Kr)=({T},{Kr})")
    pi_seq = _normalize_last_dim(pi_seq.to(device).float())

    # ---- alpha_seq ----
    if alpha is None:
        alpha_seq = None
        alpha_unc = torch.zeros(T, device=device)
    else:
        if alpha.dim() == 1:
            alpha_seq = alpha.view(1, -1).repeat(T, 1)
        else:
            alpha_seq = alpha
        if alpha_seq.shape[-1] != V:
            raise ValueError(f"alpha_seq last dim {alpha_seq.shape[-1]} != V={V}")
        alpha_seq = _normalize_last_dim(alpha_seq.to(device).float())
        # normalize entropy to [0,1]
        alpha_unc = _entropy(alpha_seq) / torch.log(torch.tensor(float(V), device=device)).clamp_min(1e-12)

    # ---- uncertainty from pi entropy (normalized to [0,1]) ----
    pi_unc = _entropy(pi_seq) / torch.log(torch.tensor(float(Kr), device=device)).clamp_min(1e-12)

    # combine uncertainty (0..1)
    unc = (pi_unc + alpha_unc) * 0.5

    # ---- URAS per window via trained Step2 ----
    m2 = _build_step2_student(cfg3, theta_global, device)
    US_list = []
    for t in range(T):
        us = m2.forward_uras(H_seq[t], pi_seq[t]).view(1, -1)  # [1,Kr*du]
        US_list.append(us)
    US_seq = torch.cat(US_list, dim=0)  # [T, D]
    D = US_seq.shape[1]

    # ---- window score: Mahalanobis distance over US distribution ----
    X = US_seq - US_seq.mean(dim=0, keepdim=True)               # [T,D]
    eps = 1e-3
    cov = (X.T @ X) / max(T - 1, 1) + eps * torch.eye(D, device=US_seq.device)
    inv_cov = torch.linalg.pinv(cov)                             # [D,D]
    e_seq = torch.sum((X @ inv_cov) * X, dim=-1)                 # [T]

    # ---- ATC: quantile threshold + uncertainty margin -> p_det_seq ----
    q = _get_quantile_level(float(cfg3.beta_det))
    thresh_base = torch.quantile(e_seq, q=q)
    std = e_seq.std().clamp_min(1e-6)

    # when uncertain, be slightly more conservative
    thresh = thresh_base + 0.10 * unc.mean() * std

    beta = _get_sigmoid_scale(float(cfg3.beta_det))
    p_det_seq = torch.sigmoid(beta * (e_seq - thresh) / std)     # [T]

    # decision uses max prob
    tau_x0 = int(torch.argmax(p_det_seq).item())
    tau_x = tau_x0 + 1
    p_det = float(p_det_seq[tau_x0].item())
    y_hat = int(p_det >= 0.5)

    # ---- QPL: top-K windows -> segments ----
    k = max(1, int(cfg3.Ka))
    k = min(k, T)
    top_idx0 = torch.topk(p_det_seq, k=k).indices.detach().cpu().tolist()  # 0-based
    top_idx1 = [int(i) + 1 for i in top_idx0]
    J_view = _merge_to_segments(top_idx1, max_gap=0)

    # ---- I_view: alpha preferred ----
    if alpha_seq is not None and alpha_seq.shape[-1] == V:
        I_view = alpha_seq.mean(dim=0).detach().cpu().tolist()
        s = sum(I_view) if sum(I_view) > 0 else 1.0
        I_view = [float(x / s) for x in I_view]
    else:
        base = pi_seq.mean(dim=0).detach().cpu().tolist()  # Kr
        I_view = [float(base[v % Kr]) for v in range(V)]
        s = sum(I_view) if sum(I_view) > 0 else 1.0
        I_view = [x / s for x in I_view]

    # ---- Unknown: explanation weak / flat / near boundary ----
    flat = float(p_det_seq.std().item()) < 1e-3
    margin = abs(p_det - 0.5)
    explain_weak = (max(I_view) < 0.6)
    flag_unknown = bool(flat or (y_hat == 1 and explain_weak) or (margin < 0.05))

    # explanations
    E_view = {
        "score_type": "mahalanobis(US)",
        "q_level": float(q),
        "thresh_base": float(thresh_base.item()),
        "thresh": float(thresh.item()),
        "unc_mean": float(unc.mean().item()),
        "e_top3": [float(v) for v in torch.topk(e_seq, k=min(3, T)).values.detach().cpu().tolist()],
        "p_top3": [float(v) for v in torch.topk(p_det_seq, k=min(3, T)).values.detach().cpu().tolist()],
    }
    r_view = {
        "pi_unc_mean": float(pi_unc.mean().item()),
        "alpha_unc_mean": float(alpha_unc.mean().item()),
        "pi_mean": [float(x) for x in pi_seq.mean(dim=0).detach().cpu().tolist()],
    }

    return Step3Outputs(
        p_det=float(p_det),
        tau_x=int(tau_x),
        y_hat=int(y_hat),
        I_view=I_view,
        J_view=J_view,
        flag_unknown=bool(flag_unknown),
        e_score=float(e_seq[tau_x0].item()),
        E_view=E_view,
        r_view=r_view,
    )
