"""
OPVC Step3 (align-final)

Goals:
- Paper-aligned at the interface/behavior level (Step3 A~D structure in method doc)
- Robust theta_global loading (supports packaged dict with {state_dict, mu, cov_inv, ...}
  and also supports plain state_dict dicts)
- JSON-friendly outputs (python primitives/lists so scripts can json.dump without to_py)

This Step3 focuses on inference-time pieces:
- Mahalanobis score on URAS/student representation (e_score)
- Detection probability p_det via sigmoid(beta_det*(e_score - nu))
- tau_x as argmax window
- Localization segments J via TopK windows -> merged contiguous segments
- Unknown flag via simple confidence heuristic (view/route concentration + p_det)

Compatibility:
- Keeps `Step3Model` name as a thin wrapper so older demos importing it won't break.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch

from opvc.contracts import Step2Config, Step3Config, Step3Outputs
from opvc.step2 import Step2Model


class Step3Model(torch.nn.Module):
    """Compatibility wrapper.

    Some earlier code imported Step3Model directly. In align-final, we implement Step3
    as a functional entry `run_step3_from_H` that returns JSON-friendly outputs.
    This module simply forwards to that functional entry.
    """

    def __init__(self, cfg: Step3Config, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device

    def forward(
        self,
        H_seq: torch.Tensor,
        pi: torch.Tensor,
        alpha: torch.Tensor,
        theta_global: Union[str, Dict[str, Any]],
    ) -> Step3Outputs:
        return run_step3_from_H(
            self.cfg, H_seq=H_seq, pi=pi, alpha=alpha, theta_global=theta_global, device=self.device
        )


# -----------------------------
# Helpers
# -----------------------------
def _to_tensor(x: Any, device: str = "cpu", dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _softmax1d(x: torch.Tensor) -> torch.Tensor:
    x = x.flatten()
    return torch.softmax(x, dim=0)


def _normalize_prob(x: torch.Tensor) -> torch.Tensor:
    x = x.flatten()
    s = float(x.sum().detach().cpu())
    if s <= 0:
        return _softmax1d(x)
    return (x / s).clamp_min(0)


def _as_python(x: Any) -> Any:
    """Convert tensors/containers to json-friendly python."""
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {k: _as_python(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_as_python(v) for v in x]
    return x


def _merge_indices_to_segments(idxs_1based: List[int]) -> List[Tuple[int, int]]:
    """Given sorted 1-based indices, merge contiguous ones into (s,e) segments."""
    if not idxs_1based:
        return []
    idxs = sorted(set(int(i) for i in idxs_1based))
    segs: List[Tuple[int, int]] = []
    s = e = idxs[0]
    for i in idxs[1:]:
        if i == e + 1:
            e = i
        else:
            segs.append((s, e))
            s = e = i
    segs.append((s, e))
    return segs


def _infer_step2_model(cfg2: Step2Config, dz: int, device: str) -> Step2Model:
    """Instantiate Step2Model with signature tolerance (Step2Model(cfg2, dz) or Step2Model(cfg2))."""
    sig = inspect.signature(Step2Model.__init__)
    params = list(sig.parameters.keys())
    # params[0] is self
    if len(params) >= 3:  # (self, cfg, dz, ...)
        m = Step2Model(cfg2, dz).to(device)
    else:
        m = Step2Model(cfg2).to(device)
    return m


def _extract_theta_pkg(theta_global: Union[str, Dict[str, Any]], device: str) -> Dict[str, Any]:
    """Load theta_global from path or accept dict; return dict."""
    if isinstance(theta_global, str):
        pkg = torch.load(theta_global, map_location=device)
        if not isinstance(pkg, dict):
            raise TypeError(f"theta_global loaded from path must be a dict, got {type(pkg)}")
        return pkg
    if not isinstance(theta_global, dict):
        raise TypeError(f"theta_global must be a path or dict, got {type(theta_global)}")
    return theta_global


def _extract_state_dict(pkg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Return a state_dict-like dict. Prefer pkg['state_dict'] if present."""
    if "state_dict" in pkg and isinstance(pkg["state_dict"], dict):
        sd = pkg["state_dict"]
    else:
        # Some older code directly saved state_dict itself
        sd = pkg
    # filter only tensor-ish values
    out = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v
    return out


def _mahalanobis_scores(
    X: torch.Tensor,
    mu: torch.Tensor,
    cov_inv: torch.Tensor,
) -> torch.Tensor:
    """Return per-row Mahalanobis distance squared."""
    # X: [T, D], mu: [D], cov_inv: [D, D]
    Xm = X - mu.view(1, -1)
    # (Xm @ cov_inv) * Xm row-sum
    return (Xm @ cov_inv * Xm).sum(dim=1)


def _estimate_mu_covinv(X: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback estimation if pkg doesn't contain mu/cov_inv."""
    mu = X.mean(dim=0)
    Xm = X - mu
    # covariance
    cov = (Xm.t() @ Xm) / max(1, X.shape[0] - 1)
    D = cov.shape[0]
    cov = cov + eps * torch.eye(D, device=X.device, dtype=X.dtype)
    cov_inv = torch.linalg.pinv(cov)
    return mu, cov_inv


def _pick_nu(pkg: Dict[str, Any], e_score: torch.Tensor) -> float:
    """Pick nu (threshold) robustly."""
    # direct
    for key in ("nu", "theta_det", "th_det"):
        if key in pkg and isinstance(pkg[key], (int, float)):
            return float(pkg[key])

    # sometimes eps is reused as threshold; only accept if it's not tiny
    if "eps" in pkg and isinstance(pkg["eps"], (int, float)) and float(pkg["eps"]) > 0.1:
        return float(pkg["eps"])

    # robust default: 0.95 quantile
    try:
        return float(torch.quantile(e_score.detach().cpu(), 0.95).item())
    except Exception:
        return float(e_score.detach().cpu().median().item())


def _topk_segments(
    scores: torch.Tensor,
    topk: int,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return top-k indices (1-based) and merged segments."""
    T = int(scores.numel())
    k = max(1, min(int(topk), T))
    vals, idxs0 = torch.topk(scores, k=k, largest=True, sorted=True)  # idxs 0-based
    idxs_1based = [int(i) + 1 for i in idxs0.detach().cpu().tolist()]
    segs = _merge_indices_to_segments(idxs_1based)
    return idxs_1based, segs


# -----------------------------
# Main entry
# -----------------------------
@torch.no_grad()
def run_step3_from_H(
    cfg3: Step3Config,
    H_seq: torch.Tensor,
    pi: torch.Tensor,
    alpha: torch.Tensor,
    theta_global: Union[str, Dict[str, Any]],
    device: str = "cpu",
) -> Step3Outputs:
    """
    Inputs:
      - H_seq: [T, da]
      - pi:    [Kr]
      - alpha: [V]
      - theta_global: packaged dict from Step2 training OR path to it

    Outputs (json-friendly):
      - p_det: float (max over windows)
      - tau_x: int (1-based argmax)
      - y_hat: int (0/1)
      - I_view: list[float] len V
      - J_view: list[(s,e)] merged segments (union of per-view topk)
      - flag_unknown: bool
      - e_score: list[float] per-window Mahalanobis score
      - E_view: dict with score_type/nu/topk/J_by_view, etc.
    """
    # ---------- shape checks ----------
    H = _to_tensor(H_seq, device=device, dtype=torch.float32)
    if H.ndim != 2:
        raise ValueError(f"H_seq must be 2D [T,da], got shape={tuple(H.shape)}")
    T, da = int(H.shape[0]), int(H.shape[1])

    pi_t = _to_tensor(pi, device=device, dtype=torch.float32).flatten()
    if pi_t.numel() != int(cfg3.Kr):
        raise ValueError(f"pi must have Kr={cfg3.Kr} elements, got {pi_t.numel()}")
    pi_n = _normalize_prob(pi_t)

    alpha_t = _to_tensor(alpha, device=device, dtype=torch.float32).flatten()
    if alpha_t.numel() != int(cfg3.V):
        raise ValueError(f"alpha must have V={cfg3.V} elements, got {alpha_t.numel()}")
    I_view = _normalize_prob(alpha_t)

    # ---------- load theta_global ----------
    pkg = _extract_theta_pkg(theta_global, device=device)
    state_dict = _extract_state_dict(pkg)

    # ---------- build Step2 model ----------
    # Step2 config: only set attributes that exist (tolerate contract drift)
    cfg2 = Step2Config()
    # safe-set common fields
    if hasattr(cfg2, "Kr"):
        cfg2.Kr = int(cfg3.Kr)
    if hasattr(cfg2, "du"):
        cfg2.du = int(cfg3.du)

    m2 = _infer_step2_model(cfg2, dz=da, device=device)
    # load weights (strict=False so it won't explode on minor drift)
    try:
        m2.load_state_dict(state_dict, strict=False)
    except Exception as e:
        # last resort: filter matching keys only
        model_keys = set(m2.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in model_keys}
        m2.load_state_dict(filtered, strict=False)

    # ---------- forward URAS/student representation ----------
    # prefer forward_uras(H,[Kr]) -> [T, D]
    if hasattr(m2, "forward_uras") and callable(getattr(m2, "forward_uras")):
        # many implementations expect pi as [B,Kr]
        pi_batch = pi_n.view(1, -1).repeat(T, 1)
        US = m2.forward_uras(H, pi_batch)
    else:
        # fallback to calling module directly
        pi_batch = pi_n.view(1, -1).repeat(T, 1)
        US = m2(H, pi_batch)

    # tolerate tuple outputs
    if isinstance(US, (tuple, list)):
        US = US[0]
    US = _to_tensor(US, device=device, dtype=torch.float32)
    if US.ndim != 2 or US.shape[0] != T:
        raise ValueError(f"Step2 forward output must be [T,D], got shape={tuple(US.shape)}")

    # ---------- Mahalanobis score ----------
    D = int(US.shape[1])
    if "mu" in pkg and "cov_inv" in pkg:
        mu = _to_tensor(pkg["mu"], device=device, dtype=torch.float32).flatten()
        cov_inv = _to_tensor(pkg["cov_inv"], device=device, dtype=torch.float32)
        if mu.numel() != D:
            # fallback estimation
            mu, cov_inv = _estimate_mu_covinv(US, eps=1e-6)
    else:
        mu, cov_inv = _estimate_mu_covinv(US, eps=float(pkg.get("eps", 1e-6) or 1e-6))

    e_score_t = _mahalanobis_scores(US, mu, cov_inv)  # [T]

    # ---------- p_det / tau_x ----------
    nu = _pick_nu(pkg, e_score_t)
    beta_det = float(getattr(cfg3, "beta_det", 1.0) or 1.0)
    p_t = torch.sigmoid(beta_det * (e_score_t - float(nu)))  # [T]

    tau0 = int(torch.argmax(p_t).item())  # 0-based
    tau_x = tau0 + 1  # 1-based
    p_det = float(p_t[tau0].item())
    y_hat = int(p_det >= 0.5)

    # ---------- localization segments ----------
    # paper-level: per-view window scoring then segments; keep union in J_view for script compatibility
    # use a small deterministic topk to keep output concise and stable
    topk = int(pkg.get("topk_loc", 3) or 3)
    topk = max(1, min(topk, T))

    J_by_view: List[List[Tuple[int, int]]] = []
    union_indices: List[int] = []

    for v in range(int(cfg3.V)):
        # simple view-weighted score
        scores_v = p_t * float(I_view[v].item())
        idxs_v, segs_v = _topk_segments(scores_v, topk=topk)
        union_indices.extend(idxs_v)
        J_by_view.append(segs_v)

    J_union = _merge_indices_to_segments(union_indices)

    # ---------- unknown flag heuristic ----------
    # If detection is high but views/routes are too uniform -> uncertain/unknown
    alpha_conf = float(I_view.max().item())  # close to 1 means confident single view
    pi_conf = float(pi_n.max().item())
    flag_unknown = bool((p_det >= 0.5) and (alpha_conf < 0.65 or pi_conf < 0.65))

    # ---------- pack extra debug info ----------
    E_view: Dict[str, Any] = {
        "score_type": "mahalanobis(US)",
        "nu": float(nu),
        "beta_det": float(beta_det),
        "topk_loc": int(topk),
        "J_by_view": J_by_view,
        "alpha_conf": float(alpha_conf),
        "pi_conf": float(pi_conf),
    }

    out = Step3Outputs(
        p_det=_as_python(p_det),
        tau_x=_as_python(tau_x),
        y_hat=_as_python(y_hat),
        I_view=_as_python(I_view),
        J_view=_as_python(J_union),
        flag_unknown=_as_python(flag_unknown),
        e_score=_as_python(e_score_t),
        E_view=_as_python(E_view),
        r_view=None,
    )
    return out
