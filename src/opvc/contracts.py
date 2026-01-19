"""opvc.contracts

Single source of truth for OPVC Step1/Step2/Step3 interfaces.

Key disambiguation (matches docs/01_notation.md and method final):
- tau (τ) as a *window index* is an integer in [1..T].
- tau_x (τ(x)) is a *sample-adaptive detection threshold* (float scalar).

All external-facing outputs are defined at WINDOW level (no within-window time-step outputs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

Tensor = torch.Tensor


def _assert_tensor(x: Any, name: str) -> Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    return x


def _assert_shape(x: Tensor, shape: Tuple[Optional[int], ...], name: str) -> None:
    _assert_tensor(x, name)
    if x.dim() != len(shape):
        raise ValueError(f"{name} dim mismatch: expected {len(shape)}D, got {x.dim()}D (shape={tuple(x.shape)})")
    for i, s in enumerate(shape):
        if s is None:
            continue
        if int(x.shape[i]) != int(s):
            raise ValueError(f"{name} shape mismatch at dim {i}: expected {s}, got {int(x.shape[i])} (shape={tuple(x.shape)})")


def _assert_finite(x: Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).sum().item()
        raise ValueError(f"{name} contains non-finite values (count={bad})")


def _assert_prob_simplex(x: Tensor, name: str, atol: float = 1e-3) -> None:
    _assert_finite(x, name)
    if (x < 0).any():
        raise ValueError(f"{name} must be >= 0")
    s = float(x.sum().detach().cpu().item())
    if abs(s - 1.0) > atol:
        raise ValueError(f"{name} must sum to 1 (got {s:.6f})")


# =========================
# Step1
# =========================

@dataclass
@dataclass
class Step1Config:
    # views / windows
    V: int
    T: int

    # deterministic window aggregation output dims per view
    d_in: List[int]

    # encoder output dim (per window, per view)
    d: int

    # aligned evidence dim
    da: int

    # routing subspaces
    Kr: int

    # quality->alpha
    tau_q: float = 1.0
    q_norm_momentum: float = 0.95  # running stats momentum for Q normalization
    q_norm_eps: float = 1e-6

    # Pearson gating
    theta: float = 0.5

    # attention (C3)
    attn_scale: Optional[float] = None


@dataclass
class Step1Metrics:
    # quality components (each [V])
    q_cov: Optional[Tensor] = None
    q_val: Optional[Tensor] = None
    q_cmp: Optional[Tensor] = None
    q_unq: Optional[Tensor] = None
    q_stb: Optional[Tensor] = None

    # fused quality score and normalized quality
    Q: Optional[Tensor] = None
    Q_hat: Optional[Tensor] = None

    # view summaries and correlation
    g_view: Optional[Tensor] = None          # [V,d] (raw) or [V,da] (aligned)
    corr_mat: Optional[Tensor] = None        # [V,V]


@dataclass
class Step1Outputs:
    # aligned evidence per view, per window
    h_aligned: Tensor        # [V,T,da]

    # reliability weights
    alpha: Tensor            # [V]

    # routing distribution
    pi: Tensor               # [Kr]

    # sample-conditioned alignment operator
    B_x: Tensor              # [da,d]

    # fused sample-level representation
    Z: Tensor                # [da]

    # fused window sequence
    H: Tensor                # [T,da]

    # optional explainability
    gate: Optional[bool] = None
    rho: Optional[float] = None

    # extra metrics (optional but strongly recommended)
    metrics: Step1Metrics = field(default_factory=Step1Metrics)

    def validate(self, cfg: Step1Config) -> None:
        _assert_shape(self.h_aligned, (cfg.V, cfg.T, cfg.da), "Step1Outputs.h_aligned")
        _assert_shape(self.alpha, (cfg.V,), "Step1Outputs.alpha")
        _assert_shape(self.pi, (cfg.Kr,), "Step1Outputs.pi")
        _assert_shape(self.B_x, (cfg.da, cfg.d), "Step1Outputs.B_x")
        _assert_shape(self.Z, (cfg.da,), "Step1Outputs.Z")
        _assert_shape(self.H, (cfg.T, cfg.da), "Step1Outputs.H")

        _assert_prob_simplex(self.alpha, "Step1Outputs.alpha")
        _assert_prob_simplex(self.pi, "Step1Outputs.pi")

        _assert_finite(self.h_aligned, "Step1Outputs.h_aligned")
        _assert_finite(self.B_x, "Step1Outputs.B_x")
        _assert_finite(self.Z, "Step1Outputs.Z")
        _assert_finite(self.H, "Step1Outputs.H")


# =========================
# Step2
# =========================

@dataclass
@dataclass
class Step2Config:
    # routing subspaces / URAS
    Kr: int
    du: int  # subspace semantic dim

    # teacher label space (multi-label)
    Ka: int = 0  # if 0, teacher supervised pretrain is skipped unless provided by caller

    # DP on behavior feature
    Cb: float = 1.0
    sigma_b0: float = 0.5  # base noise multiplier

    # federated training
    rounds: int = 1
    num_clients: int = 2
    local_epochs: int = 1
    lr: float = 1e-3

    # adaptive DP (view-wise)
    clip_min: float = 0.5
    clip_max: float = 2.0
    sigma_min: float = 0.1
    sigma_max: float = 1.0

    # redundancy projection
    proj_momentum: float = 0.9
    proj_temp: float = 5.0

    # distillation weights
    lambda_asd: float = 1.0
    lambda_nce: float = 1.0

    # temperature bounds
    tau_min: float = 0.05
    tau_max: float = 0.5

    # dynamic temperature coefficients
    kappa_u: float = 1.0
    kappa_p: float = 1.0

    # implementation switch (engineering optimization)
    # - "approx": fast path (single backward). View deltas are obtained by parameter split.
    # - "exact": per-view local updates (telescoping deltas guarantees sum_v Δ_v = Δ_total).
    view_grad_mode: str = "approx"


@dataclass
class Step2Outputs:
    # global student parameters (state_dict)
    theta_global: Dict[str, Any]

    # runtime-only helper: forward URAS with a loaded model (not serializable)
    forward_uras: Callable[[Tensor, Tensor], Tensor]

    # optional logs / explainability
    nu: Optional[float] = None
    risk: Optional[float] = None
    tau_dyn: Optional[float] = None


# =========================
# Step3
# =========================

@dataclass
class Step3Config:
    V: int
    T: int
    da: int
    Kr: int
    du: int
    Ka: int
    ds: int

    # ATC
    beta_det: float = 1.0
    q_c: float = 0.95  # quantile for tau_c
    gamma_u: float = 1.0
    gamma_p: float = 1.0
    gamma_pi: float = 1.0
    gamma_alpha: float = 1.0

    # QPL
    tau0_view: float = 0.0
    lambda_alpha: float = 1.0
    lambda_sens: float = 1.0
    lambda_risk: float = 1.0

    # mapping back to time (optional; used for producing readable results)
    delta: Optional[float] = None


@dataclass
class Step3Outputs:
    # detection probability p_det(x)
    p_det: Tensor                 # []

    # adaptive threshold tau_x (scalar threshold, NOT index)
    tau_x: Tensor                 # []

    # multi-label prediction
    y_hat: Tensor                 # [Ka]

    # view suspiciousness indicator
    I_view: Tensor                # [V] (bool)

    # suspicious window segments (1-based indices), per view
    J_view: List[List[Tuple[int, int]]]

    # unknown carrier flag
    flag_unknown: Tensor          # [] (bool)

    # optional debug
    s_score: Optional[Tensor] = None          # []
    tau_c: Optional[Tensor] = None            # []
    E_view: Optional[Tensor] = None           # [V]
    e_win: Optional[Tensor] = None            # [V,T]
    contrib_view: Optional[Tensor] = None     # [Ka,V]

    def validate(self, cfg: Step3Config) -> None:
        _assert_shape(self.p_det, tuple(), "Step3Outputs.p_det")
        _assert_shape(self.tau_x, tuple(), "Step3Outputs.tau_x")
        _assert_shape(self.y_hat, (cfg.Ka,), "Step3Outputs.y_hat")
        _assert_shape(self.I_view, (cfg.V,), "Step3Outputs.I_view")
        _assert_shape(self.flag_unknown, tuple(), "Step3Outputs.flag_unknown")

        _assert_finite(self.p_det, "Step3Outputs.p_det")
        _assert_finite(self.tau_x, "Step3Outputs.tau_x")
        _assert_finite(self.y_hat, "Step3Outputs.y_hat")

        if (self.p_det < 0).item() or (self.p_det > 1).item():
            raise ValueError(f"p_det must be in [0,1], got {float(self.p_det.item()):.6f}")

        # J_view must be well-formed segments within [1..T]
        if len(self.J_view) != cfg.V:
            raise ValueError(f"J_view must have length V={cfg.V}, got {len(self.J_view)}")
        for v, segs in enumerate(self.J_view):
            for (s, e) in segs:
                if not (1 <= int(s) <= int(e) <= cfg.T):
                    raise ValueError(f"J_view[{v}] has invalid segment ({s},{e}) for T={cfg.T}")
