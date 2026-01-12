"""
Single source of truth for step interfaces.
Dataclasses define what each step MUST output / consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

import torch

Tensor = torch.Tensor


def _assert_shape(x: Tensor, shape: Tuple[Optional[int], ...], name: str) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    if x.dim() != len(shape):
        raise ValueError(f"{name} dim mismatch: expected {len(shape)}D, got {x.dim()}D with shape {tuple(x.shape)}")
    for i, s in enumerate(shape):
        if s is None:
            continue
        if x.shape[i] != s:
            raise ValueError(f"{name} shape mismatch at dim {i}: expected {s}, got {x.shape[i]} (shape={tuple(x.shape)})")


@dataclass
class Step1Config:
    V: int
    T: int
    d_in: List[int]   # per-view Agg_v output dims
    d: int            # encoder output dim
    da: int           # aligned dim
    Kr: int           # routing subspaces
    tau_q: float = 1.0
    theta: float = 0.5


@dataclass
class Step1Metrics:
    q_cov: Optional[Tensor] = None
    q_val: Optional[Tensor] = None
    q_cmp: Optional[Tensor] = None
    q_unq: Optional[Tensor] = None
    q_stb: Optional[Tensor] = None
    Q: Optional[Tensor] = None
    g_view: Optional[Tensor] = None
    corr_mat: Optional[Tensor] = None


@dataclass
class Step1Outputs:
    """
    Final localization is defined at WINDOW level τ (tau), not time-step.
    """
    h_aligned: Tensor   # [V,T,da]
    alpha: Tensor       # [V]
    pi: Tensor          # [Kr]
    B_x: Tensor         # [da,d]
    Z: Tensor           # [da]
    H: Tensor           # [T,da]
    gate: Optional[bool] = None
    rho: Optional[float] = None
    metrics: Step1Metrics = field(default_factory=Step1Metrics)

    def validate(self, cfg: Step1Config) -> None:
        _assert_shape(self.h_aligned, (cfg.V, cfg.T, cfg.da), "Step1Outputs.h_aligned")
        _assert_shape(self.alpha, (cfg.V,), "Step1Outputs.alpha")
        _assert_shape(self.pi, (cfg.Kr,), "Step1Outputs.pi")
        _assert_shape(self.B_x, (cfg.da, cfg.d), "Step1Outputs.B_x")
        _assert_shape(self.Z, (cfg.da,), "Step1Outputs.Z")
        _assert_shape(self.H, (cfg.T, cfg.da), "Step1Outputs.H")

        if (self.alpha < 0).any():
            raise ValueError("alpha must be >= 0")
        if (self.pi < 0).any():
            raise ValueError("pi must be >= 0")

        if not torch.isclose(self.alpha.sum(), torch.tensor(1.0, device=self.alpha.device), atol=1e-3):
            raise ValueError(f"alpha must sum to 1, got {self.alpha.sum().item():.6f}")
        if not torch.isclose(self.pi.sum(), torch.tensor(1.0, device=self.pi.device), atol=1e-3):
            raise ValueError(f"pi must sum to 1, got {self.pi.sum().item():.6f}")


@dataclass
class Step2Config:
    Kr: int
    du: int
    Cb: float = 1.0
    sigma_b: float = 0.5


@dataclass
class Step2Outputs:
    theta_global: Dict[str, Any]
    forward_uras: Callable[..., Tensor]
    nu: Optional[Tensor] = None
    risk: Optional[Tensor] = None
    tau_dyn: Optional[Tensor] = None


@dataclass
class Step3Config:
    V: int
    T: int
    da: int
    Kr: int
    du: int
    Ka: int
    ds: int
    beta_det: float = 1.0


@dataclass
class Step3Outputs:
    p_det: Tensor
    tau_x: Tensor
    y_hat: Tensor
    I_view: Tensor
    J_view: List[List[Tuple[int, int]]]
    flag_unknown: Tensor

    e_score: Optional[Tensor] = None
    E_view: Optional[Tensor] = None
    r_view: Optional[Tensor] = None
