"""opvc.running_stats

Running statistics for normalization / client baselines.

We use a simple exponential moving average (EMA) running mean/var.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

Tensor = torch.Tensor


@dataclass
class RunningMeanStd:
    dim: int
    momentum: float = 0.95
    eps: float = 1e-6
    mean: Optional[Tensor] = None
    var: Optional[Tensor] = None
    initialized: bool = False

    def to(self, device: torch.device) -> "RunningMeanStd":
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.var is not None:
            self.var = self.var.to(device)
        return self

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        if x.ndim == 1:
            xb = x.view(1, -1)
        elif x.ndim == 2:
            xb = x
        else:
            raise ValueError(f"x must be [D] or [B,D], got shape={tuple(x.shape)}")
        if xb.shape[-1] != self.dim:
            raise ValueError(f"dim mismatch: expected {self.dim}, got {xb.shape[-1]}")
        m = xb.mean(dim=0)
        v = xb.var(dim=0, unbiased=False)
        if not self.initialized:
            self.mean = m.clone()
            self.var = v.clone().clamp_min(self.eps)
            self.initialized = True
            return
        assert self.mean is not None and self.var is not None
        a = float(self.momentum)
        self.mean = a * self.mean + (1 - a) * m
        self.var = a * self.var + (1 - a) * v
        self.var = self.var.clamp_min(self.eps)

    def normalize(self, x: Tensor) -> Tensor:
        if not self.initialized or self.mean is None or self.var is None:
            return x
        return (x - self.mean) / torch.sqrt(self.var + self.eps)

    def denormalize(self, x: Tensor) -> Tensor:
        if not self.initialized or self.mean is None or self.var is None:
            return x
        return x * torch.sqrt(self.var + self.eps) + self.mean
