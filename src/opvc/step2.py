"""
Step2 skeleton: privacy-aware federated pretraining.

Provides:
- Student backbone f_S
- Projection heads phi_k
- forward_uras(Z, pi) -> US
"""

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import Step2Config, Step2Outputs


class StudentBackbone(nn.Module):
    def __init__(self, da: int, dz: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(da, dz),
            nn.ReLU(),
            nn.Linear(dz, dz),
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.net(Z)


class ProjectionHead(nn.Module):
    def __init__(self, dz: int, du: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dz, dz),
            nn.ReLU(),
            nn.Linear(dz, du),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class Step2Model(nn.Module):
    def __init__(self, da: int, dz: int, cfg: Step2Config):
        super().__init__()
        self.cfg = cfg
        self.student = StudentBackbone(da=da, dz=dz)
        self.heads = nn.ModuleList([ProjectionHead(dz=dz, du=cfg.du) for _ in range(cfg.Kr)])

    def forward_uras(self, Z: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        # support [da] or [B,da]
        if Z.dim() == 1:
            Zb = Z.unsqueeze(0)
            pib = pi.unsqueeze(0) if pi.dim() == 1 else pi
        else:
            Zb, pib = Z, pi

        h = self.student(Zb)  # [B,dz]
        u_list = [self.heads[k](h) for k in range(self.cfg.Kr)]  # each [B,du]
        weighted = [pib[:, k:k+1] * u_list[k] for k in range(self.cfg.Kr)]
        US = torch.cat(weighted, dim=-1)  # [B,Kr*du]
        return US.squeeze(0) if Z.dim() == 1 else US

    def as_state_dict(self) -> Dict[str, Any]:
        return self.state_dict()


def build_step2_outputs(model: Step2Model) -> Step2Outputs:
    return Step2Outputs(theta_global=model.as_state_dict(), forward_uras=model.forward_uras)
