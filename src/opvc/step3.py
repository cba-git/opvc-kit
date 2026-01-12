"""
Step3 skeleton: detection + recognition + localization (window-level).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import Step3Config, Step1Outputs, Step2Outputs, Step3Outputs


class SCD(nn.Module):
    def __init__(self, d_in: int, ds: int):
        super().__init__()
        self.Wc = nn.Linear(d_in, ds)
        self.Ws = nn.Linear(d_in, ds)

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.Wc(u), self.Ws(u)


class Step3Model(nn.Module):
    def __init__(self, cfg: Step3Config):
        super().__init__()
        self.cfg = cfg

        u_dim = cfg.Kr * cfg.du
        self.scd = SCD(d_in=u_dim, ds=cfg.ds)

        self.det_head = nn.Linear(cfg.ds, 1)
        self.cls_head = nn.Linear(cfg.ds, cfg.Ka)

        self.Wwin = nn.Linear(cfg.da, cfg.ds, bias=False)

        # placeholders
        self.tau_c0 = nn.Parameter(torch.tensor(0.0))
        self.lamU = nn.Parameter(torch.tensor(0.1))
        self.lamC = nn.Parameter(torch.tensor(0.1))
        self.lamP = nn.Parameter(torch.tensor(0.1))
        self.lamPi = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def entropy_conf(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        V = alpha.numel()
        H = -(alpha * (alpha + eps).log()).sum()
        return 1.0 - H / torch.log(torch.tensor(float(V), device=alpha.device))

    @staticmethod
    def routing_uncertainty(pi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        Kr = pi.numel()
        H = -(pi * (pi + eps).log()).sum()
        return H / torch.log(torch.tensor(float(Kr), device=pi.device))

    def compute_tau_x(self, nu: float, risk: float, alpha: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        C_alpha = self.entropy_conf(alpha)
        u_pi = self.routing_uncertainty(pi)
        return self.tau_c0 - self.lamU * nu - self.lamC * C_alpha + self.lamP * risk + self.lamPi * u_pi

    def qpl_localize(self, step1: Step1Outputs, zs: torch.Tensor):
        V, T, _ = step1.h_aligned.shape
        z_win = F.normalize(self.Wwin(step1.h_aligned), p=2, dim=-1)  # [V,T,ds]
        zs_n = F.normalize(zs, p=2, dim=-1)                           # [ds]
        e = torch.einsum("vtd,d->vt", z_win, zs_n)                    # [V,T]
        E_view = e.mean(dim=1)                                        # [V]

        # placeholder thresholds: r_v = r0 - beta_alpha * alpha_v
        r0 = torch.tensor(0.0, device=e.device)
        beta_alpha = torch.tensor(0.5, device=e.device)
        r_view = r0 - beta_alpha * step1.alpha                        # [V]

        I_v_tau = (e >= r_view[:, None])                              # [V,T]
        J_view: List[List[Tuple[int,int]]] = []
        for v in range(V):
            intervals: List[Tuple[int,int]] = []
            on = False
            s = 0
            for tau in range(T):
                if I_v_tau[v, tau] and not on:
                    on = True
                    s = tau
                if on and (tau == T-1 or not I_v_tau[v, tau+1]):
                    intervals.append((s, tau))
                    on = False
            J_view.append(intervals)

        return e, E_view, r_view, J_view

    def forward(self, step1: Step1Outputs, step2: Step2Outputs, nu: float = 0.0, risk: float = 0.0) -> Step3Outputs:
        US = step2.forward_uras(step1.Z, step1.pi)   # [Kr*du]
        zc, zs = self.scd(US)

        s = self.det_head(zs).squeeze(-1)
        tau_x = self.compute_tau_x(nu=nu, risk=risk, alpha=step1.alpha, pi=step1.pi)
        p_det = torch.sigmoid(self.cfg.beta_det * (s - tau_x))

        y_hat = torch.sigmoid(self.cls_head(zs))     # [Ka]

        e_score, E_view, r_view, J_view = self.qpl_localize(step1, zs)
        I_view = (E_view >= r_view)

        flag_unknown = (p_det >= 0.5) & (~I_view.any())

        return Step3Outputs(p_det=p_det, tau_x=tau_x, y_hat=y_hat, I_view=I_view, J_view=J_view,
                           flag_unknown=flag_unknown, e_score=e_score, E_view=E_view, r_view=r_view)
