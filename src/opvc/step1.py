"""
Step1 skeleton: multi-view windowization -> encoder -> quality -> routing -> alignment -> gated fusion.

Convention:
- Window index is τ (tau). Total windows is T.
- Final localization is at WINDOW level (τ), not time-step.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import Step1Config, Step1Outputs, Step1Metrics


class SimpleViewEncoder(nn.Module):
    """Placeholder φ_v. Input [T,d_in_v] -> Output [T,d]."""
    def __init__(self, d_in: int, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

    def forward(self, x_win: torch.Tensor) -> torch.Tensor:
        return self.net(x_win)


class Step1Model(nn.Module):
    def __init__(self, cfg: Step1Config):
        super().__init__()
        self.cfg = cfg

        self.encoders = nn.ModuleList([SimpleViewEncoder(cfg.d_in[v], cfg.d) for v in range(cfg.V)])
        self.router = nn.Linear(cfg.V * cfg.d, cfg.Kr)
        self.B_basis = nn.Parameter(torch.randn(cfg.Kr, cfg.da, cfg.d) * 0.02)

        self.Wq = nn.Linear(cfg.da, cfg.da, bias=False)
        self.Wk = nn.Linear(cfg.da, cfg.da, bias=False)
        self.Wv = nn.Linear(cfg.da, cfg.da, bias=False)

        # ω_s logits for (cov,val,cmp,unq,stb)
        self.omega_logits = nn.Parameter(torch.zeros(5))

    # ---- Agg_v placeholder ----
    def agg_windows(self, E_v: List, T: int, d_in_v: int) -> torch.Tensor:
        return torch.zeros(T, d_in_v)

    # ---- Quality placeholder ----
    def compute_quality(self, x_win_all: torch.Tensor) -> Tuple[torch.Tensor, Step1Metrics]:
        activity = x_win_all.abs().mean(dim=(1, 2))  # [V]
        q_cov = torch.sigmoid(activity * 0.5)
        q_val = torch.sigmoid(activity * 0.3)
        q_cmp = torch.sigmoid(activity * 0.2)
        q_unq = torch.sigmoid(activity * 0.1)
        q_stb = torch.sigmoid(activity * 0.4)

        omega = F.softmax(self.omega_logits, dim=0)
        Q = omega[0]*q_cov + omega[1]*q_val + omega[2]*q_cmp + omega[3]*q_unq + omega[4]*q_stb
        alpha = F.softmax(Q / self.cfg.tau_q, dim=0)

        return alpha, Step1Metrics(q_cov=q_cov.detach(), q_val=q_val.detach(), q_cmp=q_cmp.detach(),
                                  q_unq=q_unq.detach(), q_stb=q_stb.detach(), Q=Q.detach())

    def route(self, g_view_weighted: torch.Tensor) -> torch.Tensor:
        x = g_view_weighted.reshape(-1)  # [V*d]
        return F.softmax(self.router(x), dim=0)  # [Kr]

    def build_Bx(self, pi: torch.Tensor) -> torch.Tensor:
        return torch.einsum("k,kad->ad", pi, self.B_basis)  # [da,d]

    def align(self, Bx: torch.Tensor, h_raw: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ad,vtd->vta", Bx, h_raw)  # [V,T,da]

    @staticmethod
    def pearson(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        a = a - a.mean()
        b = b - b.mean()
        denom = (a.norm(p=2) * b.norm(p=2)).clamp_min(eps)
        return (a @ b) / denom

    def compute_rho_gate(self, z_view: torch.Tensor, alpha: torch.Tensor) -> Tuple[float, bool, torch.Tensor]:
        V = z_view.shape[0]
        corr = torch.zeros(V, V, device=z_view.device)
        rho = torch.tensor(-1e9, device=z_view.device)
        for u in range(V):
            for v in range(u + 1, V):
                cuv = self.pearson(alpha[u]*z_view[u], alpha[v]*z_view[v])
                corr[u, v] = corr[v, u] = cuv
                rho = torch.maximum(rho, cuv)
        gate = bool((rho >= self.cfg.theta).item())
        return float(rho.item()), gate, corr

    def fuse_once(self, h_aligned: torch.Tensor, z_view: torch.Tensor, alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Z = torch.einsum("v,va->a", alpha, z_view)
        H = torch.einsum("v,vta->ta", alpha, h_aligned)
        return Z, H

    def fuse_with_attention(self, h_aligned: torch.Tensor, z_view: torch.Tensor, alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        V, T, da = h_aligned.shape

        z_w = alpha[:, None] * z_view
        Q = self.Wq(z_w)
        K = self.Wk(z_w)
        Vv = self.Wv(z_w)

        scores = (Q @ K.T) / (da ** 0.5)
        scores = scores.masked_fill(torch.eye(V, device=scores.device).bool(), float("-inf"))
        attn = torch.softmax(scores, dim=1)  # [V,V]

        z_prime = attn @ Vv  # [V,da]

        h_w = alpha[:, None, None] * h_aligned
        h_proj = self.Wv(h_w)
        h_prime = torch.einsum("vu,uta->vta", attn, h_proj)

        Z = torch.einsum("v,va->a", alpha, z_prime)
        H = torch.einsum("v,vta->ta", alpha, h_prime)
        return Z, H

    def forward(self, E: List[List], t0: float, delta: float) -> Step1Outputs:
        V, T = self.cfg.V, self.cfg.T

        # 1) Agg per view
        d_in_max = max(self.cfg.d_in)
        x_padded = torch.zeros(V, T, d_in_max)
        x_wins = []
        for v in range(V):
            x_v = self.agg_windows(E[v], T, self.cfg.d_in[v])  # [T,d_in_v]
            x_wins.append(x_v)
            x_padded[v, :, : self.cfg.d_in[v]] = x_v

        # 2) Encode
        h_raw = torch.stack([self.encoders[v](x_wins[v]) for v in range(V)], dim=0)  # [V,T,d]

        # 3) Quality -> alpha
        alpha, metrics = self.compute_quality(x_padded)

        # 4) View summary (pool) for routing; inject alpha
        g_view = h_raw.mean(dim=1)                 # [V,d]
        metrics.g_view = g_view.detach()
        g_view_weighted = alpha[:, None] * g_view  # [V,d]

        # 5) Router -> pi
        pi = self.route(g_view_weighted)

        # 6) Alignment
        Bx = self.build_Bx(pi)
        h_aligned = self.align(Bx, h_raw)

        # 7) View-level aligned vector
        z_view = h_aligned.mean(dim=1)  # [V,da]

        # 8) Gate
        rho, gate, corr = self.compute_rho_gate(z_view, alpha)
        metrics.corr_mat = corr.detach()

        # 9) Fusion
        if gate:
            Z, H = self.fuse_with_attention(h_aligned, z_view, alpha)
        else:
            Z, H = self.fuse_once(h_aligned, z_view, alpha)

        out = Step1Outputs(h_aligned=h_aligned, alpha=alpha, pi=pi, B_x=Bx, Z=Z, H=H,
                           gate=gate, rho=rho, metrics=metrics)
        out.validate(self.cfg)
        return out
