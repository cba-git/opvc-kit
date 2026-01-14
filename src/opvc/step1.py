from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import Step1Config, Step1Metrics, Step1Outputs


Tensor = torch.Tensor


def _as_float_tensor(x: Any, device: Optional[torch.device] = None) -> Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(x)
    t = t.to(dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def _safe_pearson_corr(x: Tensor, eps: float = 1e-8) -> Tensor:
    """
    x: [V, d]
    return corr_mat: [V, V], with safe handling for zero-variance rows (-> corr=0).
    """
    assert x.ndim == 2
    V, d = x.shape
    x0 = x - x.mean(dim=1, keepdim=True)
    var = (x0 * x0).mean(dim=1)  # [V]
    std = torch.sqrt(torch.clamp(var, min=0.0) + eps)  # [V]
    # normalize rows; if std ~ 0, keep as zeros
    mask = (std > eps).to(x.dtype)  # [V]
    xn = x0 / std.unsqueeze(1)
    xn = xn * mask.unsqueeze(1)

    corr = (xn @ xn.t()) / float(d)  # [V,V]
    corr = torch.clamp(corr, -1.0, 1.0)
    # diag should be 1 for valid rows, else 0
    diag = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))
    corr.fill_diagonal_(0.0)
    corr = corr + torch.diag(diag)
    return corr


def _rho_from_corr(corr: Tensor) -> Tensor:
    """
    corr: [V,V] (diag may be 1/0)
    rho = mean_{v<v'} |corr[v,v']|
    """
    V = corr.shape[0]
    if V <= 1:
        return corr.new_tensor(0.0)
    triu = torch.triu(corr.abs(), diagonal=1)
    denom = V * (V - 1) / 2.0
    return triu.sum() / corr.new_tensor(denom)


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_hid: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if d_hid is None:
            d_hid = max(d_out, d_in)
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Step1Model(nn.Module):
    """
    Step1 (per 方法定稿 A1~A6):
    - windowize/Agg -> x_win[v,t]         [T, d_in[v]]
    - encode         -> h_raw[v,t]        [T, d]
    - pool           -> g_view[v]         [d]
    - quality         -> q_*[v], Q[v]     [V]
    - reliability     -> alpha[v]         [V], softmax(Q)
    - routing         -> pi[k]            [Kr], softmax(router(g_all))
    - alignment       -> B_x              [da,d], sum_k pi[k] B_k
                         h_aligned[v,t]  [T,da]
    - corr/rho/gate   -> corr_mat[V,V], rho scalar, gate = rho >= theta
    - fusion:
        if gate: H[t]=sum_v alpha[v] h_aligned[v,t]
        else:   pick v* = argmax alpha, H[t]=h_aligned[v*,t]
        Z = mean_t H[t]
    """

    def __init__(self, cfg: Step1Config):
        super().__init__()
        self.cfg = cfg

        # encoders φ_v: x_win[v,t] -> h_raw[v,t]
        self.encoders = nn.ModuleList([MLP(cfg.d_in[v], cfg.d) for v in range(cfg.V)])

        # quality head: map 5-d quality vector -> scalar Q_v
        self.q_head = nn.Linear(5, 1, bias=True)
        with torch.no_grad():
            # init to equal weights (mean of 5 components)
            self.q_head.weight.fill_(1.0 / 5.0)
            self.q_head.bias.zero_()

        # router: concat g_view -> pi over Kr
        self.router = MLP(cfg.V * cfg.d, cfg.Kr, d_hid=max(cfg.Kr * 2, cfg.V * cfg.d))

        # alignment bases B_k: [Kr, da, d]
        self.B_basis = nn.Parameter(torch.randn(cfg.Kr, cfg.da, cfg.d) * 0.02)

    # ---- A1/A2: engineering-friendly Agg_v (accept already-windowized) ----
    def agg_windows(self, E_v: Any, T: int, d_in_v: int, device: Optional[torch.device]) -> Tensor:
        """
        Engineering shortcut (当前阶段你们已做对齐产物 step1_debug/CSV):
        - If caller already provides windowized features: Tensor/ndarray/list with shape [T, d_in_v], validate & return.
        - If empty/None -> zeros([T,d_in_v]).
        """
        if E_v is None:
            return torch.zeros(T, d_in_v, dtype=torch.float32, device=device)

        # empty list placeholder
        if isinstance(E_v, list) and len(E_v) == 0:
            return torch.zeros(T, d_in_v, dtype=torch.float32, device=device)

        x = _as_float_tensor(E_v, device=device)
        if tuple(x.shape) != (T, d_in_v):
            raise ValueError(f"agg_windows expects ({T},{d_in_v}), got {tuple(x.shape)}")
        return x

    # ---- A3: encode + pool ----
    def encode_view(self, x_win_v: Tensor, v: int) -> Tensor:
        # x_win_v: [T, d_in[v]] -> [T,d]
        return self.encoders[v](x_win_v)

    def pool_view(self, h_raw_v: Tensor) -> Tensor:
        # g_view[v] = mean over windows
        return h_raw_v.mean(dim=0)  # [d]

    # ---- A5: quality vector -> Q_v ----
    def quality_scores_from_xwin(self, x_win_v: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        方法定稿版质量指标计算:
        q_cov: 覆盖度 = 1 - 空窗比例
        q_val: 解析有效率 = parse_success / total
        q_cmp: 字段完整度 = 1 - 缺失率
        q_unq: 唯一性 = 1 - 重复率
        q_stb: 稳定性 = 1 / (1 + CV)，CV=std(mass)/mean(mass)
        若无解析统计字段，q_val/q_cmp/q_unq 默认为1。
        """
        T, d_in = x_win_v.shape
        mass = x_win_v.abs().sum(dim=1)
        empty = (mass <= eps).float()
        q_cov = 1.0 - empty.mean()
        q_val = torch.tensor(1.0, device=x_win_v.device)
        q_cmp = torch.tensor(1.0, device=x_win_v.device)
        q_unq = torch.tensor(1.0, device=x_win_v.device)
        mean_m = mass.mean()
        std_m = mass.std(unbiased=False)
        cv = std_m / (mean_m + eps)
        q_stb = 1.0 / (1.0 + cv)
        return q_cov, q_val, q_cmp, q_unq, q_stb

    def compute_alpha(self, q_cov: Tensor, q_val: Tensor, q_cmp: Tensor, q_unq: Tensor, q_stb: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Q[v] = q_head([q_cov,q_val,q_cmp,q_unq,q_stb])
        alpha = softmax(Q)
        """
        q_vec = torch.stack([q_cov, q_val, q_cmp, q_unq, q_stb], dim=1)  # [V,5]
        Q = self.q_head(q_vec).squeeze(1)  # [V]
        alpha = F.softmax(Q, dim=0)
        return Q, alpha

    # ---- A4: routing pi ----
    def compute_pi(self, g_view: Tensor) -> Tensor:
        """
        g_view: [V,d] -> pi: [Kr]
        """
        g_all = g_view.reshape(1, -1)  # [1,V*d]
        logits = self.router(g_all).squeeze(0)  # [Kr]
        return F.softmax(logits, dim=0)

    # ---- A4: alignment basis + aligned evidence ----
    def compute_Bx(self, pi: Tensor) -> Tensor:
        # B_x = sum_k pi_k * B_k  -> [da,d]
        return torch.einsum("k,kad->ad", pi, self.B_basis)

    def align(self, h_raw: Tensor, B_x: Tensor) -> Tensor:
        # h_raw: [V,T,d], B_x:[da,d] -> h_aligned:[V,T,da]
        return torch.einsum("vtd,ad->vta", h_raw, B_x)

    # ---- A6: corr/rho/gate + fusion ----
    def corr_rho_gate(self, g_view: Tensor, theta: float) -> Tuple[Tensor, float, bool]:
        corr = _safe_pearson_corr(g_view)  # [V,V]
        rho_t = _rho_from_corr(corr)       # scalar tensor
        rho = float(rho_t.detach().cpu().item())
        gate = (rho >= float(theta))
        return corr, rho, gate

    def fuse(self, h_aligned: Tensor, alpha: Tensor, gate: bool) -> Tuple[Tensor, Tensor]:
        """
        h_aligned: [V,T,da]
        returns Z:[da], H:[T,da]
        """
        if self.cfg.V == 1:
            H = h_aligned[0]
        else:
            if gate:
                # weighted fusion
                w = alpha.view(self.cfg.V, 1, 1)
                H = (w * h_aligned).sum(dim=0)  # [T,da]
            else:
                # distrust cross-view: pick best view
                v_star = int(torch.argmax(alpha).detach().cpu().item())
                H = h_aligned[v_star]  # [T,da]
        Z = H.mean(dim=0)  # [da]
        return Z, H

    # ---- forward ----
    def forward(self, E: Sequence[Any], t0: float, delta: float) -> Step1Outputs:
        if len(E) != self.cfg.V:
            raise ValueError(f"Expected E length {self.cfg.V}, got {len(E)}")

        device = next(self.parameters()).device

        # A1/A2 Agg
        x_win = []
        for v in range(self.cfg.V):
            x_v = self.agg_windows(E[v], self.cfg.T, self.cfg.d_in[v], device=device)
            x_win.append(x_v)
        # stack to [V,T,d_in_v] is ragged; keep list

        # A3 encode each view -> h_raw[v]:[T,d]
        h_raw_list = []
        g_list = []
        for v in range(self.cfg.V):
            h_v = self.encode_view(x_win[v], v)  # [T,d]
            h_raw_list.append(h_v)
            g_list.append(self.pool_view(h_v))   # [d]
        h_raw = torch.stack(h_raw_list, dim=0)   # [V,T,d]
        g_view = torch.stack(g_list, dim=0)      # [V,d]

        # A5 quality + alpha
        q_cov_list, q_val_list, q_cmp_list, q_unq_list, q_stb_list = [], [], [], [], []
        for v in range(self.cfg.V):
            q_cov, q_val, q_cmp, q_unq, q_stb = self.quality_scores_from_xwin(x_win[v])
            q_cov_list.append(q_cov)
            q_val_list.append(q_val)
            q_cmp_list.append(q_cmp)
            q_unq_list.append(q_unq)
            q_stb_list.append(q_stb)
        q_cov_t = torch.stack(q_cov_list)  # [V]
        q_val_t = torch.stack(q_val_list)
        q_cmp_t = torch.stack(q_cmp_list)
        q_unq_t = torch.stack(q_unq_list)
        q_stb_t = torch.stack(q_stb_list)

        Q, alpha = self.compute_alpha(q_cov_t, q_val_t, q_cmp_t, q_unq_t, q_stb_t)  # [V],[V]

        # A4 routing + alignment
        pi = self.compute_pi(g_view)      # [Kr]
        B_x = self.compute_Bx(pi)         # [da,d]
        h_aligned = self.align(h_raw, B_x)  # [V,T,da]

        # A6 corr/rho/gate + fusion
        corr_mat, rho, gate = self.corr_rho_gate(g_view, self.cfg.theta)
        Z, H = self.fuse(h_aligned, alpha, gate)

        metrics = Step1Metrics(
            q_cov=q_cov_t.detach(),
            q_val=q_val_t.detach(),
            q_cmp=q_cmp_t.detach(),
            q_unq=q_unq_t.detach(),
            q_stb=q_stb_t.detach(),
            Q=Q.detach(),
            g_view=g_view.detach(),
            corr_mat=corr_mat.detach(),
        )

        out = Step1Outputs(
            h_aligned=h_aligned,
            alpha=alpha,
            pi=pi,
            B_x=B_x,
            Z=Z,
            H=H,
            gate=gate,
            rho=rho,
            metrics=metrics,
        )
        out.validate(self.cfg)
        return out
