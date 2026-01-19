from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import Step1Config, Step1Metrics, Step1Outputs
from .data import HashingAggConfig, quality_components, window_and_agg
from .running_stats import RunningMeanStd
from .utils import rho_max_pair, safe_pearson_corr

Tensor = torch.Tensor


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        h = int(hidden) if hidden is not None else max(d_in, d_out)
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _as_float_tensor(x: Any, device: torch.device) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, device=device, dtype=torch.float32)


@dataclass
class ViewAggSpec:
    agg_cfg: HashingAggConfig
    key_fields: Optional[List[str]] = None
    dedup_fields: Optional[List[str]] = None


class Step1Model(nn.Module):
    """Step1 (Module A/B/C) implementation aligned to method final.

    Inputs:
        E: list of V views. Each view can be either:
           1) list of raw events (dicts/tuples) with timestamps, OR
           2) a pre-windowized tensor/array with shape [T, d_in[v]]
        t0: window start timestamp
        delta: window width (seconds)

    Outputs:
        Step1Outputs dataclass in contracts.py
    """

    def __init__(self, cfg: Step1Config, view_specs: Optional[List[ViewAggSpec]] = None):
        super().__init__()
        self.cfg = cfg

        if len(cfg.d_in) != cfg.V:
            raise ValueError(f"cfg.d_in must have length V={cfg.V}")

        # per-view deterministic agg spec (A2 + quality stats fields)
        if view_specs is None:
            view_specs = [ViewAggSpec(agg_cfg=HashingAggConfig(dim=cfg.d_in[v])) for v in range(cfg.V)]
        if len(view_specs) != cfg.V:
            raise ValueError(f"view_specs must have length V={cfg.V}")
        self.view_specs = view_specs

        # A3: per-view encoder phi_v: x_win[v,t] -> h_raw[v,t]
        self.encoders = nn.ModuleList([MLP(cfg.d_in[v], cfg.d) for v in range(cfg.V)])

        # A5-6: quality fusion parameters (w_q) and running stats for normalization
        self.q_head = nn.Linear(5, 1, bias=True)
        with torch.no_grad():
            self.q_head.weight.fill_(1.0 / 5.0)
            self.q_head.bias.zero_()
        self.q_rms = RunningMeanStd(dim=cfg.V, momentum=cfg.q_norm_momentum, eps=cfg.q_norm_eps)

        # B1: router is a single linear layer W_r (method final)
        self.router = nn.Linear(cfg.V * cfg.d, cfg.Kr, bias=True)

        # B2: alignment bases B_k
        self.B_basis = nn.Parameter(torch.randn(cfg.Kr, cfg.da, cfg.d) * 0.02)

        # C3: one-shot attention projections (shared W_a)
        self.W_a = nn.Linear(cfg.da, cfg.da, bias=False)

    # ---------- A1+A2: windowing + deterministic aggregation ----------
    def _windowize_and_agg_view(self, E_v: Any, v: int, t0: float, delta: float, device: torch.device) -> Tuple[Tensor, Dict[str, Any]]:
        """Return x_win [T, d_in[v]] plus debug stats dict."""
        # pre-windowized shortcut (for aligned dataset artifacts)
        if isinstance(E_v, torch.Tensor) or isinstance(E_v, (list, tuple)) and len(E_v) > 0 and not isinstance(E_v[0], (dict, tuple, list)):
            # the above heuristic is weak; use shape check below
            pass

        try:
            x = _as_float_tensor(E_v, device=device)
            if x.ndim == 2 and tuple(x.shape) == (self.cfg.T, self.cfg.d_in[v]):
                # no event-level stats; return minimal
                debug = {"pre_windowized": True, "total_events": None}
                return x, debug
        except Exception:
            # treat as raw events
            pass

        spec = self.view_specs[v]
        res = window_and_agg(
            events=E_v if isinstance(E_v, (list, tuple)) else [],
            t0=t0,
            delta=delta,
            T=self.cfg.T,
            agg_cfg=spec.agg_cfg,
            key_fields=spec.key_fields,
            dedup_fields=spec.dedup_fields,
            device=device,
        )
        debug = {
            "pre_windowized": False,
            "total_events": res.stats.total_events,
            "parse_success": res.stats.parse_success,
            "keyfield_nonempty": res.stats.keyfield_nonempty,
            "keyfield_total": res.stats.keyfield_total,
            "unique_events": res.stats.unique_events,
            "duplicate_events": res.stats.duplicate_events,
            "window_counts": res.stats.window_counts,
        }
        return res.x_win, debug

    # ---------- A3: encode ----------
    def _encode_view(self, x_win_v: Tensor, v: int) -> Tensor:
        return self.encoders[v](x_win_v)  # [T,d]

    def _pool_over_windows(self, h_v: Tensor) -> Tensor:
        return h_v.mean(dim=0)  # [d]

    # ---------- A5: quality components -> Q_v ----------
    def _compute_quality(self, window_debug: Dict[str, Any], windows_x: Tensor, eps: float = 1e-8) -> Tuple[float, float, float, float, float]:
        if window_debug.get("pre_windowized", False):
            # fallback quality from window vectors only; recommend using raw-event stats for strict audit
            mass = windows_x.abs().sum(dim=1)
            empty = (mass <= eps).float().mean().item()
            q_cov = 1.0 - float(empty)
            finite = torch.isfinite(windows_x)
            q_val = float(finite.all(dim=1).float().mean().item())
            q_cmp = float(finite.float().mean().item())
            # uniqueness: unique windows / total windows
            uniq = int(torch.unique(windows_x, dim=0).shape[0])
            q_unq = float(uniq) / float(max(1, self.cfg.T))
            # stability from per-window mass CV
            mu = float(mass.mean().item())
            sd = float(mass.std(unbiased=False).item())
            cv = sd / (mu + eps)
            q_stb = 1.0 / (1.0 + cv)
            return q_cov, q_val, q_cmp, min(1.0, max(0.0, q_unq)), min(1.0, max(0.0, q_stb))

        # strict audit metrics from raw stats (exactly as method A5)
        T = int(self.cfg.T)
        counts = window_debug.get("window_counts") or [0] * T
        empty = sum(1 for c in counts if int(c) <= 0)
        q_cov = 1.0 - float(empty) / float(max(T, 1))

        total_events = int(window_debug.get("total_events") or 0)
        parse_success = int(window_debug.get("parse_success") or 0)
        key_nonempty = int(window_debug.get("keyfield_nonempty") or 0)
        key_total = int(window_debug.get("keyfield_total") or 0)
        dup = int(window_debug.get("duplicate_events") or 0)

        if total_events <= 0:
            q_val = 0.0
            q_cmp = 0.0
            q_unq = 1.0
        else:
            q_val = float(parse_success) / float(total_events)
            q_cmp = 1.0 if key_total <= 0 else (float(key_nonempty) / float(key_total))
            q_unq = 1.0 - (float(dup) / float(total_events))

        import numpy as _np
        arr = _np.asarray(counts, dtype=_np.float32)
        mu = float(arr.mean())
        sd = float(arr.std())
        cv = sd / (mu + eps)
        q_stb = 1.0 / (1.0 + cv)

        def _clamp01(v: float) -> float:
            return float(min(1.0, max(0.0, v)))

        return _clamp01(q_cov), _clamp01(q_val), _clamp01(q_cmp), _clamp01(q_unq), _clamp01(q_stb)

    @torch.no_grad()
    def _update_quality_stats(self, Q: Tensor) -> None:
        # RunningMeanStd expects [B,V] or [V]
        self.q_rms.update(Q.detach())

    def _compute_alpha(self, q_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """q_vec: [V,5] -> Q:[V], Q_hat:[V], alpha:[V]."""
        Q = self.q_head(q_vec).squeeze(1)  # [V]
        # A5-6: normalize by running stats (training-time); if uninitialized, Q_hat=Q
        Q_hat = self.q_rms.normalize(Q)
        tau_q = max(float(self.cfg.tau_q), 1e-6)
        alpha = F.softmax(Q_hat / tau_q, dim=0)
        return Q, Q_hat, alpha

    # ---------- B1/B2: routing + alignment ----------
    def _compute_pi(self, g_view_weighted: Tensor) -> Tensor:
        logits = self.router(g_view_weighted.reshape(1, -1)).squeeze(0)  # [Kr]
        return F.softmax(logits, dim=0)

    def _compute_Bx(self, pi: Tensor) -> Tensor:
        # B_x = sum_k pi_k B_k  -> [da,d]
        return torch.einsum("k,kad->ad", pi, self.B_basis)

    def _align(self, h_raw: Tensor, B_x: Tensor) -> Tensor:
        # h_raw: [V,T,d], B_x:[da,d] -> h_aligned:[V,T,da]
        return torch.einsum("vtd,ad->vta", h_raw, B_x)

    # ---------- C1/C3: correlation + one-shot attention ----------
    def _corr_rho_gate(self, z_view: Tensor) -> Tuple[Tensor, Tensor, bool]:
        corr = safe_pearson_corr(z_view)  # [V,V]
        rho = rho_max_pair(corr)  # []
        gate = bool((rho >= float(self.cfg.theta)).detach().cpu().item())
        return corr, rho, gate

    def _one_shot_attention(self, z_view: Tensor, h_aligned: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute cross-view attention weights from z_view and inject into both z_view and h_aligned.

        Returns:
            z_hat: [V,da]
            h_hat: [V,T,da]
        """
        V, da = z_view.shape
        # project
        q = self.W_a(z_view)  # [V,da]
        k = self.W_a(z_view)  # [V,da]
        scale = float(self.cfg.attn_scale) if self.cfg.attn_scale is not None else (da ** 0.5)
        scores = (q @ k.t()) / max(scale, 1e-6)  # [V,V]
        # do not attend to self
        scores = scores - torch.diag(torch.full((V,), 1e9, device=z_view.device, dtype=z_view.dtype))
        w = F.softmax(scores, dim=1)  # [V,V]
        z_hat = w @ z_view  # [V,da]
        # inject into per-window evidence
        h_hat = torch.einsum("vw,wtd->vtd", w, h_aligned)  # [V,T,da]
        return z_hat, h_hat

    def forward(self, E: Sequence[Any], t0: float, delta: float) -> Step1Outputs:
        if len(E) != self.cfg.V:
            raise ValueError(f"Expected E length V={self.cfg.V}, got {len(E)}")
        device = next(self.parameters()).device

        # A1+A2
        x_win_list: List[Tensor] = []
        debug_list: List[Dict[str, Any]] = []
        for v in range(self.cfg.V):
            x_win_v, dbg = self._windowize_and_agg_view(E[v], v=v, t0=float(t0), delta=float(delta), device=device)
            if tuple(x_win_v.shape) != (self.cfg.T, self.cfg.d_in[v]):
                raise ValueError(f"View {v} x_win shape mismatch: expected {(self.cfg.T, self.cfg.d_in[v])}, got {tuple(x_win_v.shape)}")
            x_win_list.append(x_win_v)
            debug_list.append(dbg)

        # A3 encode + A4 pool
        h_raw_list: List[Tensor] = []
        g_list: List[Tensor] = []
        for v in range(self.cfg.V):
            h_v = self._encode_view(x_win_list[v], v=v)  # [T,d]
            h_raw_list.append(h_v)
            g_list.append(self._pool_over_windows(h_v))
        h_raw = torch.stack(h_raw_list, dim=0)  # [V,T,d]
        g_view = torch.stack(g_list, dim=0)     # [V,d]

        # A5 quality components
        q_rows: List[List[float]] = []
        for v in range(self.cfg.V):
            q_cov, q_val, q_cmp, q_unq, q_stb = self._compute_quality(debug_list[v], x_win_list[v])
            q_rows.append([q_cov, q_val, q_cmp, q_unq, q_stb])
        q_vec = torch.tensor(q_rows, device=device, dtype=torch.float32)  # [V,5]

        # A5-6 fuse quality + normalize + alpha
        Q, Q_hat, alpha = self._compute_alpha(q_vec)
        if self.training:
            self._update_quality_stats(Q)

        # A6 reliability injection into router input
        g_bar = g_view * alpha.view(self.cfg.V, 1)

        # B1 routing pi
        pi = self._compute_pi(g_bar)

        # B2/B3 alignment
        B_x = self._compute_Bx(pi)
        h_aligned = self._align(h_raw, B_x)

        # B4: aligned view vectors z_v
        z_view = h_aligned.mean(dim=1)  # [V,da]

        # C1 corr + gate
        corr_mat, rho_t, gate = self._corr_rho_gate(z_view)

        # C2 baseline fusion
        z_use = z_view
        h_use = h_aligned

        # C3 attention-enhanced if gate
        if gate and self.cfg.V > 1:
            z_hat, h_hat = self._one_shot_attention(z_view=z_view, h_aligned=h_aligned)
            g = 1.0
            z_use = g * z_hat + (1 - g) * z_view
            h_use = g * h_hat + (1 - g) * h_aligned

        # final fusion (always weighted by alpha)
        Z = torch.einsum("v,vd->d", alpha, z_use)
        H = torch.einsum("v,vtd->td", alpha, h_use)

        metrics = Step1Metrics(
            q_cov=q_vec[:, 0].detach(),
            q_val=q_vec[:, 1].detach(),
            q_cmp=q_vec[:, 2].detach(),
            q_unq=q_vec[:, 3].detach(),
            q_stb=q_vec[:, 4].detach(),
            Q=Q.detach(),
            Q_hat=Q_hat.detach(),
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
            rho=float(rho_t.detach().cpu().item()),
            metrics=metrics,
        )
        out.validate(self.cfg)
        return out
