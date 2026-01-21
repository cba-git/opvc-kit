"""opvc.step2

Step2: Privacy-aware federated pretraining (URAS + ASD + AT-InfoNCE + DP + SecureAgg + FedAvg)

This module provides:
- Student / Teacher models for URAS representation
- Deterministic behavior feature extraction b(x) (client-side) with DP sanitize (A2)
- Supervised teacher pretraining hook (A3)
- Federated simulation training entry (single-machine) implementing:
  - ASD + AT-InfoNCE losses
  - view-wise adaptive clip/noise
  - redundancy projection
  - secure aggregation simulation
  - FedAvg-style global update

Note: This code focuses on *method-faithful implementation* and clean interfaces,
not on performance tuning.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import Step2Config, Step2Outputs, Step1Outputs
from .running_stats import RunningMeanStd
from .step2_losses import (
    alpha_confidence,
    asd_loss,
    dynamic_temperature,
    info_nce_distill,
    mi_risk_estimate_from_logits,
    pi_uncertainty,
    risk_signal,
    utility_signal,
    dp_sigma,
)
from .utils import (
    safe_pearson_corr,
    state_dict_add_,
    state_dict_clip_by_l2,
    state_dict_sub,
    state_dict_scale,
    state_dict_add_noise,
    secure_agg_sum,
)

Tensor = torch.Tensor


class _MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: Optional[int] = None):
        super().__init__()
        h = int(hidden) if hidden is not None else max(d_in, d_out, 32)
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, d_out))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BehaviorFeatureExtractor(nn.Module):
    """Deterministic, auditable behavior feature b(x).

    b(x) is designed to be *non-invertible* and does not expose raw events.

    Default definition (can be replaced without touching downstream code):
    - per-view block (len=6): [alpha_v, q_cov_v, q_val_v, q_cmp_v, q_unq_v, q_stb_v]
    - global block: [pi (Kr), rho, gate]
    Total dim: 6V + Kr + 2

    DP sanitize can be applied per-view block (view-wise).
    """

    def __init__(self, V: int, Kr: int):
        super().__init__()
        self.V = int(V)
        self.Kr = int(Kr)
        self.view_block_dim = 6
        self.global_dim = self.Kr + 2
        self.dim = self.V * self.view_block_dim + self.global_dim

    def forward(self, step1_out: Step1Outputs) -> Tensor:
        a = step1_out.alpha.detach()
        pi = step1_out.pi.detach()
        rho = torch.tensor(float(step1_out.rho or 0.0), device=a.device, dtype=a.dtype)
        gate = torch.tensor(float(1.0 if step1_out.gate else 0.0), device=a.device, dtype=a.dtype)

        # quality components
        q_cov = step1_out.metrics.q_cov if step1_out.metrics.q_cov is not None else torch.zeros_like(a)
        q_val = step1_out.metrics.q_val if step1_out.metrics.q_val is not None else torch.zeros_like(a)
        q_cmp = step1_out.metrics.q_cmp if step1_out.metrics.q_cmp is not None else torch.zeros_like(a)
        q_unq = step1_out.metrics.q_unq if step1_out.metrics.q_unq is not None else torch.zeros_like(a)
        q_stb = step1_out.metrics.q_stb if step1_out.metrics.q_stb is not None else torch.zeros_like(a)

        # [V,6]
        vblk = torch.stack([a, q_cov, q_val, q_cmp, q_unq, q_stb], dim=1)
        gblk = torch.cat([pi, rho.view(1), gate.view(1)], dim=0)
        return torch.cat([vblk.reshape(-1), gblk], dim=0)

    def split_view_global(self, b: Tensor) -> Tuple[Tensor, Tensor]:
        """Return b_view_blocks [V,view_block_dim], b_global [global_dim]."""
        b = b.view(-1)
        v = b[: self.V * self.view_block_dim].reshape(self.V, self.view_block_dim)
        g = b[self.V * self.view_block_dim :]
        return v, g


@torch.no_grad()
def dp_sanitize_behavior(
    b: Tensor,
    extractor: BehaviorFeatureExtractor,
    clip_v: Tensor,
    sigma_v: Tensor,
    clip_global: float,
    sigma_global: float,
) -> Tensor:
    """DP sanitize b(x) with view-wise clip/noise."""
    device = b.device
    b_view, b_global = extractor.split_view_global(b)

    # per-view blocks
    out_blocks: List[Tensor] = []
    for v in range(extractor.V):
        x = b_view[v]
        norm = torch.linalg.vector_norm(x, ord=2).clamp_min(1e-12)
        scale = min(1.0, float(clip_v[v].item()) / float(norm.item()))
        x_clip = x * scale
        x_noisy = x_clip + torch.randn_like(x_clip) * (float(sigma_v[v].item()) * float(clip_v[v].item()))
        out_blocks.append(x_noisy)
    b_view_s = torch.stack(out_blocks, dim=0).reshape(-1)

    # global
    ng = torch.linalg.vector_norm(b_global, ord=2).clamp_min(1e-12)
    scaleg = min(1.0, float(clip_global) / float(ng.item()))
    bg = b_global * scaleg
    bg = bg + torch.randn_like(bg) * (float(sigma_global) * float(clip_global))

    return torch.cat([b_view_s, bg], dim=0)


class Step2Teacher(nn.Module):
    def __init__(self, db: int, dz: int, cfg: Step2Config, hidden: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.backbone = _MLP(db, dz, hidden=hidden)
        self.heads = nn.ModuleList([nn.Linear(dz, cfg.du) for _ in range(cfg.Kr)])
        self.cls_head = nn.Linear(dz, cfg.Ka) if cfg.Ka and cfg.Ka > 0 else None

    def forward(self, b: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        z = self.backbone(b)  # [B,dz]
        cls = self.cls_head(z) if self.cls_head is not None else None
        return z, cls

    def forward_uras(self, b: Tensor, pi: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        z, cls = self.forward(b)
        u_list = [h(z) for h in self.heads]  # Kr*[B,du]
        # weight and concat
        if pi.ndim == 1:
            pi_b = pi.view(1, -1).expand(z.shape[0], -1)
        else:
            pi_b = pi
        pi_b = pi_b / pi_b.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        parts = [pi_b[:, k:k+1] * u_list[k] for k in range(self.cfg.Kr)]
        U = torch.cat(parts, dim=-1)  # [B,Kr*du]
        return U, cls


class Step2Student(nn.Module):
    def __init__(self, da: int, cfg: Step2Config, hidden: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.da = int(da)
        self.backbone = _MLP(self.da, self.da, hidden=hidden)

        # view adapters (enable view-wise update + DP/projection)
        self.view_adapters = nn.ModuleList([nn.Linear(self.da, self.da, bias=False) for _ in range(cfg.num_clients if False else 1)])
        # we will create real adapters dynamically for V via reset_view_adapters(V)
        self.view_adapters = nn.ModuleList()

        self.heads = nn.ModuleList([nn.Linear(self.da, cfg.du) for _ in range(cfg.Kr)])

    def reset_view_adapters(self, V: int) -> None:
        self.view_adapters = nn.ModuleList([nn.Linear(self.da, self.da, bias=False) for _ in range(int(V))])
        # init as identity (stable)
        for lin in self.view_adapters:
            with torch.no_grad():
                if lin.weight.shape[0] == lin.weight.shape[1]:
                    lin.weight.copy_(torch.eye(lin.weight.shape[0]))

    def forward_uras_from_step1(self, step1_out: Step1Outputs) -> Tensor:
        # derive z_view from step1 evidence
        h_aligned = step1_out.h_aligned  # [V,T,da]
        z_view = h_aligned.mean(dim=1)   # [V,da]
        alpha = step1_out.alpha          # [V]
        pi = step1_out.pi                # [Kr]
        V = int(z_view.shape[0])
        if len(self.view_adapters) != V:
            self.reset_view_adapters(V)

        z_adapt = torch.stack([self.view_adapters[v](z_view[v]) for v in range(V)], dim=0)  # [V,da]
        Z = torch.einsum("v,vd->d", alpha, z_adapt)  # [da]

        return self.forward_uras(Z, pi)

    def forward_uras_from_step1_single_view(self, step1_out: Step1Outputs, v: int) -> Tensor:
        """URAS representation contributed by a single view v.

        This is used for cfg.view_grad_mode == "exact" to perform per-view local updates.
        """
        h_aligned = step1_out.h_aligned  # [V,T,da]
        z_view = h_aligned.mean(dim=1)   # [V,da]
        alpha = step1_out.alpha          # [V]
        pi = step1_out.pi                # [Kr]

        V = int(z_view.shape[0])
        if len(self.view_adapters) != V:
            self.reset_view_adapters(V)

        vv = int(v)
        z_adapt_v = self.view_adapters[vv](z_view[vv])  # [da]
        Z = alpha[vv] * z_adapt_v  # [da]
        return self.forward_uras(Z, pi)

    def forward_uras(self, Z: Tensor, pi: Tensor) -> Tensor:
        if Z.ndim == 1:
            Zb = Z.view(1, -1)
            squeeze = True
        else:
            Zb = Z
            squeeze = False
        h = self.backbone(Zb)  # [B,da]
        u_list = [head(h) for head in self.heads]  # Kr*[B,du]
        if pi.ndim == 1:
            pi_b = pi.view(1, -1).expand(h.shape[0], -1)
        else:
            pi_b = pi
        pi_b = pi_b / pi_b.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        parts = [pi_b[:, k:k+1] * u_list[k] for k in range(self.cfg.Kr)]
        U = torch.cat(parts, dim=-1)
        return U.squeeze(0) if squeeze else U


def pretrain_teacher_supervised(
    teacher: Step2Teacher,
    loader: Iterable[Tuple[Tensor, Tensor]],
    epochs: int,
    lr: float,
    device: torch.device,
) -> Dict[str, float]:
    """Supervised teacher pretraining on offline dataset D_T = {(b,y)}."""
    if teacher.cls_head is None:
        raise ValueError("Teacher has no cls_head: set cfg.Ka>0 for supervised pretrain")
    teacher.train()
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    log: Dict[str, float] = {}
    step = 0
    for ep in range(int(epochs)):
        losses = []
        for b, y in loader:
            b = b.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            z, logits = teacher.forward(b)
            assert logits is not None
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            step += 1
        log[f"teacher_epoch_{ep}_loss"] = float(sum(losses) / max(len(losses), 1))
    teacher.eval()
    return log


def _split_indices(N: int, num_clients: int, seed: int) -> List[Tensor]:
    g = torch.Generator()
    g.manual_seed(int(seed))
    idx = torch.randperm(N, generator=g)
    return list(torch.chunk(idx, chunks=int(num_clients)))


def _extract_view_updates(delta: Mapping[str, Tensor], V: int) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]:
    """Split delta state_dict into per-view adapter updates + shared updates."""
    view_updates: List[Dict[str, Tensor]] = [{} for _ in range(V)]
    shared: Dict[str, Tensor] = {}
    for k, v in delta.items():
        if k.startswith("view_adapters."):
            # key like view_adapters.3.weight
            parts = k.split(".")
            vid = int(parts[1])
            view_updates[vid][k] = v
        else:
            shared[k] = v
    return view_updates, shared


def _redundancy_projection(
    view_updates: List[Dict[str, Tensor]],
    g_bar: Tensor,
    proj_momentum: float,
    proj_temp: float,
    memory: Optional[List[Dict[str, Tensor]]],
) -> Tuple[List[Dict[str, Tensor]], List[Dict[str, Tensor]]]:
    """Apply redundancy projection across view updates using EMA memory m_v."""
    V = len(view_updates)
    if V <= 1:
        return view_updates, memory or view_updates

    corr = safe_pearson_corr(g_bar)  # [V,V]
    # zero out diagonal for weights
    corr = corr - torch.diag(torch.diag(corr))
    W = torch.softmax(float(proj_temp) * corr, dim=1)  # [V,V]

    if memory is None:
        memory = [{k: v.detach().clone() for k, v in upd.items()} for upd in view_updates]
    # update memory
    new_mem: List[Dict[str, Tensor]] = []
    for v in range(V):
        m_v = memory[v]
        upd_v = view_updates[v]
        merged = {}
        for k, dv in upd_v.items():
            prev = m_v.get(k, torch.zeros_like(dv))
            merged[k] = float(proj_momentum) * prev + (1 - float(proj_momentum)) * dv.detach()
        new_mem.append(merged)

    # projected updates: Δ'_v = Δ_v - sum_{v'!=v} w_{vv'} m_{v'}
    proj_updates: List[Dict[str, Tensor]] = []
    for v in range(V):
        red_dir: Dict[str, Tensor] = {}
        for vp in range(V):
            if vp == v:
                continue
            w = float(W[v, vp].detach().cpu().item())
            state_dict_add_(red_dir, new_mem[vp], alpha=w)
        dv = view_updates[v]
        dvp = {k: dv[k] - red_dir.get(k, torch.zeros_like(dv[k])) for k in dv.keys()}
        proj_updates.append(dvp)

    return proj_updates, new_mem


def train_step2_federated(
    cfg: Step2Config,
    step1_outs: Sequence[Step1Outputs],
    sensitivity_coeff: Optional[Sequence[float]] = None,
    offline_teacher_loader: Optional[Iterable[Tuple[Tensor, Tensor]]] = None,
    teacher_epochs: int = 0,
    teacher_lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Single-machine FL simulation for Step2.

    Returns:
        theta_pkg: dict suitable for torch.save (contains cfg/state_dict/teacher/proj_memory stats)
        logs: training logs
    """

    dev = torch.device(device)
    torch.manual_seed(int(seed))

    if not step1_outs:
        raise ValueError("step1_outs is empty")

    # derive dimensions
    V = int(step1_outs[0].alpha.numel())
    da = int(step1_outs[0].Z.numel())
    if sensitivity_coeff is None:
        sensitivity_coeff = [1.0] * V
    s_v = torch.tensor(list(sensitivity_coeff), device=dev, dtype=torch.float32).view(1, V)

    extractor = BehaviorFeatureExtractor(V=V, Kr=cfg.Kr)
    db = extractor.dim

    student_global = Step2Student(da=da, cfg=cfg).to(dev)
    student_global.reset_view_adapters(V)
    teacher = Step2Teacher(db=db, dz=da, cfg=cfg).to(dev)

    logs: Dict[str, Any] = {"cfg": asdict(cfg), "db": db, "da": da, "V": V}

    # optional supervised teacher pretrain
    if offline_teacher_loader is not None and teacher_epochs > 0 and (cfg.Ka and cfg.Ka > 0):
        logs["teacher_pretrain"] = pretrain_teacher_supervised(
            teacher=teacher,
            loader=offline_teacher_loader,
            epochs=teacher_epochs,
            lr=teacher_lr,
            device=dev,
        )

    # running client baselines for util/risk (used by ATC later)
    rms_util = RunningMeanStd(dim=1, momentum=0.9).to(dev)
    rms_risk = RunningMeanStd(dim=1, momentum=0.9).to(dev)

    # redundancy memory
    proj_memory: Optional[List[Dict[str, Tensor]]] = None

    N = len(step1_outs)
    shards = _split_by_host(step1_outs, num_clients=cfg.num_clients, seed=seed, host_path="meta.host")

    for r in range(int(cfg.rounds)):
        client_deltas: List[Dict[str, Tensor]] = []
        round_logs: List[Dict[str, float]] = []

        for cid, idx in enumerate(shards):

            # [AUTO] _split_by_host returns List[List[int]]; normalize idx to torch.Tensor

            if not torch.is_tensor(idx):

                idx = torch.tensor(idx, device=dev, dtype=torch.long)
            if idx.numel() == 0:
                continue
            # local copy
            student = Step2Student(da=da, cfg=cfg).to(dev)
            student.reset_view_adapters(V)
            student.load_state_dict(student_global.state_dict(), strict=True)
            student.train()

            opt = torch.optim.Adam(student.parameters(), lr=float(cfg.lr))
            # client-local stats
            alphas = torch.stack([step1_outs[int(i)].alpha.to(dev) for i in idx.tolist()], dim=0)  # [B,V]
            g_view = torch.stack([step1_outs[int(i)].metrics.g_view.to(dev) if step1_outs[int(i)].metrics.g_view is not None else torch.zeros(V, da, device=dev) for i in idx.tolist()], dim=0)
            # g_view: [B,V,d]
            g_bar = torch.einsum("bv,bvd->vd", alphas, g_view) / float(max(alphas.shape[0], 1))

            # importance i_v
            avg_alpha = alphas.mean(dim=0)  # [V]
            imp = avg_alpha / avg_alpha.max().clamp_min(1e-12)  # [V] in [0,1]

            # adaptive clip/noise (view-wise) - will refine with util/risk below
            clip_v = (cfg.clip_min + (cfg.clip_max - cfg.clip_min) * imp).detach()
            sigma_v = torch.full((V,), float(cfg.sigma_b0), device=dev)

            # local training
            losses = []
            # in "exact" mode we accumulate telescoping per-view deltas so that sum_v Δ_v == Δ_total
            per_view_delta_acc: Optional[List[Dict[str, Tensor]]] = None
            per_view_shared_acc: Optional[List[Dict[str, Tensor]]] = None
            for ep in range(int(cfg.local_epochs)):
                for i in idx.tolist():
                    s1 = step1_outs[int(i)]
                    # behavior feature + DP sanitize
                    b = extractor(s1).to(dev)
                    # temporary placeholders for sigma_v (will update using current util/risk signals)
                    b_s = dp_sanitize_behavior(
                        b=b,
                        extractor=extractor,
                        clip_v=clip_v,
                        sigma_v=sigma_v,
                        clip_global=float(cfg.Cb),
                        sigma_global=float(cfg.sigma_b0),
                    )

                    # teacher URAS on b_s
                    U_t, cls_logits = teacher.forward_uras(b_s.view(1, -1), s1.pi.to(dev).view(1, -1))
                    # student URAS (from Step1)
                    U_s = student.forward_uras_from_step1(s1).view(1, -1)

                    # signals
                    conf = alpha_confidence(s1.alpha.to(dev)).view(1)
                    dist_err = torch.linalg.vector_norm(U_s - U_t, dim=-1)  # [1]
                    util = utility_signal(dist_err, conf)
                    mi_est = mi_risk_estimate_from_logits(cls_logits) if cls_logits is not None else torch.zeros_like(util)
                    sens = (s_v * s1.alpha.to(dev).view(1, -1)).sum(dim=1)
                    risk = risk_signal(sens, mi_est)

                    tau = dynamic_temperature(
                        util=util,
                        risk=risk,
                        round_idx=r,
                        total_rounds=int(cfg.rounds),
                        tau_min=float(cfg.tau_min),
                        tau_max=float(cfg.tau_max),
                        kappa_u=float(cfg.kappa_u),
                        kappa_p=float(cfg.kappa_p),
                    )

                    # update sigma_v using view-weighted util/risk
                    u_v = (s1.alpha.to(dev) * util.view(-1)).detach()
                    r_v = (s1.alpha.to(dev) * risk.view(-1)).detach()
                    sigma_v = dp_sigma(
                        base_sigma=float(cfg.sigma_b0),
                        util=u_v,
                        risk=r_v,
                        sigma_min=float(cfg.sigma_min),
                        sigma_max=float(cfg.sigma_max),
                    )

                    if str(getattr(cfg, "view_grad_mode", "approx")) == "exact":
                        # initialize accumulators once we know V
                        if per_view_delta_acc is None:
                            per_view_delta_acc = [dict() for _ in range(V)]
                            per_view_shared_acc = [dict() for _ in range(V)]

                        # per-view local updates (telescoping): for each v, step once and record delta_v
                        for v in range(V):
                            before = {k: p.detach().clone() for k, p in student.state_dict().items()}
                            U_s_v = student.forward_uras_from_step1_single_view(s1, v=v).view(1, -1)
                            L_asd_v = asd_loss(U_s_v, U_t, lambda_sample=1.0, lambda_mean=1.0)
                            L_nce_v = info_nce_distill(U_s_v, U_t, tau=tau)
                            loss_v = float(cfg.lambda_asd) * L_asd_v + float(cfg.lambda_nce) * L_nce_v

                            opt.zero_grad()
                            loss_v.backward()
                            opt.step()

                            after = student.state_dict()
                            dv_full = state_dict_sub(after, before)
                            # split dv_full into the single-view adapter update for v, and shared update
                            v_upds_step, shared_step = _extract_view_updates(dv_full, V=V)
                            # only keep the adapter update belonging to v
                            if per_view_delta_acc is not None:
                                state_dict_add_(per_view_delta_acc[v], v_upds_step[v], alpha=1.0)
                            if per_view_shared_acc is not None:
                                state_dict_add_(per_view_shared_acc[v], shared_step, alpha=1.0)

                            losses.append(float(loss_v.detach().cpu().item()))
                    else:
                        # fast path (single backward)
                        L_asd = asd_loss(U_s, U_t, lambda_sample=1.0, lambda_mean=1.0)
                        L_nce = info_nce_distill(U_s, U_t, tau=tau)
                        loss = float(cfg.lambda_asd) * L_asd + float(cfg.lambda_nce) * L_nce

                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        losses.append(float(loss.detach().cpu().item()))

                    # update running baselines
                    rms_util.update(util.detach())
                    rms_risk.update(risk.detach())

            # compute delta(s)
            if str(getattr(cfg, "view_grad_mode", "approx")) == "exact" and per_view_delta_acc is not None and per_view_shared_acc is not None:
                v_upds = per_view_delta_acc
                # shared delta is sum of shared deltas from each per-view step (telescoping)
                shared: Dict[str, Tensor] = {}
                for sh in per_view_shared_acc:
                    state_dict_add_(shared, sh, alpha=1.0)
            else:
                delta = state_dict_sub(student.state_dict(), student_global.state_dict())
                # split into view updates + shared
                v_upds, shared = _extract_view_updates(delta, V=V)

            # redundancy projection across view updates
            v_upds, proj_memory = _redundancy_projection(
                view_updates=v_upds,
                g_bar=g_bar.detach(),
                proj_momentum=float(cfg.proj_momentum),
                proj_temp=float(cfg.proj_temp),
                memory=proj_memory,
            )

            # DP clip + noise on view updates
            v_dp: List[Dict[str, Tensor]] = []
            for v in range(V):
                dv = v_upds[v]
                dv_clip = state_dict_clip_by_l2(dv, clip=float(clip_v[v].item()))
                dv_noisy = state_dict_add_noise(dv_clip, sigma=float(sigma_v[v].item()), clip=float(clip_v[v].item()), device=dev)
                v_dp.append(dv_noisy)

            # DP clip + noise on shared update (global)
            shared_clip = state_dict_clip_by_l2(shared, clip=float(cfg.Cb))
            shared_noisy = state_dict_add_noise(shared_clip, sigma=float(cfg.sigma_b0), clip=float(cfg.Cb), device=dev)

            # merge
            merged = dict(shared_noisy)
            for dv in v_dp:
                merged.update(dv)
            client_deltas.append(merged)

            round_logs.append({"client": float(cid), "loss_mean": float(sum(losses) / max(len(losses), 1))})

        # secure aggregation sum and average
        agg_sum = secure_agg_sum(client_deltas, seed=seed + 9973 * r)
        # average
        agg_avg = state_dict_scale(agg_sum, 1.0 / float(max(len(client_deltas), 1)))

        # FedAvg update: Theta <- Theta + agg_avg
        new_state = dict(student_global.state_dict())
        state_dict_add_(new_state, agg_avg, alpha=1.0)
        student_global.load_state_dict(new_state, strict=True)
        student_global.eval()

        logs[f"round_{r}"] = round_logs

    theta_pkg = {
        "cfg2": asdict(cfg),
        "extractor": {"V": V, "Kr": cfg.Kr, "dim": db},
        "student_state_dict": {k: v.detach().cpu() for k, v in student_global.state_dict().items()},
        "teacher_state_dict": {k: v.detach().cpu() for k, v in teacher.state_dict().items()},
        "proj_memory": [
            {k: v.detach().cpu() for k, v in m.items()} for m in (proj_memory or [])
        ],
        "util_mean": float(rms_util.mean.detach().cpu().item()) if rms_util.mean is not None else 0.0,
        "risk_mean": float(rms_risk.mean.detach().cpu().item()) if rms_risk.mean is not None else 0.0,
    }

    return theta_pkg, logs


def build_step2_outputs(theta_pkg: Dict[str, Any], device: str = "cpu") -> Step2Outputs:
    """Load student from theta_pkg and expose runtime forward_uras callable."""
    cfgd = theta_pkg.get("cfg2", {})
    cfg = Step2Config(**cfgd)  # type: ignore[arg-type]
    # infer da
    sd = theta_pkg["student_state_dict"]
    # pick any weight to infer da
    any_w = next(iter(sd.values()))
    da = int(any_w.shape[0]) if any_w.ndim >= 2 else int(any_w.numel())
    student = Step2Student(da=da, cfg=cfg).to(torch.device(device))
    # reset adapters based on stored extractor V
    V = int(theta_pkg.get("extractor", {}).get("V", 1))
    student.reset_view_adapters(V)
    student.load_state_dict({k: v.to(device) for k, v in sd.items()}, strict=True)
    student.eval()

    def _forward_uras(Z: Tensor, pi: Tensor) -> Tensor:
        return student.forward_uras(Z.to(device), pi.to(device))

    return Step2Outputs(theta_global=sd, forward_uras=_forward_uras)


def _get_by_path(o, path: str):
    """Get value from Step1Outputs/dict by dotted path like 'meta.src'."""
    parts = path.split(".")
    cur = o
    for part in parts:
        if hasattr(cur, part):
            cur = getattr(cur, part)
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _split_by_host(step1_outs, num_clients: int, seed: int, host_path: str):
    """Non-IID split: assign all records of the same host into the same client."""
    import random
    rng = random.Random(seed)

    host2idx = {}
    for i, o in enumerate(step1_outs):
        v = _get_by_path(o, host_path)
        if v in (None, ""):
            raise ValueError(f"Missing host at path '{host_path}' for record index={i}. "
                             f"Your step1.jsonl must contain this field.")
        host2idx.setdefault(str(v), []).append(i)

    hosts = list(host2idx.keys())
    rng.shuffle(hosts)

    shards = [[] for _ in range(num_clients)]
    for j, h in enumerate(hosts):
        shards[j % num_clients].extend(host2idx[h])

    for s in shards:
        rng.shuffle(s)
    return shards
