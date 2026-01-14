"""
OPVC Step2 (Full-ish runnable, single-machine simulated federation)

Implements:
- URAS forward: US = concat_k (pi_k * phi_k(f_S(Z)))
- ASD + AT-InfoNCE distillation losses (see step2_losses.py)
- DP sanitize on latent Z (training mode) with adaptive sigma (utility/risk)
- Simulated multi-client local update + FedAvg aggregation -> theta_global

Contract outputs:
- Step2Outputs.theta_global : model.state_dict()
- Step2Outputs.forward_uras : callable(Z,pi)->US
- optional nu/risk/tau_dyn for logging/debug
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import Step2Config, Step2Outputs
from .step2_losses import (
    asd_loss,
    info_nce_distill,
    alpha_confidence,
    utility_signal,
    risk_signal,
    dynamic_temperature,
    dp_noise_multiplier,
)


class Step2Model(nn.Module):
    def __init__(self, cfg: Step2Config, dz: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.dz = int(dz)

        h = int(hidden) if hidden is not None else max(self.dz, 32)
        self.student = nn.Sequential(
            nn.Linear(self.dz, h),
            nn.ReLU(),
            nn.Linear(h, self.dz),
        )
        self.heads = nn.ModuleList([nn.Linear(self.dz, self.cfg.du) for _ in range(self.cfg.Kr)])

    @torch.no_grad()
    def _dp_sanitize(self, Zb: torch.Tensor, sigma_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        L2 clip to Cb and add Gaussian noise.
        sigma can be scalar or [B] broadcastable to Zb.
        """
        Cb = float(self.cfg.Cb)
        if Cb <= 0:
            return Zb

        norms = torch.linalg.vector_norm(Zb, ord=2, dim=-1, keepdim=True).clamp_min(1e-12)
        scale = (Cb / norms).clamp_max(1.0)
        Zc = Zb * scale

        sigma = float(self.cfg.sigma_b)
        if sigma_override is not None:
            # sigma_override: scalar or [B]
            if sigma_override.ndim == 0:
                sigma = float(sigma_override.item())
                noise = torch.randn_like(Zc) * (sigma * Cb)
            else:
                noise = torch.randn_like(Zc) * (sigma_override.view(-1,1) * Cb)
        else:
            noise = torch.randn_like(Zc) * (sigma * Cb)

        if (sigma_override is not None) or (float(self.cfg.sigma_b) > 0):
            Zc = Zc + noise
        return Zc

    def forward_uras(self, Z: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        # shape normalize
        if Z.dim() == 1:
            Zb, squeeze_out = Z.unsqueeze(0), True
        elif Z.dim() == 2:
            Zb, squeeze_out = Z, False
        else:
            raise ValueError("Z must be [dz] or [B,dz]")

        B = Zb.shape[0]
        if pi.dim() == 1:
            pib = pi.unsqueeze(0).expand(B, -1)
        elif pi.dim() == 2:
            pib = pi
        else:
            raise ValueError("pi must be [Kr] or [B,Kr]")

        if pib.shape[-1] != self.cfg.Kr:
            raise ValueError(f"pi last dim mismatch: {pib.shape[-1]} vs Kr={self.cfg.Kr}")

        # normalize pi
        pib = pib / pib.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # student backbone
        h = self.student(Zb)  # [B,dz]
        u_list = [head(h) for head in self.heads]  # Kr*[B,du]
        weighted = [pib[:, k:k+1] * u_list[k] for k in range(self.cfg.Kr)]
        US = torch.cat(weighted, dim=-1)  # [B,Kr*du]
        return US.squeeze(0) if squeeze_out else US

    def as_state_dict(self) -> Dict[str, Any]:
        return self.state_dict()

    def load_state_dict_strict(self, sd: Dict[str, Any]) -> None:
        self.load_state_dict(sd, strict=True)


def build_step2_outputs(model: Step2Model, nu: Optional[float] = None, risk: Optional[float] = None, tau_dyn: Optional[int] = None) -> Step2Outputs:
    return Step2Outputs(
        theta_global=model.as_state_dict(),
        forward_uras=model.forward_uras,
        nu=nu,
        risk=risk,
        tau_dyn=tau_dyn,
    )


@torch.no_grad()
def _fedavg_state_dict(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Simple FedAvg over a list of state_dicts (all same keys).
    """
    if len(state_dicts) == 1:
        return state_dicts[0]
    keys = state_dicts[0].keys()
    avg = {}
    for k in keys:
        avg[k] = torch.stack([sd[k].detach().cpu() for sd in state_dicts], dim=0).mean(dim=0)
    return avg


def simulate_federated_step2_train(
    cfg: Step2Config,
    Z_batch: torch.Tensor,          # [N,dz]
    pi_batch: torch.Tensor,         # [N,Kr]
    alpha_batch: Optional[torch.Tensor] = None,  # [N,V] optional (for confidence)
    rounds: int = 3,
    num_clients: int = 2,
    local_steps: int = 10,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[Step2Model, Dict[str, float]]:
    """
    Single-machine simulation of FL:
      - split samples into num_clients shards
      - each client trains a local copy for local_steps
      - server FedAvg parameters
    Loss = ASD + AT-InfoNCE (teacher=stopgrad on same batch for now; placeholder)
    Note: In full paper, teacher could be previous global / EMA / larger model.
    """
    torch.manual_seed(seed)

    Z_batch = Z_batch.to(device)
    pi_batch = pi_batch.to(device)

    dz = Z_batch.shape[-1]
    model_global = Step2Model(cfg, dz=dz).to(device)
    model_global.train()

    # naive shard split
    N = Z_batch.shape[0]
    idx = torch.arange(N)
    shards = torch.chunk(idx, chunks=num_clients)

    log = {}

    for r in range(rounds):
        local_sds = []
        losses = []

        for cid, sh in enumerate(shards):
            if sh.numel() == 0:
                continue

            # local copy
            m = Step2Model(cfg, dz=dz).to(device)
            m.load_state_dict_strict(model_global.state_dict())
            m.train()

            opt = torch.optim.Adam(m.parameters(), lr=lr)

            # local train
            for _ in range(local_steps):
                Z = Z_batch[sh]         # [B,dz]
                pi = pi_batch[sh]       # [B,Kr]

                # teacher placeholder: stop-grad on current global URAS
                with torch.no_grad():
                    US_t = model_global.forward_uras(Z, pi)   # [B,Kr*du]

                # utility/risk signals (lightweight)
                if alpha_batch is not None:
                    conf = alpha_confidence(alpha_batch[sh].to(device))  # [B]
                else:
                    conf = torch.ones(Z.shape[0], device=device)

                # distill error proxy: L2 between student/teacher before update
                US_s_pre = m.forward_uras(Z, pi)
                dist_err = torch.linalg.vector_norm(US_s_pre - US_t, dim=-1)  # [B]
                util = utility_signal(dist_err, conf)

                # risk proxy: use norm of Z (placeholder sensitivity)
                sens = torch.linalg.vector_norm(Z, dim=-1) / (torch.linalg.vector_norm(Z, dim=-1).mean().clamp_min(1e-6))
                risk = risk_signal(sens)

                tau = dynamic_temperature(util=util, risk=risk, round_idx=r, total_rounds=rounds)  # [B]

                # adaptive dp sigma (applied to Z)
                sigma_vec = dp_noise_multiplier(base_sigma=float(cfg.sigma_b), util=util, risk=risk)

                Z_san = m._dp_sanitize(Z, sigma_override=sigma_vec)  # [B,dz]

                US_s = m.forward_uras(Z_san, pi)  # student on sanitized latent
                # distillation losses
                L_asd = asd_loss(US_s, US_t)
                L_nce = info_nce_distill(US_s, US_t, tau=tau)

                loss = L_asd + L_nce
                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(float(loss.detach().cpu().item()))

            local_sds.append(m.state_dict())

        # server aggregation
        sd_avg = _fedavg_state_dict(local_sds)
        model_global.load_state_dict(sd_avg, strict=True)

        log[f"round_{r}_loss_mean"] = float(sum(losses) / max(len(losses), 1))

    model_global.eval()
    log["final_rounds"] = float(rounds)
    return model_global, log


# =========================
# Align-Final: Step2 training proof (teacher pretrain + adaptive DP + secure agg sim)
# =========================
from typing import Iterable, Dict, Any

def _align_final_conf_from_step1(alpha=None, rho=None, gate=None, device="cpu"):
    # conf in [0,1]: alpha entropy -> lower conf, rho lower -> lower conf, gate=True -> slightly lower
    import torch
    if alpha is None:
        conf = torch.tensor(1.0, device=device)
    else:
        a = torch.as_tensor(alpha, dtype=torch.float32, device=device).clamp_min(1e-9)
        ent = -(a * a.log()).sum()
        ent_norm = ent / max(float(torch.log(torch.tensor(a.numel(), device=device))), 1e-6)
        conf = (1.0 - ent_norm).clamp(0.0, 1.0)
    if rho is not None:
        r = torch.as_tensor(rho, dtype=torch.float32, device=device).clamp(0.0, 1.0)
        conf = conf * r
    if gate is not None and bool(gate):
        conf = conf * 0.8
    return conf

def _align_final_dp_sanitize(Z, base_clip: float, base_noise: float, conf=None, rho=None):
    import torch
    Z = Z.clone()
    B = Z.shape[0]

    if conf is None:
        u = 0.0
    else:
        u = float((1.0 - conf).clamp(0.0, 1.0).detach().cpu())
    if rho is None:
        rbar = 1.0
    else:
        rbar = float(torch.as_tensor(rho).clamp(0.0, 1.0).detach().cpu())

    # uncertainty high / consistency low -> smaller clip, larger noise
    clip = max(base_clip * (0.5 + 0.5 * rbar) * (0.5 + 0.5 * (1.0 - u)), 1e-6)
    noise = base_noise * (1.0 + u) * (1.0 + (1.0 - rbar))

    # per-sample clip
    norms = Z.norm(dim=1).clamp_min(1e-12)
    scale = (clip / norms).clamp_max(1.0)
    Z = Z * scale.view(B, 1)

    if noise > 0:
        Z = Z + torch.randn_like(Z) * (noise * clip)
    return Z

def _secure_agg_avg_state_dict(state_dicts, seed=0):
    """单机可复现 Secure Aggregation 模拟：加掩码后求和，再抵消掩码，最后求平均。"""
    import torch, random
    rng = random.Random(int(seed))
    K = len(state_dicts)
    keys = list(state_dicts[0].keys())

    masks = []
    for _ in range(K):
        # 固定 seed 使得可复现
        torch.manual_seed(rng.randint(0, 10**9))
        masks.append({k: torch.randn_like(state_dicts[0][k]) * 0.01 for k in keys})

    sum_masks = {k: sum(m[k] for m in masks) for k in keys}
    sum_enc = {k: sum(sd[k] + mk[k] for sd, mk in zip(state_dicts, masks)) for k in keys}
    sum_plain = {k: (sum_enc[k] - sum_masks[k]) for k in keys}
    avg_plain = {k: (sum_plain[k] / float(K)) for k in keys}
    return avg_plain

def pretrain_teacher(cfg, train_batches: Iterable[Dict[str, Any]], device="cpu"):
    """离线 teacher 预训练（可复现证明版）：在本地批次上训练 teacher 的子空间稳定结构。"""
    import torch
    train_batches = train_batches
    dz = int(getattr(cfg, 'dz', 0) or train_batches[0]['Z'].shape[1])
    teacher = Step2Model(cfg, dz).to(device)
    teacher.train()
    lr = float(getattr(cfg, "teacher_lr", 1e-3))
    epochs = int(getattr(cfg, "teacher_epochs", 3))
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)

    for ep in range(epochs):
        loss_sum, n = 0.0, 0
        for batch in train_batches:
            Z = batch["Z"].to(device)
            pi = batch["pi"].to(device)
            US = teacher.forward_uras(Z, pi)
            loss = US.var(dim=0).mean() + 1e-4 * (US**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach().cpu())
            n += 1
        if n:
            print(f"[teacher] ep={ep+1}/{epochs} loss={loss_sum/n:.6f}")
    return {k: v.detach().cpu() for k, v in teacher.state_dict().items()}

def simulate_federated_step2_train_align_final(
    cfg,
    train_batches: Iterable[Dict[str, Any]],
    teacher_state_dict=None,
    device="cpu",
    seed=0,
):
    """Align-final Step2：teacher蒸馏 + 自适应DP + secure-agg + fedavg，并输出 theta_global(用于Mahalanobis)。"""
    import copy, torch

    train_batches = train_batches
    dz = int(getattr(cfg, 'dz', 0) or train_batches[0]['Z'].shape[1])
    # build teacher
    if teacher_state_dict is None:
        teacher_state_dict = pretrain_teacher(cfg, train_batches, device=device)
    teacher = Step2Model(cfg, dz).to(device)
    teacher.load_state_dict(teacher_state_dict)
    teacher.eval()

    # build global student
    global_model = Step2Model(cfg, dz).to(device)
    global_model.train()

    rounds = int(getattr(cfg, "rounds", 1))
    num_clients = int(getattr(cfg, "num_clients", 4))
    local_steps = int(getattr(cfg, "local_steps", 1))
    lr = float(getattr(cfg, "lr", 1e-3))

    base_clip = float(getattr(cfg, "dp_clip", 1.0))
    base_noise = float(getattr(cfg, "dp_noise", 0.05))

    # loss helpers (fallback safe)
    try:
        from opvc.step2_losses import asd_loss, info_nce_distill, dynamic_temperature
    except Exception:
        asd_loss = None
        info_nce_distill = None
        dynamic_temperature = None

    logs = {"rounds": rounds, "num_clients": num_clients, "local_steps": local_steps, "lr": lr,
            "dp_clip": base_clip, "dp_noise": base_noise, "loss": []}

    batches = train_batches

    for r in range(rounds):
        client_states = []
        r_loss = 0.0

        for cid in range(num_clients):
            client = copy.deepcopy(global_model).to(device)
            client.train()
            opt = torch.optim.Adam(client.parameters(), lr=lr)

            for _ in range(local_steps):
                batch = batches[(r + cid) % len(batches)]
                Z = batch["Z"].to(device)
                pi = batch["pi"].to(device)
                alpha = batch.get("alpha", None)
                rho = batch.get("rho", None)
                gate = batch.get("gate", None)

                conf = _align_final_conf_from_step1(alpha=alpha, rho=rho, gate=gate, device=device)

                # DP sanitize student input
                Zs = _align_final_dp_sanitize(Z, base_clip, base_noise, conf=conf, rho=rho)

                with torch.no_grad():
                    US_t = teacher.forward_uras(Z, pi)

                US_s = client.forward_uras(Zs, pi)

                if asd_loss is not None and info_nce_distill is not None and dynamic_temperature is not None:
                    tau = float(dynamic_temperature(conf.unsqueeze(0)).detach().cpu())
                    loss = asd_loss(US_s, US_t) + info_nce_distill(US_s, US_t, tau=tau)
                else:
                    loss = torch.mean((US_s - US_t) ** 2)

                opt.zero_grad()
                loss.backward()
                opt.step()

                r_loss += float(loss.detach().cpu())

            client_states.append({k: v.detach().cpu() for k, v in client.state_dict().items()})

        # secure agg + average
        avg_state = _secure_agg_avg_state_dict(client_states, seed=seed + r)
        global_model.load_state_dict(avg_state)

        logs["loss"].append({"round": r + 1, "avg_loss": r_loss / max(num_clients * local_steps, 1)})

    # build theta_global for Mahalanobis(US): mu + cov_inv
    global_model.eval()
    US_all = []
    with torch.no_grad():
        for batch in batches:
            Z = batch["Z"].to(device)
            pi = batch["pi"].to(device)
            US = global_model.forward_uras(Z, pi)
            US_all.append(US.detach().cpu())
    US_all = torch.cat(US_all, dim=0)  # [N, D]
    mu = US_all.mean(dim=0)
    X = US_all - mu
    eps = 1e-4
    cov = (X.t() @ X) / max(US_all.shape[0] - 1, 1) + eps * torch.eye(X.shape[1])
    cov_inv = torch.linalg.pinv(cov)

    theta_global = {
        "mu": mu,
        "cov_inv": cov_inv,
        "eps": torch.tensor(eps),
        "note": "align-final theta_global (mu,cov_inv) for mahalanobis(US)"
    }

    return theta_global, logs
