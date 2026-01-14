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
