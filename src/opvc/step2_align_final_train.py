from __future__ import annotations
from typing import Iterable, Dict, Any, Tuple
import copy
import random
import torch

from opvc.step2 import Step2Model


def _make_model(cfg, dz: int, device: str):
    # 兼容不同 Step2Model 构造签名
    try:
        return Step2Model(cfg, dz).to(device)
    except TypeError:
        try:
            return Step2Model(cfg=cfg, dz=dz).to(device)
        except TypeError:
            return Step2Model(cfg).to(device)


def conf_from_step1(alpha=None, rho=None, gate=None, device="cpu") -> torch.Tensor:
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


def dp_sanitize(Z: torch.Tensor, base_clip: float, base_noise: float, conf: torch.Tensor, rho=None):
    Z = Z.clone()
    B = Z.shape[0]

    u = float((1.0 - conf).clamp(0.0, 1.0).detach().cpu())
    rbar = 1.0 if rho is None else float(torch.as_tensor(rho).clamp(0.0, 1.0).detach().cpu())

    clip = max(base_clip * (0.5 + 0.5 * rbar) * (0.5 + 0.5 * (1.0 - u)), 1e-6)
    noise = base_noise * (1.0 + u) * (1.0 + (1.0 - rbar))

    norms = Z.norm(dim=1).clamp_min(1e-12)
    scale = (clip / norms).clamp_max(1.0)
    Z = Z * scale.view(B, 1)

    if noise > 0:
        Z = Z + torch.randn_like(Z) * (noise * clip)

    return Z


def secure_agg_avg_state_dict(state_dicts, seed=0):
    rng = random.Random(int(seed))
    K = len(state_dicts)
    keys = list(state_dicts[0].keys())

    masks = []
    for _ in range(K):
        torch.manual_seed(rng.randint(0, 10**9))
        masks.append({k: torch.randn_like(state_dicts[0][k]) * 0.01 for k in keys})

    sum_masks = {k: sum(m[k] for m in masks) for k in keys}
    sum_enc = {k: sum(sd[k] + mk[k] for sd, mk in zip(state_dicts, masks)) for k in keys}
    sum_plain = {k: (sum_enc[k] - sum_masks[k]) for k in keys}
    avg_plain = {k: (sum_plain[k] / float(K)) for k in keys}
    return avg_plain


def pretrain_teacher(cfg, dz: int, train_batches: Iterable[Dict[str, Any]], device="cpu") -> Dict[str, torch.Tensor]:
    teacher = _make_model(cfg, dz, device)
    teacher.train()

    lr = float(getattr(cfg, "teacher_lr", 1e-3) or 1e-3)
    epochs = int(getattr(cfg, "teacher_epochs", 3) or 3)
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)

    batches = list(train_batches)
    for ep in range(epochs):
        loss_sum, n = 0.0, 0
        for batch in batches:
            Z = batch["Z"].to(device)
            pi = batch["pi"].to(device)
            US = teacher.forward_uras(Z, pi)
            loss = US.var(dim=0).mean() + 1e-4 * (US**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach().cpu())
            n += 1
        print(f"[teacher] ep={ep+1}/{epochs} loss={loss_sum/max(n,1):.6f}")

    return {k: v.detach().cpu() for k, v in teacher.state_dict().items()}


def train_theta_global_align_final(cfg, dz: int, train_batches: Iterable[Dict[str, Any]], device="cpu", seed=0):
    batches = list(train_batches)

    # teacher
    teacher_sd = pretrain_teacher(cfg, dz, batches, device=device)
    teacher = _make_model(cfg, dz, device)
    teacher.load_state_dict(teacher_sd)
    teacher.eval()

    # student global
    global_model = _make_model(cfg, dz, device)
    global_model.train()

    rounds = int(getattr(cfg, "rounds", 2) or 2)
    num_clients = int(getattr(cfg, "num_clients", 4) or 4)
    local_steps = int(getattr(cfg, "local_steps", 1) or 1)
    lr = float(getattr(cfg, "lr", 1e-3) or 1e-3)

    base_clip = float(getattr(cfg, "dp_clip", 1.0) or 1.0)
    base_noise = float(getattr(cfg, "dp_noise", 0.05) or 0.05)

    logs = {
        "rounds": rounds,
        "num_clients": num_clients,
        "local_steps": local_steps,
        "lr": lr,
        "dp_clip": base_clip,
        "dp_noise": base_noise,
        "loss": [],
        "note": "align-final step2 training proof (MSE distill); packaged with model weights for step3"
    }

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

                conf = conf_from_step1(alpha=alpha, rho=rho, gate=gate, device=device)
                Zs = dp_sanitize(Z, base_clip, base_noise, conf=conf, rho=rho)

                with torch.no_grad():
                    US_t = teacher.forward_uras(Z, pi)
                US_s = client.forward_uras(Zs, pi)

                loss = torch.mean((US_s - US_t) ** 2)

                opt.zero_grad()
                loss.backward()
                opt.step()

                r_loss += float(loss.detach().cpu())

            client_states.append({k: v.detach().cpu() for k, v in client.state_dict().items()})

        avg_state = secure_agg_avg_state_dict(client_states, seed=seed + r)
        global_model.load_state_dict(avg_state)

        logs["loss"].append({"round": r + 1, "avg_loss": r_loss / max(num_clients * local_steps, 1)})

    # theta_global stats for mahalanobis(US)
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

    # ✅ 打包：权重 + 统计量
    theta_pkg = {
        "version": 1,
        "state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},
        "mu": mu,
        "cov_inv": cov_inv,
        "eps": torch.tensor(eps),
        "note": "align-final theta_global package: Step2Model weights + mahalanobis stats",
    }
    return theta_pkg, logs
