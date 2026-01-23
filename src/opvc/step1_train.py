"""opvc.step1_train

Paper-level Step1 training (self-supervised) + checkpointing.

Why self-supervised?
Step1's job is to:
  - windowize + deterministic aggregation (auditable)
  - learn per-view encoders
  - learn sample-conditional routing + alignment so different views become comparable

This module implements a method-faithful, label-free pretraining objective
that encourages cross-view alignment in the shared aligned space.

Loss (default): Multi-view InfoNCE between per-view aligned vectors z_{v} and
the fused representation Z (positives are from the same sample; negatives are
other samples in the same minibatch).

This is intentionally simple but NOT a "smoke test": it trains real parameters,
saves checkpoints, and is deterministic given a seed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .contracts import Step1Config
from .step1 import Step1Model

Tensor = torch.Tensor


@dataclass
class Step1TrainConfig:
    epochs: int = 3
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.0
    tau_nce: float = 0.2
    lambda_pi_entropy: float = 0.0
    lambda_alpha_entropy: float = 0.0
    grad_clip: float = 1.0
    seed: int = 0


class EventlistJsonlDataset:
    """Reads eventlist.jsonl and yields (meta, E, t0, delta).

    Each line is expected to follow the Step0 contract:
      {meta:{host,...}, t0, delta, T, E}
    """

    def __init__(self, path: str):
        self.path = str(path)
        self._lines: List[str] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._lines.append(line)
        if not self._lines:
            raise ValueError(f"empty eventlist: {self.path}")

    def __len__(self) -> int:
        return len(self._lines)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], Any, float, float, int]:
        obj = json.loads(self._lines[int(idx)])
        meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
        # Ensure a stable node identifier exists for downstream label alignment.
        # Many datasets use "host"; some use "node". We keep both.
        if isinstance(meta, dict):
            if not meta.get("node"):
                meta["node"] = meta.get("host")
            if not meta.get("node_id"):
                meta["node_id"] = meta.get("node")
        return meta, obj["E"], float(obj["t0"]), float(obj["delta"]), int(obj["T"])


def _batch_indices(n: int, batch_size: int, seed: int) -> Iterable[List[int]]:
    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    for i in range(0, n, int(batch_size)):
        yield perm[i : i + int(batch_size)]


def step1_selfsup_loss(out_list: List[Any], tau: float, lambda_pi_entropy: float, lambda_alpha_entropy: float) -> Tensor:
    """Compute multi-view InfoNCE loss over a minibatch.

    For each sample b and each view v:
      q = normalize(z_{b,v})
      positives: k_pos = normalize(Z_b)
      negatives: {normalize(Z_{b'})}_{b'!=b}
    """
    Z = torch.stack([o.Z for o in out_list], dim=0)  # [B,da]
    Z = F.normalize(Z, dim=-1)
    B = Z.shape[0]

    # similarity between every view vector and every sample Z
    losses = []
    for b, o in enumerate(out_list):
        z_view = o.h_aligned.mean(dim=1)  # [V,da]
        z_view = F.normalize(z_view, dim=-1)
        # logits: [V,B]
        logits = (z_view @ Z.t()) / max(float(tau), 1e-6)
        # label is "this sample"
        target = torch.full((z_view.shape[0],), int(b), device=logits.device, dtype=torch.long)
        losses.append(F.cross_entropy(logits, target))

    loss = torch.stack(losses).mean()

    # optional entropies for regularization (encourage non-degenerate distributions)
    if lambda_pi_entropy > 0:
        ent = []
        for o in out_list:
            p = o.pi.clamp_min(1e-12)
            ent.append(-(p * torch.log(p)).sum())
        loss = loss - float(lambda_pi_entropy) * torch.stack(ent).mean()
    if lambda_alpha_entropy > 0:
        ent = []
        for o in out_list:
            a = o.alpha.clamp_min(1e-12)
            ent.append(-(a * torch.log(a)).sum())
        loss = loss - float(lambda_alpha_entropy) * torch.stack(ent).mean()

    return loss


def train_step1_selfsup(
    model: Step1Model,
    dataset: EventlistJsonlDataset,
    train_cfg: Step1TrainConfig,
    device: str = "cpu",
) -> Dict[str, float]:
    dev = torch.device(device)
    torch.manual_seed(int(train_cfg.seed))
    model = model.to(dev)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.lr), weight_decay=float(train_cfg.weight_decay))
    logs: Dict[str, float] = {}

    for ep in range(int(train_cfg.epochs)):
        ep_loss = 0.0
        n_batches = 0
        for idxs in _batch_indices(len(dataset), train_cfg.batch_size, seed=train_cfg.seed + 1000 * ep):
            out_list = []
            for i in idxs:
                _, E, t0, delta, T = dataset[i]
                if int(T) != int(model.cfg.T):
                    raise ValueError(f"T mismatch: record has T={T}, model.cfg.T={model.cfg.T}")
                out_list.append(model(E=E, t0=t0, delta=delta))

            loss = step1_selfsup_loss(
                out_list,
                tau=float(train_cfg.tau_nce),
                lambda_pi_entropy=float(train_cfg.lambda_pi_entropy),
                lambda_alpha_entropy=float(train_cfg.lambda_alpha_entropy),
            )

            opt.zero_grad()
            loss.backward()
            if train_cfg.grad_clip and float(train_cfg.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.grad_clip))
            opt.step()

            ep_loss += float(loss.detach().cpu().item())
            n_batches += 1

        logs[f"epoch_{ep}_loss"] = ep_loss / float(max(n_batches, 1))

    model.eval()
    return logs


def save_step1_ckpt(path: str, model: Step1Model, train_cfg: Optional[Step1TrainConfig] = None) -> None:
    ckpt = {
        "cfg1": asdict(model.cfg),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "q_rms": model.q_rms.state_dict(),
        "train_cfg": asdict(train_cfg) if train_cfg is not None else None,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(p))


def load_step1_ckpt(path: str, device: str = "cpu") -> Step1Model:
    dev = torch.device(device)
    ckpt = torch.load(path, map_location=dev)
    if not isinstance(ckpt, dict) or "cfg1" not in ckpt or "state_dict" not in ckpt:
        raise TypeError(f"invalid step1 ckpt: {path}")
    cfg1 = Step1Config(**ckpt["cfg1"])  # type: ignore[arg-type]
    model = Step1Model(cfg1).to(dev)
    model.load_state_dict({k: v.to(dev) for k, v in ckpt["state_dict"].items()}, strict=True)
    if "q_rms" in ckpt:
        model.q_rms.load_state_dict(ckpt["q_rms"], device=dev)
    model.eval()
    return model
