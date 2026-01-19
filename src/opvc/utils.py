"""opvc.utils

Small utilities used across steps.

Note: keep dependencies minimal (stdlib + torch + numpy).
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch

Tensor = torch.Tensor


def seed_all(seed: int) -> None:
    """Best-effort global seeding for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_py(x: Any) -> Any:
    """Convert tensors / dataclasses / containers to JSON-friendly python types."""
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return x.detach().cpu().tolist()
    if is_dataclass(x):
        return {k: to_py(v) for k, v in asdict(x).items()}
    if isinstance(x, Mapping):
        return {k: to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]
    return x


def safe_pearson_corr(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Row-wise Pearson correlation, robust to zero-variance rows.

    Args:
        x: [V, D]
    Returns:
        corr: [V, V] in [-1,1], diagonal is 1 for valid rows else 0.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be [V,D], got shape={tuple(x.shape)}")
    V, D = x.shape
    x0 = x - x.mean(dim=1, keepdim=True)
    var = (x0 * x0).mean(dim=1)  # [V]
    std = torch.sqrt(torch.clamp(var, min=0.0) + eps)
    mask = (std > eps).to(x.dtype)  # [V]
    xn = (x0 / std.unsqueeze(1)) * mask.unsqueeze(1)
    corr = (xn @ xn.t()) / float(max(int(D), 1))
    corr = torch.clamp(corr, -1.0, 1.0)
    diag = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))
    corr.fill_diagonal_(0.0)
    corr = corr + torch.diag(diag)
    return corr


def rho_max_pair(corr: Tensor) -> Tensor:
    """rho(x) = max_{v<v'} corr[v,v'] (as in method final)."""
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"corr must be square [V,V], got {tuple(corr.shape)}")
    V = int(corr.shape[0])
    if V <= 1:
        return corr.new_tensor(0.0)
    triu = torch.triu(corr, diagonal=1)
    return triu.max()


def merge_contiguous_segments(idxs_1based: Sequence[int]) -> List[Tuple[int, int]]:
    """Merge 1-based indices into contiguous (start,end) segments."""
    if not idxs_1based:
        return []
    idxs = sorted(set(int(i) for i in idxs_1based))
    segs: List[Tuple[int, int]] = []
    s = e = idxs[0]
    for i in idxs[1:]:
        if i == e + 1:
            e = i
        else:
            segs.append((s, e))
            s = e = i
    segs.append((s, e))
    return segs


def state_dict_add_(dst: MutableMapping[str, Tensor], src: Mapping[str, Tensor], alpha: float = 1.0) -> None:
    for k, v in src.items():
        if k not in dst:
            dst[k] = v.detach().clone()
        else:
            dst[k] = dst[k] + alpha * v.detach()


def state_dict_scale(sd: Mapping[str, Tensor], alpha: float) -> Dict[str, Tensor]:
    return {k: v.detach() * alpha for k, v in sd.items()}


def state_dict_sub(a: Mapping[str, Tensor], b: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    return {k: a[k].detach() - b[k].detach() for k in a.keys()}


def state_dict_l2norm(sd: Mapping[str, Tensor]) -> Tensor:
    s = None
    for v in sd.values():
        vv = v.detach().float().reshape(-1)
        s = vv.pow(2).sum() if s is None else (s + vv.pow(2).sum())
    return torch.sqrt(s.clamp_min(0.0)) if s is not None else torch.tensor(0.0)


def state_dict_clip_by_l2(sd: Mapping[str, Tensor], clip: float, eps: float = 1e-12) -> Dict[str, Tensor]:
    norm = state_dict_l2norm(sd)
    scale = min(1.0, float(clip) / float(norm.detach().cpu().item() + eps))
    return state_dict_scale(sd, scale)


def state_dict_add_noise(sd: Mapping[str, Tensor], sigma: float, clip: float, device: torch.device) -> Dict[str, Tensor]:
    """Gaussian noise for DP on updates: N(0, (sigma*clip)^2)."""
    out: Dict[str, Tensor] = {}
    for k, v in sd.items():
        noise = torch.randn_like(v, device=device) * (float(sigma) * float(clip))
        out[k] = v.detach().to(device) + noise
    return out


def secure_agg_sum(updates: Sequence[Mapping[str, Tensor]], seed: int = 0) -> Dict[str, Tensor]:
    """Deterministic secure-aggregation *simulation*.

    Each client adds a pseudo-random mask; masks cancel when summed.
    This is NOT a cryptographic implementation, but matches the method's interface
    requirement and yields reproducible logs.
    """
    if not updates:
        return {}
    keys = list(updates[0].keys())
    device = next(iter(updates[0].values())).device
    # generate masks pairwise based on seed and (i,j)
    masks: List[Dict[str, Tensor]] = []
    for i in range(len(updates)):
        m: Dict[str, Tensor] = {}
        for k in keys:
            m[k] = torch.zeros_like(updates[0][k], device=device)
        masks.append(m)
    for i in range(len(updates)):
        for j in range(i + 1, len(updates)):
            g = torch.Generator(device=device)
            g.manual_seed(int(seed) + 10007 * i + 10009 * j)
            for k in keys:
                r = torch.randn(updates[0][k].shape, device=device, dtype=updates[0][k].dtype, generator=g)
                masks[i][k] = masks[i][k] + r
                masks[j][k] = masks[j][k] - r
    summed: Dict[str, Tensor] = {k: torch.zeros_like(updates[0][k], device=device) for k in keys}
    for i, upd in enumerate(updates):
        for k in keys:
            summed[k] = summed[k] + upd[k].detach() + masks[i][k]
    return summed
