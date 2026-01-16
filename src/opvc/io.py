"""opvc.io

JSON helpers for artifacts.

These functions are primarily for scripts/ demos, keeping core modules pure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .contracts import Step1Outputs, Step1Metrics

Tensor = torch.Tensor


def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(obj: Any, path: str) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def step1_outputs_from_dict(d: Dict[str, Any], device: str = "cpu") -> Step1Outputs:
    """Construct Step1Outputs from a dict with numpy/list/tensor values."""
    dev = torch.device(device)
    h_aligned = torch.as_tensor(d["h_aligned"], dtype=torch.float32, device=dev)
    alpha = torch.as_tensor(d["alpha"], dtype=torch.float32, device=dev)
    pi = torch.as_tensor(d["pi"], dtype=torch.float32, device=dev)
    B_x = torch.as_tensor(d["B_x"], dtype=torch.float32, device=dev)
    Z = torch.as_tensor(d["Z"], dtype=torch.float32, device=dev)
    H = torch.as_tensor(d["H"], dtype=torch.float32, device=dev)

    metrics_d = d.get("metrics", {}) or {}
    metrics = Step1Metrics(
        q_cov=torch.as_tensor(metrics_d.get("q_cov"), dtype=torch.float32, device=dev) if metrics_d.get("q_cov") is not None else None,
        q_val=torch.as_tensor(metrics_d.get("q_val"), dtype=torch.float32, device=dev) if metrics_d.get("q_val") is not None else None,
        q_cmp=torch.as_tensor(metrics_d.get("q_cmp"), dtype=torch.float32, device=dev) if metrics_d.get("q_cmp") is not None else None,
        q_unq=torch.as_tensor(metrics_d.get("q_unq"), dtype=torch.float32, device=dev) if metrics_d.get("q_unq") is not None else None,
        q_stb=torch.as_tensor(metrics_d.get("q_stb"), dtype=torch.float32, device=dev) if metrics_d.get("q_stb") is not None else None,
        Q=torch.as_tensor(metrics_d.get("Q"), dtype=torch.float32, device=dev) if metrics_d.get("Q") is not None else None,
        Q_hat=torch.as_tensor(metrics_d.get("Q_hat"), dtype=torch.float32, device=dev) if metrics_d.get("Q_hat") is not None else None,
        g_view=torch.as_tensor(metrics_d.get("g_view"), dtype=torch.float32, device=dev) if metrics_d.get("g_view") is not None else None,
        corr_mat=torch.as_tensor(metrics_d.get("corr_mat"), dtype=torch.float32, device=dev) if metrics_d.get("corr_mat") is not None else None,
    )

    out = Step1Outputs(
        h_aligned=h_aligned,
        alpha=alpha,
        pi=pi,
        B_x=B_x,
        Z=Z,
        H=H,
        gate=bool(d.get("gate")) if d.get("gate") is not None else None,
        rho=float(d.get("rho")) if d.get("rho") is not None else None,
        metrics=metrics,
    )
    return out
