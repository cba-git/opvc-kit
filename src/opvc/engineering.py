from __future__ import annotations

def _as_scalar(x, *, name: str = "value"):
    """Interface adapter: parse scalar from list/tuple/np/torch without changing any model/formula."""
    if x is None:
        raise TypeError(f"{name} is None")
    # unwrap singletons like [4] / (4,) etc.
    if isinstance(x, (list, tuple)):
        if len(x) != 1:
            raise TypeError(f"{name} expected scalar-like (len==1), got len={len(x)}: {x!r}")
        x = x[0]
    # torch / numpy scalar
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            x = x.item()
        except Exception:
            pass
    return x

def _as_int(x, *, name: str = "value") -> int:
    return int(_as_scalar(x, name=name))

def _as_float(x, *, name: str = "value") -> float:
    return float(_as_scalar(x, name=name))


from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .data import HashingAggConfig, window_and_agg


@dataclass
class AlignedCsvConfig:
    # 时间窗口设置：我们这里用 tau 做离散时间
    # 你现在的 CSV tau=1,2,3...，我们令 ts=tau-1，使得 t0=0 起步
    t0: float = 0.0
    delta: float = 1.0
    # 你希望窗口长度 T（如果为 None，就自动按最大 tau 推）
    T: Optional[int] = None

    # 4-view 定义（跟你方法截图一致）
    # v0/v1/v2 来自 dsttype_csv 的 file/netflow/other
    # v3 来自 opcnt_csv 的 operation
    V: int = 4

    # hashing aggregator 参数
    d_in: int = 32
    seed: int = 0
    include_field_names: bool = True
    use_signed_hash: bool = True

    # opcnt 的 cnt 作为权重是否使用
    use_opcnt_weight: bool = True


def _infer_T_from_tau(dst_df: pd.DataFrame, opc_df: pd.DataFrame) -> int:
    tmax = 0
    if "tau" in dst_df.columns and len(dst_df) > 0:
        tmax = max(tmax, int(dst_df["tau"].max()))
    if "tau" in opc_df.columns and len(opc_df) > 0:
        tmax = max(tmax, int(opc_df["tau"].max()))
    # tau=1 => ts=0，所以 T=tmax
    return max(1, tmax)


def aligned_rows_to_events_by_view(
    dst_rows: Sequence[Mapping[str, Any]],
    opc_rows: Sequence[Mapping[str, Any]],
    cfg: AlignedCsvConfig,
) -> List[List[Dict[str, Any]]]:
    """
    Convert *aligned* CSV rows into 4-view event streams.

    Each event must carry a numeric timestamp field "ts",
    so opvc.data can bucket it into windows.

    We use: ts = tau - 1, with t0=0, delta=1.

    Views:
      v0: dsttype/file
      v1: dsttype/netflow
      v2: dsttype/other
      v3: opcnt/operation
    """
    Ev: List[List[Dict[str, Any]]] = [[] for _ in range(cfg.V)]

    # dst_rows schema: subject_id, tau, file_cnt, netflow_cnt, other_cnt
    for r in dst_rows:
        tau = int(r["tau"])
        ts = float(tau - 1)

        for name, col, vid in [
            ("file", "file_cnt", 0),
            ("netflow", "netflow_cnt", 1),
            ("other", "other_cnt", 2),
        ]:
            c = int(r.get(col, 0) or 0)
            if c <= 0:
                continue
            Ev[vid].append(
                {
                    "ts": ts,
                    "src": f"dsttype/{name}",
                    "cnt": c,
                }
            )

    # opc_rows schema: subject_id, tau, operation, cnt
    for r in opc_rows:
        tau = int(r["tau"])
        ts = float(tau - 1)
        op = str(r.get("operation", "UNK"))
        c = int(r.get("cnt", 0) or 0)
        if c <= 0:
            continue

        e = {
            "ts": ts,
            "src": "opcnt/operation",
            "op": op,
        }
        if cfg.use_opcnt_weight:
            e["cnt"] = c
        Ev[3].append(e)

    return Ev


def run_step1_from_aligned_csv(
    dsttype_csv: str,
    opcnt_csv: str,
    cfg: Optional[AlignedCsvConfig] = None,
    subject_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main engineering entrypoint:
    aligned CSVs -> events -> window+agg -> Step1-ready dict.

    Returns dict:
      {
        "t0": ...,
        "delta": ...,
        "cfg1": {...},
        "step1_out": {"h_aligned": [[...], ...], ...}
      }
    """
    cfg = cfg or AlignedCsvConfig()

    dst_df = pd.read_csv(dsttype_csv)
    opc_df = pd.read_csv(opcnt_csv)

    if subject_id is not None:
        dst_df = dst_df[dst_df["subject_id"] == subject_id]
        opc_df = opc_df[opc_df["subject_id"] == subject_id]

    if cfg.T is None:
        T = _infer_T_from_tau(dst_df, opc_df)
    else:
        T = int(cfg.T)

    # group by subject_id
    dst_groups = dict(tuple(dst_df.groupby("subject_id")))
    opc_groups = dict(tuple(opc_df.groupby("subject_id")))

    subjects = sorted(set(dst_groups.keys()) | set(opc_groups.keys()))
    if len(subjects) == 0:
        raise ValueError("No subject_id found in CSVs")

    # hashing config for each view
    agg_cfg = HashingAggConfig(
        dim=int(cfg.d_in),
        fields=["src", "op"],
        seed=int(cfg.seed),
        include_field_names=bool(cfg.include_field_names),
        use_signed_hash=bool(cfg.use_signed_hash),
    )

    h_aligned_all: List[List[List[float]]] = []  # [N, V, T, d_in]
    per_subject_stats: List[Dict[str, Any]] = []

    for sid in subjects:
        dst_rows = dst_groups.get(sid, pd.DataFrame()).to_dict("records")
        opc_rows = opc_groups.get(sid, pd.DataFrame()).to_dict("records")

        Ev = aligned_rows_to_events_by_view(dst_rows, opc_rows, cfg=cfg)

        # per view: window+agg
        h_views: List[List[List[float]]] = []
        view_stats: List[Dict[str, Any]] = []
        for v in range(cfg.V):
            res = window_and_agg(
                events=Ev[v],
                t0=cfg.t0,
                delta=cfg.delta,
                T=T,
                agg_cfg=agg_cfg,
                key_fields=["src", "op"],
                dedup_fields=["src", "op"],
                device=None,
            )
            # res.x_win: [T, d_in]
            h_views.append(res.x_win.detach().cpu().tolist())
            view_stats.append(
                {
                    "v": v,
                    "total_events": int(res.stats.total_events),
                    "parse_success": int(res.stats.parse_success),
                    "unique_events": int(res.stats.unique_events),
                    "duplicate_events": int(res.stats.duplicate_events),
                }
            )

        h_aligned_all.append(h_views)
        per_subject_stats.append({"subject_id": sid, "views": view_stats})

    out = {
        "t0": float(cfg.t0),
        "delta": float(cfg.delta),
        "cfg1": {
            "V": int(cfg.V),
            "T": int(T),
            "d_in": [int(cfg.d_in)] * int(cfg.V),
        },
        "step1_out": {
            "h_aligned": h_aligned_all,
        },
        "meta": {
            "subjects": subjects,
            "per_subject_stats": per_subject_stats,
        },
    }

# === AUTO-FILL STEP1_OUT DEFAULTS (for CSV->Step1 JSON) ===
    # Ensure JSON contains all fields required by opvc.io.step1_outputs_from_dict.
    # We only have h_aligned from deterministic window+agg, so fill others with safe defaults.
    step1_out = out.get("step1_out", {})
    cfg1 = out.get("cfg1", {})

    V = int(cfg1.get("V", 1))
    T = int(cfg1.get("T", 1))

    # cfg1["d_in"] may be list (per-view) or scalar
    d_in_val = cfg1.get("d_in", 1)
    if isinstance(d_in_val, list) and len(d_in_val) > 0:
        d_in = int(d_in_val[0])
    else:
        d_in = int(d_in_val)

    Kr = int(cfg1.get("Kr", 1))
    cfg1.setdefault("d", d_in)
    cfg1.setdefault("da", d_in)
    cfg1.setdefault("Kr", Kr)

    if "alpha" not in step1_out:
        step1_out["alpha"] = [1.0 / max(V, 1)] * max(V, 1)
    if "pi" not in step1_out:
        step1_out["pi"] = [1.0 / max(Kr, 1)] * max(Kr, 1)
    if "B_x" not in step1_out:
        # identity-like (da x d); here da==d==d_in
        step1_out["B_x"] = [[1.0 if i == j else 0.0 for j in range(d_in)] for i in range(d_in)]
    if "Z" not in step1_out:
        # zeros (T x V x Kr)
        step1_out["Z"] = [[[0.0 for _ in range(Kr)] for _ in range(V)] for _ in range(T)]
    if "H" not in step1_out:
        # zeros (T x V x Kr)
        step1_out["H"] = [[[0.0 for _ in range(Kr)] for _ in range(V)] for _ in range(T)]

    out["cfg1"] = cfg1
    out["step1_out"] = step1_out
    return out



# =============================================================================
# CSV -> Step2 adapter (Scheme 1: Step2 standard input is i.i.d. batch)
#
# Step2 expects:
#   - Z_batch    : [N, dz]
#   - pi_batch   : [N, Kr] or [Kr] (broadcastable)
#   - alpha_batch: [N, V]  or [V]  (broadcastable)
#
# We DO NOT change any model/formula here. This is purely an interface adapter
# between step1_out (from aligned CSV pipeline) and step2 training/simulation.
# =============================================================================

import warnings
from typing import Tuple

import torch


def adapt_step1_to_step2(step1_out, cfg2, *, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert Step1Outputs to Step2-friendly i.i.d. batch tensors.

    - step1_out.Z is typically [T, V, dz] or [T, V, ..., dz]. We flatten leading dims -> [N, dz].
    - step1_out.pi can be scalar/len1/[Kr]/[N,Kr]. We broadcast to [N,Kr].
    - step1_out.alpha can be [V] or [N,V]. We broadcast to [N,V].

    This adapter is ONLY for input-shape compatibility; it does not alter model equations.
    """
    # --- Z -> [N, dz]
    Z = step1_out.Z
    if not torch.is_tensor(Z):
        Z = torch.as_tensor(Z)
    Z = Z.to(device)

    if Z.dim() == 1:
        # [dz] -> [1,dz]
        Z_b = Z.view(1, -1)
    elif Z.dim() == 2:
        # [N,dz]
        Z_b = Z
    else:
        # [..., dz] -> [N, dz]
        dz = Z.shape[-1]
        Z_b = Z.reshape(-1, dz).contiguous()

    N = Z_b.shape[0]
    dz = Z_b.shape[1]

    # --- alpha -> [N, V]
    alpha = step1_out.alpha
    if not torch.is_tensor(alpha):
        alpha = torch.as_tensor(alpha)
    alpha = alpha.to(device)

    # infer V
    V = getattr(cfg2, "V", None)
    if V is None:
        # fallback: use alpha last dim
        V = int(alpha.numel()) if alpha.dim() == 1 else int(alpha.shape[-1])

    if alpha.dim() == 0:
        # scalar -> [V]
        alpha_v = alpha.view(1).repeat(V)
        warnings.warn("step1_out.alpha is scalar; broadcasting to uniform [V]. This is adapter fallback.")
    elif alpha.dim() == 1:
        if alpha.numel() == 1:
            alpha_v = alpha.view(1).repeat(V)
            warnings.warn("step1_out.alpha has length 1; broadcasting to uniform [V]. This is adapter fallback.")
        else:
            alpha_v = alpha
            if alpha_v.numel() != V:
                warnings.warn(f"alpha length ({alpha_v.numel()}) != cfg2.V ({V}); proceeding but check your pipeline.")
    else:
        # [N,V] or [*,V] -> flatten leading -> [N,V] if possible
        if alpha.shape[-1] != V:
            warnings.warn(f"alpha last dim ({alpha.shape[-1]}) != cfg2.V ({V}); proceeding but check your pipeline.")
        alpha_v = alpha.reshape(-1, alpha.shape[-1])

    if alpha_v.dim() == 1:
        alpha_b = alpha_v.view(1, -1).expand(N, -1).contiguous()
    else:
        # if already [N,V] but N mismatch, try broadcast or trim
        if alpha_v.shape[0] == N:
            alpha_b = alpha_v.contiguous()
        elif alpha_v.shape[0] == 1:
            alpha_b = alpha_v.expand(N, -1).contiguous()
            warnings.warn("alpha has only 1 row; broadcasting to [N,V]. This is adapter fallback.")
        else:
            # last resort: repeat/cycle to match N
            reps = (N + alpha_v.shape[0] - 1) // alpha_v.shape[0]
            alpha_b = alpha_v.repeat(reps, 1)[:N].contiguous()
            warnings.warn("alpha rows != N; repeating/cycling to match N. Check your pipeline.")

    # --- pi -> [N, Kr]
    pi = step1_out.pi
    if not torch.is_tensor(pi):
        pi = torch.as_tensor(pi)
    pi = pi.to(device)

    Kr = getattr(cfg2, "Kr", None)
    if Kr is None:
        # fallback: infer from pi if possible
        if pi.dim() >= 1 and pi.numel() > 1:
            Kr = int(pi.shape[-1]) if pi.dim() > 1 else int(pi.numel())
        else:
            Kr = 4  # conservative fallback; but cfg2.Kr should exist normally
            warnings.warn("cfg2.Kr not found; defaulting Kr=4. Please verify Step2Config has Kr.")

    if pi.dim() == 0:
        # scalar -> [Kr]
        pi_k = pi.view(1).repeat(Kr)
        warnings.warn("step1_out.pi is scalar; broadcasting to uniform [Kr]. This is adapter fallback.")
    elif pi.dim() == 1:
        if pi.numel() == 1:
            pi_k = pi.view(1).repeat(Kr)
            warnings.warn("step1_out.pi has length 1; broadcasting to uniform [Kr]. This is adapter fallback.")
        else:
            pi_k = pi
            if pi_k.numel() != Kr:
                warnings.warn(f"pi length ({pi_k.numel()}) != cfg2.Kr ({Kr}); proceeding but check your pipeline.")
    else:
        # [N,Kr] or [*,Kr]
        pi_k = pi.reshape(-1, pi.shape[-1])
        if pi_k.shape[-1] != Kr:
            warnings.warn(f"pi last dim ({pi_k.shape[-1]}) != cfg2.Kr ({Kr}); proceeding but check your pipeline.")

    if pi_k.dim() == 1:
        pi_b = pi_k.view(1, -1).expand(N, -1).contiguous()
    else:
        if pi_k.shape[0] == N:
            pi_b = pi_k.contiguous()
        elif pi_k.shape[0] == 1:
            pi_b = pi_k.expand(N, -1).contiguous()
            warnings.warn("pi has only 1 row; broadcasting to [N,Kr]. This is adapter fallback.")
        else:
            reps = (N + pi_k.shape[0] - 1) // pi_k.shape[0]
            pi_b = pi_k.repeat(reps, 1)[:N].contiguous()
            warnings.warn("pi rows != N; repeating/cycling to match N. Check your pipeline.")

    # sanity
    if Z_b.dim() != 2:
        raise ValueError(f"Adapter produced Z_batch with dim {Z_b.dim()}, expected 2.")
    if pi_b.dim() != 2:
        raise ValueError(f"Adapter produced pi_batch with dim {pi_b.dim()}, expected 2.")
    if alpha_b.dim() != 2:
        raise ValueError(f"Adapter produced alpha_batch with dim {alpha_b.dim()}, expected 2.")

    return Z_b, pi_b, alpha_b


def run_step2_from_step1_json(
    step1_json_path: str,
    *,
    device: str = "cpu",
    rounds: int = 1,
    num_clients: int = 2,
    local_steps: int = 1,
    lr: float = 1e-3,
    seed: int = 0,
):
    """
    Smoke-run Step2 from a saved Step1 JSON (produced by run_step1_from_aligned_csv + save_json).

    NOTE:
    - rounds/num_clients/local_steps are training-simulation parameters, not model-formula changes.
    - For full experiments, pass your real rounds/num_clients/local_steps.
    """
    from opvc.io import load_json, step1_outputs_from_dict
    from opvc.step2 import Step2Config, simulate_federated_step2_train

    d = load_json(step1_json_path)
    out1 = step1_outputs_from_dict(d["step1_out"], device=device)

    cfg2 = Step2Config()
    Z_batch, pi_batch, alpha_batch = adapt_step1_to_step2(out1, cfg2, device=device)

    model, stats = simulate_federated_step2_train(
        cfg2,
        Z_batch=Z_batch,
        pi_batch=pi_batch,
        alpha_batch=alpha_batch,
        rounds=rounds,
        num_clients=num_clients,
        local_steps=local_steps,
        lr=lr,
        seed=seed,
        device=device,
    )
    return model, stats

# =============================
# Scheme-1: Step2 adapter + runner
# =============================

# =============================
# Scheme-1: Step2 adapter + runner
# =============================
# We DO NOT change any model/formula here.
# This is purely an interface adapter between:
#   step1_out (from aligned CSV pipeline)  ->  step2 training/simulation
# so the method stays fully aligned to the paper formulation.

import warnings
from typing import Tuple

import torch

from .io import load_json, step1_outputs_from_dict
from .step2 import Step2Config, simulate_federated_step2_train


def _scheme1_step2_adapter_from_step1(out1, cfg: Step2Config):
    """
    Scheme-1: Step2 standard input is an i.i.d batch.
      - Z_batch: [N, dz]
      - pi_batch: [N, Kr] or [Kr] (broadcast)
      - alpha_batch: [N, V] or [V] (broadcast)

    Our step1_out is produced from aligned CSV with shapes like:
      - Z: [T, V, dz] or [T, V, 1] (depends on pipeline)
      - pi: [Kr] or [1]
      - alpha: [V]
    We flatten (T,V,*) into N=T*V samples. No formula change.
    """

    # ---- Z -> [N, dz] ----
    Z = out1.Z
    if Z.dim() == 1:
        # [dz] -> [1,dz]
        Z_batch = Z.unsqueeze(0)
    elif Z.dim() == 2:
        # [N,dz]
        Z_batch = Z
    elif Z.dim() >= 3:
        # e.g. [T,V,dz] or [T,V,1] -> flatten to [N,dz]
        dz = Z.shape[-1]
        Z_batch = Z.reshape(-1, dz).contiguous()
    else:
        raise ValueError(f"Unexpected Z dim: {Z.dim()} shape={tuple(Z.shape)}")

    N = Z_batch.shape[0]

    # ---- pi -> [N,Kr] ----
    pi = out1.pi
    Kr = int(getattr(cfg, "Kr"))

    if pi.dim() == 0:
        pi = pi.view(1)
    if pi.numel() == 1:
        # no real routing prior in this CSV: broadcast uniform [Kr]
        warnings.warn(
            "step1_out.pi has length 1; broadcasting to uniform [Kr]. This is adapter fallback.",
            UserWarning,
        )
        pi = torch.ones(Kr, dtype=pi.dtype, device=pi.device) / float(Kr)

    if pi.dim() == 1:
        # [Kr] -> [N,Kr]
        pi_batch = pi.unsqueeze(0).expand(N, -1).contiguous()
    elif pi.dim() == 2:
        pi_batch = pi
        if pi_batch.shape[0] != N:
            # if pi is [T*V,Kr] mismatched N, force broadcast first row
            pi_batch = pi_batch[:1].expand(N, -1).contiguous()
    else:
        raise ValueError(f"Unexpected pi dim: {pi.dim()} shape={tuple(pi.shape)}")

    # ---- alpha -> [N,V] ----
    alpha = out1.alpha
    if alpha.dim() == 0:
        alpha = alpha.view(1)
    if alpha.dim() == 1:
        alpha_batch = alpha.unsqueeze(0).expand(N, -1).contiguous()
    elif alpha.dim() == 2:
        alpha_batch = alpha
        if alpha_batch.shape[0] != N:
            alpha_batch = alpha_batch[:1].expand(N, -1).contiguous()
    else:
        raise ValueError(f"Unexpected alpha dim: {alpha.dim()} shape={tuple(alpha.shape)}")

    return Z_batch, pi_batch, alpha_batch


def run_step2_from_step1_json(
    step1_json_path: str,
    device: str = "cpu",
    rounds: int = 1,
    num_clients: int = 2,
    local_steps: int = 1,
    lr: float = 1e-3,
    seed: int = 0,
):
    """
    Convenience runner:
      step1_json -> Step1Outputs -> Scheme-1 adapter -> step2 simulate train
    """
    d = load_json(step1_json_path)
    out1 = step1_outputs_from_dict(d["step1_out"], device=device)

    cfg = Step2Config()
    Z_batch, pi_batch, alpha_batch = _scheme1_step2_adapter_from_step1(out1, cfg)

    model, stats = simulate_federated_step2_train(
        cfg,
        Z_batch=Z_batch,
        pi_batch=pi_batch,
        alpha_batch=alpha_batch,
        rounds=rounds,
        num_clients=num_clients,
        local_steps=local_steps,
        lr=lr,
        seed=seed,
        device=device,
    )
    return model, stats



# =============================================================================
# Scheme-1 Step3 adapter + runner (NO model/formula changes)
# This is purely an interface adapter between step1_out (from aligned CSV pipeline)
# and Step3 inference runner. Step3 expects H_seq as 2D [T, da].
# If step1_out.H is [T, V, da], we reduce over V (mean) -> [T, da].
# =============================================================================
import warnings
from typing import Optional

import torch

from .contracts import Step3Config
from .step3 import run_step3_from_H


def _adapt_step1_to_step3_inputs(out1, cfg3: Step3Config, device: str = "cpu"):
    """
    Adapter ONLY. Does not touch Step3 math.
    - H:  [T,da] required by Step3
          if [T,V,da] -> mean over V -> [T,da] (view aggregation at interface)
    - pi: [Kr] (or broadcastable)
    - alpha: [V] (or broadcastable)
    """
    dev = torch.device(device)

    # ---- H_seq ----
    H = out1.H
    if not isinstance(H, torch.Tensor):
        H = torch.as_tensor(H)
    H = H.to(dev)

    if H.dim() == 2:
        H_seq = H  # [T,da]
    elif H.dim() == 3:
        # [T,V,da] -> [T,da]
        warnings.warn(
            "step1_out.H is [T,V,da]; reducing over V by mean to produce Step3 H_seq [T,da]. "
            "This is an interface aggregation (Scheme-1), not a model/formula change."
        )
        H_seq = H.mean(dim=1)
    else:
        raise ValueError(f"step1_out.H must be 2D [T,da] or 3D [T,V,da]; got shape={tuple(H.shape)}")

    # ---- pi ----
    pi = out1.pi
    if not isinstance(pi, torch.Tensor):
        pi = torch.as_tensor(pi)
    pi = pi.to(dev)

    # allow scalar/len1 fallback -> broadcast to [Kr]
    if pi.dim() == 0:
        pi = pi.view(1)
    if pi.numel() == 1:
        warnings.warn(
            "step1_out.pi has length 1; broadcasting to uniform [Kr]. This is adapter fallback."
        )
        pi = pi.repeat(int(cfg3.Kr))
    if pi.dim() != 1 or pi.shape[0] != int(cfg3.Kr):
        raise ValueError(f"Step3 expects pi as [Kr]; got shape={tuple(pi.shape)}, Kr={int(cfg3.Kr)}")

    # ---- alpha ----
    alpha = out1.alpha
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.as_tensor(alpha)
    alpha = alpha.to(dev)

    if alpha.dim() == 0:
        alpha = alpha.view(1)
    # alpha can be [V] or len1 fallback
    if alpha.numel() == 1 and int(cfg3.V) > 1:
        warnings.warn(
            "step1_out.alpha has length 1; broadcasting to uniform [V]. This is adapter fallback."
        )
        alpha = alpha.repeat(int(cfg3.V))
    if alpha.dim() != 1 or alpha.shape[0] != int(cfg3.V):
        raise ValueError(f"Step3 expects alpha as [V]; got shape={tuple(alpha.shape)}, V={int(cfg3.V)}")

    return H_seq, pi, alpha


def _shape_or_scalar(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return tuple(x.shape)
    except Exception:
        pass
    # python number / numpy scalar / 0-d torch tensor都走到这
    try:
        return float(x)
    except Exception:
        return str(type(x))

def run_step3_from_step1_json(
    step1_json_path: str,
    theta_global: dict,
    *,
    device: str = "cpu",
    beta_det: float = 1.0,
    du: int = 8,
    Ka: int = 8,
    ds: int = 8,
):
    """
    Load Step1Outputs from JSON, build Step3Config from shapes, adapt interfaces, run Step3.

    NOTE: This function does NOT modify Step3/Step2 math. It only adapts shapes.
    """
    from .io import load_json, step1_outputs_from_dict

    d = load_json(step1_json_path)
    out1 = step1_outputs_from_dict(d["step1_out"], device=device)

    # infer from tensors
    # out1.H: [T,da] or [T,V,da]
    H = out1.H
    T = int(H.shape[0])
    if H.dim() == 2:
        da_infer = int(H.shape[1])
        V_infer = int(out1.alpha.shape[-1]) if hasattr(out1, "alpha") else 1
    else:
        V_infer = int(H.shape[1])
        da_infer = int(H.shape[2])

    V = V_infer
    da = da_infer

    # Kr: prefer pi lastdim, else fallback to V
    Kr = int(out1.pi.shape[-1]) if hasattr(out1, "pi") and getattr(out1.pi, "dim", lambda: 0)() >= 1 else V

    cfg3 = Step3Config(V=V, T=T, da=da, Kr=Kr, du=int(du), Ka=int(Ka), ds=int(ds), beta_det=float(beta_det))
    H_seq, pi, alpha = _adapt_step1_to_step3_inputs(out1, cfg3, device=device)

    out3 = run_step3_from_H(cfg3, H_seq=H_seq, pi=pi, alpha=alpha, theta_global=theta_global, device=device)
    return out3, cfg3
