from __future__ import annotations

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
    return out

