"""opvc.data

Engineering / data-side deterministic utilities:
- windowing events into T windows with width delta
- deterministic aggregation Agg_v: events -> fixed-dim vectors
- quality statistics required by Step1 A5 (cov/val/cmp/unq/stb)

This module is intentionally lightweight and auditable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

Tensor = torch.Tensor


def build_window_ranges(t0: float, delta: float, T: int) -> List[Tuple[float, float]]:
    """Return list of half-open windows W_tau = [t0+(tau-1)Δ, t0+tauΔ)."""
    t0 = float(t0)
    delta = float(delta)
    return [(t0 + i * delta, t0 + (i + 1) * delta) for i in range(int(T))]


def _event_ts(e: Any) -> float:
    if isinstance(e, Mapping):
        for k in ("ts", "timestamp", "time", "t"):
            if k in e:
                return float(e[k])
    if isinstance(e, (list, tuple)) and len(e) >= 1:
        return float(e[0])
    raise KeyError("event must contain timestamp under key ts/timestamp/time/t or be (ts,...) tuple")


def assign_events_to_windows(events: Sequence[Any], t0: float, delta: float, T: int) -> List[List[Any]]:
    """Assign events into T windows by timestamp."""
    T = int(T)
    buckets: List[List[Any]] = [[] for _ in range(T)]
    if not events:
        return buckets
    t0 = float(t0)
    delta = float(delta)
    for e in events:
        ts = _event_ts(e)
        k = int(np.floor((ts - t0) / delta))
        if 0 <= k < T:
            buckets[k].append(e)
    return buckets


def _stable_hash_to_int(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16)


@dataclass
class HashingAggConfig:
    dim: int
    fields: Optional[List[str]] = None
    seed: int = 0
    include_field_names: bool = True
    use_signed_hash: bool = True


def agg_hashing(events: Sequence[Any], cfg: HashingAggConfig) -> Tensor:
    """Deterministic feature hashing aggregator.

    For each event, we turn selected fields into tokens and count into hash buckets.
    This is a generic auditable Agg_v; you can replace with a domain-specific aggregator
    without changing Step1/2/3 model code.
    """
    x = torch.zeros(int(cfg.dim), dtype=torch.float32)
    if not events:
        return x
    fields = cfg.fields
    for e in events:
        if not isinstance(e, Mapping):
            # try to stringify whole event
            tok = str(e)
            idx = (_stable_hash_to_int(f"{cfg.seed}:{tok}") % cfg.dim)
            sgn = -1.0 if (cfg.use_signed_hash and ((_stable_hash_to_int(tok) >> 1) & 1)) else 1.0
            x[idx] += sgn
            continue

        if fields is None:
            items = list(e.items())
        else:
            items = [(k, e.get(k, None)) for k in fields]

        for k, v in items:
            if v is None:
                continue
            if isinstance(v, (list, tuple, set)):
                vals = list(v)
            else:
                vals = [v]
            for vv in vals:
                if vv is None:
                    continue
                if cfg.include_field_names:
                    tok = f"{k}={vv}"
                else:
                    tok = str(vv)
                idx = (_stable_hash_to_int(f"{cfg.seed}:{tok}") % cfg.dim)
                sgn = -1.0 if (cfg.use_signed_hash and ((_stable_hash_to_int(tok) >> 1) & 1)) else 1.0
                x[idx] += sgn
    return x


@dataclass
class QualityStats:
    total_events: int = 0
    parse_success: int = 0
    keyfield_nonempty: int = 0
    keyfield_total: int = 0
    unique_events: int = 0
    duplicate_events: int = 0
    window_counts: Optional[List[int]] = None


def compute_quality_stats(
    windows: Sequence[Sequence[Any]],
    key_fields: Optional[Sequence[str]] = None,
    dedup_fields: Optional[Sequence[str]] = None,
) -> QualityStats:
    """Compute auditable quality statistics from windowed events."""
    stats = QualityStats()
    stats.window_counts = [len(w) for w in windows]
    key_fields = list(key_fields) if key_fields is not None else []
    dedup_fields = list(dedup_fields) if dedup_fields is not None else []
    seen = set()

    for w in windows:
        for e in w:
            stats.total_events += 1
            if isinstance(e, Mapping):
                # parse success marker
                ok = e.get("parse_ok", e.get("ok", None))
                if ok is None:
                    # heuristic: has timestamp and at least one other key
                    ok = True if len(e.keys()) >= 2 else False
                if bool(ok):
                    stats.parse_success += 1

                if key_fields:
                    for k in key_fields:
                        stats.keyfield_total += 1
                        v = e.get(k, None)
                        if v is not None and str(v) != "":
                            stats.keyfield_nonempty += 1

                if dedup_fields:
                    key = tuple((k, e.get(k, None)) for k in dedup_fields)
                else:
                    # fallback: hash of sorted items (excluding ts to make duplicates stable)
                    items = sorted((kk, vv) for kk, vv in e.items() if kk not in ("ts", "timestamp", "time", "t"))
                    key = tuple(items)
                seen.add(key)
            else:
                seen.add(str(e))

    stats.unique_events = len(seen)
    stats.duplicate_events = max(0, stats.total_events - stats.unique_events)
    return stats


def quality_components(
    windows: Sequence[Sequence[Any]],
    stats: QualityStats,
    eps: float = 1e-8,
) -> Tuple[float, float, float, float, float]:
    """Compute q_cov/q_val/q_cmp/q_unq/q_stb from windowed events + stats."""
    T = max(1, len(windows))
    empty = sum(1 for w in windows if len(w) == 0)
    q_cov = 1.0 - float(empty) / float(T)

    if stats.total_events <= 0:
        q_val = 0.0
        q_cmp = 0.0
        q_unq = 1.0
    else:
        q_val = float(stats.parse_success) / float(stats.total_events)
        if stats.keyfield_total <= 0:
            q_cmp = 1.0
        else:
            q_cmp = float(stats.keyfield_nonempty) / float(stats.keyfield_total)
        dup_rate = float(stats.duplicate_events) / float(stats.total_events)
        q_unq = 1.0 - dup_rate

    # stability based on window count CV
    counts = np.asarray(stats.window_counts if stats.window_counts is not None else [0] * T, dtype=np.float32)
    mu = float(counts.mean())
    sd = float(counts.std())
    cv = sd / (mu + eps)
    q_stb = 1.0 / (1.0 + cv)

    # clamp into [0,1]
    def clamp01(x: float) -> float:
        return float(min(1.0, max(0.0, x)))

    return clamp01(q_cov), clamp01(q_val), clamp01(q_cmp), clamp01(q_unq), clamp01(q_stb)


@dataclass
class WindowAggResult:
    x_win: Tensor                  # [T, d_in]
    windows: List[List[Any]]        # events per window
    stats: QualityStats             # raw statistics


def window_and_agg(
    events: Sequence[Any],
    t0: float,
    delta: float,
    T: int,
    agg_cfg: HashingAggConfig,
    key_fields: Optional[Sequence[str]] = None,
    dedup_fields: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
) -> WindowAggResult:
    """Full A1+A2 engineering step for one view."""
    windows = assign_events_to_windows(events, t0=float(t0), delta=float(delta), T=int(T))
    stats = compute_quality_stats(windows, key_fields=key_fields, dedup_fields=dedup_fields)
    x_list = [agg_hashing(w, agg_cfg) for w in windows]
    x_win = torch.stack(x_list, dim=0)  # [T,dim]
    if device is not None:
        x_win = x_win.to(device)
    return WindowAggResult(x_win=x_win, windows=windows, stats=stats)
