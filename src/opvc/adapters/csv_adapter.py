from __future__ import annotations

from ..host import IPV4_RE, host_from_node_str

import csv
import math
from typing import Any, Dict, List, Optional, Tuple

from .base import DatasetAdapter

UNIT_SCALE = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}


class CSVDatasetAdapter(DatasetAdapter):
    def _ts_to_seconds(self, ts_raw: float) -> float:
        unit = (self.cfg.timestamp or {}).get("unit", "ns")
        scale = UNIT_SCALE.get(unit)
        if scale is None:
            raise ValueError(f"Unknown timestamp unit: {unit}")
        ts = float(ts_raw) * float(scale)
        # config says to_seconds: True/False; if False, still treat as seconds already
        # (kept for compatibility)
        return ts

    def iter_eventlist_records(
        self,
        delta: float,
        T: int,
        t0: Optional[float] = None,
        max_rows: Optional[int] = None,
        segment_by_host: bool = True,
        segment_mode: str = "per_host",
        max_records: Optional[int] = None,
        max_segments_per_host: Optional[int] = None,
    ):
        cols = self.cfg.columns
        ts_col = cols.get("ts")
        op_col = cols.get("op")
        if not ts_col:
            raise ValueError("cfg.columns.ts is required")
        if not op_col:
            raise ValueError("cfg.columns.op is required")

        views = self.cfg.views or []
        if not views:
            raise ValueError("cfg.views is empty")

        allowed_ops = (self.cfg.filters or {}).get("allowed_ops", None)
        allowed_ops_set = set(allowed_ops) if isinstance(allowed_ops, list) else None

        # Paper-level sample definition: sample = host × fixed time segment.
        # We therefore produce MULTIPLE records from a single CSV.
        # Each record covers exactly [t0, t0 + T*delta) seconds for one host.

        # --- host inference per row ---

        host_col = None
        # prefer explicit config: columns.host OR filters.host_column
        if isinstance(cols, dict) and cols.get("host"):
            host_col = cols.get("host")
        if isinstance(self.cfg.filters, dict) and self.cfg.filters.get("host_column"):
            host_col = self.cfg.filters.get("host_column")

        entity_cols = [vw.get("entity") for vw in views if isinstance(vw, dict) and vw.get("entity")]
        candidates = [c for c in [host_col, cols.get("src_node"), cols.get("dst_node"), *entity_cols] if c]

        def _infer_host_from_row(row: Dict[str, Any]) -> str:
            for c in candidates:
                v = row.get(c)
                if v is None:
                    continue
                h = host_from_node_str(str(v))
                if h:
                    return h
                m = IPV4_RE.search(str(v))
                if m:
                    return m.group(0)
            return "unknown"

        # read + normalize rows
        rows: List[Tuple[str, float, Dict[str, Any]]] = []
        ts_all: List[float] = []

        with open(self.cfg.path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            n = 0
            for row in r:
                if max_rows is not None and n >= int(max_rows):
                    break

                raw = row.get(ts_col)
                if raw is None or raw == "":
                    continue

                op = row.get(op_col, "")
                if allowed_ops_set is not None and op not in allowed_ops_set:
                    continue

                try:
                    ts = self._ts_to_seconds(float(raw))
                except Exception:
                    continue

                host = _infer_host_from_row(row) if segment_by_host else "global"

                # Common fields (keep strings; hashing aggregator in Step1 can consume them)
                base: Dict[str, Any] = {}
                for k, v in row.items():
                    if k == ts_col:
                        continue
                    if v is None:
                        continue
                    vv = v.strip() if isinstance(v, str) else v
                    if vv == "":
                        continue
                    base[k] = vv

                rows.append((host, float(ts), base))
                ts_all.append(float(ts))
                n += 1

        if not rows:
            raise SystemExit("[ERR] no valid timestamps after filtering")

        # deterministic ordering (host, ts)
        rows.sort(key=lambda x: (x[0], x[1]))

        seg_len = float(delta) * float(int(T))
        if seg_len <= 0:
            raise ValueError("delta*T must be > 0")

        # compute base t0 per host or global
        if t0 is not None:
            base_t0_global = float(t0)
        else:
            base_t0_global = math.floor(min(ts_all) / float(delta)) * float(delta)

        host_min: Dict[str, float] = {}
        if segment_mode not in ("per_host", "global"):
            raise ValueError("segment_mode must be 'per_host' or 'global'")
        if segment_mode == "per_host" and t0 is None:
            for h, ts, _ in rows:
                host_min[h] = ts if (h not in host_min or ts < host_min[h]) else host_min[h]

        def _base_t0_for_host(h: str) -> float:
            if segment_mode == "global" or t0 is not None:
                return base_t0_global
            # per-host aligned to delta
            mt = host_min.get(h, base_t0_global)
            return math.floor(mt / float(delta)) * float(delta)

        # groups[(host, seg_idx)] -> E lists
        groups: Dict[Tuple[str, int], List[List[Dict[str, Any]]]] = {}
        meta_stats: Dict[Tuple[str, int], Dict[str, Any]] = {}
        per_host_seg_count: Dict[str, int] = {}

        src_node_col = cols.get("src_node")
        dst_node_col = cols.get("dst_node")

        for host, ts, base in rows:
            bt0 = _base_t0_for_host(host)
            seg_idx = int(math.floor((ts - bt0) / seg_len)) if seg_len > 0 else 0
            if seg_idx < 0:
                continue

            # enforce per-host segment cap (deterministic)
            if max_segments_per_host is not None:
                seen = per_host_seg_count.get(host, 0)
                if seg_idx >= int(max_segments_per_host):
                    continue
                if seen < seg_idx + 1:
                    per_host_seg_count[host] = seg_idx + 1

            key = (host, seg_idx)
            if key not in groups:
                groups[key] = [[] for _ in range(len(views))]
                meta_stats[key] = {"min_ts": ts, "max_ts": ts, "n_rows": 0}
            ms = meta_stats[key]
            ms["min_ts"] = ts if ts < float(ms["min_ts"]) else float(ms["min_ts"])
            ms["max_ts"] = ts if ts > float(ms["max_ts"]) else float(ms["max_ts"])
            ms["n_rows"] = int(ms["n_rows"]) + 1

            # Build per-view events.
            for i, vw in enumerate(views):
                ent_col = vw.get("entity") if isinstance(vw, dict) else None
                ent_val = base.get(ent_col, None) if ent_col else None
                e: Dict[str, Any] = {"ts": ts, "parse_ok": True}
                e.update(base)
                e["view"] = (vw.get("name") if isinstance(vw, dict) else None) or f"v{i}"
                if ent_val is not None:
                    e["entity"] = ent_val
                if src_node_col and dst_node_col and ent_col:
                    if ent_col == src_node_col:
                        e["peer"] = base.get(dst_node_col, None)
                    elif ent_col == dst_node_col:
                        e["peer"] = base.get(src_node_col, None)
                groups[key][i].append(e)

        # yield records in deterministic order
        keys = sorted(groups.keys(), key=lambda x: (x[0], x[1]))
        out_n = 0
        for host, seg_idx in keys:
            if max_records is not None and out_n >= int(max_records):
                break
            bt0 = _base_t0_for_host(host)
            seg_start = bt0 + float(seg_idx) * seg_len
            E = groups[(host, seg_idx)]
            ms = meta_stats[(host, seg_idx)]

            rec = {
                "meta": {
                    "dataset": self.cfg.name,
                    "src": self.cfg.path,
                    "host": host,
                    # Some downstream label files / evaluation scripts may refer to the
                    # sample's machine identifier as "node" instead of "host".
                    # We store a redundant alias for robustness.
                    "node": host,
                    "node_id": host,
                    "segment_idx": int(seg_idx),
                    "segment_mode": segment_mode,
                    "segment_len_s": float(seg_len),
                    "sample_id": f"{host}__{int(seg_start)}__{int(T)}w",
                    "min_ts": float(ms.get("min_ts", seg_start)),
                    "max_ts": float(ms.get("max_ts", seg_start + seg_len)),
                    "n_rows_used": int(ms.get("n_rows", 0)),
                    "V": int(len(E)),
                    "views": views,
                },
                "t0": float(seg_start),
                "delta": float(delta),
                "T": int(T),
                "E": E,
            }
            yield rec
            out_n += 1
