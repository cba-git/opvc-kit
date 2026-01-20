from __future__ import annotations

import csv
import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .base import DatasetAdapter, DatasetConfig

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

        # We will build ONE record (one sample) from the whole CSV (or max_rows).
        # Later you can extend to multiple samples by splitting time ranges.
        E: List[List[Dict[str, Any]]] = [[] for _ in range(len(views))]
        ts_list: List[float] = []

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

                ts = self._ts_to_seconds(float(raw))
                ts_list.append(ts)

                # Common fields
                # NOTE: keep strings; hashing aggregator in Step1 can consume them.
                base = {}
                for k, v in row.items():
                    if k == ts_col:
                        continue
                    if v is None:
                        continue
                    vv = v.strip() if isinstance(v, str) else v
                    if vv == "":
                        continue
                    base[k] = vv

                # Build per-view events. Make views actually different by:
                # - putting the configured entity column into a normalized key "entity"
                # - putting the opposite node into "peer" when possible
                src_node_col = cols.get("src_node")
                dst_node_col = cols.get("dst_node")

                for i, vw in enumerate(views):
                    ent_col = vw.get("entity")
                    ent_val = row.get(ent_col, None) if ent_col else None
                    e = {"ts": ts, "parse_ok": True}
                    e.update(base)
                    e["view"] = vw.get("name", f"v{i}")
                    if ent_val is not None:
                        e["entity"] = ent_val

                    # peer heuristic for common src/dst datasets
                    if src_node_col and dst_node_col and ent_col:
                        if ent_col == src_node_col:
                            e["peer"] = row.get(dst_node_col, None)
                        elif ent_col == dst_node_col:
                            e["peer"] = row.get(src_node_col, None)

                    E[i].append(e)

                n += 1

        if not ts_list:
            raise SystemExit("[ERR] no valid timestamps after filtering")

        min_ts = min(ts_list)
        max_ts = max(ts_list)

        if t0 is None:
            t0_use = math.floor(min_ts / float(delta)) * float(delta)
        else:
            t0_use = float(t0)

        rec = {
            "meta": {
                "dataset": self.cfg.name,
                "src": self.cfg.path,
                "min_ts": float(min_ts),
                "max_ts": float(max_ts),
                "n_rows_used": int(len(ts_list)),
                "V": int(len(E)),
                "views": views,
            },
            "t0": float(t0_use),
            "delta": float(delta),
            "T": int(T),
            "E": E,
        }
        yield rec
