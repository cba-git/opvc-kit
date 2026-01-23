#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, csv
from pathlib import Path
from typing import Any, Dict, Optional

UNIT_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}

def load_dataset_cfg(p: str) -> Dict[str, Any]:
    obj = json.loads(Path(p).read_text(encoding="utf-8"))
    if "name" not in obj or "path" not in obj:
        raise SystemExit("[ERR] dataset_cfg missing required keys: name/path")
    obj.setdefault("format", "csv")
    obj.setdefault("columns", {})
    obj.setdefault("timestamp", {})
    obj.setdefault("views", [])
    obj.setdefault("filters", {})
    return obj

def scan_minmax_csv(csv_path: str, ts_col: str, unit: str, max_rows: Optional[int] = None):
    scale = UNIT_SCALE.get(unit)
    if scale is None:
        raise SystemExit(f"[ERR] unknown timestamp unit: {unit} (supported: {list(UNIT_SCALE.keys())})")

    min_ts = None
    max_ts = None
    n = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if ts_col not in (r.fieldnames or []):
            raise SystemExit(f"[ERR] ts_col '{ts_col}' not found in csv header. fields={r.fieldnames}")

        for row in r:
            if max_rows is not None and n >= int(max_rows):
                break
            v = row.get(ts_col)
            if v is None or v == "":
                continue
            try:
                ts = float(v) * float(scale)  # -> seconds
            except Exception:
                continue

            min_ts = ts if (min_ts is None or ts < min_ts) else min_ts
            max_ts = ts if (max_ts is None or ts > max_ts) else max_ts
            n += 1

    if min_ts is None or max_ts is None or n == 0:
        raise SystemExit("[ERR] no valid timestamps scanned")
    return float(min_ts), float(max_ts), int(n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_cfg", required=True, help="configs/datasets/*.json")
    ap.add_argument("--out_profile", required=True, help="output profile json")
    ap.add_argument("--delta_s", type=float, default=60.0, help="recommended delta (seconds)")
    ap.add_argument("--T_fixed_2h", type=int, default=120, help="T for fixed 2h window when delta=60")
    ap.add_argument("--max_rows", type=int, default=None, help="optional scan cap (for huge csv)")
    ap.add_argument("--out_pipeline_cfg", default=None, help="if set, also write configs/pipelines/*.json")
    args = ap.parse_args()

    ds = load_dataset_cfg(args.dataset_cfg)
    if (ds.get("format") or "").lower() != "csv":
        raise SystemExit(f"[ERR] only csv supported in this profiler, got format={ds.get('format')}")

    cols = ds.get("columns", {})
    ts_col = cols.get("ts")
    if not ts_col:
        raise SystemExit("[ERR] dataset_cfg.columns.ts is required")
    unit = (ds.get("timestamp") or {}).get("unit", "ns")

    csv_path = ds["path"]
    if not Path(csv_path).exists():
        raise SystemExit(f"[ERR] csv not found: {csv_path}")

    min_ts, max_ts, n_rows = scan_minmax_csv(csv_path, ts_col=ts_col, unit=unit, max_rows=args.max_rows)

    delta = float(args.delta_s)
    t0 = math.floor(min_ts / delta) * delta
    span = max_ts - t0
    T_cover_all = int(math.ceil(span / delta)) if span > 0 else 1

    out = {
        "dataset": {
            "name": ds.get("name"),
            "path": csv_path,
            "format": ds.get("format"),
            "ts_col": ts_col,
            "ts_unit": unit,
        },
        "scan": {
            "min_ts_s": float(min_ts),
            "max_ts_s": float(max_ts),
            "n_rows_used": int(n_rows),
        },
        "recommend": {
            "delta_s": float(delta),
            "t0_s": float(t0),
            "T_cover_all": int(T_cover_all),
            "T_fixed_2h": int(args.T_fixed_2h),
        },
    }

    op = Path(args.out_profile)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[OK] wrote", op)

    if args.out_pipeline_cfg:
        # 默认一套“论文口径 + 可复用”的 pipeline 参数（后续你可以按需要改默认值）
        pipe = {
            "name": f"{ds.get('name')}_auto",
            "dataset_cfg": str(Path(args.dataset_cfg)),
            "eventlist": {
                "delta": int(delta) if abs(delta - int(delta)) < 1e-9 else float(delta),
                # paper setting: fixed segment length; defaults to 2h if delta=60s
                "T": int(args.T_fixed_2h),
                # if provided, this is the global alignment; otherwise build_eventlist can align per-host
                "t0": int(t0) if abs(t0 - int(t0)) < 1e-9 else float(t0),
                "max_rows": None,
                "segment_by_host": True,
                "segment_mode": "per_host",
                "max_records": None,
                "max_segments_per_host": None,
            },
            "step1": {
                "d_in": 256, "d": 64, "da": 32, "Kr": 4,
                "tau_q": 1.0, "theta": 0.5, "device": "cpu",
            },
            "step2": {
                "du": 16, "Ka": 0, "rounds": 1, "num_clients": 2,
                "local_epochs": 1, "lr": 1e-3, "Cb": 1.0, "sigma_b0": 0.5,
                "seed": 0, "device": "cpu",
            },
            "step3": {
                "ds": 16, "beta_det": 1.0,
                "Ka_override": 1,  # cfg2.Ka=0 时用于 smoke test
                "device": "cpu",
            },
        }

        pp = Path(args.out_pipeline_cfg)
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text(json.dumps(pipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("[OK] wrote", pp)

if __name__ == "__main__":
    main()
