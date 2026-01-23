#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opvc.adapters.base import DatasetConfig
from opvc.adapters.registry import get_adapter

def load_cfg(p: str) -> DatasetConfig:
    obj = json.loads(Path(p).read_text(encoding="utf-8"))
    return DatasetConfig(
        name=obj["name"],
        format=obj.get("format", "csv"),
        path=obj["path"],
        columns=obj.get("columns", {}),
        timestamp=obj.get("timestamp", {}),
        views=obj.get("views", []),
        filters=obj.get("filters", {}),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_cfg", required=True, help="configs/datasets/*.json")
    ap.add_argument("--out", required=True, help="eventlist jsonl (one record per line)")
    ap.add_argument("--delta", type=float, required=True, help="window width in seconds")
    ap.add_argument("--T", type=int, required=True, help="number of windows")
    ap.add_argument("--t0", type=float, default=None, help="optional window start in seconds")
    ap.add_argument("--max_rows", type=int, default=None, help="limit rows read from csv")
    ap.add_argument("--segment_by_host", type=int, default=1, help="1=sample=host×time-segment (paper setting); 0=single record")
    ap.add_argument("--segment_mode", default="per_host", choices=["per_host", "global"], help="how to align segment t0 when t0 is not provided")
    ap.add_argument("--max_records", type=int, default=None, help="optional cap on number of emitted records")
    ap.add_argument("--max_segments_per_host", type=int, default=None, help="optional cap on segments per host")
    args = ap.parse_args()

    cfg = load_cfg(args.dataset_cfg)
    adp = get_adapter(cfg)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with outp.open("w", encoding="utf-8") as f:
        for rec in adp.iter_eventlist_records(
            delta=args.delta,
            T=args.T,
            t0=args.t0,
            max_rows=args.max_rows,
            segment_by_host=bool(int(args.segment_by_host)),
            segment_mode=str(args.segment_mode),
            max_records=args.max_records,
            max_segments_per_host=args.max_segments_per_host,
        ):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] wrote {n} records -> {outp}")

if __name__ == "__main__":
    main()
