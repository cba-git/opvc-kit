#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    p = Path(args.in_jsonl)
    n = 0
    det = []
    host_stats = defaultdict(lambda: {"n": 0, "det_gt_0.5": 0, "unknown": 0})

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            n += 1
            p_det = float(o.get("p_det", 0.0))
            det.append(p_det)
            meta = o.get("meta", {}) or {}
            host = str(meta.get("host", "unknown"))
            st = host_stats[host]
            st["n"] += 1
            if p_det > 0.5:
                st["det_gt_0.5"] += 1
            if bool(o.get("flag_unknown", False)):
                st["unknown"] += 1

    if n == 0:
        raise SystemExit("[ERR] empty input")

    det_sorted = sorted(det)
    def q(qv: float) -> float:
        i = int(qv * (len(det_sorted) - 1))
        return float(det_sorted[max(0, min(i, len(det_sorted) - 1))])

    print(f"[OK] n={n}")
    print("p_det quantiles:")
    for qq in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
        print(f"  q{int(qq*100):02d}: {q(qq):.6f}")

    rows = []
    for host, st in host_stats.items():
        rows.append((st["det_gt_0.5"], st["unknown"], st["n"], host))
    rows.sort(reverse=True)

    print("\nTop hosts by (#p_det>0.5, #unknown, #samples):")
    for i, (a, u, nn, host) in enumerate(rows[: int(args.topk)]):
        print(f"  {i+1:02d}. host={host}  det>0.5={a}/{nn}  unknown={u}/{nn}")


if __name__ == "__main__":
    main()
