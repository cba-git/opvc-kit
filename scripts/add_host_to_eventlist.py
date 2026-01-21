#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read eventlist.jsonl, infer a "host" per record, write back with meta.host.

Heuristic (safe default for now):
- If edges contain strings like 'net:10.20.2.66:5010->...' (either in list edge or dict edge),
  extract IPv4 candidates and pick the most frequent as host.
- If no IP found, host='local'.

Usage:
  python3 scripts/add_host_to_eventlist.py --in in.jsonl --out out.jsonl
"""
import argparse, json, re
from collections import Counter
from pathlib import Path

IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

def iter_edge_strings(e):
    # edge can be list/tuple (mixed) or dict
    if isinstance(e, (list, tuple)):
        for x in e:
            if isinstance(x, str):
                yield x
    elif isinstance(e, dict):
        for k, v in e.items():
            if isinstance(v, str):
                yield v

def infer_host_from_record(rec):
    E = rec.get("E", [])
    c = Counter()
    for e in E:
        for s in iter_edge_strings(e):
            # prioritize net:* strings but also allow any string containing IP
            ips = IPV4_RE.findall(s)
            for ip in ips:
                c[ip] += 1
    if c:
        return c.most_common(1)[0][0]
    return "local"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.outp)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with inp.open("r", encoding="utf-8") as fi, outp.open("w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            # set meta.host if missing
            host = meta.get("host")
            if not host:
                host = infer_host_from_record(rec)
                meta["host"] = host
            rec["meta"] = meta
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] wrote {n} lines -> {outp}")
    # quick peek
    print("[OK] sample meta.host:", host)

if __name__ == "__main__":
    main()
