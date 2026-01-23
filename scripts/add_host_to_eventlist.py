#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts.add_host_to_eventlist

Read an eventlist.jsonl and ensure each record contains:
- meta.host
- meta.node
- meta.node_id

This is a *data repair* utility. It does not modify any model logic.

Heuristic (safe default):
- Count all IPv4 strings inside record["E"] (nested structures allowed).
- Use the most frequent IP as host; if none is found, fall back to "local".

Usage:
  python3 scripts/add_host_to_eventlist.py --in in.jsonl --out out.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make "import opvc" work when running from repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opvc.host import infer_ipv4_from_any


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.outp)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    last_host = None

    with inp.open("r", encoding="utf-8") as fi, outp.open("w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            meta = rec.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            meta = dict(meta)

            # set meta.host if missing
            host = meta.get("host")
            if not host:
                host = infer_ipv4_from_any(rec.get("E"), default="local")
                meta["host"] = host

            # explicit node aliases for downstream label alignment
            meta.setdefault("node", meta.get("host"))
            meta.setdefault("node_id", meta.get("node"))

            rec["meta"] = meta
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n += 1
            last_host = host

    print(f"[OK] wrote {n} lines -> {outp}")
    if last_host is not None:
        print("[OK] sample meta.host:", last_host)


if __name__ == "__main__":
    main()
