#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch src/opvc/step1.py so that meta from input record (rec.get("meta")) is preserved.

We do NOT assume the exact code structure; we search a safe anchor:
- Find a line that assigns meta = {...}  (or meta = dict(...))
- Immediately after that assignment, inject:
    if isinstance(rec, dict) and isinstance(rec.get("meta"), dict):
        meta = {**rec["meta"], **meta}

This keeps step1's meta keys while ensuring upstream meta (e.g., host) survives.
"""
import argparse, re, shutil
from datetime import datetime
from pathlib import Path

DEFAULT_FILE = "src/opvc/step1.py"

INJECT = """\
    # --- injected: preserve adapter/eventlist meta (e.g., meta.host) ---
    if isinstance(rec, dict) and isinstance(rec.get("meta"), dict):
        meta = {**rec["meta"], **meta}
"""

def backup(p: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = p.with_suffix(p.suffix + f".bak_meta_{ts}")
    shutil.copy2(p, bak)
    return bak

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--rec-var", default="rec", help="name of record variable in the loop (default: rec)")
    args = ap.parse_args()

    p = Path(args.file)
    txt = p.read_text(encoding="utf-8").splitlines(True)

    # If already patched, exit
    if "preserve adapter/eventlist meta" in "".join(txt):
        print(f"[OK] already patched: {p}")
        return

    # try find "meta = {"
    meta_assign_idx = None
    meta_assign_pat = re.compile(r"^\s*meta\s*=\s*(\{|\bdict\s*\()")
    for i, line in enumerate(txt):
        if meta_assign_pat.search(line):
            meta_assign_idx = i
            break

    if meta_assign_idx is None:
        print("ERROR: could not find a 'meta = {...}' or 'meta = dict(...)' assignment to patch.")
        print("HINT: open src/opvc/step1.py and search 'meta =' then tell me the surrounding 10 lines.")
        raise SystemExit(1)

    # infer indentation from meta assignment line
    indent = re.match(r"^(\s*)", txt[meta_assign_idx]).group(1)
    inj = INJECT.replace("rec", args.rec_var)
    inj = "".join([indent + l if l.strip() else l for l in inj.splitlines(True)])

    bak = backup(p)
    # insert right after the meta assignment
    txt.insert(meta_assign_idx + 1, inj)
    p.write_text("".join(txt), encoding="utf-8")

    print(f"[OK] backup -> {bak}")
    print(f"[OK] patched -> {p} (insert after line {meta_assign_idx+1})")
    print("[OK] step1 will now preserve upstream meta (including meta.host).")

if __name__ == "__main__":
    main()
