#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot patch to make host-based split stable:

1) scripts/run_step1.py
   - guarantee meta.host exists in step1.jsonl
   - prefer input meta.host; otherwise infer from E; fallback "local"

2) scripts/run_step2.py
   - preserve obj["meta"] into Step1Outputs.meta so src/opvc/step2.py can read meta.host via _get_by_path

3) src/opvc/step2.py
   - _split_by_host returns list[list[int]]; convert idx list -> torch.tensor before idx.numel()/tolist()

All patches are:
- idempotent (safe to run multiple times)
- backup original file with .bak_YYYYmmdd_HHMMSS
- py_compile check for patched python files

Usage:
  cd ~/opvc-kit
  python3 scripts/patch_host_support.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import sys


def backup(p: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = p.with_suffix(p.suffix + f".bak_{ts}")
    shutil.copy2(p, bak)
    return bak


def py_compile(p: Path) -> None:
    subprocess.check_call([sys.executable, "-m", "py_compile", str(p)])


def patch_run_step1(p: Path) -> bool:
    """
    Insert host inference helpers + ensure rec["meta"]["host"] exists.
    We patch by:
      - adding helper block once (marker)
      - before writing rec: enforce host in rec["meta"]
    """
    txt = p.read_text(encoding="utf-8")
    if "host inference helpers (auto-added)" in txt and "guarantee meta.host" in txt:
        print(f"[SKIP] {p} already patched")
        return False

    # 1) insert helper block after imports (best-effort)
    helper = r'''
# --- host inference helpers (auto-added) ---
import re as _re
_IPV4_RE = _re.compile(r"\b(?:(?:\d{1,3}\.){3}\d{1,3})\b")

def _iter_edge_strings(_e):
    # E can be list/tuple (mixed) or dict or str
    if _e is None:
        return
    if isinstance(_e, str):
        yield _e
        return
    if isinstance(_e, (list, tuple)):
        for x in _e:
            if isinstance(x, str):
                yield x
            elif isinstance(x, dict):
                for _, v in x.items():
                    if isinstance(v, str):
                        yield v
            elif isinstance(x, (list, tuple)):
                # nested
                for y in x:
                    if isinstance(y, str):
                        yield y
                    elif isinstance(y, dict):
                        for _, v in y.items():
                            if isinstance(v, str):
                                yield v
    elif isinstance(_e, dict):
        for _, v in _e.items():
            if isinstance(v, str):
                yield v

def _infer_host_from_E(E):
    # pick most frequent IPv4 in edge strings; fallback "local"
    if not E:
        return "local"
    cnt = {}
    for s in _iter_edge_strings(E):
        for ip in _IPV4_RE.findall(s):
            cnt[ip] = cnt.get(ip, 0) + 1
    if not cnt:
        return "local"
    return max(cnt.items(), key=lambda kv: kv[1])[0]
# --- end host inference helpers ---
'''.lstrip("\n")

    # place helper after the first import block
    ins_pos = None
    m = re.search(r"(?m)^(import .+|from .+ import .+)\s*$", txt)
    if m:
        # insert after last consecutive import line
        last = None
        for mm in re.finditer(r"(?m)^(import .+|from .+ import .+)\s*$", txt):
            last = mm
        if last:
            ins_pos = last.end()

    if ins_pos is None:
        ins_pos = 0

    txt2 = txt[:ins_pos] + "\n\n" + helper + "\n" + txt[ins_pos:]

    # 2) ensure host right before writing the output record
    # We anchor at: fout.write(json.dumps(rec,...)) or f.write(...) style.
    # In your file, it's: fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write_pat = re.compile(r'(?m)^(?P<indent>\s*)fout\.write\(\s*json\.dumps\(\s*rec\s*,\s*ensure_ascii=False\)\s*\+\s*"\\n"\s*\)\s*$')
    m2 = write_pat.search(txt2)
    if not m2:
        # fallback: any write(json.dumps(rec,...))
        write_pat2 = re.compile(r'(?m)^(?P<indent>\s*).*(?:write)\(json\.dumps\(\s*rec\b.*\)\s*\+\s*"\\n"\s*\)\s*$')
        m2 = write_pat2.search(txt2)

    if not m2:
        raise SystemExit(f"[ERR] cannot locate fout.write(json.dumps(rec...)) in {p}. Please paste the write-line block.")

    indent = m2.group("indent")
    guard = (
        f"{indent}# guarantee meta.host (from input meta.host or inferred from E)\n"
        f"{indent}try:\n"
        f"{indent}    _m = rec.get('meta') or {{}}\n"
        f"{indent}    _inm = (obj.get('meta') or {{}}) if isinstance(obj, dict) else {{}}\n"
        f"{indent}    if not _m.get('host'):\n"
        f"{indent}        _m['host'] = _inm.get('host') or _infer_host_from_E(obj.get('E') if isinstance(obj, dict) else None)\n"
        f"{indent}    rec['meta'] = _m\n"
        f"{indent}except Exception:\n"
        f"{indent}    pass\n"
    )

    # insert guard immediately before the write line
    txt3 = txt2[:m2.start()] + guard + txt2[m2.start():]

    bak = backup(p)
    p.write_text(txt3, encoding="utf-8")
    py_compile(p)
    print(f"[OK] patched {p} (backup -> {bak})")
    print("[OK] run_step1 will now guarantee meta.host (from input meta.host or inferred from E).")
    return True


def patch_run_step2(p: Path) -> bool:
    """
    Replace:
      outs.append(step1_outputs_from_dict(s1d, device=args.device))
    With:
      s1 = step1_outputs_from_dict(...)
      setattr(s1,"meta", obj.get("meta", {}) or {})
      outs.append(s1)
    """
    txt = p.read_text(encoding="utf-8")
    if "preserve obj['meta'] into Step1Outputs.meta" in txt or "setattr(s1, \"meta\"" in txt:
        print(f"[SKIP] {p} already patched")
        return False

    pat = re.compile(
        r'(?m)^(?P<indent>\s*)outs\.append\(\s*step1_outputs_from_dict\(\s*s1d\s*,\s*device=args\.device\s*\)\s*\)\s*$'
    )
    m = pat.search(txt)
    if not m:
        raise SystemExit(f"[ERR] cannot find outs.append(step1_outputs_from_dict(s1d, device=args.device)) in {p}")

    indent = m.group("indent")
    repl = (
        f"{indent}# [AUTO] preserve obj['meta'] into Step1Outputs.meta for host split\n"
        f"{indent}s1 = step1_outputs_from_dict(s1d, device=args.device)\n"
        f"{indent}try:\n"
        f"{indent}    setattr(s1, \"meta\", obj.get(\"meta\", {{}}) or {{}})\n"
        f"{indent}except Exception:\n"
        f"{indent}    pass\n"
        f"{indent}outs.append(s1)\n"
    )

    txt2 = txt[:m.start()] + repl + txt[m.end():]

    bak = backup(p)
    p.write_text(txt2, encoding="utf-8")
    py_compile(p)
    print(f"[OK] patched {p} (backup -> {bak})")
    print("[OK] run_step2 will now preserve obj['meta'] into Step1Outputs.meta")
    return True


def patch_step2_py(p: Path) -> bool:
    """
    In src/opvc/step2.py, inside train_step2_federated:
      for cid, idx in enumerate(shards):
          if idx.numel() == 0:
    insert conversion idx=list -> torch.tensor before idx.numel().
    """
    txt = p.read_text(encoding="utf-8")
    if "idx list -> torch.tensor" in txt:
        print(f"[SKIP] {p} already patched")
        return False

    # locate the block: for cid, idx in enumerate(shards): \n <indent>if idx.numel() == 0:
    pat = re.compile(
        r'(?m)^(?P<indent>\s*)for\s+cid\s*,\s*idx\s+in\s+enumerate\(shards\)\s*:\s*\n(?P=indent)\s+if\s+idx\.numel\(\)\s*==\s*0\s*:\s*$'
    )
    m = pat.search(txt)
    if not m:
        raise SystemExit(f"[ERR] cannot locate 'for cid, idx in enumerate(shards)' + 'if idx.numel()==0' pattern in {p}")

    indent = m.group("indent")
    inner = indent + "    "
    inject = (
        f"{indent}for cid, idx in enumerate(shards):\n"
        f"{inner}# [AUTO] idx list -> torch.tensor (for numel/tolist)\n"
        f"{inner}if not torch.is_tensor(idx):\n"
        f"{inner}    idx = torch.tensor(idx, device=dev, dtype=torch.long)\n"
        f"{inner}if idx.numel() == 0:\n"
    )

    # replace only the two lines header+if with injected+if
    start = m.start()
    end = m.end()
    txt2 = txt[:start] + inject + txt[end:]  # keeps the remainder of the function intact

    bak = backup(p)
    p.write_text(txt2, encoding="utf-8")
    py_compile(p)
    print(f"[OK] patched {p} (backup -> {bak})")
    print("[OK] step2.py: idx list -> torch.tensor")
    return True


def repo_root() -> Path:
    # assume script lives under <root>/scripts/
    here = Path(__file__).resolve()
    root = here.parent.parent
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="opvc-kit repo root (default: auto from this script path)")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else repo_root()
    files = {
        "run_step1": root / "scripts" / "run_step1.py",
        "run_step2": root / "scripts" / "run_step2.py",
        "step2": root / "src" / "opvc" / "step2.py",
    }

    for k, p in files.items():
        if not p.exists():
            raise SystemExit(f"[ERR] missing {k} file: {p}")

    changed = False
    changed |= patch_run_step1(files["run_step1"])
    changed |= patch_run_step2(files["run_step2"])
    changed |= patch_step2_py(files["step2"])

    print("\n[DONE] patch_host_support complete." if changed else "\n[DONE] nothing to patch (already up-to-date).")

    print("\n--- Verification commands ---")
    print("1) After you run step1, verify meta.host exists for all lines:")
    print(r'''python3 - <<'PY'
import json, collections
p = r"/path/to/step1.jsonl"
cnt=collections.Counter(); n=0; missing=0
with open(p,"r",encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        n+=1
        o=json.loads(line)
        h=(o.get("meta") or {}).get("host")
        if not h: missing+=1
        cnt[str(h)] += 1
print("lines =", n)
print("missing_meta.host =", missing)
print("host_counts =", dict(cnt))
PY''')
    print("\n2) Run a quick syntax check:")
    print("python3 -m py_compile scripts/run_step1.py scripts/run_step2.py src/opvc/step2.py")


if __name__ == "__main__":
    main()
