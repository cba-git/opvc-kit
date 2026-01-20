#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def _abspath(p: str | None) -> str | None:
    if p is None:
        return None
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((REPO / pp).resolve())

def run(cmd, env=None):
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd, env=env)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline_cfg", required=True, help="configs/pipelines/*.json")
    ap.add_argument("--workdir", default=None, help="where to write artifacts; default=outputs/<name>")
    ap.add_argument("--keep", action="store_true", help="keep intermediate files (default keeps anyway)")
    args = ap.parse_args()

    cfg_path = Path(args.pipeline_cfg)
    if not cfg_path.is_absolute():
        cfg_path = (REPO / cfg_path).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    name = cfg.get("name", cfg_path.stem)
    workdir = Path(args.workdir) if args.workdir else (REPO / "outputs" / name)
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = _abspath(cfg["dataset_cfg"])

    ev = cfg.get("eventlist", {})
    s1 = cfg.get("step1", {})
    s2 = cfg.get("step2", {})
    s3 = cfg.get("step3", {})

    # outputs
    eventlist_out = workdir / "eventlist.jsonl"
    step1_out = workdir / "step1.jsonl"
    theta_out = workdir / "step2_theta.pt"
    step3_out = workdir / "step3_out.json"

    env = os.environ.copy()
    # make subprocess scripts import src/
    env["PYTHONPATH"] = str(REPO / "src") + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    # 0) build_eventlist
    cmd0 = [
        sys.executable, str(REPO / "scripts" / "build_eventlist.py"),
        "--dataset_cfg", dataset_cfg,
        "--out", str(eventlist_out),
        "--delta", str(ev["delta"]),
        "--T", str(ev["T"]),
    ]
    if ev.get("t0", None) is not None:
        cmd0 += ["--t0", str(ev["t0"])]
    if ev.get("max_rows", None) is not None:
        cmd0 += ["--max_rows", str(ev["max_rows"])]
    run(cmd0, env=env)

    # 1) step1
    cmd1 = [
        sys.executable, str(REPO / "scripts" / "run_step1.py"),
        "--in_eventlist", str(eventlist_out),
        "--out_step1", str(step1_out),
        "--d_in", str(s1.get("d_in", 256)),
        "--d", str(s1.get("d", 64)),
        "--da", str(s1.get("da", 32)),
        "--Kr", str(s1.get("Kr", 4)),
        "--tau_q", str(s1.get("tau_q", 1.0)),
        "--theta", str(s1.get("theta", 0.5)),
        "--device", str(s1.get("device", "cpu")),
    ]
    run(cmd1, env=env)

    # 2) step2
    cmd2 = [
        sys.executable, str(REPO / "scripts" / "run_step2.py"),
        "--step1_jsonl", str(step1_out),
        "--out_pt", str(theta_out),
        "--du", str(s2.get("du", 16)),
        "--Ka", str(s2.get("Ka", 0)),
        "--rounds", str(s2.get("rounds", 1)),
        "--num_clients", str(s2.get("num_clients", 2)),
        "--local_epochs", str(s2.get("local_epochs", 1)),
        "--lr", str(s2.get("lr", 1e-3)),
        "--Cb", str(s2.get("Cb", 1.0)),
        "--sigma_b0", str(s2.get("sigma_b0", 0.5)),
        "--seed", str(s2.get("seed", 0)),
        "--device", str(s2.get("device", "cpu")),
    ]
    run(cmd2, env=env)

    # 3) step3
    cmd3 = [
        sys.executable, str(REPO / "scripts" / "run_step3.py"),
        "--step1_json", str(step1_out),   # jsonl OK: script uses first line by default
        "--theta", str(theta_out),
        "--out", str(step3_out),
        "--ds", str(s3.get("ds", 16)),
        "--beta_det", str(s3.get("beta_det", 1.0)),
        "--device", str(s3.get("device", "cpu")),
    ]
    if s3.get("Ka_override", None) is not None:
        cmd3 += ["--Ka", str(s3["Ka_override"])]
    run(cmd3, env=env)

    print("\n[OK] pipeline done")
    print("workdir:", workdir)
    print("eventlist:", eventlist_out)
    print("step1:", step1_out)
    print("theta:", theta_out)
    print("step3:", step3_out)

if __name__ == "__main__":
    main()
