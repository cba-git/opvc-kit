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

    ev = cfg.get("eventlist", {}) or {}
    # Support both legacy flat step configs and the newer {model, train, device} layout.
    s1_sec = cfg.get("step1", {}) or {}
    s2_sec = cfg.get("step2", {}) or {}
    s3_sec = cfg.get("step3", {}) or {}

    s1_model = (s1_sec.get("model") if isinstance(s1_sec, dict) else None) or s1_sec
    s2_model = (s2_sec.get("model") if isinstance(s2_sec, dict) else None) or s2_sec
    s3_model = (s3_sec.get("model") if isinstance(s3_sec, dict) else None) or s3_sec

    s1_train = s1_sec.get("train", {}) if isinstance(s1_sec, dict) else {}
    s2_train = s2_sec.get("train", {}) if isinstance(s2_sec, dict) else {}
    s3_train = s3_sec.get("train", {}) if isinstance(s3_sec, dict) else {}

    def _g(*maps, key: str, default=None):
        for m in maps:
            if isinstance(m, dict) and key in m and m[key] is not None:
                return m[key]
        return default

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
        sys.executable,
        str(REPO / "scripts" / "build_eventlist.py"),
        "--dataset_cfg",
        dataset_cfg,
        "--out",
        str(eventlist_out),
        "--delta",
        str(ev["delta"]),
        "--T",
        str(ev["T"]),
    ]
    if ev.get("t0", None) is not None:
        cmd0 += ["--t0", str(ev["t0"])]
    if ev.get("max_rows", None) is not None:
        cmd0 += ["--max_rows", str(ev["max_rows"])]

    # paper-level sample definition (host×fixed segment)
    if ev.get("segment_by_host", None) is not None:
        cmd0 += ["--segment_by_host", str(int(bool(ev["segment_by_host"])))]
    if ev.get("segment_mode", None) is not None:
        cmd0 += ["--segment_mode", str(ev["segment_mode"])]
    if ev.get("max_records", None) is not None:
        cmd0 += ["--max_records", str(ev["max_records"])]
    if ev.get("max_segments_per_host", None) is not None:
        cmd0 += ["--max_segments_per_host", str(ev["max_segments_per_host"])]

    run(cmd0, env=env)

    # 1) step1
    cmd1 = [
        sys.executable,
        str(REPO / "scripts" / "run_step1.py"),
        "--in_eventlist",
        str(eventlist_out),
        "--out_step1",
        str(step1_out),
        "--d_in",
        str(_g(s1_model, key="d_in", default=256)),
        "--d",
        str(_g(s1_model, key="d", default=64)),
        "--da",
        str(_g(s1_model, key="da", default=32)),
        "--Kr",
        str(_g(s1_model, key="Kr", default=4)),
        "--tau_q",
        str(_g(s1_model, key="tau_q", default=1.0)),
        "--theta",
        str(_g(s1_model, key="theta", default=0.5)),
        "--q_norm_momentum",
        str(_g(s1_model, key="q_norm_momentum", default=0.95)),
        "--q_norm_eps",
        str(_g(s1_model, key="q_norm_eps", default=1e-6)),
        "--device",
        str(_g(s1_sec, s1_model, key="device", default="cpu")),
    ]
    attn_scale = _g(s1_model, key="attn_scale", default=None)
    if attn_scale is not None:
        cmd1 += ["--attn_scale", str(attn_scale)]
    if _g(s1_sec, key="ckpt", default=None):
        cmd1 += ["--ckpt", _abspath(_g(s1_sec, key="ckpt", default=None))]
    run(cmd1, env=env)

    # 2) step2
    cmd2 = [
        sys.executable,
        str(REPO / "scripts" / "run_step2.py"),
        "--step1_jsonl",
        str(step1_out),
        "--out_pt",
        str(theta_out),
        "--du",
        str(_g(s2_model, key="du", default=16)),
        "--Ka",
        str(_g(s2_model, key="Ka", default=0)),
        "--rounds",
        str(_g(s2_train, s2_model, key="rounds", default=1)),
        "--num_clients",
        str(_g(s2_train, s2_model, key="num_clients", default=2)),
        "--local_epochs",
        str(_g(s2_train, s2_model, key="local_epochs", default=1)),
        "--lr",
        str(_g(s2_train, s2_model, key="lr", default=1e-3)),
        "--Cb",
        str(_g(s2_train, s2_model, key="Cb", default=1.0)),
        "--sigma_b0",
        str(_g(s2_train, s2_model, key="sigma_b0", default=0.5)),
        "--clip_min",
        str(_g(s2_train, s2_model, key="clip_min", default=0.5)),
        "--clip_max",
        str(_g(s2_train, s2_model, key="clip_max", default=2.0)),
        "--sigma_min",
        str(_g(s2_train, s2_model, key="sigma_min", default=0.2)),
        "--sigma_max",
        str(_g(s2_train, s2_model, key="sigma_max", default=1.0)),
        "--proj_momentum",
        str(_g(s2_train, s2_model, key="proj_momentum", default=0.9)),
        "--proj_temp",
        str(_g(s2_train, s2_model, key="proj_temp", default=1.0)),
        "--view_grad_mode",
        str(_g(s2_train, s2_model, key="view_grad_mode", default="approx")),
        "--lambda_asd",
        str(_g(s2_train, s2_model, key="lambda_asd", default=1.0)),
        "--lambda_nce",
        str(_g(s2_train, s2_model, key="lambda_nce", default=1.0)),
        "--tau_min",
        str(_g(s2_train, s2_model, key="tau_min", default=0.1)),
        "--tau_max",
        str(_g(s2_train, s2_model, key="tau_max", default=1.0)),
        "--kappa_u",
        str(_g(s2_train, s2_model, key="kappa_u", default=1.0)),
        "--kappa_p",
        str(_g(s2_train, s2_model, key="kappa_p", default=1.0)),
        "--dp_delta",
        str(_g(s2_train, s2_model, key="dp_delta", default=1e-5)),
        "--seed",
        str(_g(s2_train, s2_model, key="seed", default=0)),
        "--device",
        str(_g(s2_sec, s2_model, key="device", default="cpu")),
    ]

    teacher_pt = _g(s2_train, s2_sec, key="teacher_pt", default=None)
    if teacher_pt:
        cmd2 += ["--teacher_pt", _abspath(teacher_pt)]
        cmd2 += ["--teacher_batch_size", str(_g(s2_train, key="teacher_batch_size", default=64))]
        cmd2 += ["--teacher_epochs", str(_g(s2_train, key="teacher_epochs", default=0))]
        cmd2 += ["--teacher_lr", str(_g(s2_train, key="teacher_lr", default=1e-3))]
        cmd2 += ["--teacher_selfsup_epochs", str(_g(s2_train, key="teacher_selfsup_epochs", default=0))]
        cmd2 += ["--teacher_selfsup_lr", str(_g(s2_train, key="teacher_selfsup_lr", default=1e-3))]
        cmd2 += ["--teacher_selfsup_tau", str(_g(s2_train, key="teacher_selfsup_tau", default=0.2))]
        cmd2 += ["--teacher_selfsup_noise_std", str(_g(s2_train, key="teacher_selfsup_noise_std", default=0.01))]
        cmd2 += ["--teacher_selfsup_dropout", str(_g(s2_train, key="teacher_selfsup_dropout", default=0.1))]
        cmd2 += ["--teacher_selfsup_lambda", str(_g(s2_train, key="teacher_selfsup_lambda", default=1.0))]
        cmd2 += ["--teacher_selfsup_sup_lambda", str(_g(s2_train, key="teacher_selfsup_sup_lambda", default=0.0))]
    run(cmd2, env=env)

    # 3) step3
    cmd3 = [
        sys.executable,
        str(REPO / "scripts" / "run_step3.py"),
        "--step1_json",
        str(step1_out),  # jsonl OK: script uses first line by default
        "--theta",
        str(theta_out),
        "--out",
        str(step3_out),
        "--ds",
        str(_g(s3_model, key="ds", default=16)),
        "--beta_det",
        str(_g(s3_model, key="beta_det", default=1.0)),
        "--q_c",
        str(_g(s3_model, key="q_c", default=0.95)),
        "--gamma_u",
        str(_g(s3_model, key="gamma_u", default=1.0)),
        "--gamma_p",
        str(_g(s3_model, key="gamma_p", default=1.0)),
        "--gamma_pi",
        str(_g(s3_model, key="gamma_pi", default=1.0)),
        "--gamma_alpha",
        str(_g(s3_model, key="gamma_alpha", default=1.0)),
        "--tau0_view",
        str(_g(s3_model, key="tau0_view", default=0.0)),
        "--lambda_alpha",
        str(_g(s3_model, key="lambda_alpha", default=1.0)),
        "--lambda_sens",
        str(_g(s3_model, key="lambda_sens", default=1.0)),
        "--lambda_risk",
        str(_g(s3_model, key="lambda_risk", default=1.0)),
        "--device",
        str(_g(s3_sec, s3_model, key="device", default="cpu")),
    ]
    if _g(s3_model, key="delta", default=None) is not None:
        cmd3 += ["--delta", str(_g(s3_model, key="delta", default=None))]
    if _g(s3_sec, key="ckpt", default=None):
        cmd3 += ["--step3_ckpt", _abspath(_g(s3_sec, key="ckpt", default=None))]
    ka_override = _g(s3_sec, key="Ka_override", default=None)
    if ka_override is None:
        ka_override = _g(s3_model, key="Ka", default=None)
    if ka_override is not None:
        cmd3 += ["--Ka", str(int(ka_override))]
    run(cmd3, env=env)

    print("\n[OK] pipeline done")
    print("workdir:", workdir)
    print("eventlist:", eventlist_out)
    print("step1:", step1_out)
    print("theta:", theta_out)
    print("step3:", step3_out)


if __name__ == "__main__":
    main()
