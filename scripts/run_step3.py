#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys, json

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opvc.contracts import Step3Config
from opvc.io import load_json, step1_outputs_from_dict
from opvc.step3 import run_step3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1_json", required=True, help="a single step1 json (or a jsonl file; defaults to first line)")
    ap.add_argument("--theta", required=True, help="theta_pkg .pt from step2")
    ap.add_argument("--out", required=True, help="output json")
    ap.add_argument("--ds", type=int, default=16)
    ap.add_argument("--beta_det", type=float, default=1.0)
    ap.add_argument("--Ka", type=int, default=None, help="override Ka for smoke test when cfg2.Ka=0")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # load step1: accept json or jsonl(first line)
    p = Path(args.step1_json)
    txt = p.read_text(encoding="utf-8").strip()
    if txt.startswith("{"):
        d = json.loads(txt)
    else:
        # jsonl
        first = txt.splitlines()[0]
        d = json.loads(first)

    s1d = d.get("step1", d.get("step1_out", d))
    step1_out = step1_outputs_from_dict(s1d, device=args.device)

    V = int(step1_out.alpha.numel())
    T = int(step1_out.H.shape[0])
    da = int(step1_out.H.shape[1])
    Kr = int(step1_out.pi.numel())

    theta_pkg = torch.load(args.theta, map_location=args.device)
    cfg2 = theta_pkg.get("cfg2", {})
    du = int(cfg2.get("du", 8))
    Ka = int(args.Ka) if args.Ka is not None else int(cfg2.get("Ka", 1))
    Ka = max(1, Ka)

    cfg3 = Step3Config(V=V, T=T, da=da, Kr=Kr, du=du, Ka=Ka, ds=args.ds, beta_det=args.beta_det)
    out = run_step3(cfg3, step1_out=step1_out, theta_pkg=theta_pkg, sensitivity_coeff=[1.0] * V, device=args.device)

    payload = {
        "cfg3": {
            "V": V, "T": T, "da": da, "Kr": Kr, "du": du, "Ka": Ka,
            "ds": int(args.ds), "beta_det": float(args.beta_det),
            "q_c": float(getattr(cfg3, "q_c", 0.95)),
            "gamma_u": float(getattr(cfg3, "gamma_u", 1.0)),
            "gamma_p": float(getattr(cfg3, "gamma_p", 1.0)),
            "gamma_pi": float(getattr(cfg3, "gamma_pi", 1.0)),
            "gamma_alpha": float(getattr(cfg3, "gamma_alpha", 1.0)),
            "tau0_view": float(getattr(cfg3, "tau0_view", 0.0)),
            "lambda_alpha": float(getattr(cfg3, "lambda_alpha", 1.0)),
            "lambda_sens": float(getattr(cfg3, "lambda_sens", 1.0)),
            "lambda_risk": float(getattr(cfg3, "lambda_risk", 1.0)),
            "delta": getattr(cfg3, "delta", None),
        },
        "p_det": float(out.p_det.detach().cpu().item()),
        "tau_x": float(out.tau_x.detach().cpu().item()),
        "y_hat": out.y_hat.detach().cpu().view(-1).tolist(),
        "I_view": out.I_view.detach().cpu().tolist(),
        "J_view": out.J_view,
        "flag_unknown": bool(out.flag_unknown.detach().cpu().item()),
        "debug": {
            "s_score": float(out.s_score.detach().cpu().item()),
            "tau_c": float(out.tau_c.detach().cpu().item()),
            "E_view": out.E_view.detach().cpu().tolist(),
        },
    }

    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] wrote", op)

if __name__ == "__main__":
    main()
