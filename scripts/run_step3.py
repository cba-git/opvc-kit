#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opvc.contracts import Step3Config
from opvc.io import step1_outputs_from_dict
from opvc.step3 import run_step3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1_json", required=True, help="a single step1 json (or a jsonl file; defaults to first line)")
    ap.add_argument("--theta", required=True, help="theta_pkg .pt from step2")
    ap.add_argument("--out", required=True, help="output json")

    # core arch
    ap.add_argument("--ds", type=int, default=16)
    ap.add_argument("--beta_det", type=float, default=1.0)
    ap.add_argument("--Ka", type=int, default=None, help="override Ka for smoke test when cfg2.Ka=0")
    ap.add_argument("--step3_ckpt", default=None, help="optional trained Step3Core checkpoint (.pt)")

    # hyper-params (defaults match Step3Config)
    ap.add_argument("--q_c", type=float, default=0.95, help="client quantile for baseline threshold")
    ap.add_argument("--gamma_u", type=float, default=1.0)
    ap.add_argument("--gamma_p", type=float, default=1.0)
    ap.add_argument("--gamma_pi", type=float, default=1.0)
    ap.add_argument("--gamma_alpha", type=float, default=1.0)
    ap.add_argument("--tau0_view", type=float, default=0.0)
    ap.add_argument("--lambda_alpha", type=float, default=1.0)
    ap.add_argument("--lambda_sens", type=float, default=1.0)
    ap.add_argument("--lambda_risk", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=None, help="optional window size (only for time mapping in outputs)")

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

    # Step3 config: if a trained Step3Core ckpt is provided, prefer its architecture (ds/Ka) to avoid shape mismatches.
    if args.step3_ckpt:
        ck = torch.load(args.step3_ckpt, map_location=args.device)
        if isinstance(ck, dict) and "cfg3" in ck:
            cfg3 = Step3Config(**ck["cfg3"])  # type: ignore[arg-type]
            # sanity: must match current Step1/Step2 dims
            if cfg3.V != V or cfg3.T != T or cfg3.da != da or cfg3.Kr != Kr:
                raise SystemExit(
                    f"[ERR] step3_ckpt cfg mismatch: ckpt(V={cfg3.V},T={cfg3.T},da={cfg3.da},Kr={cfg3.Kr}) "
                    f"vs input(V={V},T={T},da={da},Kr={Kr})"
                )
            if cfg3.du != du:
                raise SystemExit(f"[ERR] step3_ckpt cfg mismatch: ckpt.du={cfg3.du} vs step2.du={du}")
            # allow overriding beta_det and Ka via CLI
            cfg3.beta_det = float(args.beta_det)
            if args.Ka is not None:
                cfg3.Ka = int(Ka)
        else:
            cfg3 = Step3Config(V=V, T=T, da=da, Kr=Kr, du=du, Ka=Ka, ds=args.ds, beta_det=args.beta_det)
    else:
        cfg3 = Step3Config(V=V, T=T, da=da, Kr=Kr, du=du, Ka=Ka, ds=args.ds, beta_det=args.beta_det)

    # Apply hyper-parameter overrides (no behavior change if you keep defaults)
    cfg3.q_c = float(args.q_c)
    cfg3.gamma_u = float(args.gamma_u)
    cfg3.gamma_p = float(args.gamma_p)
    cfg3.gamma_pi = float(args.gamma_pi)
    cfg3.gamma_alpha = float(args.gamma_alpha)
    cfg3.tau0_view = float(args.tau0_view)
    cfg3.lambda_alpha = float(args.lambda_alpha)
    cfg3.lambda_sens = float(args.lambda_sens)
    cfg3.lambda_risk = float(args.lambda_risk)

    # carry delta for time mapping (prefer CLI, else reuse step1 record delta)
    if args.delta is not None:
        cfg3.delta = float(args.delta)
    else:
        try:
            cfg3.delta = float(d.get("delta")) if d.get("delta") is not None else None
        except Exception:
            cfg3.delta = None

    out = run_step3(
        cfg3,
        step1_out=step1_out,
        theta_pkg=theta_pkg,
        sensitivity_coeff=[1.0] * V,
        core_ckpt=args.step3_ckpt,
        device=args.device,
    )

    payload = {
        "cfg3": {
            "V": V,
            "T": T,
            "da": da,
            "Kr": Kr,
            "du": du,
            "Ka": int(cfg3.Ka),
            "ds": int(cfg3.ds),
            "beta_det": float(cfg3.beta_det),
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
