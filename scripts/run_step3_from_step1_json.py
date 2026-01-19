#!/usr/bin/env python3
"""Run Step3 inference from Step1 debug JSON and theta_pkg."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opvc.contracts import Step3Config
from opvc.io import load_json, step1_outputs_from_dict
from opvc.step3 import run_step3
from opvc.utils import to_py


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1_json", required=True)
    ap.add_argument("--theta", required=True, help="Path to theta_pkg .pt saved from Step2")
    ap.add_argument("--out", required=True, help="Output json path")
    ap.add_argument("--ds", type=int, default=16)
    ap.add_argument("--beta_det", type=float, default=1.0)
    args = ap.parse_args()

    d = load_json(args.step1_json)
    s1d = d.get("step1_out", d)
    step1_out = step1_outputs_from_dict(s1d, device="cpu")

    V = int(step1_out.alpha.numel())
    T = int(step1_out.H.shape[0])
    da = int(step1_out.H.shape[1])
    Kr = int(step1_out.pi.numel())

    theta_pkg = torch.load(args.theta, map_location="cpu")
    cfg2 = theta_pkg.get("cfg2", {})
    du = int(cfg2.get("du", 8))
    Ka = int(cfg2.get("Ka", 5))

    cfg3 = Step3Config(V=V, T=T, da=da, Kr=Kr, du=du, Ka=Ka, ds=args.ds, beta_det=args.beta_det)
    out = run_step3(cfg3, step1_out=step1_out, theta_pkg=theta_pkg, sensitivity_coeff=[1.0] * V)

    payload = {
        "cfg3": to_py(cfg3),
        "p_det": to_py(out.p_det),
        "tau_x": to_py(out.tau_x),
        "y_hat": to_py(out.y_hat),
        "I_view": to_py(out.I_view),
        "J_view": out.J_view,
        "flag_unknown": to_py(out.flag_unknown),
        "debug": {
            "s_score": to_py(out.s_score),
            "tau_c": to_py(out.tau_c),
            "E_view": to_py(out.E_view),
        },
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(__import__("json").dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK] wrote", op)


if __name__ == "__main__":
    main()
