#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from opvc.contracts import Step3Config
from opvc.io import step1_outputs_from_dict
from opvc.step3 import run_step3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1_jsonl", required=True)
    ap.add_argument("--theta", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--step3_ckpt", default=None)
    ap.add_argument("--ds", type=int, default=16, help="used only if no step3_ckpt is provided")
    ap.add_argument("--beta_det", type=float, default=1.0)
    ap.add_argument("--Ka", type=int, default=None, help="override Ka (smoke test)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_records", type=int, default=None)
    args = ap.parse_args()

    theta_pkg = torch.load(args.theta, map_location=args.device)
    cfg2 = theta_pkg.get("cfg2", {})
    du = int(cfg2.get("du", 8))
    Ka_theta = int(cfg2.get("Ka", 1))

    # if ckpt exists, use its cfg3
    cfg3_from_ckpt = None
    if args.step3_ckpt:
        ck = torch.load(args.step3_ckpt, map_location=args.device)
        if isinstance(ck, dict) and "cfg3" in ck:
            cfg3_from_ckpt = ck["cfg3"]

    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(args.step1_jsonl, "r", encoding="utf-8") as fi, outp.open("w", encoding="utf-8") as fo:
        for line in fi:
            if args.max_records is not None and n >= int(args.max_records):
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
            s1d = obj.get("step1", obj.get("step1_out", obj))
            step1_out = step1_outputs_from_dict(s1d, device=args.device)

            V = int(step1_out.alpha.numel())
            T = int(step1_out.H.shape[0])
            da = int(step1_out.H.shape[1])
            Kr = int(step1_out.pi.numel())

            if cfg3_from_ckpt is not None:
                cfg3 = Step3Config(**cfg3_from_ckpt)  # type: ignore[arg-type]
                cfg3.beta_det = float(args.beta_det)
            else:
                Ka = int(args.Ka) if args.Ka is not None else max(1, Ka_theta)
                cfg3 = Step3Config(V=V, T=T, da=da, Kr=Kr, du=du, Ka=Ka, ds=int(args.ds), beta_det=float(args.beta_det))

            out = run_step3(cfg3, step1_out=step1_out, theta_pkg=theta_pkg, sensitivity_coeff=[1.0] * V, core_ckpt=args.step3_ckpt, device=args.device)

            rec = {
                "meta": meta,
                "t0": obj.get("t0"),
                "delta": obj.get("delta"),
                "p_det": float(out.p_det.detach().cpu().item()),
                "tau_x": float(out.tau_x.detach().cpu().item()),
                "y_hat": out.y_hat.detach().cpu().view(-1).tolist(),
                "I_view": out.I_view.detach().cpu().tolist(),
                "J_view": out.J_view,
                "flag_unknown": bool(out.flag_unknown.detach().cpu().item()),
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] wrote {n} records -> {outp}")


if __name__ == "__main__":
    main()
