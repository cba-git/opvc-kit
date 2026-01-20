#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from opvc.contracts import Step1Config
from opvc.step1 import Step1Model

def to_py(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_eventlist", required=True, help="eventlist jsonl from scripts/build_eventlist.py")
    ap.add_argument("--out_step1", required=True, help="step1 outputs jsonl")
    ap.add_argument("--d_in", type=int, default=256)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--da", type=int, default=32)
    ap.add_argument("--Kr", type=int, default=4)
    ap.add_argument("--tau_q", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    inp = Path(args.in_eventlist)
    outp = Path(args.out_step1)
    outp.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model = None
    cfg = None
    n = 0

    with inp.open("r", encoding="utf-8") as fin, outp.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            E = obj["E"]
            V = len(E)
            T = int(obj["T"])
            t0 = float(obj["t0"])
            delta = float(obj["delta"])

            if model is None:
                cfg = Step1Config(
                    V=V, T=T,
                    d_in=[int(args.d_in)] * V,
                    d=int(args.d),
                    da=int(args.da),
                    Kr=int(args.Kr),
                    tau_q=float(args.tau_q),
                    theta=float(args.theta),
                )
                model = Step1Model(cfg).to(device).eval()

            with torch.no_grad():
                out = model(E, t0=t0, delta=delta)

            rec = {
                "meta": obj.get("meta", {}),
                "t0": t0,
                "delta": delta,
                "T": T,
                "V": V,
                "cfg": {
                    "d_in": [int(args.d_in)] * V,
                    "d": int(args.d),
                    "da": int(args.da),
                    "Kr": int(args.Kr),
                    "tau_q": float(args.tau_q),
                    "theta": float(args.theta),
                },
                "step1": {
                    "alpha": to_py(out.alpha),
                    "pi": to_py(out.pi),
                    "B_x": to_py(out.B_x),
                    "Z": to_py(out.Z),
                    "H": to_py(out.H),
                    "h_aligned": to_py(out.h_aligned),
                    "gate": bool(out.gate) if out.gate is not None else None,
                    "rho": float(out.rho) if out.rho is not None else None,
                    "metrics": {
                        "q_cov": to_py(out.metrics.q_cov),
                        "q_val": to_py(out.metrics.q_val),
                        "q_cmp": to_py(out.metrics.q_cmp),
                        "q_unq": to_py(out.metrics.q_unq),
                        "q_stb": to_py(out.metrics.q_stb),
                        "Q": to_py(out.metrics.Q),
                        "Q_hat": to_py(out.metrics.Q_hat),
                        "g_view": to_py(out.metrics.g_view),
                        "corr_mat": to_py(out.metrics.corr_mat),
                    },
                },
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] wrote {n} lines -> {outp}")

if __name__ == "__main__":
    main()
