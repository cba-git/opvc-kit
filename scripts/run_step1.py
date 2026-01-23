#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from opvc.contracts import Step1Config
from opvc.step1 import Step1Model
from opvc.step1_train import load_step1_ckpt
from opvc.host import infer_host_from_eventlist_E
from opvc.utils import to_py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_eventlist", required=True, help="eventlist jsonl from scripts/build_eventlist.py")
    ap.add_argument("--out_step1", required=True, help="step1 outputs jsonl")
    ap.add_argument("--ckpt", default=None, help="optional Step1 checkpoint (.pt). If omitted, uses random init (NOT paper-level).")

    # Only used when --ckpt is not provided
    ap.add_argument("--d_in", type=int, default=256, help="per-view aggregator output dim (will be replicated to V views)")
    ap.add_argument("--d", type=int, default=64, help="encoder hidden dim")
    ap.add_argument("--da", type=int, default=32, help="aligned space dim")
    ap.add_argument("--Kr", type=int, default=4, help="number of routing bases")
    ap.add_argument("--tau_q", type=float, default=1.0, help="quality softmax temperature")
    ap.add_argument("--theta", type=float, default=0.5, help="Pearson gate threshold")

    # (Optional) extra hyper-params (default values match Step1Config)
    ap.add_argument("--q_norm_momentum", type=float, default=0.95, help="running RMS momentum for Q normalization")
    ap.add_argument("--q_norm_eps", type=float, default=1e-6, help="running RMS eps for Q normalization")
    ap.add_argument("--attn_scale", type=float, default=None, help="override attention scale; default sqrt(da)")

    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no_h_aligned", action="store_true", help="do not write h_aligned to jsonl (saves disk)")
    args = ap.parse_args()

    inp = Path(args.in_eventlist)
    outp = Path(args.out_step1)
    outp.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model: Step1Model | None = None
    cfg: Step1Config | None = None
    n = 0

    with inp.open("r", encoding="utf-8") as fin, outp.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            E = obj["E"]
            meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
            V = len(E)
            T = int(obj["T"])
            t0 = float(obj["t0"])
            delta = float(obj["delta"])

            if model is None:
                if args.ckpt:
                    model = load_step1_ckpt(args.ckpt, device=str(device))
                    cfg = model.cfg
                    if cfg.V != V or cfg.T != T:
                        raise ValueError(f"Step1 ckpt cfg (V={cfg.V},T={cfg.T}) does not match input (V={V},T={T})")
                else:
                    cfg = Step1Config(
                        V=V,
                        T=T,
                        d_in=[int(args.d_in)] * V,
                        d=int(args.d),
                        da=int(args.da),
                        Kr=int(args.Kr),
                        tau_q=float(args.tau_q),
                        theta=float(args.theta),
                        q_norm_momentum=float(args.q_norm_momentum),
                        q_norm_eps=float(args.q_norm_eps),
                        attn_scale=None if args.attn_scale is None else float(args.attn_scale),
                    )
                    model = Step1Model(cfg).to(device).eval()
                    print("[WARN] --ckpt not provided: Step1 uses random initialization; results are NOT paper-level.")

            assert model is not None and cfg is not None
            model = model.to(device).eval()

            with torch.no_grad():
                out = model(E, t0=t0, delta=delta)

            # preserve input meta, ensure meta.host exists
            if not isinstance(meta, dict):
                meta = {}
            meta = dict(meta)
            if not meta.get("host"):
                meta["host"] = infer_host_from_eventlist_E(obj.get("E"), default="unknown")
            # Keep an explicit "node" alias (often used by downstream label files).
            if not meta.get("node"):
                meta["node"] = meta.get("host")
            if not meta.get("node_id"):
                meta["node_id"] = meta.get("node")

            rec = {
                "meta": meta,
                "t0": t0,
                "delta": delta,
                "T": T,
                "V": V,
                "cfg": {
                    "d_in": list(cfg.d_in),
                    "d": int(cfg.d),
                    "da": int(cfg.da),
                    "Kr": int(cfg.Kr),
                    "tau_q": float(cfg.tau_q),
                    "theta": float(cfg.theta),
                    "q_norm_momentum": float(getattr(cfg, "q_norm_momentum", 0.95)),
                    "q_norm_eps": float(getattr(cfg, "q_norm_eps", 1e-6)),
                    "attn_scale": getattr(cfg, "attn_scale", None),
                    "ckpt": args.ckpt,
                },
                "step1": {
                    "alpha": to_py(out.alpha),
                    "pi": to_py(out.pi),
                    "B_x": to_py(out.B_x),
                    "Z": to_py(out.Z),
                    "H": to_py(out.H),
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
            if not args.no_h_aligned:
                rec["step1"]["h_aligned"] = to_py(out.h_aligned)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] wrote {n} lines -> {outp}")


if __name__ == "__main__":
    main()
