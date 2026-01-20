#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from opvc.contracts import Step2Config
from opvc.io import step1_outputs_from_dict
from opvc.step2 import train_step2_federated

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1_jsonl", required=True, help="output of scripts/run_step1.py")
    ap.add_argument("--out_pt", required=True, help="theta_pkg .pt")
    ap.add_argument("--du", type=int, default=16)
    ap.add_argument("--Ka", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--num_clients", type=int, default=2)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--Cb", type=float, default=1.0)
    ap.add_argument("--sigma_b0", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # load step1_outs
    outs = []
    with open(args.step1_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            s1d = obj.get("step1", obj.get("step1_out", obj))
            outs.append(step1_outputs_from_dict(s1d, device=args.device))

    if not outs:
        raise SystemExit("[ERR] empty step1_jsonl")

    Kr = int(outs[0].pi.numel())

    cfg2 = Step2Config(
        Kr=Kr,
        du=int(args.du),
        Ka=int(args.Ka),
        Cb=float(args.Cb),
        sigma_b0=float(args.sigma_b0),
        rounds=int(args.rounds),
        num_clients=int(args.num_clients),
        local_epochs=int(args.local_epochs),
        lr=float(args.lr),
    )

    theta_pkg, logs = train_step2_federated(
        cfg=cfg2,
        step1_outs=outs,
        seed=int(args.seed),
        device=str(args.device),
    )

    outp = Path(args.out_pt)
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(theta_pkg, str(outp))

    # also save logs
    logp = outp.with_suffix(outp.suffix + ".log.json")
    logp.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] saved:", outp)
    print("[OK] logs:", logp)

if __name__ == "__main__":
    main()
