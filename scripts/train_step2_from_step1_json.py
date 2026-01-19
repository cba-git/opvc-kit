#!/usr/bin/env python3
"""Train Step2 (federated simulation) from a Step1 debug JSON.

Expected JSON structure:
  - either contains key 'step1_out' (as produced in artifacts/step1_out_debug_full.json)
  - or directly contains keys like h_aligned/alpha/pi/...

This script is for reproducibility / debugging only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opvc.contracts import Step2Config
from opvc.io import load_json, step1_outputs_from_dict
from opvc.step2 import train_step2_federated
from opvc.utils import to_py


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1_json", required=True, help="Path to step1 debug json (must include h_aligned)")
    ap.add_argument("--out", required=True, help="Output theta_pkg .pt path")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--clients", type=int, default=2)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--Kr", type=int, default=3)
    ap.add_argument("--du", type=int, default=8)
    ap.add_argument("--Ka", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = load_json(args.step1_json)
    s1d = d.get("step1_out", d)
    step1_out = step1_outputs_from_dict(s1d, device="cpu")

    cfg2 = Step2Config(Kr=args.Kr, du=args.du, Ka=args.Ka, rounds=args.rounds, num_clients=args.clients, local_epochs=args.local_epochs)
    # replicate the single sample to create a small dataset
    step1_outs = [step1_out for _ in range(max(4, args.clients * 2))]

    theta_pkg, logs = train_step2_federated(cfg2, step1_outs=step1_outs, seed=args.seed, device="cpu")
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(theta_pkg, outp)
    (outp.with_suffix(outp.suffix + ".log.json")).write_text(json.dumps(to_py(logs), indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK] wrote", outp)


if __name__ == "__main__":
    main()
