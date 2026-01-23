#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opvc.contracts import Step1Config
from opvc.step1 import Step1Model
from opvc.step1_train import EventlistJsonlDataset, Step1TrainConfig, train_step1_selfsup, save_step1_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eventlist_jsonl", required=True, help="eventlist.jsonl from scripts/build_eventlist.py")
    ap.add_argument("--out_ckpt", required=True, help="output .pt checkpoint")
    ap.add_argument("--d_in", type=int, default=256)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--da", type=int, default=32)
    ap.add_argument("--Kr", type=int, default=4)
    ap.add_argument("--tau_q", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=0.5)
    # training
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--tau_nce", type=float, default=0.2)
    ap.add_argument("--lambda_pi_entropy", type=float, default=0.0)
    ap.add_argument("--lambda_alpha_entropy", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log_json", default=None, help="optional training log output")
    args = ap.parse_args()

    ds = EventlistJsonlDataset(args.eventlist_jsonl)
    # infer V,T from first record
    meta0, E0, t0, delta, T = ds[0]
    V = len(E0)
    cfg1 = Step1Config(
        V=V,
        T=int(T),
        d_in=[int(args.d_in)] * V,
        d=int(args.d),
        da=int(args.da),
        Kr=int(args.Kr),
        tau_q=float(args.tau_q),
        theta=float(args.theta),
    )
    model = Step1Model(cfg1)

    train_cfg = Step1TrainConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        tau_nce=float(args.tau_nce),
        lambda_pi_entropy=float(args.lambda_pi_entropy),
        lambda_alpha_entropy=float(args.lambda_alpha_entropy),
        grad_clip=float(args.grad_clip),
        seed=int(args.seed),
    )

    logs = train_step1_selfsup(model=model, dataset=ds, train_cfg=train_cfg, device=args.device)
    save_step1_ckpt(args.out_ckpt, model=model, train_cfg=train_cfg)

    if args.log_json:
        Path(args.log_json).write_text(json.dumps(logs, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK] saved ckpt:", args.out_ckpt)
    print("[OK] logs:", logs)


if __name__ == "__main__":
    main()
