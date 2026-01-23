#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

    # core dims
    ap.add_argument("--du", type=int, default=16)
    ap.add_argument("--Ka", type=int, default=0, help="attack label dim; Ka=1 is binary; Ka>1 multi-label; Ka=0 disables teacher supervised head")

    # federated training
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--num_clients", type=int, default=2)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)

    # DP base params
    ap.add_argument("--Cb", type=float, default=1.0)
    ap.add_argument("--sigma_b0", type=float, default=0.5)

    # DP/view-wise bounds (optional; default values match Step2Config)
    ap.add_argument("--clip_min", type=float, default=0.5)
    ap.add_argument("--clip_max", type=float, default=2.0)
    ap.add_argument("--sigma_min", type=float, default=0.2)
    ap.add_argument("--sigma_max", type=float, default=1.0)

    # view-gradient projection (optional)
    ap.add_argument("--proj_momentum", type=float, default=0.9)
    ap.add_argument("--proj_temp", type=float, default=1.0)
    ap.add_argument("--view_grad_mode", default="approx", choices=["approx", "exact"])

    # distillation weights (optional)
    ap.add_argument("--lambda_asd", type=float, default=1.0)
    ap.add_argument("--lambda_nce", type=float, default=1.0)

    # dynamic temperature (optional)
    ap.add_argument("--tau_min", type=float, default=0.1)
    ap.add_argument("--tau_max", type=float, default=1.0)
    ap.add_argument("--kappa_u", type=float, default=1.0)
    ap.add_argument("--kappa_p", type=float, default=1.0)

    # DP accounting delta (epsilon is logged in theta_pkg)
    ap.add_argument("--dp_delta", type=float, default=1e-5, help="target delta for DP accounting (epsilon is logged in theta_pkg)")

    # optional offline teacher dataset (.pt) + pretrain settings
    ap.add_argument("--teacher_pt", default=None, help="optional offline teacher dataset (.pt). Expected dict with keys {'b','y'}")
    ap.add_argument("--teacher_batch_size", type=int, default=64)

    # optional: offline teacher supervised pretrain
    ap.add_argument("--teacher_epochs", type=int, default=0, help="epochs for offline teacher supervised pretrain (requires labels)")
    ap.add_argument("--teacher_lr", type=float, default=1e-3)

    # optional: offline teacher self-supervised pretrain
    ap.add_argument("--teacher_selfsup_epochs", type=int, default=0, help="epochs for offline teacher self-supervised pretrain")
    ap.add_argument("--teacher_selfsup_lr", type=float, default=1e-3)
    ap.add_argument("--teacher_selfsup_tau", type=float, default=0.2)
    ap.add_argument("--teacher_selfsup_noise_std", type=float, default=0.01)
    ap.add_argument("--teacher_selfsup_dropout", type=float, default=0.1)
    ap.add_argument("--teacher_selfsup_lambda", type=float, default=1.0)
    ap.add_argument("--teacher_selfsup_sup_lambda", type=float, default=0.0, help="if labels exist, weight of supervised BCE inside selfsup pretrain")

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
            s1 = step1_outputs_from_dict(s1d, device=args.device)
            # preserve meta (e.g., meta.host) from step1.jsonl record
            try:
                setattr(s1, "meta", obj.get("meta", {}) or {})
            except Exception:
                pass
            outs.append(s1)

    if not outs:
        raise SystemExit("[ERR] empty step1_jsonl")

    Kr = int(outs[0].pi.numel())

    # optional offline teacher dataset (labeled or unlabeled)
    offline_loader = None
    if args.teacher_pt:
        pkg = torch.load(args.teacher_pt, map_location="cpu")
        if isinstance(pkg, (tuple, list)) and len(pkg) == 2:
            b, y = pkg
        elif isinstance(pkg, dict):
            b = pkg.get("b", pkg.get("B"))
            y = pkg.get("y", pkg.get("Y"))
        else:
            raise SystemExit("[ERR] teacher_pt must be dict or (b,y) tuple")
        if b is None:
            raise SystemExit("[ERR] teacher_pt missing key 'b'")
        b = torch.as_tensor(b, dtype=torch.float32)
        if y is not None:
            y = torch.as_tensor(y, dtype=torch.float32)
            if y.ndim == 1:
                y = y.view(-1, 1)
            if int(args.Ka) <= 0:
                # infer Ka from teacher labels
                args.Ka = int(y.shape[1])
            ds = torch.utils.data.TensorDataset(b, y)
        else:
            # unlabeled dataset: used only for self-supervised teacher pretrain
            ds = torch.utils.data.TensorDataset(b)
        offline_loader = torch.utils.data.DataLoader(ds, batch_size=int(args.teacher_batch_size), shuffle=True)

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
        clip_min=float(args.clip_min),
        clip_max=float(args.clip_max),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        proj_momentum=float(args.proj_momentum),
        proj_temp=float(args.proj_temp),
        lambda_asd=float(args.lambda_asd),
        lambda_nce=float(args.lambda_nce),
        tau_min=float(args.tau_min),
        tau_max=float(args.tau_max),
        kappa_u=float(args.kappa_u),
        kappa_p=float(args.kappa_p),
        view_grad_mode=str(args.view_grad_mode),
    )

    theta_pkg, logs = train_step2_federated(
        cfg=cfg2,
        step1_outs=outs,
        offline_teacher_loader=offline_loader,
        teacher_epochs=int(args.teacher_epochs),
        teacher_lr=float(args.teacher_lr),
        teacher_selfsup_epochs=int(args.teacher_selfsup_epochs),
        teacher_selfsup_lr=float(args.teacher_selfsup_lr),
        teacher_selfsup_tau=float(args.teacher_selfsup_tau),
        teacher_selfsup_noise_std=float(args.teacher_selfsup_noise_std),
        teacher_selfsup_dropout=float(args.teacher_selfsup_dropout),
        teacher_selfsup_lambda=float(args.teacher_selfsup_lambda),
        teacher_selfsup_sup_lambda=float(args.teacher_selfsup_sup_lambda),
        dp_delta=float(args.dp_delta),
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
