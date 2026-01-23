#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from opvc.contracts import Step2Config, Step3Config
from opvc.io import step1_outputs_from_dict
from opvc.step2 import Step2Student
from opvc.step3 import Step3Core
from opvc.step3_train import train_step3_supervised, save_step3_core_ckpt


def load_labels(path: str) -> Dict[str, torch.Tensor]:
    """Load labels from labels.jsonl.

    Supported formats per line:
      - {"sample_id": "...", "y": 0/1}                 (binary)
      - {"sample_id": "...", "y": [0/1, 0/1, ...]}     (multi-hot)
      - {"sample_id": "...", "y": [idx1, idx2, ...]}   (indices)

    For binary training (Ka==1), multi-hot/indices will be auto-collapsed to
    attack vs benign (any-positive => 1).
    """
    out: Dict[str, torch.Tensor] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = str(obj.get("sample_id", ""))
            if not sid:
                continue
            y = obj.get("y")
            if y is None:
                continue
            t = torch.as_tensor(y, dtype=torch.float32)
            out[sid] = t
    if not out:
        raise SystemExit("[ERR] empty labels")
    return out


def to_multi_hot(y: torch.Tensor, Ka: int) -> torch.Tensor:
    Ka = int(Ka)
    # Binary special-case: represent as a single probability/label.
    if Ka == 1:
        # empty list -> benign
        if y.ndim == 1 and y.numel() == 0:
            return torch.zeros((1,), dtype=torch.float32)
        if y.ndim == 0:
            v = float(y.item())
            if v in (0.0, 1.0):
                return torch.tensor([v], dtype=torch.float32)
            # treat non-zero index as attack
            return torch.tensor([1.0 if int(v) != 0 else 0.0], dtype=torch.float32)
        if y.ndim == 1 and y.numel() == 1:
            v = float(y.view(-1)[0].item())
            if v in (0.0, 1.0):
                return torch.tensor([v], dtype=torch.float32)
        # multi-hot -> any positive means attack
        if y.ndim == 1 and y.numel() > 1 and ((y == 0) | (y == 1)).all():
            return torch.tensor([1.0 if float(y.sum().item()) > 0.5 else 0.0], dtype=torch.float32)
        # indices list -> non-empty means attack
        if y.ndim == 1:
            return torch.tensor([1.0 if int(y.numel()) > 0 else 0.0], dtype=torch.float32)
        raise ValueError(f"Unsupported y shape for Ka=1: {tuple(y.shape)}")

    if y.ndim == 0:
        idxs = [int(y.item())]
        mh = torch.zeros((Ka,), dtype=torch.float32)
        for i in idxs:
            if 0 <= i < Ka:
                mh[i] = 1.0
        return mh
    if y.ndim == 1 and y.numel() == Ka and ((y == 0) | (y == 1)).all():
        return y.float()
    # treat as list of indices
    idxs = [int(i) for i in y.view(-1).tolist()]
    mh = torch.zeros((Ka,), dtype=torch.float32)
    for i in idxs:
        if 0 <= i < Ka:
            mh[i] = 1.0
    return mh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1_jsonl", required=True, help="step1.jsonl from scripts/run_step1.py")
    ap.add_argument("--theta", required=True, help="theta_pkg .pt from Step2")
    ap.add_argument("--labels_jsonl", required=True, help="labels.jsonl with sample_id->y")
    ap.add_argument("--out_ckpt", required=True, help="output step3_core.ckpt.pt")
    ap.add_argument("--Ka", type=int, default=None, help="number of classes. If omitted, inferred from first y")
    ap.add_argument("--ds", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_dac", type=float, default=1.0)
    ap.add_argument("--lambda_decouple", type=float, default=1.0)
    ap.add_argument("--lambda_proto", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log_json", default=None)
    args = ap.parse_args()

    labels = load_labels(args.labels_jsonl)
    theta_pkg = torch.load(args.theta, map_location=args.device)
    cfg2 = Step2Config(**theta_pkg.get("cfg2", {}))  # type: ignore[arg-type]

    # build student (da/V are inferred from the first matched Step1 record for correctness)
    student_sd = theta_pkg.get("student_state_dict") or theta_pkg.get("state_dict") or {}
    student = None
    da = None

    # collect (U,y)
    U_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    V = None
    T = None
    Kr = None
    du = int(cfg2.du)

    with open(args.step1_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
            sid = str((meta or {}).get("sample_id", ""))
            if not sid or sid not in labels:
                continue
            s1d = obj.get("step1", obj.get("step1_out", obj))
            s1 = step1_outputs_from_dict(s1d, device=args.device)
            # attach meta to support possible future splits
            setattr(s1, "meta", meta)

            if V is None:
                V = int(s1.alpha.numel())
                T = int(s1.H.shape[0])
                da = int(s1.H.shape[1])
                Kr = int(s1.pi.numel())
                student = Step2Student(da=int(da), cfg=cfg2).to(torch.device(args.device))
                student.reset_view_adapters(V)
                student.load_state_dict({k: v.to(args.device) for k, v in student_sd.items()}, strict=True)
                student.eval()

            assert student is not None
            U = student.forward_uras_from_step1(s1).detach().cpu().float()
            U_list.append(U)

            # infer Ka
            y_raw = labels[sid]
            if args.Ka is None:
                # binary labels (scalar 0/1 or [] for benign)
                if y_raw.ndim == 0:
                    args.Ka = 1
                elif y_raw.ndim == 1 and y_raw.numel() <= 1:
                    args.Ka = 1
                elif y_raw.ndim == 1 and y_raw.numel() > 1 and ((y_raw == 0) | (y_raw == 1)).all():
                    args.Ka = int(y_raw.numel())
                else:
                    # assume indices; need explicit Ka
                    raise SystemExit("[ERR] --Ka is required when labels are class indices")
            y_list.append(to_multi_hot(y_raw, int(args.Ka)))

    if not U_list:
        raise SystemExit("[ERR] no matched samples between step1_jsonl and labels_jsonl")
    assert V is not None and T is not None and Kr is not None and da is not None
    Ka = int(args.Ka)

    U_all = torch.stack(U_list, dim=0)
    Y_all = torch.stack(y_list, dim=0)
    ds = torch.utils.data.TensorDataset(U_all, Y_all)
    loader = torch.utils.data.DataLoader(ds, batch_size=int(args.batch_size), shuffle=True)

    cfg3 = Step3Config(V=int(V), T=int(T), da=int(da), Kr=int(Kr), du=int(du), Ka=int(Ka), ds=int(args.ds))
    core = Step3Core(cfg3)

    logs = train_step3_supervised(
        cfg3=cfg3,
        core=core,
        loader=loader,
        epochs=int(args.epochs),
        lr=float(args.lr),
        lambda_dac=float(args.lambda_dac),
        lambda_decouple=float(args.lambda_decouple),
        lambda_proto=float(args.lambda_proto),
        device=str(args.device),
    )

    save_step3_core_ckpt(args.out_ckpt, cfg3=cfg3, core=core, extra={"n_samples": len(U_list)})
    if args.log_json:
        Path(args.log_json).write_text(json.dumps(logs, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK] saved:", args.out_ckpt)
    print("[OK] logs:", logs)


if __name__ == "__main__":
    main()
