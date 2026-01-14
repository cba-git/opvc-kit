import os, sys, json, time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
from opvc.contracts import Step2Config
from opvc.step2_align_final_train import train_theta_global_align_final

ART = Path("/home/caa/opvc-lab/artifacts")
STEP1_OUT = ART / "step1_out_from_aligned_opcnt_T24.json"

def _to(x, device="cpu"):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def _safe_set(cfg, k, v):
    if hasattr(cfg, k):
        setattr(cfg, k, v)

def main():
    device = "cpu"
    d = json.loads(STEP1_OUT.read_text(encoding="utf-8"))
    out1 = d.get("out1", d)

    H = _to(out1["H"], device=device)              # [T, da]
    T, dz = H.shape

    pi1 = _to(out1["pi"], device=device).view(-1)  # [Kr]
    Kr = int(pi1.numel())
    pi = pi1.view(1, Kr).repeat(T, 1)              # [T,Kr]

    alpha = out1.get("alpha", None)
    rho = out1.get("rho", None)
    gate = out1.get("gate", None)

    cfg = Step2Config()
    _safe_set(cfg, "Kr", Kr)
    # 训练证明超参（你需要更强再调大）
    _safe_set(cfg, "teacher_epochs", 3)
    _safe_set(cfg, "teacher_lr", 1e-3)
    _safe_set(cfg, "rounds", 2)
    _safe_set(cfg, "num_clients", 4)
    _safe_set(cfg, "local_steps", 1)
    _safe_set(cfg, "lr", 1e-3)
    _safe_set(cfg, "dp_clip", 1.0)
    _safe_set(cfg, "dp_noise", 0.05)

    print(f"[cfg] Kr={Kr} dz={dz} dp_clip={getattr(cfg,'dp_clip',None)} dp_noise={getattr(cfg,'dp_noise',None)}")

    train_batches = [{
        "Z": H,
        "pi": pi,
        "alpha": alpha,
        "rho": rho,
        "gate": gate,
    }]

    theta_pkg, logs = train_theta_global_align_final(cfg, dz=dz, train_batches=train_batches, device=device, seed=0)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_pt = ART / f"step2_theta_global_trained_align_final_{ts}.pt"
    out_log = ART / f"step2_train_log_align_final_{ts}.json"

    torch.save(theta_pkg, out_pt)
    out_log.write_text(json.dumps(logs, indent=2, ensure_ascii=False), encoding="utf-8")

    canonical = ART / "step2_theta_global_trained.pt"
    if canonical.exists():
        bak = ART / f"step2_theta_global_trained.pt.bak_{ts}"
        canonical.replace(bak)
        print(f"[OK] backup old canonical -> {bak}")
    torch.save(theta_pkg, canonical)

    print(f"[OK] wrote {out_pt}")
    print(f"[OK] wrote {out_log}")
    print(f"[OK] updated canonical {canonical}")

if __name__ == "__main__":
    main()
