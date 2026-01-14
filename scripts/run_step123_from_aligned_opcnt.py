import json, os, time
import dataclasses
import torch

from opvc.contracts import Step1Config, Step3Config
from opvc.step1 import Step1Model
from opvc.step3 import run_step3_from_H

def to_py(x):
    # dataclass -> dict
    if dataclasses.is_dataclass(x):
        return {k: to_py(v) for k, v in dataclasses.asdict(x).items()}
    # torch.Tensor -> list/number
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    # common containers
    if isinstance(x, dict):
        return {k: to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]
    # primitives
    return x

def main():
    INP = os.path.expanduser("~/opvc-lab/artifacts/step1_in_from_aligned_opcnt_T24.json")
    TH  = os.path.expanduser("~/opvc-lab/artifacts/step2_theta_global_trained.pt")
    O1  = os.path.expanduser("~/opvc-lab/artifacts/step1_out_from_aligned_opcnt_T24.json")
    OALL= os.path.expanduser("~/opvc-lab/artifacts/step123_out_from_aligned_opcnt_T24.json")

    d = json.load(open(INP, "r"))
    X_views = d["X_views"]                 # [V,T,num_ops]
    meta = d["meta"]
    V = int(meta["V"]); T = int(meta["T"]); Din = int(meta["num_ops"])

    # Build E: list of tensors [T,Din] per view
    E = [torch.tensor(X_views[v], dtype=torch.float32) for v in range(V)]

    # Step1 config (use your default dims consistent with your debug setup)
    cfg1 = Step1Config(V=V, T=T, d_in=[Din]*V, d=16, da=8, Kr=4, tau_q=1.0, theta=0.5)
    m1 = Step1Model(cfg1).eval()

    # Use your known alignment time params
    t0 = 1568676300.0   # 2019-09-16 23:25:00 UTC in seconds
    delta = 300.0

    out1 = m1.forward(E=E, t0=float(t0), delta=float(delta))

    # Save Step1 output
    payload1 = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "cfg1": to_py(cfg1.__dict__), "out1": to_py(out1.__dict__)}
    json.dump(payload1, open(O1, "w"), indent=2)
    print("[OK] wrote", O1)

    # Extract H/pi/alpha from Step1 outputs (support both dict and attribute styles)
    out1d = out1.__dict__
    H = torch.as_tensor(out1d.get("H"), dtype=torch.float32)          # [T,da]
    pi = torch.as_tensor(out1d.get("pi"), dtype=torch.float32)        # [Kr]
    alpha = torch.as_tensor(out1d.get("alpha"), dtype=torch.float32)  # [V]

    theta_global = torch.load(TH, map_location="cpu")

    cfg3 = Step3Config(V=V, T=int(H.shape[0]), da=int(H.shape[1]), Kr=int(pi.numel()), du=8, Ka=3, ds=32, beta_det=0.95)
    out3 = run_step3_from_H(cfg3, H_seq=H, pi=pi, alpha=alpha, theta_global=theta_global, device="cpu")

    payload = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {"step1_in": INP, "theta_global": TH},
        "cfg1": cfg1.__dict__,
        "cfg3": {"V":V,"T":int(cfg3.T),"da":int(cfg3.da),"Kr":int(cfg3.Kr),"du":8,"Ka":int(cfg3.Ka),"ds":int(cfg3.ds),"beta_det":float(cfg3.beta_det)},
        "step3": {
            "p_det": out3.p_det, "tau_x": out3.tau_x, "y_hat": out3.y_hat,
            "I_view": out3.I_view, "J_view": out3.J_view, "flag_unknown": out3.flag_unknown,
            "e_score": out3.e_score, "E_view": out3.E_view, "r_view": out3.r_view,
        }
    }
    json.dump(payload, open(OALL, "w"), indent=2)
    print("[OK] wrote", OALL)
    print("p_det =", out3.p_det, "y_hat =", out3.y_hat)
    print("tau_x =", out3.tau_x, "J_view =", out3.J_view)
    print("I_view =", out3.I_view, "flag_unknown =", out3.flag_unknown)
    print("score_type =", (out3.E_view or {}).get("score_type"))

if __name__ == "__main__":
    main()
