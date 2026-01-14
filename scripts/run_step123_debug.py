import json, os, time
import torch
from opvc.contracts import Step3Config
from opvc.step3 import run_step3_from_H

def main():
    s1 = os.path.expanduser("~/opvc-lab/artifacts/step1_out_debug.json")
    th = os.path.expanduser("~/opvc-lab/artifacts/step2_theta_global_trained.pt")
    out = os.path.expanduser("~/opvc-lab/artifacts/step123_out_debug.json")

    d = json.load(open(s1, "r"))
    H = torch.tensor(d["H"], dtype=torch.float32)
    pi = torch.tensor(d["pi"], dtype=torch.float32)
    alpha = torch.tensor(d["alpha"], dtype=torch.float32)

    T, da = H.shape
    Kr = int(pi.numel())
    V = int(alpha.numel())

    cfg3 = Step3Config(V=V, T=int(T), da=int(da), Kr=int(Kr), du=8, Ka=3, ds=32, beta_det=0.95)
    theta_global = torch.load(th, map_location="cpu")

    out3 = run_step3_from_H(cfg3, H_seq=H, pi=pi, alpha=alpha, theta_global=theta_global, device="cpu")

    payload = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {"step1_out_debug": s1, "theta_global": th},
        "cfg3": {"V":V,"T":int(T),"da":int(da),"Kr":Kr,"du":8,"Ka":int(cfg3.Ka),"ds":int(cfg3.ds),"beta_det":float(cfg3.beta_det)},
        "outputs": {
            "p_det": out3.p_det,
            "tau_x": out3.tau_x,
            "y_hat": out3.y_hat,
            "I_view": out3.I_view,
            "J_view": out3.J_view,
            "flag_unknown": out3.flag_unknown,
            "e_score": out3.e_score,
            "E_view": out3.E_view,
            "r_view": out3.r_view,
        }
    }

    json.dump(payload, open(out, "w"), indent=2)
    print("[OK] wrote", out)
    print("p_det =", out3.p_det, "y_hat =", out3.y_hat)
    print("tau_x =", out3.tau_x, "J_view =", out3.J_view)
    print("I_view =", out3.I_view, "flag_unknown =", out3.flag_unknown)
    print("score_type =", (out3.E_view or {}).get("score_type"))

if __name__ == "__main__":
    main()
