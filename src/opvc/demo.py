"""
Runnable demo verifying interfaces end-to-end with random tensors.

Run:
  python -m opvc.demo
"""

from opvc.contracts import Step1Config, Step2Config, Step3Config
from opvc.step1 import Step1Model
from opvc.step2 import Step2Model, build_step2_outputs
from opvc.step3 import Step3Model


def main():
    V, T = 4, 8
    d_in = [6, 5, 7, 4]
    d, da = 16, 12
    Kr, du = 3, 10
    Ka, ds = 5, 14

    cfg1 = Step1Config(V=V, T=T, d_in=d_in, d=d, da=da, Kr=Kr, tau_q=1.0, theta=0.2)
    cfg2 = Step2Config(Kr=Kr, du=du)
    cfg3 = Step3Config(V=V, T=T, da=da, Kr=Kr, du=du, Ka=Ka, ds=ds)

    step1 = Step1Model(cfg1)
    step2m = Step2Model(da=da, dz=18, cfg=cfg2)
    step2 = build_step2_outputs(step2m)
    step3 = Step3Model(cfg3)

    E = [[] for _ in range(V)]  # placeholder events

    out1 = step1(E=E, t0=0.0, delta=60.0)
    print("[OK] Step1 outputs:")
    print("  h_aligned:", tuple(out1.h_aligned.shape))
    print("  alpha sum:", out1.alpha.sum().item())
    print("  pi sum:", out1.pi.sum().item())
    print("  Z:", tuple(out1.Z.shape))
    print("  H:", tuple(out1.H.shape))
    print("  gate:", out1.gate, "rho:", out1.rho)

    out3 = step3(step1=out1, step2=step2, nu=0.0, risk=0.0)
    print("[OK] Step3 outputs:")
    print("  p_det:", float(out3.p_det.item()))
    print("  tau_x:", float(out3.tau_x.item()))
    print("  y_hat:", tuple(out3.y_hat.shape))
    print("  I_view:", tuple(out3.I_view.shape), "any=", bool(out3.I_view.any().item()))
    print("  J_view:", out3.J_view)
    print("  flag_unknown:", bool(out3.flag_unknown.item()))


if __name__ == "__main__":
    main()
