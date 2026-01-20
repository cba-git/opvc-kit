"""opvc.demo

End-to-end smoke test for Step1/Step2/Step3.

Run:
  python -m opvc.demo

This uses small synthetic data to verify:
- contracts validate
- Step1 windowing + quality + routing + alignment + gating
- Step2 federated training path produces theta_pkg
- Step3 produces p_det / y_hat / I_view / J_view

This is NOT a benchmark.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

import torch

from .contracts import Step1Config, Step2Config, Step3Config
from .data import HashingAggConfig
from .step1 import Step1Model, ViewAggSpec
from .step2 import train_step2_federated
from .step3 import run_step3


def _make_events(t0: float, delta: float, T: int, n: int, view: int) -> List[Dict[str, Any]]:
    evs: List[Dict[str, Any]] = []
    for i in range(n):
        ts = t0 + random.random() * (delta * T)
        evs.append(
            {
                "ts": ts,
                "parse_ok": True,
                "pid": int(1000 + view * 10 + i),
                "op": random.choice(["read", "write", "exec", "connect"]),
                "path": f"/tmp/v{view}/f{i%3}",
                "dst": random.choice(["10.0.0.1", "10.0.0.2", "8.8.8.8"]),
            }
        )
    return evs


def main() -> None:
    random.seed(0)
    torch.manual_seed(0)

    V, T = 4, 8
    delta = 60.0
    d_in = [32, 32, 32, 32]
    d, da = 16, 12
    Kr, du = 3, 8
    Ka, ds = 5, 16

    cfg1 = Step1Config(V=V, T=T, d_in=d_in, d=d, da=da, Kr=Kr, tau_q=1.0, theta=0.1)

    view_specs = []
    for v in range(V):
        agg_cfg = HashingAggConfig(dim=d_in[v], fields=["op", "path", "dst", "pid"], seed=13 + v)
        view_specs.append(
            ViewAggSpec(
                agg_cfg=agg_cfg,
                key_fields=["op", "path"],
                dedup_fields=["op", "path", "dst"],
            )
        )

    step1 = Step1Model(cfg1, view_specs=view_specs)

    # generate a small dataset of step1_outs
    step1_outs = []
    for s in range(10):
        t0 = float(s) * 10.0
        E = [_make_events(t0, delta, T, n=30 + 5 * v, view=v) for v in range(V)]
        out1 = step1(E=E, t0=t0, delta=delta)
        step1_outs.append(out1)

    print("[OK] Step1 sample:")
    o = step1_outs[0]
    print("  alpha:", [round(x, 4) for x in o.alpha.detach().cpu().tolist()])
    print("  pi:", [round(x, 4) for x in o.pi.detach().cpu().tolist()])
    print("  gate:", o.gate, "rho:", round(float(o.rho or 0.0), 4))
    print("  H:", tuple(o.H.shape), "Z:", tuple(o.Z.shape))

    # Step2 federated train (tiny)
    cfg2 = Step2Config(
        Kr=Kr,
        du=du,
        Ka=Ka,
        rounds=1,
        num_clients=2,
        local_epochs=1,
        lr=1e-3,
        Cb=1.0,
        sigma_b0=0.2,
    )
    theta_pkg, logs = train_step2_federated(cfg2, step1_outs=step1_outs, seed=0, device="cpu")
    print("[OK] Step2 theta_pkg keys:", sorted(list(theta_pkg.keys())))
    print("  logs:", {k: v for k, v in logs.items() if k.startswith("round_")})

    # Step3 run
    cfg3 = Step3Config(V=V, T=T, da=da, Kr=Kr, du=du, Ka=Ka, ds=ds, beta_det=1.0)
    out3 = run_step3(cfg3, step1_out=step1_outs[0], theta_pkg=theta_pkg, sensitivity_coeff=[1.0] * V)

    print("[OK] Step3 outputs:")
    print("  p_det:", round(float(out3.p_det.item()), 4))
    print("  tau_x:", round(float(out3.tau_x.item()), 4))
    print("  y_hat (first 3):", [round(x, 4) for x in out3.y_hat.detach().cpu().tolist()[:3]])
    print("  I_view:", out3.I_view.detach().cpu().tolist())
    print("  J_view:", out3.J_view)
    print("  flag_unknown:", bool(out3.flag_unknown.item()))


if __name__ == "__main__":
    main()
