"""opvc.step3

Step3: Attack-adaptive detection / recognition / localization (SCD + ATC + DAC + QPL)

Aligned to method final:
- SCD: style/content decomposition of URAS
- ATC: detection with client baseline threshold tau_c and sample-adaptive tau_x
- DAC: multi-label recognition + (optional) prototype constraint
- QPL: view-window localization at window level, producing I_view and J_view segments

This module provides a clean inference entry:
    run_step3(cfg3, step1_out, theta_pkg, sensitivity_coeff=...)

The theta_pkg is the serialized artifact produced by Step2 (train_step2_federated).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import Step3Config, Step3Outputs, Step2Config, Step1Outputs
from .step2 import BehaviorFeatureExtractor, Step2Student, Step2Teacher, dp_sanitize_behavior
from .step2_losses import alpha_confidence, pi_uncertainty, utility_signal, risk_signal, mi_risk_estimate_from_logits
from .utils import merge_contiguous_segments

Tensor = torch.Tensor


def _load_theta_pkg(theta: Union[str, Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    if isinstance(theta, str):
        pkg = torch.load(theta, map_location=device)
        if not isinstance(pkg, dict):
            raise TypeError(f"theta_pkg must be a dict, got {type(pkg)}")
        return pkg
    if not isinstance(theta, dict):
        raise TypeError(f"theta must be a path or dict, got {type(theta)}")
    return theta


def _load_step3_core_ckpt(core_ckpt: Union[str, Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    if isinstance(core_ckpt, str):
        pkg = torch.load(core_ckpt, map_location=device)
        if not isinstance(pkg, dict):
            raise TypeError(f"step3_core_ckpt must be a dict, got {type(pkg)}")
        return pkg
    if not isinstance(core_ckpt, dict):
        raise TypeError(f"core_ckpt must be a path or dict, got {type(core_ckpt)}")
    return core_ckpt


def _infer_da_from_state_dict(sd: Dict[str, Tensor]) -> int:
    """Infer da (input aligned dim) from saved student state_dict."""
    # backbone first linear: weight shape [hidden, da]
    for k, v in sd.items():
        if "backbone" in k and isinstance(v, torch.Tensor) and v.ndim == 2:
            return int(v.shape[1])
    # fallback: any 2D weight assuming [out,in]
    for v in sd.values():
        if isinstance(v, torch.Tensor) and v.ndim == 2:
            return int(v.shape[1])
    # last fallback
    v0 = next(iter(sd.values()))
    return int(v0.numel())


def _calibrate_tau_c(benign_scores: Tensor, q_c: float) -> Tensor:
    if benign_scores.numel() == 0:
        return torch.tensor(0.0, device=benign_scores.device)
    q = float(q_c)
    q = min(1.0, max(0.0, q))
    return torch.quantile(benign_scores, q)


class Step3Core(nn.Module):
    def __init__(self, cfg3: Step3Config):
        super().__init__()
        self.cfg3 = cfg3
        Du = int(cfg3.Kr) * int(cfg3.du)

        # SCD projections
        self.W_c = nn.Linear(Du, cfg3.ds)
        self.W_s = nn.Linear(Du, cfg3.ds)

        # ATC detection head
        self.det_head = nn.Linear(cfg3.ds, 1)

        # DAC classification head
        self.cls_head = nn.Linear(cfg3.ds, cfg3.Ka)

        # QPL evidence projection
        self.W_h = nn.Linear(cfg3.da, cfg3.ds)

        # prototypes (optional training use)
        self.prototypes = nn.Parameter(torch.randn(cfg3.Ka, cfg3.ds) * 0.01)

    def scd(self, U: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (z_content, z_style).

        NOTE: We only apply batch centering when batch size > 1.
        For B==1 (common in inference), centering would collapse z to zeros.
        """
        if U.ndim == 1:
            Ub = U.view(1, -1)
            squeeze = True
        else:
            Ub = U
            squeeze = False
        zc = self.W_c(Ub)
        zs = self.W_s(Ub)
        # Avoid collapse when B==1.
        if Ub.shape[0] > 1:
            zc = zc - zc.mean(dim=0, keepdim=True)
            zs = zs - zs.mean(dim=0, keepdim=True)
        return (zc.squeeze(0), zs.squeeze(0)) if squeeze else (zc, zs)

    def detect_score(self, z_style: Tensor) -> Tensor:
        return self.det_head(z_style).view(())  # scalar

    def detect_prob(self, s_score: Tensor, tau_x: Tensor) -> Tensor:
        return torch.sigmoid(float(self.cfg3.beta_det) * (s_score - tau_x))

    def classify(self, z_style: Tensor) -> Tensor:
        return torch.sigmoid(self.cls_head(z_style))  # [Ka]

    def qpl_scores(self, h_aligned: Tensor, z_style: Tensor) -> Tuple[Tensor, Tensor]:
        """Return e_win [V,T] and style evidence vectors per view [V,ds]."""
        V, T, da = h_aligned.shape
        # project each window evidence into style evidence space
        proj = self.W_h(h_aligned)  # [V,T,ds]
        proj_n = F.normalize(proj, dim=-1)
        z_n = F.normalize(z_style.view(1, 1, -1), dim=-1)  # [1,1,ds]
        e_win = (proj_n * z_n).sum(dim=-1)  # [V,T]
        e_view_vec = proj_n.mean(dim=1)  # [V,ds]
        return e_win, e_view_vec

    def view_contrib(self, e_view_vec: Tensor) -> Tensor:
        """View contribution distribution per class: r_{k,v} = softmax(w_k^T e_v)."""
        # cls weights: [Ka,ds]
        w = self.cls_head.weight  # [Ka,ds]
        scores = torch.einsum("kd,vd->kv", w, e_view_vec)  # [Ka,V]
        return torch.softmax(scores, dim=-1)


@torch.no_grad()
def run_step3(
    cfg3: Step3Config,
    step1_out: Step1Outputs,
    theta_pkg: Union[str, Dict[str, Any]],
    sensitivity_coeff: Optional[Sequence[float]] = None,
    benign_scores_for_calibration: Optional[Tensor] = None,
    core_ckpt: Optional[Union[str, Dict[str, Any]]] = None,
    device: str = "cpu",
) -> Step3Outputs:
    """Main Step3 entry."""
    dev = torch.device(device)
    pkg = _load_theta_pkg(theta_pkg, device=dev)

    cfg2 = Step2Config(**pkg.get("cfg2", {}))  # type: ignore[arg-type]
    student_sd = pkg.get("student_state_dict") or pkg.get("state_dict") or pkg
    teacher_sd = pkg.get("teacher_state_dict", {})

    # infer dims
    da = _infer_da_from_state_dict(student_sd)
    V = int(step1_out.alpha.numel())
    # extract behavior feature dims
    ext_info = pkg.get("extractor", {"V": V, "Kr": cfg2.Kr, "dim": 6 * V + cfg2.Kr + 2})
    extractor = BehaviorFeatureExtractor(V=V, Kr=cfg2.Kr)
    db = int(ext_info.get("dim", extractor.dim))

    # load student / teacher
    student = Step2Student(da=da, cfg=cfg2).to(dev)
    student.reset_view_adapters(V)
    student.load_state_dict({k: v.to(dev) for k, v in student_sd.items()}, strict=True)
    student.eval()

    teacher = Step2Teacher(db=db, dz=da, cfg=cfg2).to(dev)
    if teacher_sd:
        teacher.load_state_dict({k: v.to(dev) for k, v in teacher_sd.items()}, strict=False)
    teacher.eval()

    # build URAS (student + teacher) to compute nu and risk
    U_s = student.forward_uras_from_step1(step1_out).to(dev)  # [Kr*du]
    b = extractor(step1_out).to(dev)
    # for inference we can reuse base DP params
    clip_v = torch.full((V,), float(cfg2.Cb), device=dev)
    sigma_v = torch.full((V,), float(cfg2.sigma_b0), device=dev)
    b_s = dp_sanitize_behavior(b, extractor, clip_v, sigma_v, clip_global=float(cfg2.Cb), sigma_global=float(cfg2.sigma_b0))
    U_t, cls_logits = teacher.forward_uras(b_s.view(1, -1), step1_out.pi.to(dev).view(1, -1))
    U_t = U_t.view(-1)

    # Step3 core model
    core = Step3Core(cfg3).to(dev)
    core.eval()
    if core_ckpt is not None:
        ck = _load_step3_core_ckpt(core_ckpt, device=dev)
        sd = ck.get("state_dict") or ck.get("step3_state_dict")
        if sd is None:
            raise ValueError("core_ckpt missing 'state_dict'")
        core.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=True)

    # SCD
    z_c, z_s = core.scd(U_s)

    # ATC signals
    c_alpha = alpha_confidence(step1_out.alpha.to(dev)).view(())  # scalar
    u_pi = pi_uncertainty(step1_out.pi.to(dev)).view(())          # scalar
    dist_err = torch.linalg.vector_norm((U_s - U_t), ord=2).view(())
    nu = utility_signal(dist_err, c_alpha).view(())

    if sensitivity_coeff is None:
        sensitivity_coeff = [1.0] * V
    s_v = torch.tensor(list(sensitivity_coeff), device=dev, dtype=torch.float32)
    sens = torch.einsum("v,v->", s_v, step1_out.alpha.to(dev))  # scalar
    mi_est = mi_risk_estimate_from_logits(cls_logits).view(()) if cls_logits is not None else torch.tensor(0.0, device=dev)
    P = risk_signal(sens.view(1), mi_est.view(1)).view(())  # scalar

    # client baselines (if provided, use; else fall back to current => correction zero)
    nu_bar = float(pkg.get("util_mean", float(nu.detach().cpu().item())))
    P_bar = float(pkg.get("risk_mean", float(P.detach().cpu().item())))
    u_pi_bar = float(pkg.get("u_pi_mean", float(u_pi.detach().cpu().item())))
    c_alpha_bar = float(pkg.get("c_alpha_mean", float(c_alpha.detach().cpu().item())))

    # tau_c calibration (if provided)
    if "tau_c" in pkg:
        tau_c = torch.tensor(float(pkg["tau_c"]), device=dev)
    elif benign_scores_for_calibration is not None:
        tau_c = _calibrate_tau_c(benign_scores_for_calibration.to(dev), cfg3.q_c)
    else:
        tau_c = torch.tensor(0.0, device=dev)

    # ATC: tau_x = tau_c + gamma_u*(nu_bar-nu) + gamma_p*(P-P_bar) + gamma_pi*(u_pi-u_pi_bar) + gamma_alpha*(c_alpha_bar-c_alpha)
    tau_x = tau_c         + float(cfg3.gamma_u) * (torch.tensor(nu_bar, device=dev) - nu)         + float(cfg3.gamma_p) * (P - torch.tensor(P_bar, device=dev))         + float(cfg3.gamma_pi) * (u_pi - torch.tensor(u_pi_bar, device=dev))         + float(cfg3.gamma_alpha) * (torch.tensor(c_alpha_bar, device=dev) - c_alpha)

    s_score = core.detect_score(z_s)
    p_det = core.detect_prob(s_score, tau_x)

    # DAC
    y_hat = core.classify(z_s)

    # QPL
    e_win, e_view_vec = core.qpl_scores(step1_out.h_aligned.to(dev), z_s)
    E_view = e_win.mean(dim=1)  # [V]
    alpha = step1_out.alpha.to(dev)
    alpha_bar = torch.full_like(alpha, 1.0 / float(max(V, 1)))
    tau_v = torch.tensor(float(cfg3.tau0_view), device=dev)         + float(cfg3.lambda_alpha) * (alpha_bar - alpha)         + float(cfg3.lambda_sens) * s_v         + float(cfg3.lambda_risk) * P

    I_view = (E_view > tau_v)

    # window-level segments per view
    J_view: List[List[Tuple[int, int]]] = []
    for v in range(V):
        idxs = (e_win[v] > tau_v[v]).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
        idxs_1based = [int(i) + 1 for i in idxs]
        J_view.append(merge_contiguous_segments(idxs_1based))

    flag_unknown = (p_det > 0.5) & (~I_view.any())

    contrib = core.view_contrib(e_view_vec)

    out = Step3Outputs(
        p_det=p_det.view(()),
        tau_x=tau_x.view(()),
        y_hat=y_hat.view(cfg3.Ka),
        I_view=I_view.to(torch.bool),
        J_view=J_view,
        flag_unknown=flag_unknown.to(torch.bool).view(()),
        s_score=s_score.view(()),
        tau_c=tau_c.view(()),
        E_view=E_view.detach(),
        e_win=e_win.detach(),
        contrib_view=contrib.detach(),
    )
    out.validate(cfg3)
    return out
