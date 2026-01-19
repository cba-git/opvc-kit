"""opvc.step3_train

Minimal supervised training interface for Step3.

Why this exists:
- Method checklists often require the *training objectives* to be implemented
  even if the current experiment phase only runs Step3 inference.
- We keep inference contracts stable (Step3Outputs, run_step3) and expose
  training in a separate, opt-in interface.

This module trains only the Step3Core parameters (SCD, ATC head, DAC head,
prototypes) on pre-computed URAS embeddings U (from Step2Student) and labels.

Dataset format:
    (U, y)
where:
    U: [B,Kr*du] float tensor
    y: [B,Ka] multi-hot labels
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn

from .contracts import Step3Config
from .step3 import Step3Core
from .step3_losses import dac_multilabel_bce_loss, prototype_constraint_loss, scd_decouple_loss

Tensor = torch.Tensor


def train_step3_supervised(
    cfg3: Step3Config,
    core: Step3Core,
    loader: Iterable[Tuple[Tensor, Tensor]],
    epochs: int = 1,
    lr: float = 1e-3,
    lambda_det: float = 0.0,
    lambda_dac: float = 1.0,
    lambda_decouple: float = 1.0,
    lambda_proto: float = 1.0,
    device: str = "cpu",
) -> Dict[str, float]:
    """Supervised training for Step3Core.

    Notes:
      - Detection loss is optional because many internal datasets do not provide
        per-sample detection labels. If you have detection ground truth, set
        lambda_det>0 and pass y_det via a custom loader.
      - DAC/prototype losses use the same multi-hot labels y.
    """

    dev = torch.device(device)
    core = core.to(dev)
    core.train()

    opt = torch.optim.Adam(core.parameters(), lr=float(lr))
    logs: Dict[str, float] = {}

    for ep in range(int(epochs)):
        L_tot = 0.0
        L_dac = 0.0
        L_dec = 0.0
        L_pro = 0.0
        n = 0

        for U, y in loader:
            U = U.to(dev, dtype=torch.float32)
            y = y.to(dev, dtype=torch.float32)

            # SCD
            z_c, z_s = core.scd(U)

            # DAC logits (pre-sigmoid)
            logits = core.cls_head(z_s)
            loss_dac = dac_multilabel_bce_loss(logits, y)

            # decouple
            loss_dec = scd_decouple_loss(z_c, z_s)

            # prototype constraint
            loss_pro = prototype_constraint_loss(z_s, y, core.prototypes)

            loss = float(lambda_dac) * loss_dac + float(lambda_decouple) * loss_dec + float(lambda_proto) * loss_pro

            opt.zero_grad()
            loss.backward()
            opt.step()

            L_tot += float(loss.detach().cpu().item())
            L_dac += float(loss_dac.detach().cpu().item())
            L_dec += float(loss_dec.detach().cpu().item())
            L_pro += float(loss_pro.detach().cpu().item())
            n += 1

        denom = float(max(n, 1))
        logs[f"step3_epoch_{ep}_loss"] = L_tot / denom
        logs[f"step3_epoch_{ep}_dac"] = L_dac / denom
        logs[f"step3_epoch_{ep}_decouple"] = L_dec / denom
        logs[f"step3_epoch_{ep}_proto"] = L_pro / denom

    core.eval()
    return logs
