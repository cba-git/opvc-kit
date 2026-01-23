"""opvc.dp_accountant

Lightweight DP accounting helpers.

Important:
This kit simulates federated training in a single process. We therefore cannot
provide a *cryptographically-end-to-end* privacy proof out of the box.

What we CAN do (paper-level engineering):
  - record DP mechanism hyperparameters (clip, sigma, delta)
  - provide conservative epsilon estimates for Gaussian mechanisms under
    repeated composition (client-level releases).

If you need a tighter accountant (RDP / subsampled Gaussian), integrate a
dedicated DP accounting library and keep these helpers as a fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def gaussian_mechanism_eps(sigma: float, delta: float) -> float:
    """Standard (approx) Gaussian mechanism bound.

    Assumes L2 sensitivity is 1 after clipping and noise is N(0, (sigma)^2).
    Uses the classical bound: eps = sqrt(2 log(1.25/delta)) / sigma.
    """
    sigma = float(sigma)
    delta = float(delta)
    if sigma <= 0:
        return float("inf")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt(2.0 * math.log(1.25 / delta)) / sigma


def advanced_compose_eps(eps: float, delta_per_step: float, steps: int, delta_total: float) -> float:
    """Advanced composition (Dwork-Roth style) for repeated mechanisms.

    Args:
      eps: per-step epsilon
      delta_per_step: per-step delta
      steps: number of compositions
      delta_total: target total delta (must be > steps*delta_per_step)

    Returns:
      epsilon_total
    """
    eps = float(eps)
    delta_per_step = float(delta_per_step)
    steps = int(steps)
    delta_total = float(delta_total)
    if steps <= 0:
        return 0.0
    if not (0 < delta_per_step < 1) or not (0 < delta_total < 1):
        raise ValueError("delta must be in (0,1)")
    # allocate the remainder to the advanced-composition term
    rem = delta_total - steps * delta_per_step
    if rem <= 0:
        # fall back to naive composition
        return steps * eps
    return math.sqrt(2.0 * steps * math.log(1.0 / rem)) * eps + steps * eps * (math.exp(eps) - 1.0)


@dataclass
class GaussianDPReport:
    delta: float
    steps: int
    sigma: float
    eps_per_step: float
    eps_total: float
