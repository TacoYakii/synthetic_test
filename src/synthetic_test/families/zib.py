"""Zero-Inflated Beta shape family — mass at zero + Beta on ``(0, capacity]``.

Motivated by wind-power output distributions: real observations frequently
include exact zeros (turbine off / wind below cut-in) with continuous
production above.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import beta as beta_dist

from synthetic_test.families.base import ShapeFamily, enforce_monotone


@dataclass
class ZIBForecaster(ShapeFamily):
    """Point mass at 0 with weight ``zero_prob``; Beta(a, b) rescaled to
    ``(0, capacity]`` with weight ``1 - zero_prob``.

    Beta shape parameters are chosen so that the *conditional* mean
    (given non-zero) equals ``max(observed + bias, floor) / capacity``
    (clipped to ``(0, 1)``). ``concentration = a + b`` is a fixed
    dispersion knob — higher = tighter around the conditional mean.

    Note:
        Assumes ``observed >= 0``. Use only when the ground-truth field is
        non-negative and bounded above by ``capacity``.

    Attributes:
        bias_std: Std of the mean-zero Gaussian bias.
        zero_prob: Probability mass at exact zero. Must be in ``[0, 1)``.
        concentration: Beta ``a + b``; larger = tighter.
        capacity: Upper bound for the continuous support.
        floor: Minimum conditional mean (kept strictly positive).
    """

    bias_std: float = 0.0
    zero_prob: float = 0.2
    concentration: float = 4.0
    capacity: float = 1.0
    floor: float = 1e-3
    name: str = field(default="zib")

    def __post_init__(self):
        if not (0.0 <= self.zero_prob < 1.0):
            raise ValueError(f"zero_prob must be in [0, 1); got {self.zero_prob}")
        if self.concentration <= 0:
            raise ValueError(f"concentration must be positive; got {self.concentration}")
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive; got {self.capacity}")

    def quantiles(self, observed, quantile_levels, rng):
        target = observed
        if self.bias_std > 0:
            target = target + rng.normal(scale=self.bias_std, size=observed.shape)
        cond_mean = np.clip(target / self.capacity, self.floor, 1.0 - self.floor)
        a = cond_mean * self.concentration
        b = (1.0 - cond_mean) * self.concentration
        q = np.asarray(quantile_levels)
        # For q <= zero_prob → the zero atom; otherwise Beta ppf on rescaled q.
        above = np.clip((q - self.zero_prob) / max(1.0 - self.zero_prob, 1e-12), 0.0, 1.0)
        # Beta.ppf broadcast: (T, N, 1) shape params × (Q,) → (T, N, Q)
        beta_q = beta_dist.ppf(above, a[..., None], b[..., None])
        out = beta_q * self.capacity
        out = np.where(q <= self.zero_prob, 0.0, out)
        return enforce_monotone(out)
