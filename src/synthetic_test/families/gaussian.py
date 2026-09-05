"""Gaussian shape family — the canonical baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from synthetic_test.families.base import ShapeFamily


@dataclass
class GaussianForecaster(ShapeFamily):
    """Gaussian(observed + bias_noise, spread²).

    Attributes:
        bias_std: Std of the (per-cell) mean-zero Gaussian added to the
            observed value before it becomes the forecast location.
            Zero for a well-calibrated forecaster.
        spread: Std of the forecast distribution itself.
    """

    bias_std: float = 0.0
    spread: float = 1.0
    name: str = field(default="gaussian")

    def quantiles(self, observed, quantile_levels, rng):
        loc = observed + rng.normal(scale=self.bias_std, size=observed.shape) if self.bias_std > 0 else observed
        # ppf broadcasts (T, N, 1) with (Q,) → (T, N, Q)
        z = norm.ppf(quantile_levels)  # (Q,)
        return loc[..., None] + self.spread * z
