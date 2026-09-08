"""Gaussian shape family — the canonical baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from synthetic_test import calibration as _cal
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

    @classmethod
    def from_calibration(
        cls,
        error_scale: float,
        mode: _cal.CalibrationSpec = "calibrated",
        overconfident_ratio: float = _cal.DEFAULT_OVERCONFIDENT_RATIO,
        overdispersed_ratio: float = _cal.DEFAULT_OVERDISPERSED_RATIO,
    ) -> "GaussianForecaster":
        """Build a Gaussian forecaster with a declared calibration relationship.

        The actual forecast error has std ``error_scale``; the declared
        forecast std is ``error_scale * ratio(mode)``. Pass ``mode`` as
        a positive float to set the spread ratio directly (natural sweep
        axis).
        """
        r = _cal.ratio(mode, overconfident_ratio, overdispersed_ratio)
        return cls(bias_std=float(error_scale), spread=float(error_scale) * r)
