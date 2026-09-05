"""Gamma shape family — positive support, right-skewed (lighter tail than LogNormal)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import gamma

from synthetic_test.families.base import ShapeFamily


@dataclass
class GammaForecaster(ShapeFamily):
    """Gamma(shape, rate) with rate chosen so that mean = target.

    For a Gamma with shape ``k`` and rate ``β``: mean = ``k/β``, var = ``k/β²``.
    We fix ``k`` (dispersion knob) and set ``β = k / target_mean``.
    Larger ``shape`` → less right-skew, tighter around mean.

    Attributes:
        bias_std: Std of the mean-zero Gaussian bias.
        shape: Gamma shape parameter ``k``. Must be positive.
        floor: Minimum allowed target mean (Gamma needs strictly positive).
    """

    bias_std: float = 0.0
    shape: float = 2.0
    floor: float = 1e-3
    name: str = field(default="gamma")

    def __post_init__(self):
        if self.shape <= 0:
            raise ValueError(f"shape must be positive; got {self.shape}")

    def quantiles(self, observed, quantile_levels, rng):
        target_mean = observed
        if self.bias_std > 0:
            target_mean = target_mean + rng.normal(scale=self.bias_std, size=observed.shape)
        target_mean = np.maximum(target_mean, self.floor)
        # scale = 1/rate = target_mean / shape
        z = gamma.ppf(quantile_levels, a=self.shape)  # (Q,) with scale=1
        scale = target_mean / self.shape
        return scale[..., None] * z
