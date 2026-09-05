"""LogNormal shape family — positive support, right-skewed."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import lognorm

from synthetic_test.families.base import ShapeFamily


@dataclass
class LogNormalForecaster(ShapeFamily):
    """LogNormal parameterised to match the observed value as its mean.

    For each cell, we set the log-mean ``μ`` so that
    ``E[X] = exp(μ + sigma²/2) = max(observed + bias, floor)``.
    ``sigma`` (log-space std) is fixed; larger ``sigma`` → heavier right tail.

    Attributes:
        bias_std: Std of the mean-zero Gaussian bias added to ``observed``.
        sigma: Log-space std (dispersion knob). Larger = more right-skew.
        floor: Minimum allowed value for the target mean, since LogNormal
            requires strictly positive location. Cells whose ``observed +
            bias`` fall below this are clipped up.
    """

    bias_std: float = 0.0
    sigma: float = 0.3
    floor: float = 1e-3
    name: str = field(default="lognormal")

    def quantiles(self, observed, quantile_levels, rng):
        target_mean = observed
        if self.bias_std > 0:
            target_mean = target_mean + rng.normal(scale=self.bias_std, size=observed.shape)
        target_mean = np.maximum(target_mean, self.floor)
        mu = np.log(target_mean) - 0.5 * self.sigma**2
        # lognorm.ppf(q, s=sigma, scale=exp(mu))
        z = lognorm.ppf(quantile_levels, s=self.sigma)  # (Q,) with scale=1, μ=0
        return np.exp(mu)[..., None] * z
