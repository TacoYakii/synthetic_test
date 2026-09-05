"""Student-t shape family — heavier tails than Gaussian at low ``df``."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import t as student_t

from synthetic_test.families.base import ShapeFamily


@dataclass
class StudentTForecaster(ShapeFamily):
    """Student-t(df) centered at observed + bias, scaled by ``spread``.

    Marginal std is ``spread * sqrt(df / (df - 2))`` for ``df > 2``.

    Attributes:
        bias_std: Std of the mean-zero Gaussian bias.
        spread: Scale parameter of the t distribution (NOT the resulting std).
        df: Degrees of freedom. Must be > 2 for finite variance.
    """

    bias_std: float = 0.0
    spread: float = 1.0
    df: float = 5.0
    name: str = field(default="student_t")

    def __post_init__(self):
        if self.df <= 2:
            raise ValueError(f"df must be > 2 for finite variance; got {self.df}")

    def quantiles(self, observed, quantile_levels, rng):
        loc = observed + rng.normal(scale=self.bias_std, size=observed.shape) if self.bias_std > 0 else observed
        z = student_t.ppf(quantile_levels, df=self.df)  # (Q,)
        return loc[..., None] + self.spread * z
