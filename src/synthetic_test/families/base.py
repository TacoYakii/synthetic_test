"""Abstract base class for shape families.

A ``ShapeFamily`` is a synthetic base forecaster: given a ground-truth
value at each ``(time, node)`` cell, it emits a probabilistic forecast as
a length-``Q`` vector of quantile values.

Different families produce systematically different marginal ``(mean, std)``
profiles from the same underlying ground-truth field, which is what
drives the sweep over the dispersion budget ratio and other scenario
characteristics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ShapeFamily(ABC):
    """Base class for synthetic probabilistic base forecasters.

    Subclasses hold their hyperparameters as dataclass fields and
    implement :meth:`quantiles`.

    Attributes:
        name: Short registry key, unique across families.
    """

    name: str

    @abstractmethod
    def quantiles(
        self,
        observed: np.ndarray,
        quantile_levels: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Emit a probabilistic forecast.

        Args:
            observed: Ground-truth values, shape ``(T, N)``.
            quantile_levels: Probability levels in ``(0, 1)``, shape ``(Q,)``.
            rng: RNG for stochastic families (bias noise, sampling).

        Returns:
            Quantile values of the forecast, shape ``(T, N, Q)``, monotone
            non-decreasing along the last axis.
        """
        ...

    def clone_with(self, **overrides) -> "ShapeFamily":
        """Return a copy of this family with selected fields overridden."""
        from dataclasses import replace

        return replace(self, **overrides)


def enforce_monotone(quantiles: np.ndarray) -> np.ndarray:
    """Sort along the last axis to eliminate residual quantile crossing.

    Useful for families whose ppf could be non-monotone due to numerical
    edge effects near the tails.
    """
    return np.sort(quantiles, axis=-1)
