"""Per-scenario characteristics — the descriptors that may predict
"angular wins" outcomes.

The primary registered characteristic is the **budget ratio**
``(V_μ + V_σ) / V_V = 1 − V_H / V_V`` (Meng & Taylor / Jeon lineage).
This value is bounded to ``[0, 1]`` and measures *angular headroom* at
each bottom node — i.e. how much room the combining kernel has to
interpolate between horizontal (comonotonic quantile average) and
vertical (linear pool / CDF average).

**Important semantic caveat:** ratio close to 0 makes an angular win
algebraically impossible, but a high ratio does NOT imply an angular
win. It is a necessary-for-relevance measure, not a sufficient one.

Additional characteristics (diversity, hierarchy depth statistics,
base-error correlation, bias regime, …) can be registered via
:func:`register`; the sweep runner records all of them per scenario.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from src.models.reconciliation import TemporalHierarchy


class Characteristic(Protocol):
    """A scalar-valued function of a synthetic scenario.

    Signature:
        (forecast[T, N, Q], observed[T, N], hierarchy, quantile_levels[Q]) -> float

    Must be a pure function (no I/O, no side effects). Return NaN if the
    characteristic is undefined for the given inputs.
    """

    def __call__(
        self,
        forecast: np.ndarray,
        observed: np.ndarray,
        hierarchy: TemporalHierarchy,
        quantile_levels: np.ndarray,
    ) -> float: ...


REGISTRY: dict[str, Characteristic] = {}


def register(name: str) -> Callable[[Characteristic], Characteristic]:
    """Decorator to register a new characteristic under ``name``."""

    def _wrap(fn: Characteristic) -> Characteristic:
        if name in REGISTRY:
            raise ValueError(f"characteristic {name!r} already registered")
        REGISTRY[name] = fn
        return fn

    return _wrap


def compute_all(
    forecast: np.ndarray,
    observed: np.ndarray,
    hierarchy: TemporalHierarchy,
    quantile_levels: np.ndarray,
    only: list[str] | None = None,
) -> dict[str, float]:
    """Run every registered characteristic (or a subset) and return the results."""
    keys = list(REGISTRY) if only is None else only
    out: dict[str, float] = {}
    for k in keys:
        out[k] = float(REGISTRY[k](forecast, observed, hierarchy, quantile_levels))
    return out


# =====================================================================
# Moment estimators from quantile forecasts
# =====================================================================


def marginal_moments(
    quantiles: np.ndarray,
    quantile_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate ``(mean, std)`` of the distribution encoded by
    per-cell quantile vectors.

    Uses the equally-weighted pseudo-sample approximation, treating each
    quantile value as a sample with weight ``1/Q``. This matches
    ``windpower_forecasting``'s ``quantiles_as_samples`` convention and
    is exact for the midpoint-rule inversion of a uniform quantile grid.

    Args:
        quantiles: shape ``(..., Q)``.
        quantile_levels: shape ``(Q,)`` — unused here (kept for interface
            uniformity; future estimators may weight by ``levels`` deltas).

    Returns:
        Tuple of arrays each shape ``quantiles.shape[:-1]``.
    """
    mean = quantiles.mean(axis=-1)
    var = quantiles.var(axis=-1)
    return mean, np.sqrt(var)


# =====================================================================
# Budget ratio
# =====================================================================


@dataclass
class BudgetRatioBreakdown:
    """Per-bottom-node, per-time components of the budget ratio.

    All arrays share shape ``(T, num_low)`` (or the reduction thereof).
    """

    V_mu: np.ndarray
    V_sigma: np.ndarray
    V_H: np.ndarray
    V_V: np.ndarray
    ratio: np.ndarray


def budget_ratio_breakdown(
    forecast: np.ndarray,
    hierarchy: TemporalHierarchy,
    quantile_levels: np.ndarray,
    weights: str | np.ndarray = "uniform",
) -> BudgetRatioBreakdown:
    """Full ``(V_μ, V_σ, V_H, V_V, ratio)`` decomposition per ``(t, bottom_i)``.

    For each bottom node ``i``, its parents (ancestors, including self)
    are the row indices ``j`` where ``S[j, i] > 0``. With weights
    ``p_j`` (default uniform over the parent set), define:

    * ``V_μ = Σ p_j (μ_j - μ̄)² `` — between-parent mean dispersion (weighted)
    * ``V_σ = Σ p_j (σ_j - σ̄)² `` — between-parent std dispersion (weighted)
    * ``V_H = (Σ p_j σ_j)²`` — horizontal (comonotonic) combining variance
    * ``V_V = Σ p_j σ_j² + V_μ`` — vertical (LP / CDF-average) marginal variance
    * ``ratio = (V_μ + V_σ) / V_V = 1 − V_H / V_V``

    Args:
        forecast: shape ``(T, N, Q)``.
        hierarchy: gives ``S`` and ``num_low``.
        quantile_levels: shape ``(Q,)``.
        weights: either ``"uniform"`` (default) or a callable / matrix
            ``P`` of shape ``(num_low, N)`` specifying per-bottom-parent
            weights. Rows must be normalized.

    Returns:
        :class:`BudgetRatioBreakdown` with arrays shape ``(T, num_low)``.
    """
    mean_all, std_all = marginal_moments(forecast, quantile_levels)  # (T, N)
    S = hierarchy.S  # (N, m)
    m = hierarchy.num_low
    T = forecast.shape[0]
    V_mu = np.empty((T, m))
    V_sigma = np.empty((T, m))
    V_H = np.empty((T, m))
    V_V = np.empty((T, m))

    for i in range(m):
        parent_idx = np.where(S[:, i] > 0)[0]
        if parent_idx.size == 0:
            V_mu[:, i] = V_sigma[:, i] = V_H[:, i] = V_V[:, i] = np.nan
            continue
        if isinstance(weights, str) and weights == "uniform":
            p = np.full(parent_idx.size, 1.0 / parent_idx.size)
        elif isinstance(weights, np.ndarray):
            p_row = np.asarray(weights)[i, parent_idx]
            s = p_row.sum()
            if s <= 0:
                V_mu[:, i] = V_sigma[:, i] = V_H[:, i] = V_V[:, i] = np.nan
                continue
            p = p_row / s
        else:
            raise ValueError(f"unsupported weights argument: {weights!r}")

        mu = mean_all[:, parent_idx]  # (T, K)
        sd = std_all[:, parent_idx]  # (T, K)
        mu_bar = (mu * p).sum(axis=1)
        sd_bar = (sd * p).sum(axis=1)
        V_mu[:, i] = (p * (mu - mu_bar[:, None]) ** 2).sum(axis=1)
        V_sigma[:, i] = (p * (sd - sd_bar[:, None]) ** 2).sum(axis=1)
        V_H[:, i] = sd_bar**2
        V_V[:, i] = (p * sd**2).sum(axis=1) + V_mu[:, i]

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(V_V > 0, (V_mu + V_sigma) / V_V, np.nan)
    return BudgetRatioBreakdown(V_mu=V_mu, V_sigma=V_sigma, V_H=V_H, V_V=V_V, ratio=ratio)


@register("budget_ratio")
def budget_ratio(
    forecast: np.ndarray,
    observed: np.ndarray,
    hierarchy: TemporalHierarchy,
    quantile_levels: np.ndarray,
) -> float:
    """Median-over-``(t, bottom_i)`` budget ratio, using uniform parent weights.

    This matches the "validation split median budget ratio" summary used
    in the ``windpower_forecasting`` empirical hint. Returns a scalar in
    ``[0, 1]``. NaN if the breakdown is entirely undefined.
    """
    br = budget_ratio_breakdown(forecast, hierarchy, quantile_levels)
    finite = br.ratio[np.isfinite(br.ratio)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


# =====================================================================
# Diversity between ancestors (paper H3 candidate)
# =====================================================================


@register("ancestor_diversity")
def ancestor_diversity(
    forecast: np.ndarray,
    observed: np.ndarray,
    hierarchy: TemporalHierarchy,
    quantile_levels: np.ndarray,
) -> float:
    """Mean over bottom nodes of ``σ_between / σ_within`` across ancestors.

    Diagnostic candidate for paper H3 — measures how differentiated the
    ancestor forecasts are relative to their intrinsic spread. Larger =
    more diverse pool of parent forecasts.
    """
    mean_all, std_all = marginal_moments(forecast, quantile_levels)  # (T, N)
    S = hierarchy.S
    m = hierarchy.num_low
    per_node: list[float] = []
    for i in range(m):
        parent_idx = np.where(S[:, i] > 0)[0]
        if parent_idx.size < 2:
            continue
        mu = mean_all[:, parent_idx]  # (T, K)
        sigma_within = std_all[:, parent_idx].mean(axis=1)  # (T,)
        sigma_between = mu.std(axis=1)  # (T,)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(sigma_within > 0, sigma_between / sigma_within, np.nan)
        per_node.append(np.nanmean(r))
    if not per_node:
        return float("nan")
    return float(np.nanmean(per_node))


# =====================================================================
# Simple hierarchy characteristic examples
# =====================================================================


@register("mean_parents_per_bottom")
def mean_parents_per_bottom(
    forecast: np.ndarray,
    observed: np.ndarray,
    hierarchy: TemporalHierarchy,
    quantile_levels: np.ndarray,
) -> float:
    """Mean number of ancestors (incl. self) per bottom node."""
    S = hierarchy.S
    counts = (S > 0).sum(axis=0)  # (num_low,)
    return float(counts.mean())


__all__ = [
    "REGISTRY",
    "Characteristic",
    "BudgetRatioBreakdown",
    "budget_ratio_breakdown",
    "marginal_moments",
    "compute_all",
    "register",
]
