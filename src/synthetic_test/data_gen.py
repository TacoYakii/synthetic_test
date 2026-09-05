"""Generate synthetic :class:`ReconciliationData` scenarios.

Pipeline:

1. Draw coherent bottom-level ground truth ``y_bottom`` shape ``(T, m)``
   from a pluggable ``truth_fn``.
2. Project to all hierarchy nodes: ``observed = y_bottom @ S.T`` shape ``(T, N)``.
3. Assign one :class:`ShapeFamily` to each of the ``N`` nodes by sampling
   from a :class:`MixtureSpec`.
4. For each node, that family emits a probabilistic forecast, producing
   ``forecast`` shape ``(T, N, Q)``.
5. Bundle into :class:`ReconciliationData` and a :class:`Scenario` metadata
   record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from src.models.reconciliation import ReconciliationData, TemporalHierarchy

from synthetic_test.families.base import ShapeFamily
from synthetic_test.mixture import MixtureSpec

DEFAULT_QUANTILE_LEVELS = np.linspace(0.005, 0.995, 100)
"""Default probability grid at which synthetic base forecasts are emitted.

Note:
    ``windpower_forecasting``'s ``AngularCombine`` kernel expects the input
    forecast's last-axis length to equal ``ReconciliationConfig.n_samples``
    and its quantile positions to match ``linspace(q_start, q_end, n_samples)``.
    Use :func:`synthetic_test.sweep.default_presets` (or construct configs
    yourself) with the same ``quantile_levels`` you pass to
    :func:`generate_scenario` to keep the grids aligned.
"""


TruthFn = Callable[[TemporalHierarchy, int, np.random.Generator], np.ndarray]
"""Callable ``(hierarchy, T, rng) -> y_bottom[T, num_low]`` producing bottom-
level ground truth."""


def seasonal_truth(
    period: float = 24.0,
    amplitude: float = 2.0,
    baseline: float = 5.0,
    noise_std: float = 0.5,
    non_negative: bool = False,
    per_node_offset_scale: float = 0.5,
) -> TruthFn:
    """Return a ``TruthFn`` producing a seasonal + noise ground truth series.

    Args:
        period: Cycle length in units of ``t`` steps.
        amplitude: Sine amplitude of the seasonal component.
        baseline: Mean level.
        noise_std: Std of the per-cell Gaussian noise.
        non_negative: If True, clip the output to ``max(0, ·)``.
        per_node_offset_scale: Std of a fixed per-bottom-node offset,
            giving different bottom cells slightly different means.
    """

    def _fn(hierarchy: TemporalHierarchy, T: int, rng: np.random.Generator) -> np.ndarray:
        m = hierarchy.num_low
        t = np.arange(T)[:, None]  # (T, 1)
        offsets = rng.normal(scale=per_node_offset_scale, size=(1, m))  # (1, m)
        y = baseline + offsets + amplitude * np.sin(2 * np.pi * t / period)
        y = y + rng.normal(scale=noise_std, size=(T, m))
        if non_negative:
            y = np.maximum(y, 0.0)
        return y

    return _fn


@dataclass
class Scenario:
    """Bundled synthetic experiment inputs and their provenance.

    Attributes:
        data: The :class:`ReconciliationData` consumed by
            ``HierarchicalReconciliation``.
        hierarchy: Hierarchy used to generate this scenario.
        mixture: Mixture spec used to assign families to nodes.
        family_by_node: Length-``N`` list of the actual family instance
            assigned to each node (post-sampling).
        seed: RNG seed used.
        quantile_levels: Probability levels used to emit quantiles.
        y_bottom: Ground-truth bottom-level series ``(T, m)`` (kept for
            characteristic computation and diagnostics).
    """

    data: ReconciliationData
    hierarchy: TemporalHierarchy
    mixture: MixtureSpec
    family_by_node: list[ShapeFamily]
    seed: int
    quantile_levels: np.ndarray
    y_bottom: np.ndarray

    def describe(self) -> dict:
        """Serializable summary (for logging and long-format tables)."""
        return {
            "mixture_label": self.mixture.label,
            "num_node": self.hierarchy.num_node,
            "num_low": self.hierarchy.num_low,
            "T": self.data.forecast.shape[0],
            "Q": self.data.forecast.shape[-1] if self.data.forecast.ndim == 3 else 1,
            "seed": self.seed,
            "family_counts": _count_families(self.family_by_node),
        }


def _count_families(assignment: list[ShapeFamily]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fam in assignment:
        counts[fam.name] = counts.get(fam.name, 0) + 1
    return counts


def generate_scenario(
    hierarchy: TemporalHierarchy,
    mixture: MixtureSpec,
    T: int,
    seed: int,
    truth_fn: TruthFn | None = None,
    quantile_levels: np.ndarray = DEFAULT_QUANTILE_LEVELS,
) -> Scenario:
    """Generate one synthetic experiment scenario.

    Args:
        hierarchy: Temporal hierarchy defining ``S`` and ``num_low``.
        mixture: Distribution over shape families used to assign one family
            per hierarchy node.
        T: Number of timepoints.
        seed: RNG seed. Everything downstream (truth, family assignment,
            forecast bias noise) is derived from this seed.
        truth_fn: Ground-truth generator. Defaults to a Gaussian seasonal
            series suitable for unconstrained families.
        quantile_levels: Probability levels at which to emit the forecast
            quantiles.

    Returns:
        A :class:`Scenario` bundling the reconciliation-ready data and the
        provenance metadata needed by the characteristics/sweep modules.
    """
    if truth_fn is None:
        truth_fn = seasonal_truth()
    rng = np.random.default_rng(seed)

    y_bottom = truth_fn(hierarchy, T, rng)  # (T, m)
    if y_bottom.shape != (T, hierarchy.num_low):
        raise ValueError(
            f"truth_fn returned shape {y_bottom.shape}, expected {(T, hierarchy.num_low)}"
        )

    observed = y_bottom @ hierarchy.S.T  # (T, N)

    fam_assignment = mixture.assign(hierarchy.num_node, rng)

    q_levels = np.asarray(quantile_levels, dtype=float)
    Q = len(q_levels)
    forecast = np.empty((T, hierarchy.num_node, Q), dtype=float)
    for n, fam in enumerate(fam_assignment):
        # Family sees a (T, 1) view of the observed values at node n.
        node_obs = observed[:, n : n + 1]
        f = fam.quantiles(node_obs, q_levels, rng)  # (T, 1, Q)
        forecast[:, n : n + 1, :] = f

    data = ReconciliationData(
        keys=list(range(T)),
        forecast=forecast,
        observed=observed,
    )
    return Scenario(
        data=data,
        hierarchy=hierarchy,
        mixture=mixture,
        family_by_node=fam_assignment,
        seed=seed,
        quantile_levels=q_levels,
        y_bottom=y_bottom,
    )


__all__ = [
    "DEFAULT_QUANTILE_LEVELS",
    "Scenario",
    "TruthFn",
    "seasonal_truth",
    "generate_scenario",
]
