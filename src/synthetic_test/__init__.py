"""synthetic_test — synthetic experiment harness for reconciliation research.

Primary question: **under which scenario conditions does angular reconciliation
outperform analytical (weighted / QA / linear-pool) baselines?**

Scope: this package generates synthetic hierarchical probabilistic forecasts,
computes per-scenario characteristics (leading candidate: dispersion budget
ratio), sweeps reconciliation presets from the sibling
``windpower_forecasting`` package, and produces analysis of *when* angular
wins. It does **not** implement or customize reconciliation itself.
"""

from synthetic_test import (
    calibration,
    characteristics,
    data_gen,
    families,
    hierarchy,
    mixture,
    plots,
    stats,
    sweep,
)

__all__ = [
    "calibration",
    "characteristics",
    "data_gen",
    "families",
    "hierarchy",
    "mixture",
    "plots",
    "stats",
    "sweep",
]
