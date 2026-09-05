"""Sweep runner — orchestrates scenarios × presets and records per-node CRPS
plus all registered characteristics.

All fit / optimizer decisions are passed through to
``windpower_forecasting``'s ``HierarchicalReconciliation`` unchanged. This
module owns no reconciliation logic — it only feeds inputs and collects
outputs.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from src.models.reconciliation import (
    HierarchicalReconciliation,
    ReconciliationConfig,
    ReconciliationData,
    ReconciliationFitConfig,
)
from src.utils.metrics import crps_quantile

from synthetic_test import characteristics as chars_mod
from synthetic_test.data_gen import DEFAULT_QUANTILE_LEVELS, Scenario


def default_presets(
    quantile_levels: np.ndarray | None = None,
    mc_samples: int = 500,
) -> dict[str, ReconciliationConfig]:
    """Return the three primary combiner comparators used in the paper.

    - ``weighted`` — QA-style analytical baseline (pcv + weighted quantile average)
    - ``linear_pool`` — vertical / CDF-average combining (θ=90° angular degenerate)
    - ``shared_angular`` — scalar θ shared across bottom nodes, the paper's main proposal

    Args:
        quantile_levels: Probability grid the reconciler is configured to
            consume/emit. **Must equal** the grid used by
            :func:`synthetic_test.data_gen.generate_scenario` since
            ``AngularCombine`` expects ``forecast.shape[-1] == n_samples``
            and positions at ``linspace(q_start, q_end, n_samples)``.
            Defaults to :data:`synthetic_test.data_gen.DEFAULT_QUANTILE_LEVELS`.
        mc_samples: MC size for angular combining draws.
    """
    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILE_LEVELS
    n_samples = len(quantile_levels)
    q_start = float(quantile_levels[0])
    q_end = float(quantile_levels[-1])
    common = dict(
        projection_structure="pcv",
        base_arrangement="ranked",
        post_dependence="ranked",
        n_samples=n_samples,
        mc_samples=mc_samples,
        q_start=q_start,
        q_end=q_end,
    )
    return {
        "weighted": ReconciliationConfig(combining_method="weighted", **common),
        "linear_pool": ReconciliationConfig(combining_method="linear_pool", **common),
        "shared_angular": ReconciliationConfig(combining_method="shared_angular", **common),
    }


def split_scenario(scenario: Scenario, val_fraction: float) -> tuple[ReconciliationData, ReconciliationData]:
    """Split a scenario time-wise into (val, test) ReconciliationData."""
    T = scenario.data.forecast.shape[0]
    T_val = max(1, int(round(T * val_fraction)))
    val = ReconciliationData(
        keys=scenario.data.keys[:T_val],
        forecast=scenario.data.forecast[:T_val],
        observed=scenario.data.observed[:T_val],
    )
    test = ReconciliationData(
        keys=scenario.data.keys[T_val:],
        forecast=scenario.data.forecast[T_val:],
        observed=scenario.data.observed[T_val:],
    )
    return val, test


@dataclass
class PresetResult:
    """Outputs of running one preset on one scenario.

    Attributes:
        per_time_per_node_crps: Shape ``(T_test, N)``. Used for DM tests
            and for scenario-level aggregation.
        per_node_mean_crps: Shape ``(N,)``, ``= per_time_per_node_crps.mean(0)``.
        succeeded: False if the fit or reconcile raised; arrays are then NaN.
        error: String rendering of the exception, if any.
    """

    per_time_per_node_crps: np.ndarray
    per_node_mean_crps: np.ndarray
    succeeded: bool = True
    error: str = ""


def evaluate_preset(
    scenario: Scenario,
    config: ReconciliationConfig,
    fit_config: ReconciliationFitConfig | None = None,
    val_fraction: float = 0.5,
) -> PresetResult:
    """Fit on the val split; return per-(t, n) CRPS on the test split.

    The per-time array powers Diebold-Mariano and other paired tests; the
    per-node mean is the standard summary used by the sweep DataFrame.
    """
    val_data, test_data = split_scenario(scenario, val_fraction)
    N = scenario.hierarchy.num_node
    T_test = test_data.forecast.shape[0]

    try:
        model = HierarchicalReconciliation(S=scenario.hierarchy.S, config=config)
        model.fit(val_data, fit_config=fit_config)
        coherent = model.reconcile(test_data)  # (T_test, N, Q_out)
        if coherent.ndim != 3:
            raise ValueError(
                f"reconcile returned ndim={coherent.ndim}; expected probabilistic (T,N,Q)."
            )
        q_val = model.q_val
        if coherent.shape[2] != len(q_val):
            q_val = np.linspace(float(q_val[0]), float(q_val[-1]), coherent.shape[2])
        per_tn = np.empty((T_test, N), dtype=float)
        for n in range(N):
            per_tn[:, n] = crps_quantile(
                q_val, coherent[:, n, :], test_data.observed[:, n], reduction="obs",
            )
        return PresetResult(
            per_time_per_node_crps=per_tn,
            per_node_mean_crps=per_tn.mean(axis=0),
        )
    except Exception as e:  # noqa: BLE001
        nan_tn = np.full((T_test, N), np.nan, dtype=float)
        return PresetResult(
            per_time_per_node_crps=nan_tn,
            per_node_mean_crps=np.full(N, np.nan, dtype=float),
            succeeded=False,
            error=f"{type(e).__name__}: {e}",
        )


@dataclass
class SweepResult:
    """Bundled sweep outputs.

    Attributes:
        long_df: Long-format DataFrame — one row per
            ``(scenario, preset, node)``.
        per_cell: Optional dict ``{(scenario_id, preset_name): PresetResult}``
            preserving the per-(t, n) CRPS tensors. Present only when
            ``store_per_time=True`` on :func:`run_sweep`.
    """

    long_df: pd.DataFrame
    per_cell: dict[tuple[str, str], PresetResult] = field(default_factory=dict)

    def save(self, path: Path | str) -> None:
        """Save the DataFrame as CSV and the per-cell dict as a pickle
        alongside it."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.long_df.to_csv(path / "sweep_long.csv", index=False)
        if self.per_cell:
            with open(path / "per_cell.pkl", "wb") as f:
                pickle.dump(self.per_cell, f)


def run_sweep(
    scenarios: Sequence[Scenario],
    presets: dict[str, ReconciliationConfig],
    fit_config: ReconciliationFitConfig | None = None,
    val_fraction: float = 0.5,
    characteristics_only: list[str] | None = None,
    scenario_ids: Sequence[str] | None = None,
    verbose: bool = False,
    store_per_time: bool = False,
) -> SweepResult:
    """Run ``scenarios × presets`` and return a :class:`SweepResult`.

    Long-format DataFrame columns::

        scenario_id, mixture_label, seed,
        preset,
        node, level, horizon,
        crps,
        <one column per registered characteristic — scenario-broadcast>

    Args:
        scenarios: Synthetic scenarios (produced by ``data_gen``).
        presets: Mapping ``preset_name -> ReconciliationConfig``.
        fit_config: Optional fit config passed through unchanged to
            ``HierarchicalReconciliation.fit``.
        val_fraction: Fraction of the scenario timeline used for fitting.
        characteristics_only: Restrict characteristic computation to a
            subset (defaults to all registered).
        scenario_ids: Optional explicit ids for scenarios; defaults to
            ``["scenario_{k}" for k in range(len(scenarios))]``.
        verbose: Print per-cell status.
        store_per_time: If True, populate ``SweepResult.per_cell`` with
            per-(t, n) CRPS tensors — required for DM tests but memory-heavy
            on large sweeps (``O(scenarios × presets × T × N)``).
    """
    if scenario_ids is None:
        scenario_ids = [f"scenario_{k}" for k in range(len(scenarios))]
    if len(scenario_ids) != len(scenarios):
        raise ValueError("scenario_ids length must match scenarios length")

    rows: list[dict] = []
    per_cell: dict[tuple[str, str], PresetResult] = {}
    for sid, scenario in zip(scenario_ids, scenarios):
        chars = chars_mod.compute_all(
            forecast=scenario.data.forecast,
            observed=scenario.data.observed,
            hierarchy=scenario.hierarchy,
            quantile_levels=scenario.quantile_levels,
            only=characteristics_only,
        )
        for preset_name, config in presets.items():
            if verbose:
                print(f"[sweep] {sid} × {preset_name}", flush=True)
            result = evaluate_preset(
                scenario, config, fit_config=fit_config, val_fraction=val_fraction,
            )
            if verbose and not result.succeeded:
                print(f"  ! failed: {result.error}", flush=True)
            if store_per_time:
                per_cell[(sid, preset_name)] = result
            for n in range(scenario.hierarchy.num_node):
                row = {
                    "scenario_id": sid,
                    "mixture_label": scenario.mixture.label,
                    "seed": scenario.seed,
                    "preset": preset_name,
                    "node": n,
                    "level": int(scenario.hierarchy.levels[n]),
                    "horizon": int(scenario.hierarchy.horizons[n]),
                    "crps": float(result.per_node_mean_crps[n]),
                }
                row.update(chars)
                rows.append(row)

    return SweepResult(long_df=pd.DataFrame(rows), per_cell=per_cell)


def pairwise_gap(
    df: pd.DataFrame,
    reference_preset: str,
    challenger_preset: str,
    aggregate_by: Iterable[str] = ("scenario_id",),
) -> pd.DataFrame:
    """Compute per-scenario (or user-specified aggregation) CRPS gap
    ``challenger - reference``.

    Negative gap means the challenger has *lower* CRPS (i.e. wins).
    Scenario-level characteristics are carried through unchanged since
    they are constant per scenario.
    """
    keys = list(aggregate_by)
    kept_chars = [c for c in df.columns if c in chars_mod.REGISTRY]
    grouped = df.groupby(keys + ["preset"], as_index=False)["crps"].mean()
    wide = grouped.pivot(index=keys, columns="preset", values="crps").reset_index()
    if reference_preset not in wide.columns or challenger_preset not in wide.columns:
        missing = {reference_preset, challenger_preset} - set(wide.columns)
        raise KeyError(f"missing presets in sweep df: {missing}")
    wide["gap"] = wide[challenger_preset] - wide[reference_preset]
    wide["challenger_wins"] = wide["gap"] < 0
    if kept_chars:
        char_vals = df.groupby(keys, as_index=False)[kept_chars].first()
        wide = wide.merge(char_vals, on=keys, how="left")
    return wide


__all__ = [
    "PresetResult",
    "SweepResult",
    "default_presets",
    "split_scenario",
    "evaluate_preset",
    "run_sweep",
    "pairwise_gap",
]
