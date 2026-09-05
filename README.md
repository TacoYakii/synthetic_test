# synthetic_test

Synthetic experiment harness for hierarchical probabilistic forecast reconciliation research.

**Primary question:** under which scenario conditions does angular reconciliation outperform analytical (weighted / QA / linear-pool) baselines?

## Scope

This package generates synthetic hierarchical probabilistic forecasts, computes per-scenario characteristics, sweeps reconciliation presets, and analyzes *when* angular wins. It does **not** implement reconciliation itself — all fit / optimizer / reconciliation logic is delegated to the sibling [`windpower_forecasting`](../windpower_forecasting) package.

The leading characteristic is the **dispersion budget ratio**:

$$
\rho = \frac{V_\mu + V_\sigma}{V_V} = 1 - \frac{V_H}{V_V} \in [0, 1]
$$

The H2 sweep scans this axis and records per-node CRPS across presets.

## Layout

```
src/synthetic_test/
├── data_gen.py          # scenario generation
├── hierarchy.py         # aggregation structures
├── mixture.py           # K-way shape mixing (extensible via families/)
├── families/            # registerable distribution families
├── characteristics.py   # per-scenario measurables (budget ratio, ...)
├── sweep.py             # scenarios × presets runner
├── stats.py
└── plots.py
```

## Install

Requires Python 3.13 and `uv`. The `windpower_forecasting` sibling repo must be checked out at `../windpower_forecasting`.

```bash
uv sync
```

## Presets

Three combiners are compared out of the box (`synthetic_test.sweep.default_presets`):

- `weighted` — QA-style analytical baseline (pcv + weighted quantile average)
- `linear_pool` — vertical / CDF-average combining (θ = 90° angular degenerate)
- `shared_angular` — scalar θ shared across bottom nodes; the paper's main proposal
