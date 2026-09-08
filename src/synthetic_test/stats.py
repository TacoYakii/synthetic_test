"""Statistical validation of sweep results.

Three levels of testing, oriented around the project's #1 question
("when does angular reconciliation win?"):

* **Per-scenario:** Diebold-Mariano test on the per-timestep CRPS
  difference series (``challenger − reference``). One p-value per scenario.
  Answers *does the challenger beat the reference on this specific scenario?*

* **Cross-scenario aggregation:** paired Wilcoxon signed-rank test on
  scenario-level gap values, plus a BCa bootstrap CI on the median gap.
  Answers *does the challenger tend to beat the reference across a batch
  of scenarios?*

* **Characteristic effect:** win rate + Wilson CI per characteristic bin,
  a nonparametric Jonckheere-Terpstra monotone trend test, Spearman
  correlation, and OLS regression of gap on characteristic(s) with HC3
  robust standard errors. Answers *what feature of a scenario predicts a
  challenger win?* — the project's primary question.

Utilities return typed result objects; write callers that render these to
tables / plots rather than parsing strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import norm

from synthetic_test.sweep import PresetResult


# =====================================================================
# Level 1: per-scenario Diebold-Mariano
# =====================================================================


@dataclass
class DMResult:
    """Diebold-Mariano test on paired per-timestep CRPS.

    Attributes:
        statistic: DM test statistic (approximately N(0, 1) under H0 of
            equal predictive accuracy).
        p_value: Two-sided asymptotic p-value.
        mean_diff: Mean of ``challenger − reference`` per-timestep CRPS.
        n: Effective sample size.
        h: Forecast horizon lag used in the variance estimator.
    """

    statistic: float
    p_value: float
    mean_diff: float
    n: int
    h: int = 1


def diebold_mariano(
    challenger_per_time: np.ndarray,
    reference_per_time: np.ndarray,
    h: int = 1,
) -> DMResult:
    """Two-sided DM test on per-timestep CRPS differences at horizon ``h``.

    Loss differential ``d_t = L(challenger)_t − L(reference)_t``; the DM
    statistic is ``mean(d) / sqrt(long-run-var(d) / T)`` with a Bartlett-
    Newey-West type variance using window ``h − 1``. Setting ``h = 1``
    reduces the long-run variance to the plain sample variance.

    Args:
        challenger_per_time: 1-D per-timestep loss series for the challenger.
        reference_per_time: 1-D per-timestep loss series for the reference.
        h: Forecast horizon (window for autocovariance sum). ``h = 1`` for
            one-step forecasts.
    """
    d = np.asarray(challenger_per_time, dtype=float) - np.asarray(
        reference_per_time, dtype=float,
    )
    d = d[np.isfinite(d)]
    T = d.size
    if T < 3:
        return DMResult(np.nan, np.nan, float("nan"), T, h)
    dm_mean = float(d.mean())
    gamma_0 = float(np.var(d, ddof=0))
    var_d = gamma_0
    for k in range(1, h):
        gamma_k = float(np.mean((d[k:] - dm_mean) * (d[:-k] - dm_mean)))
        var_d += 2 * (1 - k / h) * gamma_k
    var_d = max(var_d, 1e-30)
    stat = dm_mean / np.sqrt(var_d / T)
    p = 2.0 * (1.0 - norm.cdf(abs(stat)))
    return DMResult(statistic=float(stat), p_value=float(p), mean_diff=dm_mean, n=T, h=h)


def dm_across_scenarios(
    per_cell: dict[tuple[str, str], PresetResult],
    reference_preset: str,
    challenger_preset: str,
    aggregate: str = "mean_across_nodes",
    h: int = 1,
) -> pd.DataFrame:
    """Run per-scenario DM tests and return a summary DataFrame.

    Args:
        per_cell: The ``SweepResult.per_cell`` dict.
        reference_preset: Baseline preset name.
        challenger_preset: Challenger preset name.
        aggregate: How to reduce the ``(T, N)`` CRPS tensor to a 1-D series
            for DM. Options:
                * ``"mean_across_nodes"`` — per-t mean across all nodes.
                * ``"bottom_only"`` — mean across bottom nodes.
        h: Horizon lag for the DM variance estimator.
    """
    rows: list[dict] = []
    scenarios = sorted({sid for (sid, _) in per_cell})
    for sid in scenarios:
        key_r = (sid, reference_preset)
        key_c = (sid, challenger_preset)
        if key_r not in per_cell or key_c not in per_cell:
            continue
        r = per_cell[key_r]
        c = per_cell[key_c]
        if not (r.succeeded and c.succeeded):
            continue
        if aggregate == "mean_across_nodes":
            r_series = r.per_time_per_node_crps.mean(axis=1)
            c_series = c.per_time_per_node_crps.mean(axis=1)
        elif aggregate == "bottom_only":
            # Bottom nodes have positive S diagonal-ish structure; without
            # the hierarchy we take the last N_bottom columns by convention
            # (canonical row ordering places bottom last).
            raise NotImplementedError(
                "bottom_only requires a hierarchy handle — pass it in or use "
                "'mean_across_nodes'."
            )
        else:
            raise ValueError(f"unknown aggregate={aggregate!r}")
        dm = diebold_mariano(c_series, r_series, h=h)
        rows.append(
            {
                "scenario_id": sid,
                "reference": reference_preset,
                "challenger": challenger_preset,
                "dm_statistic": dm.statistic,
                "dm_p_value": dm.p_value,
                "mean_diff": dm.mean_diff,
                "n": dm.n,
            }
        )
    return pd.DataFrame(rows)


# =====================================================================
# Level 2: cross-scenario aggregation
# =====================================================================


@dataclass
class WilcoxonResult:
    """Wilcoxon signed-rank on paired scenario-level gaps."""

    statistic: float
    p_value: float
    median_gap: float
    n_pairs: int
    n_positive: int
    n_negative: int


def wilcoxon_paired(gap: Sequence[float]) -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank test on a gap sample.

    Positive gap = challenger loses; negative = challenger wins. Null H0:
    median gap = 0.
    """
    x = np.asarray(gap, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return WilcoxonResult(np.nan, np.nan, float("nan"), x.size, 0, 0)
    stat, p = sp_stats.wilcoxon(x, zero_method="wilcox", alternative="two-sided")
    return WilcoxonResult(
        statistic=float(stat),
        p_value=float(p),
        median_gap=float(np.median(x)),
        n_pairs=int(x.size),
        n_positive=int((x > 0).sum()),
        n_negative=int((x < 0).sum()),
    )


@dataclass
class BootstrapCI:
    """BCa bootstrap confidence interval for a statistic of the gap."""

    point: float
    lo: float
    hi: float
    alpha: float
    n_boot: int


def bootstrap_median_ci(
    gap: Sequence[float],
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for the median of the gap sample.

    Uses simple percentile bounds. For BCa, add bias-correction based on
    the sample median and jackknife acceleration (omitted here for
    simplicity — percentile CI is usually sufficient for symmetric
    distributions).
    """
    x = np.asarray(gap, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return BootstrapCI(np.nan, np.nan, np.nan, alpha, n_boot)
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return BootstrapCI(point=float(np.median(x)), lo=lo, hi=hi, alpha=alpha, n_boot=n_boot)


# =====================================================================
# Level 3: characteristic effect
# =====================================================================


@dataclass
class BinRateRow:
    left: float
    right: float
    center: float
    n: int
    n_wins: int
    win_rate: float
    ci_lo: float
    ci_hi: float


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def win_rate_by_bin(
    gap_df: pd.DataFrame,
    characteristic: str,
    n_bins: int = 5,
    alpha: float = 0.05,
    gap_col: str = "gap",
) -> pd.DataFrame:
    """Per-characteristic-bin challenger win rate + Wilson score CI."""
    x = gap_df[characteristic].to_numpy(dtype=float)
    g = gap_df[gap_col].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(g)
    if mask.sum() < n_bins:
        return pd.DataFrame(columns=[
            "left", "right", "center", "n", "n_wins", "win_rate", "ci_lo", "ci_hi",
        ])
    x = x[mask]
    g = g[mask]
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    which = np.digitize(x, edges[1:-1])
    out: list[dict] = []
    for k in range(n_bins):
        m = which == k
        n = int(m.sum())
        if n == 0:
            out.append(BinRateRow(
                left=edges[k], right=edges[k + 1],
                center=0.5 * (edges[k] + edges[k + 1]),
                n=0, n_wins=0, win_rate=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
            ).__dict__)
            continue
        n_wins = int((g[m] < 0).sum())
        rate = n_wins / n
        lo, hi = _wilson_ci(n_wins, n, alpha=alpha)
        out.append(BinRateRow(
            left=float(edges[k]), right=float(edges[k + 1]),
            center=0.5 * (float(edges[k]) + float(edges[k + 1])),
            n=n, n_wins=n_wins, win_rate=rate, ci_lo=lo, ci_hi=hi,
        ).__dict__)
    return pd.DataFrame(out)


@dataclass
class TrendResult:
    """Monotone trend / correlation between a characteristic and the gap."""

    spearman_rho: float
    spearman_p: float
    jonckheere_z: float
    jonckheere_p: float
    n: int


def _jonckheere_terpstra(bins_data: list[np.ndarray]) -> tuple[float, float]:
    """Compute normal-approx Jonckheere-Terpstra statistic and two-sided p."""
    k = len(bins_data)
    n = [len(b) for b in bins_data]
    if k < 2 or any(x < 1 for x in n):
        return (float("nan"), float("nan"))
    J = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            for a in bins_data[i]:
                J += float((bins_data[j] > a).sum() + 0.5 * (bins_data[j] == a).sum())
    N = sum(n)
    mean_J = (N**2 - sum(x**2 for x in n)) / 4.0
    var_J = (N**2 * (2 * N + 3) - sum(x**2 * (2 * x + 3) for x in n)) / 72.0
    if var_J <= 0:
        return (float("nan"), float("nan"))
    z = (J - mean_J) / np.sqrt(var_J)
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return (float(z), float(p))


def trend_test(
    gap_df: pd.DataFrame,
    characteristic: str,
    gap_col: str = "gap",
    n_bins: int = 5,
) -> TrendResult:
    """Nonparametric characteristic-effect summary.

    * Spearman rank correlation between characteristic and gap.
    * Jonckheere-Terpstra ordered-alternative test on the ``characteristic``
      bins with the negated ``gap`` (so a negative-going trend in gap =
      increasing challenger wins as characteristic grows).
    """
    x = gap_df[characteristic].to_numpy(dtype=float)
    g = gap_df[gap_col].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(g)
    x = x[mask]
    g = g[mask]
    if x.size < 3:
        return TrendResult(np.nan, np.nan, np.nan, np.nan, x.size)
    rho, p_rho = sp_stats.spearmanr(x, g)
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    which = np.digitize(x, edges[1:-1])
    bins = [(-g)[which == k] for k in range(n_bins)]
    bins = [b for b in bins if b.size > 0]
    z_j, p_j = _jonckheere_terpstra(bins)
    return TrendResult(
        spearman_rho=float(rho), spearman_p=float(p_rho),
        jonckheere_z=float(z_j), jonckheere_p=float(p_j),
        n=int(x.size),
    )


@dataclass
class RegressionRow:
    variable: str
    coef: float
    se: float
    t_stat: float
    p_value: float
    ci_lo: float
    ci_hi: float


def regression_gap_on_chars(
    gap_df: pd.DataFrame,
    characteristics: Sequence[str],
    gap_col: str = "gap",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """OLS ``gap ~ 1 + characteristics`` with HC3 robust standard errors.

    Standalone numpy implementation to avoid a statsmodels dependency.
    Returns coefficient table with variable name, coef, SE, t, p, CI.
    """
    cols = list(characteristics)
    df = gap_df[[gap_col] + cols].dropna()
    if len(df) < len(cols) + 2:
        return pd.DataFrame(columns=[
            "variable", "coef", "se", "t_stat", "p_value", "ci_lo", "ci_hi",
        ])
    y = df[gap_col].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(dtype=float) for c in cols])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    n, p = X.shape
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    h = np.clip(h, 0.0, 1 - 1e-9)
    # HC3 sandwich: S = X' diag(r_i^2 / (1-h_i)^2) X
    w = resid**2 / (1 - h) ** 2
    S = X.T @ (w[:, None] * X)
    cov_hc3 = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov_hc3))
    dof = n - p
    from scipy.stats import t as tdist

    t_stat = beta / np.where(se > 0, se, np.nan)
    p_val = 2 * (1 - tdist.cdf(np.abs(t_stat), df=dof))
    t_crit = tdist.ppf(1 - alpha / 2, df=dof)
    ci_lo = beta - t_crit * se
    ci_hi = beta + t_crit * se
    names = ["(intercept)"] + list(cols)
    return pd.DataFrame(
        [
            {
                "variable": names[i],
                "coef": float(beta[i]),
                "se": float(se[i]),
                "t_stat": float(t_stat[i]),
                "p_value": float(p_val[i]),
                "ci_lo": float(ci_lo[i]),
                "ci_hi": float(ci_hi[i]),
            }
            for i in range(len(names))
        ]
    )


# =====================================================================
# Convenience roll-up
# =====================================================================


@dataclass
class Report:
    """Full statistical summary for one (reference, challenger) comparison."""

    reference: str
    challenger: str
    wilcoxon: WilcoxonResult
    bootstrap: BootstrapCI
    per_characteristic_trend: dict[str, TrendResult]
    per_characteristic_bins: dict[str, pd.DataFrame]
    regression: pd.DataFrame
    dm_by_scenario: pd.DataFrame | None = None

    def as_markdown(self) -> str:
        """Render a compact markdown summary."""
        lines = [
            f"### {self.challenger} vs {self.reference}",
            "",
            f"- Wilcoxon: W={self.wilcoxon.statistic:.3g}, "
            f"p={self.wilcoxon.p_value:.3g}, "
            f"median gap={self.wilcoxon.median_gap:+.4f}, "
            f"n={self.wilcoxon.n_pairs} "
            f"({self.wilcoxon.n_negative} wins / {self.wilcoxon.n_positive} losses)",
            f"- Bootstrap median CI (95%): "
            f"[{self.bootstrap.lo:+.4f}, {self.bootstrap.hi:+.4f}]",
            "",
            "**Trend by characteristic:**",
        ]
        for name, tr in self.per_characteristic_trend.items():
            lines.append(
                f"- {name}: Spearman ρ={tr.spearman_rho:+.3f} (p={tr.spearman_p:.3g}); "
                f"Jonckheere z={tr.jonckheere_z:+.3f} (p={tr.jonckheere_p:.3g})"
            )
        lines.append("")
        lines.append("**Multi-characteristic regression (OLS + HC3):**")
        for _, row in self.regression.iterrows():
            lines.append(
                f"- {row['variable']}: coef={row['coef']:+.4f} "
                f"(SE={row['se']:.4f}, p={row['p_value']:.3g})"
            )
        return "\n".join(lines)


def build_report(
    gap_df: pd.DataFrame,
    characteristics: Sequence[str],
    reference: str,
    challenger: str,
    per_cell: dict[tuple[str, str], PresetResult] | None = None,
    n_bins: int = 5,
) -> Report:
    """Compute the full statistical bundle for a (reference, challenger) pair."""
    wil = wilcoxon_paired(gap_df["gap"])
    boot = bootstrap_median_ci(gap_df["gap"])
    trends: dict[str, TrendResult] = {}
    bins: dict[str, pd.DataFrame] = {}
    for c in characteristics:
        if c not in gap_df.columns:
            continue
        trends[c] = trend_test(gap_df, characteristic=c, n_bins=n_bins)
        bins[c] = win_rate_by_bin(gap_df, characteristic=c, n_bins=n_bins)
    reg = regression_gap_on_chars(gap_df, characteristics=[c for c in characteristics if c in gap_df.columns])
    dm = (
        dm_across_scenarios(per_cell, reference_preset=reference, challenger_preset=challenger)
        if per_cell else None
    )
    return Report(
        reference=reference, challenger=challenger,
        wilcoxon=wil, bootstrap=boot,
        per_characteristic_trend=trends,
        per_characteristic_bins=bins,
        regression=reg,
        dm_by_scenario=dm,
    )


# =====================================================================
# Angular θ vs bottom-horizon hypothesis test
# =====================================================================


@dataclass
class ThetaHorizonResult:
    """Per-scenario θ-vs-horizon summary."""

    scenario_id: str
    preset: str
    spearman_rho: float
    spearman_p: float
    slope_deg_per_horizon: float
    theta_min_deg: float
    theta_max_deg: float
    theta_mean_deg: float
    theta_std_deg: float
    n_bottom: int


def theta_vs_horizon_by_scenario(
    per_cell: dict[tuple[str, str], PresetResult],
    bottom_horizons: np.ndarray,
    preset: str,
) -> pd.DataFrame:
    """Per-scenario correlation between fitted per-node θ and bottom horizon.

    Args:
        per_cell: The ``SweepResult.per_cell`` dict.
        bottom_horizons: 1-D array of length ``num_low`` giving the horizon
            (e.g. ``[1..48]`` for the canonical wind hierarchy) of each
            bottom node.
        preset: Preset name to inspect. Must be a preset that carries a
            per-node θ vector (``angular``; ``shared_angular`` will have
            a constant vector and yield ``NaN`` correlation).
    """
    rows: list[dict] = []
    horizons = np.asarray(bottom_horizons, dtype=float).ravel()
    for (sid, pname), result in per_cell.items():
        if pname != preset:
            continue
        theta = result.fitted_theta_deg
        if theta is None or not result.succeeded:
            continue
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != horizons.size:
            continue
        finite = np.isfinite(theta) & np.isfinite(horizons)
        if finite.sum() < 3 or np.nanstd(theta[finite]) < 1e-9:
            rho, p = float("nan"), float("nan")
            slope = float("nan")
        else:
            rho, p = sp_stats.spearmanr(horizons[finite], theta[finite])
            # OLS slope for a rough magnitude of the effect (deg / horizon-step)
            slope = float(np.polyfit(horizons[finite], theta[finite], 1)[0])
        rows.append(ThetaHorizonResult(
            scenario_id=sid, preset=pname,
            spearman_rho=float(rho), spearman_p=float(p),
            slope_deg_per_horizon=slope,
            theta_min_deg=float(np.nanmin(theta)),
            theta_max_deg=float(np.nanmax(theta)),
            theta_mean_deg=float(np.nanmean(theta)),
            theta_std_deg=float(np.nanstd(theta)),
            n_bottom=int(theta.size),
        ).__dict__)
    return pd.DataFrame(rows)


def theta_vs_horizon_aggregate(scenario_df: pd.DataFrame) -> dict:
    """Aggregate per-scenario θ-horizon correlations across a batch.

    Returns median Spearman ρ, fraction of scenarios with positive ρ
    (win rate for the hypothesis), and a one-sample Wilcoxon on the ρ
    sample vs 0 (are ρ's systematically positive?).
    """
    rho = scenario_df["spearman_rho"].to_numpy(dtype=float)
    rho = rho[np.isfinite(rho)]
    if rho.size < 3:
        return {"n": int(rho.size), "median_rho": float("nan"),
                "frac_positive": float("nan"), "wilcoxon_stat": float("nan"),
                "wilcoxon_p": float("nan")}
    stat, p = sp_stats.wilcoxon(rho, alternative="greater")
    return {
        "n": int(rho.size),
        "median_rho": float(np.median(rho)),
        "frac_positive": float((rho > 0).mean()),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p),
    }


__all__ = [
    "DMResult", "diebold_mariano", "dm_across_scenarios",
    "WilcoxonResult", "wilcoxon_paired",
    "BootstrapCI", "bootstrap_median_ci",
    "BinRateRow", "win_rate_by_bin",
    "TrendResult", "trend_test",
    "RegressionRow", "regression_gap_on_chars",
    "Report", "build_report",
    "ThetaHorizonResult", "theta_vs_horizon_by_scenario", "theta_vs_horizon_aggregate",
]
