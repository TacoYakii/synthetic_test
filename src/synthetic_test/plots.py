"""Plotting helpers for the sweep — dose-response of characteristic vs
CRPS gap between reconciliation presets."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def dose_response(
    gap_df: pd.DataFrame,
    characteristic: str = "budget_ratio",
    gap_col: str = "gap",
    ax: plt.Axes | None = None,
    color_by: str | None = "challenger_wins",
    scatter_kw: dict | None = None,
) -> plt.Axes:
    """Scatter of a scenario characteristic vs the pairwise CRPS gap.

    Meant to be run on the output of
    :func:`synthetic_test.sweep.pairwise_gap`. Points below the zero line
    are challenger wins.

    Args:
        gap_df: DataFrame with columns ``characteristic``, ``gap_col``,
            and optionally ``challenger_wins``.
        characteristic: Column to place on the x-axis.
        gap_col: Column with the CRPS gap on the y-axis.
        ax: Optional pre-existing axes.
        color_by: Column used to color-map points; ``None`` uses a single color.
        scatter_kw: Extra kwargs forwarded to ``ax.scatter``.

    Returns:
        The axes for further customization.
    """
    if characteristic not in gap_df.columns:
        raise KeyError(
            f"characteristic {characteristic!r} not in gap_df columns "
            f"{list(gap_df.columns)}"
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    kw = dict(alpha=0.7)
    if scatter_kw:
        kw.update(scatter_kw)

    if color_by and color_by in gap_df.columns:
        wins = gap_df[color_by].astype(bool)
        ax.scatter(
            gap_df.loc[wins, characteristic], gap_df.loc[wins, gap_col],
            color="tab:blue", label="challenger wins", **kw,
        )
        ax.scatter(
            gap_df.loc[~wins, characteristic], gap_df.loc[~wins, gap_col],
            color="tab:red", label="challenger loses", **kw,
        )
        ax.legend()
    else:
        ax.scatter(gap_df[characteristic], gap_df[gap_col], **kw)

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(characteristic)
    ax.set_ylabel(gap_col)
    ax.set_title(f"CRPS gap vs {characteristic}")
    return ax


def win_rate_by_bin(
    gap_df: pd.DataFrame,
    characteristic: str = "budget_ratio",
    n_bins: int = 8,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Discretize the characteristic into ``n_bins`` and plot the challenger
    win rate per bin. Companion to :func:`dose_response` — helpful when
    the response is noisy at the scenario level.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    x = gap_df[characteristic].to_numpy()
    finite = np.isfinite(x)
    if not finite.any():
        raise ValueError(f"no finite values in column {characteristic!r}")
    edges = np.linspace(np.nanmin(x[finite]), np.nanmax(x[finite]), n_bins + 1)
    which = np.digitize(x, edges[1:-1])
    per_bin_rate = []
    per_bin_center = []
    per_bin_count = []
    for k in range(n_bins):
        mask = (which == k) & finite
        if not mask.any():
            continue
        per_bin_rate.append(gap_df.loc[mask, "challenger_wins"].mean())
        per_bin_center.append(0.5 * (edges[k] + edges[k + 1]))
        per_bin_count.append(int(mask.sum()))
    ax.bar(per_bin_center, per_bin_rate, width=(edges[1] - edges[0]) * 0.9, alpha=0.7)
    for c, r, n in zip(per_bin_center, per_bin_rate, per_bin_count):
        ax.text(c, r, f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel(characteristic)
    ax.set_ylabel("challenger win rate")
    ax.set_title(f"win rate vs {characteristic} (bins={n_bins})")
    return ax


__all__ = ["dose_response", "win_rate_by_bin"]
