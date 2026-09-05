"""Helpers to construct :class:`TemporalHierarchy` instances for tests.

Wraps ``windpower_forecasting``'s ``TemporalHierarchy`` — this module only
provides convenience constructors for the small canonical shapes we sweep
over.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from src.models.reconciliation import TemporalHierarchy


def build_summation_matrix(levels: Sequence[int], max_horizon: int) -> np.ndarray:
    """Construct the summation matrix ``S`` for a temporal hierarchy.

    Row order matches
    :meth:`TemporalHierarchy.build_temporal`::

        for L in levels:
            for h in 1..(max_horizon // L):
                row = 1s over bottom horizons [(h-1)L, hL)

    Args:
        levels: Aggregation levels in hours; each must divide ``max_horizon``.
            The bottom level ``L=1`` must be included for ``S`` to be a
            complete "aggregate + bottom" matrix.
        max_horizon: Total number of bottom-level timesteps in the hierarchy.

    Returns:
        Dense summation matrix, shape ``(num_node, num_low)``.
    """
    m = int(max_horizon)
    rows: list[np.ndarray] = []
    for L in levels:
        L = int(L)
        if m % L != 0:
            raise ValueError(f"level {L} must divide max_horizon {m}.")
        num_h = m // L
        for h in range(1, num_h + 1):
            row = np.zeros(m, dtype=float)
            row[(h - 1) * L : h * L] = 1.0
            rows.append(row)
    return np.stack(rows, axis=0)


def build_temporal(levels: Sequence[int], max_horizon: int) -> TemporalHierarchy:
    """Build a canonical temporal ``TemporalHierarchy``.

    Levels are used verbatim in their given order; convention is
    coarsest-first ending in ``1``. ``max_horizon`` is the bottom-level span.
    """
    S = build_summation_matrix(levels, max_horizon)
    return TemporalHierarchy.build_temporal(S=S, levels=list(levels), max_horizon=int(max_horizon))


def canonical_wind() -> TemporalHierarchy:
    """Return the 124-node canonical wind-power temporal hierarchy.

    Levels ``[48, 24, 16, 12, 8, 6, 4, 3, 2, 1]``, ``max_horizon = 48``.
    Matches the shape used in the ``windpower_forecasting`` paper experiments.
    """
    return build_temporal(levels=[48, 24, 16, 12, 8, 6, 4, 3, 2, 1], max_horizon=48)


def small_toy() -> TemporalHierarchy:
    """Tiny hierarchy for smoke tests. Levels ``[4, 2, 1]``, ``max_horizon = 4``
    (num_node=7, num_low=4)."""
    return build_temporal(levels=[4, 2, 1], max_horizon=4)


def ancestors(hierarchy: TemporalHierarchy, bottom_index: int) -> np.ndarray:
    """Return row indices in ``S`` that are ancestors (or self) of a bottom node.

    A node ``n`` is an ancestor of bottom node ``i`` iff ``S[n, i] > 0``.
    """
    return np.where(hierarchy.S[:, bottom_index] > 0)[0]


__all__ = [
    "TemporalHierarchy",
    "build_summation_matrix",
    "build_temporal",
    "canonical_wind",
    "small_toy",
    "ancestors",
]
